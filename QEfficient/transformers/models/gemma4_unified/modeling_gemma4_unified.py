# -----------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------
"""QEff support for HF Gemma4 Unified / encoder-free Gemma4 models.

This module intentionally mirrors the existing QEff Gemma4 split:

  vision QPC:
      pixel_values, image_position_ids -> vision_embeds

  language QPC:
      input_ids, vision_embeds, image_idx, mm_token_type_ids, position_ids,
      past_key_values -> logits, retained vision_embeds, retained image_idx,
      retained KV cache

The important difference from previous Gemma4 is vision.  Gemma4 Unified has
no heavy vision tower.  The image path is raw merged patches directly through
`model.model.embed_vision`.
"""

import os
from collections import UserDict, defaultdict
from pathlib import Path
from typing import List, Optional, Type, Union

import numpy as np
import onnx
import torch
import torch.nn as nn
import yaml
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedForCausalLM,
    Gemma4UnifiedForConditionalGeneration,
    Gemma4UnifiedRMSNorm,
    Gemma4UnifiedTextAttention,
    Gemma4UnifiedTextDecoderLayer,
    Gemma4UnifiedTextModel,
    Gemma4UnifiedTextModelOutputWithPast,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from QEfficient.base.onnx_transforms import FP16ClipTransform
from QEfficient.customop.rms_norm import CustomRMSNormFunc
from QEfficient.transformers.cache_utils import QEffGemma4DynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils import constants

_FP16_CLAMP_MIN = -65504.0
_FP16_CLAMP_MAX = 65504.0
_DISABLE_EXPORT_FP16_CLAMP = False


def _is_onnx_export() -> bool:
    return torch.onnx.is_in_onnx_export()


def _clamp_to_fp16_range(hidden_states: torch.Tensor) -> torch.Tensor:
    if not _is_onnx_export() or _DISABLE_EXPORT_FP16_CLAMP:
        return hidden_states
    return hidden_states.clamp(_FP16_CLAMP_MIN, _FP16_CLAMP_MAX)


def _saturating_residual_add(residual: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    if not _is_onnx_export() or _DISABLE_EXPORT_FP16_CLAMP:
        return residual + hidden_states
    return (residual.float() + hidden_states.float()).clamp(_FP16_CLAMP_MIN, _FP16_CLAMP_MAX).to(hidden_states.dtype)


def _build_additive_attention_mask(
    position_ids: torch.Tensor,
    target_length,
    dtype: torch.dtype,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    causal_mask = _create_causal_mask(
        position_ids=position_ids,
        target_length=target_length,
        sliding_window=sliding_window,
    )
    return causal_mask.to(dtype=dtype) * torch.finfo(dtype).min


def _build_bidirectional_vision_attention_mask(
    position_ids: torch.Tensor,
    mm_token_type_ids: Optional[torch.Tensor],
    target_length: int,
    dtype: torch.dtype,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """Export-safe Gemma4 Unified vision-bidirectional attention mask.

    Vision/video tokens inside the same contiguous multimodal block attend
    bidirectionally.  Text tokens remain causal/sliding.
    """
    base_mask = _create_causal_mask(
        position_ids=position_ids,
        target_length=target_length,
        sliding_window=sliding_window,
    )

    if mm_token_type_ids is None:
        return base_mask.to(dtype=dtype) * torch.finfo(dtype).min

    # HF uses multimodal type ids.  Keep image/video as bidirectional blocks.
    # Common convention in your old Gemma4 wrapper:
    #   1 -> image, 2 -> video.  Audio is deliberately left causal initially.
    is_vision = (mm_token_type_ids == 1) | (mm_token_type_ids == 2)

    is_prev_vision = torch.roll(is_vision, shifts=1, dims=-1)
    is_prev_vision[..., 0] = False
    new_vision_starts = is_vision & ~is_prev_vision
    vision_group_ids = torch.cumsum(new_vision_starts.to(torch.int64), dim=1) - 1
    vision_group_ids = torch.where(is_vision, vision_group_ids, torch.full_like(vision_group_ids, -1))

    kv_indices = torch.arange(target_length, device=vision_group_ids.device, dtype=torch.int64).view(1, -1)
    seq_len_limit = torch.full_like(kv_indices, vision_group_ids.shape[1] - 1)
    safe_kv_indices = torch.minimum(kv_indices, seq_len_limit)
    kv_group_ids = torch.gather(vision_group_ids, 1, safe_kv_indices.expand(vision_group_ids.shape[0], -1))
    kv_group_ids = torch.where(kv_indices < vision_group_ids.shape[1], kv_group_ids, torch.full_like(kv_group_ids, -1))

    same_group = (vision_group_ids.unsqueeze(-1) == kv_group_ids.unsqueeze(1)) & (vision_group_ids.unsqueeze(-1) >= 0)
    attention_mask = base_mask & ~same_group.unsqueeze(1)
    return attention_mask.to(dtype=dtype) * torch.finfo(dtype).min


class QEffGemma4UnifiedCustomRMSNormAIC(Gemma4UnifiedRMSNorm):
    """Gemma4 Unified RMSNorm using compiler-known CustomRMSNorm op."""

    def __qeff_init__(self):
        if getattr(self, "with_scale", True) or hasattr(self, "_qeff_unit_weight"):
            return
        weight = getattr(self, "weight", None)
        if weight is not None:
            self.register_buffer("_qeff_unit_weight", torch.ones_like(weight))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not _is_onnx_export():
            normed_output = self._norm(hidden_states.float())
            if getattr(self, "with_scale", True):
                normed_output = normed_output * self.weight.float()
            return normed_output.type_as(hidden_states)

        if getattr(self, "with_scale", True):
            weight = self.weight
        else:
            weight = getattr(self, "_qeff_unit_weight", None)
            if weight is None:
                weight = hidden_states.new_ones(hidden_states.shape[-1])
        return CustomRMSNormFunc.apply(hidden_states, weight, self.eps)


class QEffGemma4UnifiedTextAttention(Gemma4UnifiedTextAttention):
    """AOT-friendly eager attention for Gemma4 Unified text layers."""

    def __qeff_init__(self):
        for norm_name in ("q_norm", "k_norm", "v_norm"):
            norm = getattr(self, norm_name, None)
            if norm is not None and not getattr(norm, "with_scale", True) and not hasattr(norm, "_qeff_unit_weight"):
                norm.register_buffer("_qeff_unit_weight", torch.ones(self.head_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        shared_kv_states: Optional[dict[str, tuple[torch.Tensor, torch.Tensor]]] = None,
        past_key_values: Optional[Cache] = None,
        position_ids: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        if shared_kv_states is None:
            shared_kv_states = {}

        if self.is_kv_shared_layer:
            key_states, value_states = shared_kv_states[self.layer_type]
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
            key_states = key_states.transpose(1, 2)
            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

        if past_key_values is not None and not self.is_kv_shared_layer:
            # QEffGemma4DynamicCache accepts the same basic update pattern used
            # by previous Gemma4.  Keep position_ids in cache_kwargs for the
            # sliding-window path when available.
            try:
                key_states, value_states = past_key_values.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    {"position_ids": position_ids},
                )
            except TypeError:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        if self.store_full_length_kv:
            shared_kv_states[self.layer_type] = key_states, value_states

        if mm_token_type_ids is not None and hidden_states.shape[1] != 1:
            attention_mask = _build_bidirectional_vision_attention_mask(
                position_ids=position_ids,
                mm_token_type_ids=mm_token_type_ids,
                target_length=key_states.shape[-2],
                dtype=query_states.dtype,
                sliding_window=self.sliding_window,
            )

        if attention_mask is not None and attention_mask.dtype == torch.bool:
            attention_mask = (~attention_mask).to(dtype=query_states.dtype) * torch.finfo(query_states.dtype).min

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QEffGemma4UnifiedTextDecoderLayer(Gemma4UnifiedTextDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        shared_kv_states: Optional[dict[str, tuple[torch.Tensor, torch.Tensor]]] = None,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = _clamp_to_fp16_range(hidden_states)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = _saturating_residual_add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = _saturating_residual_add(residual, hidden_states)
        hidden_states *= self.layer_scalar
        return hidden_states


class QEffGemma4UnifiedTextModel(Gemma4UnifiedTextModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and isinstance(past_key_values, Cache) and not isinstance(past_key_values, QEffGemma4DynamicCache):
            past_key_values = QEffGemma4DynamicCache.from_cache(self.config, past_key_values)
        elif use_cache and not isinstance(past_key_values, Cache):
            past_key_values = QEffGemma4DynamicCache.from_legacy_cache(self.config, past_key_values)
        elif use_cache and past_key_values is None:
            past_key_values = QEffGemma4DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        hidden_states = inputs_embeds

        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        shared_kv_states = kwargs.pop("shared_kv_states", UserDict())

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_type = self.config.layer_types[i]
            layer_attention_mask = attention_mask

            use_mm_bidirectional_mask = (
                kwargs.get("mm_token_type_ids") is not None
                and inputs_embeds.shape[1] != 1
                and getattr(self.config, "use_bidirectional_attention", None) == "vision"
            )

            if isinstance(attention_mask, dict):
                layer_attention_mask = attention_mask[layer_type]
            elif use_mm_bidirectional_mask:
                # Let QEff attention rebuild the correct export-safe vision mask
                # after it knows the actual KV length.
                layer_attention_mask = None
            else:
                sliding_window = self.config.sliding_window if layer_type == "sliding_attention" else None
                target_length = inputs_embeds.shape[1]
                if past_key_values is not None and len(past_key_values.layers) > i:
                    layer_keys = past_key_values.layers[i].keys
                    if layer_keys is not None and layer_keys.numel() > 0:
                        target_length = layer_keys.shape[-2]
                layer_attention_mask = _build_additive_attention_mask(
                    position_ids=position_ids,
                    target_length=target_length,
                    dtype=hidden_states.dtype,
                    sliding_window=sliding_window,
                )

            hidden_states = decoder_layer(
                hidden_states,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings[layer_type],
                attention_mask=layer_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        next_cache = past_key_values.to_legacy_cache() if use_cache else None
        output = Gemma4UnifiedTextModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            shared_kv_states=shared_kv_states if kwargs.get("return_shared_kv_states", False) else None,
        )
        return output if return_dict else output.to_tuple()


class QEffGemma4UnifiedForCausalLM(Gemma4UnifiedForCausalLM):
    _NPI_FP32_OPS = {"Cast", "Pow", "ReduceMean", "Add", "Mul", "Div", "Softmax", "Tanh", "Clip"}
    _NPI_ATTENTION_NAMES = ("query_states", "key_states", "value_states", "key", "value")
    _NPI_EXCLUDED_OPS = {
        "Constant",
        "ConstantOfShape",
        "Concat",
        "CustomRMSNorm",
        "Equal",
        "Gather",
        "MatMul",
        "Range",
        "Reshape",
        "Shape",
        "Slice",
        "Transpose",
        "Unsqueeze",
    }

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffGemma4UnifiedTextDecoderLayer}

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        comp_ctx_lengths_prefill: Optional[List[int]] = None,
        comp_ctx_lengths_decode: Optional[List[int]] = None,
        continuous_batching: bool = False,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
        **kwargs,
    ):
        del kwargs
        batch_size = batch_size if batch_size else 1
        prefill_seq_len = prefill_seq_len if prefill_seq_len else 32
        ctx_len = ctx_len if ctx_len else constants.INTERN_CTX_LEN
        kv_cache_batch_size = kv_cache_batch_size or full_batch_size or batch_size

        def build_prefill_spec(comp_ctx_lengths: Optional[int] = None):
            spec = {
                "batch_size": 1 if continuous_batching else batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "sliding_window": self.config.sliding_window,
            }
            if comp_ctx_lengths is not None:
                spec["comp_ctx_lengths"] = comp_ctx_lengths
            if continuous_batching:
                spec["full_batch_size"] = kv_cache_batch_size
            else:
                spec["batch_size"] = kv_cache_batch_size
            if full_batch_size:
                spec["full_batch_exec_size"] = full_batch_size
            return spec

        def build_decode_spec(comp_ctx_lengths: Optional[int] = None):
            spec = {
                "batch_size": full_batch_size if continuous_batching else batch_size,
                "seq_len": "1",
                "ctx_len": ctx_len,
                "sliding_window": self.config.sliding_window,
            }
            if comp_ctx_lengths is not None:
                spec["comp_ctx_lengths"] = comp_ctx_lengths
            if continuous_batching:
                spec["full_batch_size"] = kv_cache_batch_size
            else:
                spec["batch_size"] = kv_cache_batch_size
            return spec

        if comp_ctx_lengths_prefill and comp_ctx_lengths_decode:
            specializations = [build_prefill_spec(length) for length in comp_ctx_lengths_prefill]
            specializations.extend(build_decode_spec(length) for length in comp_ctx_lengths_decode)
            return specializations
        return [build_prefill_spec(), build_decode_spec()]

    def get_pkv_dynamic_axes(self, retain_full_kv: Optional[bool] = False, continuous_batching: Optional[bool] = False):
        del retain_full_kv
        return [
            (
                {0: "full_batch_size" if continuous_batching else "batch_size", 2: "sliding_window"}
                if layer_type == "sliding_attention"
                else {0: "full_batch_size" if continuous_batching else "batch_size", 2: "ctx_len"}
            )
            for layer_type in self.config.layer_types
        ]

    def get_onnx_dynamic_axes(self, comp_ctx_lengths: Optional[List[int]] = None, continuous_batching: bool = False):
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
        }
        if continuous_batching:
            dynamic_axes["batch_index"] = {0: "batch_size"}
        for i, ctx_axis in enumerate(self.get_pkv_dynamic_axes(continuous_batching=continuous_batching)):
            for kv in ("key", "value"):
                dynamic_axes[f"past_{kv}.{i}"] = ctx_axis
        if comp_ctx_lengths is not None:
            dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}
        return dynamic_axes

    def get_dummy_pkv_cache(self, config, batch_size, seq_len):
        past_key_values = []
        for layer_type in config.layer_types:
            if layer_type == "sliding_attention":
                n_heads = config.num_key_value_heads
                d_head = config.head_dim
                layer_seq_len = min(config.sliding_window, seq_len)
            else:
                use_alternative_attention = getattr(config, "attention_k_eq_v", False)
                n_heads = (
                    config.num_global_key_value_heads
                    if use_alternative_attention and getattr(config, "num_global_key_value_heads", None) is not None
                    else config.num_key_value_heads
                )
                d_head = config.global_head_dim if getattr(config, "global_head_dim", None) else config.head_dim
                layer_seq_len = seq_len
            cache_shape = [batch_size, n_heads, layer_seq_len, d_head]
            past_key_values.append(
                (
                    torch.zeros(cache_shape, dtype=torch.float32),
                    torch.zeros(cache_shape, dtype=torch.float32),
                )
            )
        return past_key_values

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        del attention_mask, labels, logits_to_keep
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        if position_ids is not None:
            logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
            hidden_states = hidden_states[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        else:
            hidden_states = hidden_states[:, -1:, :]
        logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        logits = logits.float()
        return CausalLMOutputWithPast(logits=logits, past_key_values=outputs.past_key_values)


class QEffGemma4UnifiedDecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.language_model = model.model.language_model
        self.input_embeddings = model.get_input_embeddings()
        self.config = model.config
        self.lm_head = model.lm_head

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffGemma4UnifiedTextDecoderLayer}

    def forward(
        self,
        input_ids,
        vision_embeds,
        position_ids,
        image_idx,
        past_key_values,
        mm_token_type_ids=None,
        batch_index: Optional[torch.LongTensor] = None,
        comp_ctx_lengths: Optional[List[int]] = None,
        **kwargs,
    ):
        del batch_index, comp_ctx_lengths, kwargs
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            past_key_values = QEffGemma4DynamicCache.from_legacy_cache(self.language_model.config, past_key_values)

        if mm_token_type_ids is not None and mm_token_type_ids.shape == input_ids.shape:
            special_image_mask = (mm_token_type_ids == 1) | (mm_token_type_ids == 2)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        llm_input_ids = input_ids.clone()
        llm_input_ids[special_image_mask] = self.config.text_config.pad_token_id
        inputs_embeds = self.input_embeddings(llm_input_ids)

        next_image_idx = image_idx
        if input_ids.shape[1] != 1 and special_image_mask.any() and vision_embeds is None:
            raise RuntimeError(
                "Image placeholder tokens were found in decoder input, but `vision_embeds` is missing. "
                "Run the Gemma4 Unified vision encoder path first."
            )

        if vision_embeds is not None and input_ids.shape[1] != 1 and special_image_mask.any():
            if vision_embeds.dim() == 2:
                vision_embeds = vision_embeds.unsqueeze(0)
            if next_image_idx is None:
                next_image_idx = torch.zeros((1, 1), dtype=torch.int64, device=inputs_embeds.device)

            indices1 = special_image_mask.to(torch.int64).cumsum(1) - 1
            indices1 = torch.where(indices1 != -1, indices1 + next_image_idx.to(indices1.device), indices1)
            indices0 = torch.arange(special_image_mask.shape[0], device=special_image_mask.device).view(-1, 1)
            safe_indices1 = torch.where(indices1 < 0, torch.zeros_like(indices1), indices1)
            gathered_vision_embeds = vision_embeds[indices0, safe_indices1]
            inputs_embeds = torch.where(special_image_mask.unsqueeze(-1), gathered_vision_embeds, inputs_embeds)
            next_image_idx = (indices1.max() + 1).reshape(1, 1)

        global _DISABLE_EXPORT_FP16_CLAMP
        restore_disable_clamp = _DISABLE_EXPORT_FP16_CLAMP
        if _is_onnx_export():
            _DISABLE_EXPORT_FP16_CLAMP = True
        try:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                mm_token_type_ids=mm_token_type_ids,
            )
        finally:
            _DISABLE_EXPORT_FP16_CLAMP = restore_disable_clamp

        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs[0][torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states)
        if self.config.text_config.final_logit_softcapping is not None:
            logits = logits / self.config.text_config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.text_config.final_logit_softcapping
        logits = logits.float()

        if next_image_idx is None:
            next_image_idx = torch.zeros((1, 1), dtype=torch.int64, device=logits.device)
        return logits, vision_embeds, next_image_idx, outputs.past_key_values


class QEffGemma4UnifiedEncoderWrapper(nn.Module):
    """Fixed-shape AOT wrapper for encoder-free Gemma4 Unified vision."""

    def __init__(self, model):
        super().__init__()
        self.embed_vision = model.model.embed_vision
        self.config = model.config
        vision_config = getattr(model.config, "vision_config", None)
        self.mm_tokens_per_image = getattr(model.config, "mm_tokens_per_image", None)
        if self.mm_tokens_per_image is None and vision_config is not None:
            self.mm_tokens_per_image = getattr(
                vision_config,
                "num_soft_tokens",
                getattr(vision_config, "max_soft_tokens", None),
            )
        if self.mm_tokens_per_image is None:
            self.mm_tokens_per_image = getattr(model.config, "num_image_tokens", 256)

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {self.embed_vision.__class__}

    def forward(self, pixel_values, image_position_ids):
        # HF eager path flattens/removes padding via `vision_outputs[~padding_mask]`.
        # Do not do that for AOT.  Keep [B, V, H], zero padded slots, and let the
        # language graph consume a fixed number of image placeholders.
        vision_embeds = self.embed_vision(pixel_values, image_position_ids)
        padding_mask = (image_position_ids == -1).all(dim=-1)
        vision_embeds = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(vision_embeds), vision_embeds)
        if vision_embeds.dim() == 2:
            vision_embeds = vision_embeds.unsqueeze(0)
        return vision_embeds[:, : self.mm_tokens_per_image, :]


class QEffGemma4UnifiedForConditionalGeneration(Gemma4UnifiedForConditionalGeneration):
    _VISION_NPI_FP32_OPS = {"Add", "CustomRMSNorm", "ReduceMean", "Div", "Pow", "Sqrt"}
    _NPI_FP32_OPS = QEffGemma4UnifiedForCausalLM._NPI_FP32_OPS
    _NPI_EXCLUDED_OPS = QEffGemma4UnifiedForCausalLM._NPI_EXCLUDED_OPS

    def _get_vision_max_patches(self) -> int:
        vision_config = getattr(self.config, "vision_config", self.config)
        return getattr(
            vision_config,
            "num_soft_tokens",
            getattr(vision_config, "max_soft_tokens", getattr(self.config, "mm_tokens_per_image", 256)),
        )

    def _get_mm_tokens_per_image(self) -> int:
        vision_config = getattr(self.config, "vision_config", None)
        return getattr(
            self.config,
            "mm_tokens_per_image",
            getattr(vision_config, "num_soft_tokens", getattr(vision_config, "max_soft_tokens", 256)),
        )

    def _get_patch_dim(self) -> int:
        vision_config = getattr(self.config, "vision_config", self.config)
        patch_size = getattr(vision_config, "patch_size", 16)
        pooling_kernel_size = getattr(vision_config, "pooling_kernel_size", 3)
        model_patch_size = patch_size * pooling_kernel_size
        return model_patch_size * model_patch_size * 3

    def get_qeff_vision_encoder(self):
        return QEffGemma4UnifiedEncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEffGemma4UnifiedDecoderWrapper(self)

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffGemma4UnifiedTextDecoderLayer}

    def generate_npi_file(self, onnx_path: Union[str, Path], model_name: Optional[str] = None) -> str:
        del model_name
        onnx_path = Path(onnx_path)
        npi_path = onnx_path.with_name(f"{onnx_path.stem}_gemma4_unified_npi.yaml")
        model = onnx.load(str(onnx_path), load_external_data=False)
        fp32_names = []
        for node in model.graph.node:
            if node.op_type in self._NPI_EXCLUDED_OPS:
                continue
            fp32_names.extend(
                out_name for out_name in node.output if out_name and not out_name.endswith("_RetainedState")
            )
        for function in model.functions:
            if "DecoderLayer" not in function.name:
                continue
            for node in function.node:
                if node.op_type in self._NPI_EXCLUDED_OPS:
                    continue
                fp32_names.extend(output_name for output_name in node.output if output_name)
        fp32_names = [name for name in list(dict.fromkeys(fp32_names)) if "MatMul" not in name]
        with open(npi_path, "w") as fp:
            yaml.safe_dump({"FP32NodeInstanceNames": fp32_names}, fp, sort_keys=False)
        return str(npi_path)

    def generate_vision_npi_file(self, onnx_path: Union[str, Path], model_name: Optional[str] = None) -> str:
        del model_name
        onnx_path = Path(onnx_path)
        npi_path = onnx_path.with_name(f"{onnx_path.stem}_gemma4_unified_vision_npi.yaml")
        model = onnx.load(str(onnx_path), load_external_data=False)
        fp32_names = []
        for node in model.graph.node:
            if node.op_type not in self._VISION_NPI_FP32_OPS:
                continue
            fp32_names.extend(output_name for output_name in node.output if output_name)
        with open(npi_path, "w") as fp:
            yaml.safe_dump({"FP32NodeInstanceNames": list(dict.fromkeys(fp32_names))}, fp, sort_keys=False)
        return str(npi_path)

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        img_size: int = 0,
        comp_ctx_lengths_prefill: Optional[List[int]] = None,
        comp_ctx_lengths_decode: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
        **compiler_options,
    ):
        del img_size
        prefill_seq_len = prefill_seq_len if prefill_seq_len else 32
        ctx_len = ctx_len if ctx_len else constants.INTERN_CTX_LEN
        max_patches = self._get_vision_max_patches()
        mm_tokens_per_image = self._get_mm_tokens_per_image()
        vision = [{"batch_size": batch_size, "max_patches": max_patches}]

        def build_lang_prefill_spec(comp_ctx_lengths: Optional[int] = None):
            spec = {
                "batch_size": 1 if continuous_batching else batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "sliding_window": self.model.language_model.config.sliding_window,
                "vision_batch_size": batch_size,
                "vision_tokens": mm_tokens_per_image,
            }
            if comp_ctx_lengths is not None:
                spec["comp_ctx_lengths"] = comp_ctx_lengths
            if continuous_batching:
                spec["full_batch_size"] = kv_cache_batch_size or batch_size
            else:
                spec["batch_size"] = kv_cache_batch_size or batch_size
            if full_batch_size:
                spec["full_batch_exec_size"] = full_batch_size
            return spec

        def build_lang_decode_spec(comp_ctx_lengths: Optional[int] = None):
            spec = {
                "batch_size": full_batch_size if continuous_batching else batch_size,
                "seq_len": "1",
                "ctx_len": ctx_len,
                "sliding_window": self.model.language_model.config.sliding_window,
                "vision_batch_size": batch_size,
                "vision_tokens": mm_tokens_per_image,
            }
            if comp_ctx_lengths is not None:
                spec["comp_ctx_lengths"] = comp_ctx_lengths
            if continuous_batching:
                spec["full_batch_size"] = kv_cache_batch_size or batch_size
            else:
                spec["batch_size"] = kv_cache_batch_size or batch_size
            return spec

        if comp_ctx_lengths_prefill and comp_ctx_lengths_decode:
            lang = [build_lang_prefill_spec(length) for length in comp_ctx_lengths_prefill]
            lang.extend(build_lang_decode_spec(length) for length in comp_ctx_lengths_decode)
        else:
            lang = [build_lang_prefill_spec(), build_lang_decode_spec()]

        if kv_offload:
            return {"vision": vision, "lang": lang}, compiler_options
        return lang, compiler_options

    def get_onnx_dynamic_axes(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
    ):
        vision_dynamic_axes = {
            "pixel_values": {0: "batch_size", 1: "max_patches"},
            "image_position_ids": {0: "batch_size", 1: "max_patches"},
        }
        lang_dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "vision_embeds": {0: "vision_batch_size", 1: "vision_tokens"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "mm_token_type_ids": {0: "batch_size", 1: "seq_len"},
        }
        if continuous_batching:
            lang_dynamic_axes["batch_index"] = {0: "batch_size"}
        for i in range(self.model.language_model.config.num_hidden_layers):
            layer_type = self.model.language_model.config.layer_types[i]
            if layer_type == "sliding_attention":
                ctx_axis = {0: "full_batch_size" if continuous_batching else "batch_size", 2: "sliding_window"}
            else:
                ctx_axis = {0: "full_batch_size" if continuous_batching else "batch_size", 2: "ctx_len"}
            for kv in ("key", "value"):
                lang_dynamic_axes[f"past_{kv}.{i}"] = ctx_axis
        if comp_ctx_lengths is not None:
            lang_dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}
        if kv_offload:
            return {"vision": vision_dynamic_axes, "lang": lang_dynamic_axes}
        return {**vision_dynamic_axes, **lang_dynamic_axes}

    def get_output_names(self, kv_offload: bool = False):
        vision_output_names = ["vision_embeds"]
        lang_output_names = ["logits", "vision_embeds_RetainedState", "image_idx_output"]
        for i in range(self.model.language_model.config.num_hidden_layers):
            for kv in ("key", "value"):
                lang_output_names.append(f"past_{kv}.{i}_RetainedState")
        if kv_offload:
            return {"vision": vision_output_names, "lang": lang_output_names}
        return lang_output_names

    def get_dummy_pkv_cache(self, config, batch_size, seq_len):
        return QEffGemma4UnifiedForCausalLM.get_dummy_pkv_cache(self, config, batch_size, seq_len)

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        prefill_seq_len: Optional[int] = None,
    ):
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        fbs = constants.ONNX_EXPORT_EXAMPLE_FBS
        max_patches = self._get_vision_max_patches()
        mm_tokens_per_image = self._get_mm_tokens_per_image()
        seq_len = prefill_seq_len or max(constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN, mm_tokens_per_image + 32)
        patch_dim = self._get_patch_dim()

        image_position_ids = torch.full((bs, max_patches, 2), -1, dtype=torch.int64)
        side = int(max_patches**0.5)
        valid = side * side
        xs = torch.arange(side, dtype=torch.int64).view(1, -1).expand(side, -1).reshape(-1)
        ys = torch.arange(side, dtype=torch.int64).view(-1, 1).expand(-1, side).reshape(-1)
        valid_positions = torch.stack((xs, ys), dim=-1)
        image_position_ids[:, :valid, :] = valid_positions.unsqueeze(0)

        input_ids = torch.zeros((bs, seq_len), dtype=torch.int64)
        mm_token_type_ids = torch.zeros((bs, seq_len), dtype=torch.int64)
        text_prefix_len = min(5, seq_len)
        image_start = text_prefix_len
        image_end = min(image_start + mm_tokens_per_image, seq_len)
        input_ids[:, image_start:image_end] = self.config.image_token_id
        mm_token_type_ids[:, image_start:image_end] = 1

        vision_inputs = {
            "pixel_values": torch.zeros((bs, max_patches, patch_dim), dtype=torch.float32),
            "image_position_ids": image_position_ids,
        }
        lang_inputs = {
            "input_ids": input_ids,
            "vision_embeds": torch.zeros((bs, mm_tokens_per_image, self.model.language_model.config.hidden_size)),
            "position_ids": torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(bs, 1),
            "image_idx": torch.zeros((1, 1), dtype=torch.int64),
            "mm_token_type_ids": mm_token_type_ids,
            "past_key_values": self.get_dummy_pkv_cache(
                config=self.model.language_model.config,
                batch_size=fbs if continuous_batching else bs,
                seq_len=seq_len,
            ),
        }
        if comp_ctx_lengths is not None:
            lang_inputs["comp_ctx_lengths"] = torch.randint(0, 100, (40,), dtype=torch.int8)
        if kv_offload:
            return {"vision": vision_inputs, "lang": lang_inputs}
        return {**vision_inputs, **lang_inputs}

    def remove_fp16clip_transform_if_disabled(self, effective_fp16clip: bool):
        if not effective_fp16clip:
            if hasattr(self, "lang_model") and hasattr(self.lang_model, "_onnx_transforms"):
                self.lang_model._onnx_transforms = [
                    t for t in self.lang_model._onnx_transforms if t is not FP16ClipTransform
                ]
            if getattr(self, "vision_model", None) is not None and hasattr(self.vision_model, "_onnx_transforms"):
                self.vision_model._onnx_transforms = [
                    t for t in self.vision_model._onnx_transforms if t is not FP16ClipTransform
                ]

    def normalize_generated_ids(self, generated_ids):
        array = np.asarray(generated_ids)
        if array.dtype == object:
            array = np.asarray([np.asarray(row).reshape(-1) for row in generated_ids], dtype=np.int64)
        array = np.asarray(array)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        elif array.ndim > 2:
            array = array.reshape(array.shape[0], -1)
        return array.astype(np.int64, copy=False)

    def effective_lens(
        self, prefill_seq_len: int, ctx_len: int, prompt_len: int, generation_len: int, skip_vision: bool
    ):
        effective_ctx_len = max(ctx_len, prompt_len + generation_len)
        effective_prefill_seq_len = prefill_seq_len
        return effective_prefill_seq_len, effective_ctx_len
