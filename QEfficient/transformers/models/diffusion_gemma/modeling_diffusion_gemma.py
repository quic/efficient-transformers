# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from typing import List, Optional, Type

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
    DiffusionGemmaEncoderTextAttention,
    DiffusionGemmaEncoderTextLayer,
    DiffusionGemmaEncoderTextModel,
    DiffusionGemmaForBlockDiffusion,
    DiffusionGemmaRMSNorm,
    DiffusionGemmaTextExperts,
    DiffusionGemmaTextRouter,
    apply_rotary_pos_emb,
)

from QEfficient.customop.rms_norm import CustomRMSNormFunc
from QEfficient.transformers.cache_utils import QEffGemma4DynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils import constants

_FP16_CLAMP_MIN = -65504.0
_FP16_CLAMP_MAX = 65504.0

EXPERT_BLOCKING_NUM_NSP = int(os.environ.get("EXPERT_BLOCKING_NUM_NSP", "16"))


def _is_onnx_export() -> bool:
    return torch.onnx.is_in_onnx_export()


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Dynamic-shape-safe repeat_kv: uses -1 in reshape to avoid baking seq_len."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, _, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, -1, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, -1, head_dim)

def _clamp_to_fp16_range(t: torch.Tensor) -> torch.Tensor:
    if not _is_onnx_export():
        return t
    return t.clamp(_FP16_CLAMP_MIN, _FP16_CLAMP_MAX)


def _saturating_residual_add(residual: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    """Mirrors gemma4's helper — fp32 sum then clamp back to fp16 range.

    Required for the canvas-decode path: without this clamp the decoder's residual
    stream can exceed fp16 ±65504 on hardware, producing inf in subsequent attn
    matmuls and NaN softmax rows. The encoder's layer-entry clamp kept its residual
    bounded; the decoder has no such guard.
    """
    if not _is_onnx_export():
        return residual + hidden_states
    return (residual.float() + hidden_states.float()).clamp(_FP16_CLAMP_MIN, _FP16_CLAMP_MAX).to(hidden_states.dtype)


class QEffDiffusionGemmaRMSNorm(DiffusionGemmaRMSNorm):
    """Export-safe RMSNorm.

    For ``with_scale=True`` modules, exports as the AIC ``CustomRMSNorm`` op
    (the proven compiler path). For ``with_scale=False`` modules (v_norm,
    self_conditioning.post_norm, router.norm with no parent-registered unit
    buffer), use the HF parent forward directly — it produces basic
    ``Pow/ReduceMean/Sqrt/Mul`` ONNX ops that the compiler handles correctly.

    The alternative (``CustomRMSNormFunc.apply(x, new_ones(dim), eps)``) exports a
    runtime ``Shape→Gather→Concat→ConstantOfShape`` chain feeding the CustomRMSNorm
    weight input, which computes the wrong norm on hardware. Bypassing the custom
    op when ``with_scale=False`` matches the eager forward exactly.
    """

    def __qeff_init__(self):
        if not getattr(self, "with_scale", True) and hasattr(self, "weight") and not hasattr(self, "_qeff_unit_weight"):
            self.register_buffer("_qeff_unit_weight", torch.ones_like(self.weight))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not _is_onnx_export():
            return super().forward(hidden_states)
        if getattr(self, "with_scale", True):
            return CustomRMSNormFunc.apply(hidden_states, self.weight, self.eps)
        weight = getattr(self, "_qeff_unit_weight", None)
        if weight is not None and weight.shape[-1] != 1:
            return CustomRMSNormFunc.apply(hidden_states, weight, self.eps)
        return super().forward(hidden_states)


# ---------------------------------------------------------------------------
# Router — same as gemma4 (drop router_probabilities return, compile-safe topk)
# ---------------------------------------------------------------------------


class QEffDiffusionGemmaTextRouter(DiffusionGemmaTextRouter):
    def __qeff_init__(self):
        if (
            hasattr(self, "norm")
            and not getattr(self.norm, "with_scale", True)
            and not hasattr(self.norm, "_qeff_unit_weight")
        ):
            self.norm.register_buffer("_qeff_unit_weight", torch.ones(self.hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size

        router_probabilities = nn.functional.softmax(self.proj(hidden_states), dim=-1)
        top_k_weights, top_k_index = torch.topk(
            router_probabilities,
            k=self.config.top_k_experts,
            dim=-1,
        )
        top_k_weights = top_k_weights / torch.einsum("bk->b", top_k_weights).unsqueeze(-1)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        return router_probabilities, top_k_weights, top_k_index


# ---------------------------------------------------------------------------
# Experts — batched BMM (same structure as QEffGemma4TextExperts)
# ---------------------------------------------------------------------------


class QEffDiffusionGemmaTextExperts(DiffusionGemmaTextExperts):
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        gate_up_proj_t = self.gate_up_proj.transpose(1, 2)
        gate_up_out = torch.matmul(hidden_states, gate_up_proj_t).permute(1, 0, 2)
        gate, up = gate_up_out.chunk(2, dim=-1)
        activated = self.act_fn(gate) * up

        down_proj_t = self.down_proj.transpose(1, 2)
        experts_out = torch.matmul(activated.permute(1, 0, 2), down_proj_t).permute(1, 0, 2)
        # Avoid scatter_add_ which traces to ScatterElements(reduction='add') in ONNX
        # and compiles incorrectly on AI 100 (large per-layer cosine error compounding
        # over 30 layers). Use broadcast equality + weighted sum instead (no scatter).
        # top_k_index: [tokens, top_k], top_k_weights: [tokens, top_k]
        # one_hot[t,k,e] = (top_k_index[t,k] == e)
        expert_ids = torch.arange(self.num_experts, device=top_k_index.device, dtype=top_k_index.dtype)
        one_hot = (top_k_index.unsqueeze(-1) == expert_ids.view(1, 1, -1)).to(top_k_weights.dtype)
        # expert_weights[t, e] = sum_k(one_hot[t,k,e] * top_k_weights[t,k])
        expert_weights = torch.einsum("tke,tk->te", one_hot, top_k_weights)
        weighted_experts = experts_out.transpose(1, 2)  # [tokens, hidden, num_experts]
        combine_weights = expert_weights.to(experts_out.dtype).unsqueeze(-1)  # [tokens, num_experts, 1]
        return torch.bmm(weighted_experts, combine_weights).squeeze(-1)


class QEffDiffusionGemmaEncoderTextAttention(DiffusionGemmaEncoderTextAttention):
    """Shared encoder/decoder attention over physical retained KV buffers."""

    def __qeff_init__(self):
        if hasattr(self, "v_norm") and not getattr(self.v_norm, "with_scale", True):
            if not hasattr(self.v_norm, "_qeff_unit_weight"):
                self.v_norm.register_buffer(
                    "_qeff_unit_weight",
                    torch.ones(self.head_dim, dtype=torch.float32),
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position_ids: Optional[torch.LongTensor] = None,
        use_physical_kv: bool = False,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

        key_states = self.k_norm(key_states)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        if past_key_values is not None:
            # Keep the QEff cache update so prefill/commit scatters the current
            # block into retained state. Decode supplies all -1 cache positions,
            # which leaves retained state untouched. Attention deliberately reads
            # the physical buffers rather than the update return value: that return
            # contains a gather/reordered view unsuitable for the unified QPC.
            if use_physical_kv:
                write_positions = cache_position_ids if cache_position_ids is not None else position_ids
                past_key_values.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    {"position_ids": write_positions},
                )
                layer = past_key_values.layers[self.layer_idx]
                retained_key_states = layer.keys
                retained_value_states = layer.values
                key_states = torch.cat([retained_key_states, key_states], dim=2)
                value_states = torch.cat([retained_value_states, value_states], dim=2)
            else:
                key_states, value_states = past_key_values.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    {"position_ids": cache_position_ids if cache_position_ids is not None else position_ids},
                )

        key_states_for_attn = _repeat_kv(key_states, self.num_key_value_groups)
        value_states_for_attn = _repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states_for_attn.transpose(2, 3)) * self.scaling
        if self.config.final_logit_softcapping is not None:
            pass  # softcapping is on logits only, not on attn_weights for this arch
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states_for_attn)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QEffDiffusionGemmaEncoderTextLayer(DiffusionGemmaEncoderTextLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position_ids: Optional[torch.LongTensor] = None,
        decoder_layer_scalar: Optional[torch.Tensor] = None,
        is_encode: Optional[torch.LongTensor] = None,
        use_physical_kv: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = _clamp_to_fp16_range(hidden_states)
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position_ids=cache_position_ids,
            use_physical_kv=use_physical_kv,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = _saturating_residual_add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

        hidden_states_flat = residual.reshape(-1, residual.shape[-1])
        _, top_k_weights, top_k_index = self.router(hidden_states_flat)
        hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
        hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)
        hidden_states_2 = hidden_states_2.reshape(residual.shape)
        hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

        hidden_states = _saturating_residual_add(hidden_states_1, hidden_states_2)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = _saturating_residual_add(residual, hidden_states)
        if decoder_layer_scalar is not None and is_encode is not None:
            layer_scalar = torch.where(is_encode.bool(), self.layer_scalar, decoder_layer_scalar)
        else:
            layer_scalar = self.layer_scalar
        hidden_states *= layer_scalar
        return hidden_states


class QEffDiffusionGemmaEncoderTextModel(DiffusionGemmaEncoderTextModel):
    """
    QEff-patched encoder text model.

    Replaces HF's dynamic mask creation with static QEff-friendly masks;
    uses QEffGemma4DynamicCache for the encoder KV cache.
    """

    def _precomputed_rope_gather(self, position_ids: torch.Tensor, layer_type: str, dtype: torch.dtype):
        """Gather-based RoPE for full_attention layers.

        Precomputes the cos/sin table at max_position_embeddings once, then indexes
        by position_ids via Gather. This avoids the runtime MatMul(inv_freq, positions)
        which mis-compiles when inv_freq has trailing zeros (partial_rotary_factor < 1).
        """
        inv_freq = getattr(self.rotary_emb, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self.rotary_emb, f"{layer_type}_attention_scaling")
        max_pos = min(self.config.max_position_embeddings, 4096)
        all_pos = torch.arange(max_pos, device=inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(all_pos, inv_freq.float())  # [max_pos, D]
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_pos, 2D]
        cos_table = (emb.cos() * attention_scaling).to(dtype)  # [max_pos, 2D]
        sin_table = (emb.sin() * attention_scaling).to(dtype)
        # Gather: position_ids [B, S] → cos [B, S, 2D]
        pos_clamped = position_ids.clamp(min=0, max=max_pos - 1)
        cos = cos_table[pos_clamped]  # [B, S, 2D]
        sin = sin_table[pos_clamped]
        return cos, sin


class QEffDiffusionGemmaVisionEncoderWrapper(nn.Module):
    """
    Standalone vision encoder wrapper for dual-QPC export.

    Runs vision_tower + embed_vision from the encoder model and clips
    outputs to FP16 range.
    """

    def __init__(self, model: "QEffDiffusionGemmaForBlockDiffusion"):
        super().__init__()
        self.model = model
        self.mm_tokens_per_image = getattr(model.config, "mm_tokens_per_image", 256)

    def get_submodules_for_export(self) -> Type[nn.Module]:
        encoder_model = self.model.model.encoder
        return {encoder_model.vision_tower.encoder.layers[0].__class__}

    def forward(self, pixel_values: torch.Tensor, image_position_ids: torch.Tensor) -> torch.Tensor:
        encoder_model = self.model.model.encoder
        vision_tower = encoder_model.vision_tower
        padding_positions = (image_position_ids == -1).all(dim=-1)

        inputs_embeds = vision_tower.patch_embedder(pixel_values, image_position_ids, padding_positions)
        valid_tokens = ~padding_positions
        vision_attention_mask = (~valid_tokens).unsqueeze(1).unsqueeze(2).to(dtype=inputs_embeds.dtype)
        vision_attention_mask = vision_attention_mask * torch.finfo(inputs_embeds.dtype).min
        vision_attention_mask = vision_attention_mask.expand(-1, 1, inputs_embeds.shape[1], -1)

        hidden_states = inputs_embeds
        position_embeddings = vision_tower.encoder.rotary_emb(hidden_states, image_position_ids)
        for layer in vision_tower.encoder.layers[: vision_tower.encoder.config.num_hidden_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=vision_attention_mask,
                position_embeddings=position_embeddings,
                position_ids=image_position_ids,
            )

        output_length = getattr(vision_tower.config, "default_output_length", None)
        if output_length is None:
            output_length = pixel_values.shape[-2] // (
                vision_tower.config.pooling_kernel_size * vision_tower.config.pooling_kernel_size
            )
        hidden_states, _ = vision_tower.pooler(
            hidden_states=hidden_states,
            pixel_position_ids=image_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )
        if vision_tower.config.standardize:
            hidden_states = (hidden_states - vision_tower.std_bias) * vision_tower.std_scale

        vision_embeds = encoder_model.embed_vision(inputs_embeds=hidden_states)
        if vision_embeds.dim() == 2:
            vision_embeds = vision_embeds.unsqueeze(0)

        # clamp vision projector output to FP16 range
        vision_embeds = vision_embeds.clamp(-60000.0, 60000.0)
        return vision_embeds[:, : self.mm_tokens_per_image, :]


class QEffDiffusionGemmaUnifiedWrapper(nn.Module):
    """Single-QPC shared DiffusionGemma transformer path.

    The host supplies the active token block, absolute RoPE positions, cache write
    positions, and additive masks for full/sliding layers. ``is_encode`` remains a
    scalar mode input only for the distinct layer-scale and self-conditioning
    computations; it never selects logits, image indices, or retained KV tensors.
    """

    supports_autoregressive_generate = False

    def __init__(self, model: "QEffDiffusionGemmaForBlockDiffusion"):
        super().__init__()
        self.model = model
        self.config = model.config
        self.text_config = model.config.text_config

    def get_submodules_for_export(self):
        return {QEffDiffusionGemmaEncoderTextLayer}

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        cache_position_ids: torch.LongTensor,
        full_attention_mask: torch.Tensor,
        sliding_attention_mask: torch.Tensor,
        vision_embeds: Optional[torch.Tensor] = None,
        image_idx: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        self_conditioning_logits: Optional[torch.FloatTensor] = None,
        is_encode: Optional[torch.LongTensor] = None,
        use_self_conditioning: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        del batch_index, kwargs

        if past_key_values is not None and not isinstance(past_key_values, QEffGemma4DynamicCache):
            past_key_values = QEffGemma4DynamicCache.from_legacy_cache(self.text_config, past_key_values)
        if past_key_values is None:
            past_key_values = QEffGemma4DynamicCache(config=self.text_config)

        if is_encode is None:
            is_encode = torch.ones(1, dtype=torch.int64, device=input_ids.device)
        if use_self_conditioning is None:
            use_self_conditioning = torch.zeros(1, dtype=torch.int64, device=input_ids.device)

        inputs_embeds, next_image_idx = self.model._inject_vision_embeds(input_ids, vision_embeds, image_idx)
        decoder = self.model.model.decoder
        if self_conditioning_logits is None:
            soft_embeddings = torch.zeros_like(inputs_embeds)
        else:
            soft_embeddings = torch.matmul(
                self_conditioning_logits.softmax(dim=-1, dtype=torch.float32).to(decoder.embed_tokens.weight.dtype),
                decoder.embed_tokens.weight,
            ) * decoder.embed_tokens.embed_scale.to(inputs_embeds.dtype)
        use_sc = use_self_conditioning.bool().view(1, 1, 1)
        soft_embeddings = torch.where(use_sc, soft_embeddings, torch.zeros_like(soft_embeddings))
        conditioned_embeds = decoder.self_conditioning(inputs_embeds, soft_embeddings)
        hidden_states = torch.where(is_encode.bool().view(1, 1, 1), inputs_embeds, conditioned_embeds)

        language_model = self.model.model.encoder.language_model
        position_embeddings = {}
        for layer_type in language_model.unique_layer_types:
            if layer_type == "full_attention" and _is_onnx_export():
                position_embeddings[layer_type] = language_model._precomputed_rope_gather(
                    position_ids, layer_type, hidden_states.dtype
                )
            else:
                position_embeddings[layer_type] = language_model.rotary_emb(hidden_states, position_ids, layer_type)

        for layer_index, encoder_layer in enumerate(language_model.layers[: self.text_config.num_hidden_layers]):
            layer_type = self.text_config.layer_types[layer_index]
            attention_mask = sliding_attention_mask if layer_type == "sliding_attention" else full_attention_mask
            decoder_layer_scalar = decoder.layers[layer_index].layer_scalar
            hidden_states = encoder_layer(
                hidden_states,
                position_embeddings=position_embeddings[layer_type],
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position_ids=cache_position_ids,
                past_key_values=past_key_values,
                decoder_layer_scalar=decoder_layer_scalar,
                is_encode=is_encode,
                use_physical_kv=True,
                mm_token_type_ids=mm_token_type_ids,
            )

        hidden_states = language_model.norm(_clamp_to_fp16_range(hidden_states))
        canvas_logits = self.model._apply_logit_softcapping(self.model.lm_head(hidden_states).float())
        pkv = [
            (past_key_values.layers[layer_index].keys, past_key_values.layers[layer_index].values)
            for layer_index in range(self.text_config.num_hidden_layers)
        ]
        return canvas_logits, next_image_idx, pkv

    def get_dummy_inputs(self, **kwargs):
        encoder = QEffDiffusionGemmaEncoderPrefillWrapper(self.model)
        encoder_inputs = encoder.get_dummy_inputs()
        batch_size, block_length = encoder_inputs["input_ids"].shape
        text_config = self.text_config
        full_kv_length = encoder_inputs["past_key_values"][
            next(index for index, layer_type in enumerate(text_config.layer_types) if layer_type == "full_attention")
        ][0].shape[-2]
        sliding_kv_length = encoder_inputs["past_key_values"][
            next(index for index, layer_type in enumerate(text_config.layer_types) if layer_type == "sliding_attention")
        ][0].shape[-2]
        return {
            **encoder_inputs,
            "cache_position_ids": encoder_inputs["position_ids"].clone(),
            "full_attention_mask": torch.zeros(
                (batch_size, 1, block_length, full_kv_length + block_length), dtype=torch.float32
            ),
            "sliding_attention_mask": torch.zeros(
                (batch_size, 1, block_length, sliding_kv_length + block_length), dtype=torch.float32
            ),
            "self_conditioning_logits": torch.zeros(
                (batch_size, block_length, text_config.vocab_size), dtype=torch.float32
            ),
            "is_encode": torch.ones(1, dtype=torch.int64),
            "use_self_conditioning": torch.zeros(1, dtype=torch.int64),
        }

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        canvas_length: Optional[int] = None,
        **compiler_options,
    ):
        prefill_seq_len = prefill_seq_len or 32
        canvas_length = canvas_length or getattr(self.config, "canvas_length", prefill_seq_len)
        if canvas_length != prefill_seq_len:
            raise ValueError("The shared unified QPC requires matching prefill and canvas lengths.")
        ctx_len = ctx_len or constants.INTERN_CTX_LEN
        return [
            {
                "_graph_name": "Unified",
                "batch_size": batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "sliding_window": self.text_config.sliding_window,
                "full_kv_plus_seq_len": ctx_len + prefill_seq_len,
                "sliding_kv_plus_seq_len": self.text_config.sliding_window + prefill_seq_len,
                "vision_batch_size": batch_size,
                "vision_tokens": self.model._get_mm_tokens_per_image(),
            }
        ], compiler_options

    def get_onnx_dynamic_axes(self, **kwargs):
        axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "cache_position_ids": {0: "batch_size", 1: "seq_len"},
            "vision_embeds": {0: "vision_batch_size", 1: "vision_tokens"},
            "mm_token_type_ids": {0: "batch_size", 1: "seq_len"},
            "full_attention_mask": {0: "batch_size", 2: "seq_len", 3: "full_kv_plus_seq_len"},
            "sliding_attention_mask": {0: "batch_size", 2: "seq_len", 3: "sliding_kv_plus_seq_len"},
            "self_conditioning_logits": {0: "batch_size", 1: "seq_len"},
        }
        for layer_index, layer_type in enumerate(self.text_config.layer_types):
            ctx_axis = {0: "batch_size", 2: "sliding_window" if layer_type == "sliding_attention" else "ctx_len"}
            for kv_name in ("key", "value"):
                axes[f"past_{kv_name}.{layer_index}"] = ctx_axis
        return axes

    def get_output_names(self, **kwargs):
        names = ["canvas_logits", "image_idx_output"]
        for layer_index in range(self.text_config.num_hidden_layers):
            for kv_name in ("key", "value"):
                names.append(f"past_{kv_name}.{layer_index}_RetainedState")
        return names


class QEffDiffusionGemmaEncoderPrefillWrapper(nn.Module):
    """Standalone encoder-prefill QPC: prompt(+vision) → filled KV cache.

    No decoder path, no is_encode gate. The encoder writes the KV cache via
    past_key_values.update(); we return the filled KV as _RetainedState outputs.
    """

    def __init__(self, model: "QEffDiffusionGemmaForBlockDiffusion"):
        super().__init__()
        self.model = model
        self.config = model.config
        self.text_config = model.config.text_config

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffDiffusionGemmaEncoderTextLayer}

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        vision_embeds: Optional[torch.Tensor] = None,
        image_idx: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ):
        del kwargs
        text_cfg = self.config.text_config
        if past_key_values is not None and not isinstance(past_key_values, QEffGemma4DynamicCache):
            past_key_values = QEffGemma4DynamicCache.from_legacy_cache(text_cfg, past_key_values)

        inputs_embeds, next_image_idx = self.model._inject_vision_embeds(input_ids, vision_embeds, image_idx)

        enc_outputs = self.model.model.encoder.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            mm_token_type_ids=None,
        )

        hidden_states = enc_outputs.last_hidden_state
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        last_hidden = hidden_states[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        enc_logits = self.model._apply_logit_softcapping(self.model.lm_head(last_hidden).float())

        pkv = [
            (past_key_values.layers[i].keys, past_key_values.layers[i].values)
            for i in range(text_cfg.num_hidden_layers)
        ]
        return enc_logits, next_image_idx, pkv

    # -- export metadata (single Prefill specialization) --

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        canvas_length: Optional[int] = None,
        **compiler_options,
    ):
        prefill_seq_len = prefill_seq_len or 32
        ctx_len = ctx_len or constants.INTERN_CTX_LEN
        text_cfg = self.config.text_config
        mm_tokens_per_image = self.model._get_mm_tokens_per_image()
        spec = {
            "batch_size": batch_size,
            "seq_len": prefill_seq_len,
            "ctx_len": ctx_len,
            "sliding_window": text_cfg.sliding_window,
            "vision_batch_size": batch_size,
            "vision_tokens": mm_tokens_per_image,
        }
        return [spec], compiler_options

    def get_onnx_dynamic_axes(self, **kwargs):
        text_cfg = self.config.text_config
        axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "vision_embeds": {0: "vision_batch_size", 1: "vision_tokens"},
            "mm_token_type_ids": {0: "batch_size", 1: "seq_len"},
        }
        for i, layer_type in enumerate(text_cfg.layer_types):
            ctx_axis = {0: "batch_size", 2: "sliding_window" if layer_type == "sliding_attention" else "ctx_len"}
            for kv in ("key", "value"):
                axes[f"past_{kv}.{i}"] = ctx_axis
        return axes

    def get_output_names(self, **kwargs):
        text_cfg = self.config.text_config
        # enc_logits is a graph-liveness anchor (not consumed by the runner) — see forward().
        # past_*_keyout / past_*_valout are the encoder-filled KV emitted as REGULAR outputs
        # (not _RetainedState). Single-spec encoder QPCs have their _RetainedState pathway
        # dead-elimed by qaic-compile (no in-graph consumer); regular outputs are user-visible
        # and survive. The runner host-copies them into the decoder QPC's past_*.{i} inputs.
        names = ["enc_logits", "image_idx_output"]
        for i in range(text_cfg.num_hidden_layers):
            for kv in ("key", "value"):
                names.append(f"past_{kv}.{i}_out")
        return names

    def get_dummy_inputs(self, **kwargs):
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        mm_tokens_per_image = self.model._get_mm_tokens_per_image()
        text_cfg = self.config.text_config
        seq_len = max(constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN, mm_tokens_per_image + 32)

        input_ids = torch.zeros((bs, seq_len), dtype=torch.int64)
        mm_token_type_ids = torch.zeros((bs, seq_len), dtype=torch.int64)
        text_prefix_len = min(5, seq_len)
        image_start = text_prefix_len
        image_end = min(image_start + mm_tokens_per_image, seq_len)
        input_ids[:, image_start:image_end] = self.config.image_token_id
        mm_token_type_ids[:, image_start:image_end] = 1

        return {
            "input_ids": input_ids,
            "position_ids": torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(bs, 1),
            "vision_embeds": torch.zeros((bs, mm_tokens_per_image, text_cfg.hidden_size), dtype=torch.float32),
            "image_idx": torch.zeros((1, 1), dtype=torch.int64),
            "mm_token_type_ids": mm_token_type_ids,
            "past_key_values": self.model.get_dummy_pkv_cache(config=text_cfg, batch_size=bs, seq_len=seq_len),
        }


# ---------------------------------------------------------------------------
# Top-level model class — registered to AutoModelForImageTextToText
# ---------------------------------------------------------------------------


class QEffDiffusionGemmaForBlockDiffusion(DiffusionGemmaForBlockDiffusion):
    """
    QEff-patched DiffusionGemmaForBlockDiffusion.

    Registered to AutoModelForImageTextToText.

    Supports:
      - Single-QPC: encoder prefill + decoder canvas-denoise in one compiled graph
      - Dual-QPC  : separate vision and language QPCs (kv_offload=True)
    """

    def get_qeff_vision_encoder(self) -> QEffDiffusionGemmaVisionEncoderWrapper:
        return QEffDiffusionGemmaVisionEncoderWrapper(self)

    def get_qeff_language_decoder(self) -> QEffDiffusionGemmaUnifiedWrapper:
        return QEffDiffusionGemmaUnifiedWrapper(self)

    def get_qeff_unified_wrapper(self) -> QEffDiffusionGemmaUnifiedWrapper:
        """Single-QPC unified wrapper (encoder-prefill + canvas-decode in one QPC)."""
        return QEffDiffusionGemmaUnifiedWrapper(self)

    def get_qeff_encoder_prefill(self) -> QEffDiffusionGemmaEncoderPrefillWrapper:
        """Disaggregated dual-QPC: standalone encoder-prefill QPC."""
        return QEffDiffusionGemmaEncoderPrefillWrapper(self)

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffDiffusionGemmaEncoderTextLayer}

    def _get_mm_tokens_per_image(self) -> int:
        return getattr(
            self.config.vision_config,
            "default_output_length",
            getattr(self.config, "mm_tokens_per_image", 256),
        )

    def _get_vision_max_patches(self) -> int:
        pooling_kernel_size = getattr(self.config.vision_config, "pooling_kernel_size", 3)
        default_output_length = getattr(self.config.vision_config, "default_output_length", 280)
        return default_output_length * pooling_kernel_size * pooling_kernel_size

    def get_dummy_pkv_cache(self, config, batch_size: int, seq_len: int):
        past_key_values = []
        for layer_type in config.layer_types:
            if layer_type == "sliding_attention":
                n_heads = config.num_key_value_heads
                d_head = config.head_dim
                layer_seq_len = min(config.sliding_window, seq_len)
            else:
                n_heads = config.num_global_key_value_heads or config.num_key_value_heads
                d_head = getattr(config, "global_head_dim", None) or config.head_dim
                layer_seq_len = seq_len
            cache_shape = [batch_size, n_heads, layer_seq_len, d_head]
            past_key_values.append(
                (torch.zeros(cache_shape, dtype=torch.float32), torch.zeros(cache_shape, dtype=torch.float32))
            )
        return past_key_values

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        canvas_length: Optional[int] = None,
        img_size: Optional[int] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
        comp_ctx_lengths_prefill: Optional[List[int]] = None,
        comp_ctx_lengths_decode: Optional[List[int]] = None,
        **compiler_options,
    ):
        prefill_seq_len = prefill_seq_len or 32
        ctx_len = ctx_len or constants.INTERN_CTX_LEN
        canvas_length = canvas_length or getattr(self.config, "canvas_length", 256)
        mm_tokens_per_image = self._get_mm_tokens_per_image()
        max_patches = self._get_vision_max_patches()
        text_cfg = self.config.text_config

        vision = [{"batch_size": batch_size, "max_patches": max_patches}]

        def build_encoder_prefill_spec(comp_ctx_lengths=None):
            spec = {
                "_graph_name": "Prefill",
                "batch_size": 1 if continuous_batching else batch_size,
                # seq_len=prefill_seq_len uniquely identifies the encoder specialization
                "seq_len": prefill_seq_len,
                "canvas_len": canvas_length,
                "ctx_len": ctx_len,
                "sliding_window": text_cfg.sliding_window,
                "vision_batch_size": batch_size,
                "vision_tokens": mm_tokens_per_image,
                "is_encode": 1,
            }
            if comp_ctx_lengths is not None:
                spec["comp_ctx_lengths"] = comp_ctx_lengths
            if continuous_batching:
                spec["full_batch_size"] = kv_cache_batch_size or batch_size
            else:
                spec["batch_size"] = kv_cache_batch_size or batch_size
            return spec

        def build_decoder_canvas_spec(comp_ctx_lengths=None):
            # seq_len=1 for decoder: the compiler uses this shape change to dispatch
            # (seq_len differs from encoder's prefill_seq_len, uniquely identifying this spec)
            spec = {
                "_graph_name": "Decode",
                "batch_size": full_batch_size if continuous_batching else batch_size,
                "seq_len": 1,
                "canvas_len": canvas_length,
                "ctx_len": ctx_len,
                "sliding_window": text_cfg.sliding_window,
                "vision_batch_size": batch_size,
                "vision_tokens": mm_tokens_per_image,
                "is_encode": 0,
            }
            if comp_ctx_lengths is not None:
                spec["comp_ctx_lengths"] = comp_ctx_lengths
            if continuous_batching:
                spec["full_batch_size"] = kv_cache_batch_size or batch_size
            else:
                spec["batch_size"] = kv_cache_batch_size or batch_size
            return spec

        if comp_ctx_lengths_prefill and comp_ctx_lengths_decode:
            lang = [build_encoder_prefill_spec(ccl) for ccl in comp_ctx_lengths_prefill]
            lang.extend(build_decoder_canvas_spec(ccl) for ccl in comp_ctx_lengths_decode)
        else:
            lang = [build_encoder_prefill_spec(), build_decoder_canvas_spec()]

        if kv_offload:
            return {"vision": vision, "lang": lang}, compiler_options
        return lang, compiler_options

    def get_onnx_dynamic_axes(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
    ):
        text_cfg = self.config.text_config

        vision_dynamic_axes = {
            "pixel_values": {0: "batch_size", 1: "max_patches"},
            "image_position_ids": {0: "batch_size", 1: "max_patches"},
        }
        lang_dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "decoder_input_ids": {0: "batch_size", 1: "canvas_len"},
            "vision_embeds": {0: "vision_batch_size", 1: "vision_tokens"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "mm_token_type_ids": {0: "batch_size", 1: "seq_len"},
            "decoder_position_ids": {0: "batch_size", 1: "canvas_len"},
            "self_conditioning_logits": {0: "batch_size", 1: "canvas_len"},
        }
        if continuous_batching:
            lang_dynamic_axes["batch_index"] = {0: "batch_size"}

        for i, layer_type in enumerate(text_cfg.layer_types):
            if layer_type == "sliding_attention":
                ctx_axis = {
                    0: "full_batch_size" if continuous_batching else "batch_size",
                    2: "sliding_window",
                }
            else:
                ctx_axis = {
                    0: "full_batch_size" if continuous_batching else "batch_size",
                    2: "ctx_len",
                }
            for kv in ("key", "value"):
                lang_dynamic_axes[f"past_{kv}.{i}"] = ctx_axis

        if comp_ctx_lengths is not None:
            lang_dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}

        if kv_offload:
            return {"vision": vision_dynamic_axes, "lang": lang_dynamic_axes}
        return {**vision_dynamic_axes, **lang_dynamic_axes}

    def get_output_names(self, kv_offload: bool = False):
        text_cfg = self.config.text_config
        vision_output_names = ["vision_embeds"]
        # Unified output names for both encoder (canvas_len=1) and decoder (canvas_len=canvas_length).
        # Encoder: canvas_logits[bs,1,vocab] holds the first-token logit (TTFT).
        # Decoder: canvas_logits[bs,canvas_length,vocab] holds the denoised token logits.
        lang_output_names = [
            "canvas_logits",
            "vision_embeds_RetainedState",
            "image_idx_output",
        ]
        for i in range(text_cfg.num_hidden_layers):
            for kv in ("key", "value"):
                lang_output_names.append(f"past_{kv}.{i}_RetainedState")
        if kv_offload:
            return {"vision": vision_output_names, "lang": lang_output_names}
        return lang_output_names

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
    ):
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        fbs = constants.ONNX_EXPORT_EXAMPLE_FBS
        mm_tokens_per_image = self._get_mm_tokens_per_image()
        max_patches = self._get_vision_max_patches()
        canvas_length = getattr(self.config, "canvas_length", 256)
        text_cfg = self.config.text_config
        seq_len = max(constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN, mm_tokens_per_image + 32)
        patch_dim = getattr(self.config.vision_config, "patch_size", 16) ** 2 * 3

        # Build image_position_ids
        image_position_ids = torch.full((bs, max_patches, 2), -1, dtype=torch.int64)
        pooled_side = int(mm_tokens_per_image**0.5)
        patch_side = pooled_side * getattr(self.config.vision_config, "pooling_kernel_size", 3)
        xs = torch.arange(patch_side, dtype=torch.int64).view(1, -1).expand(patch_side, -1).reshape(-1)
        ys = torch.arange(patch_side, dtype=torch.int64).view(-1, 1).expand(-1, patch_side).reshape(-1)
        valid_positions = torch.stack((xs, ys), dim=-1)
        image_position_ids[:, : valid_positions.shape[0], :] = valid_positions.unsqueeze(0)

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
        # is_encode=1 (encoder sentinel) for the export trace.
        # Both encoder AND decoder paths are executed during torch.onnx.export because
        # is_encode flows through the graph as a real tensor input (not a Python bool).
        # The compiler constant-folds is_encode=1/0 per specialization.
        lang_inputs = {
            "input_ids": input_ids,
            "vision_embeds": torch.zeros((bs, mm_tokens_per_image, text_cfg.hidden_size), dtype=torch.float32),
            "position_ids": torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(bs, 1),
            "image_idx": torch.zeros((1, 1), dtype=torch.int64),
            "mm_token_type_ids": mm_token_type_ids,
            "decoder_input_ids": torch.zeros((bs, canvas_length), dtype=torch.int64),
            "decoder_position_ids": torch.arange(canvas_length, dtype=torch.int64).view(1, canvas_length).repeat(bs, 1),
            "self_conditioning_logits": torch.zeros((bs, canvas_length, text_cfg.vocab_size), dtype=torch.float32),
            "is_encode": torch.ones(1, dtype=torch.int64),
            "use_self_conditioning": torch.ones(1, dtype=torch.int64),
            "past_key_values": self.get_dummy_pkv_cache(
                config=text_cfg,
                batch_size=fbs if continuous_batching else bs,
                seq_len=seq_len,
            ),
        }
        if comp_ctx_lengths is not None:
            lang_inputs["comp_ctx_lengths"] = torch.randint(0, 100, (40,), dtype=torch.int8)
        if kv_offload:
            return {"vision": vision_inputs, "lang": lang_inputs}
        return {**vision_inputs, **lang_inputs}

    def _apply_logit_softcapping(self, logits: torch.Tensor) -> torch.Tensor:
        if self.final_logit_softcapping is not None:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping
        return logits

    def _inject_vision_embeds(
        self,
        input_ids: torch.LongTensor,
        vision_embeds: Optional[torch.Tensor],
        image_idx: Optional[torch.Tensor],
    ):
        """Inject vision features into the text embeddings at image-token positions.

        Shared by all three export paths (single-QPC forward, dual-QPC LangWrapper,
        disaggregated encoder-prefill). Uses torch.where instead of masked_scatter and
        clamps the gather index to avoid -1, both for export safety.
        """
        encoder_model = self.model.encoder
        lang_model = encoder_model.language_model
        text_cfg = self.config.text_config

        special_image_mask = input_ids == self.config.image_token_id
        llm_input_ids = input_ids.clone()
        llm_input_ids[special_image_mask] = text_cfg.pad_token_id
        inputs_embeds = lang_model.embed_tokens(llm_input_ids)

        next_image_idx = image_idx
        if input_ids.shape[1] != 1 and special_image_mask.any() and vision_embeds is not None:
            if vision_embeds.dim() == 2:
                vision_embeds = vision_embeds.unsqueeze(0)
            if next_image_idx is None:
                next_image_idx = torch.zeros((1, 1), dtype=torch.int64, device=inputs_embeds.device)

            indices1 = special_image_mask.to(torch.int64).cumsum(1) - 1
            indices1 = torch.where(
                indices1 >= 0,
                indices1 + next_image_idx.to(indices1.device),
                indices1,
            )
            indices0 = torch.arange(special_image_mask.shape[0], device=inputs_embeds.device).view(-1, 1)
            safe_indices1 = indices1.clamp(min=0)
            gathered_vision_embeds = vision_embeds[indices0, safe_indices1]
            inputs_embeds = torch.where(special_image_mask.unsqueeze(-1), gathered_vision_embeds, inputs_embeds)
            next_image_idx = (indices1.max() + 1).reshape(1, 1)

        if next_image_idx is None:
            next_image_idx = torch.zeros((1, 1), dtype=torch.int64, device=inputs_embeds.device)

        return inputs_embeds, next_image_idx

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        cache_position_ids: torch.LongTensor,
        full_attention_mask: torch.Tensor,
        sliding_attention_mask: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        vision_embeds: Optional[torch.Tensor] = None,
        image_idx: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        self_conditioning_logits: Optional[torch.FloatTensor] = None,
        is_encode: Optional[torch.LongTensor] = None,
        use_self_conditioning: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """Compatibility entry point for the shared unified-QPC contract."""
        canvas_logits, next_image_idx, past_key_values = self.get_qeff_unified_wrapper()(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_position_ids=cache_position_ids,
            full_attention_mask=full_attention_mask,
            sliding_attention_mask=sliding_attention_mask,
            past_key_values=past_key_values,
            vision_embeds=vision_embeds,
            image_idx=image_idx,
            mm_token_type_ids=mm_token_type_ids,
            self_conditioning_logits=self_conditioning_logits,
            is_encode=is_encode,
            use_self_conditioning=use_self_conditioning,
            batch_index=batch_index,
            **kwargs,
        )
        return canvas_logits, vision_embeds, next_image_idx, past_key_values
