# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from typing import Optional, Type, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs


def qeff_apply_deepseek_v4_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> torch.Tensor:
    """ONNX-friendly DeepSeek-V4 interleaved RoPE.

    Upstream Transformers expands the half-sized cos/sin tables with
    ``repeat_interleave`` and then rotates with a helper that stacks/slices the
    already-expanded tensor. The legacy torch ONNX tracer can infer an extra
    broadcast rank for that sequence in DeepSeek-V4 and then fail while lowering
    the final ``cat([nope, rotated])``. Keep the same math but operate on paired
    even/odd channels directly, so cos/sin stay half-width and broadcast with a
    stable rank during export.
    """
    cos = cos.reshape(cos.shape[0], cos.shape[1], -1).unsqueeze(unsqueeze_dim)
    sin = sin.reshape(sin.shape[0], sin.shape[1], -1).unsqueeze(unsqueeze_dim)
    rope_dim = cos.shape[-1] * 2
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rope_even = rope[..., 0::2].float()
    rope_odd = rope[..., 1::2].float()
    rotated_even = rope_even * cos - rope_odd * sin
    rotated_odd = rope_odd * cos + rope_even * sin
    rotated = torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2).to(x.dtype)
    return torch.cat((nope, rotated), dim=-1)


def patch_deepseek_v4_rotary_for_qeff_export() -> None:
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as hf_deepseek_v4

    if getattr(hf_deepseek_v4.apply_rotary_pos_emb, "_qeff_patched", False):
        return
    qeff_apply_deepseek_v4_rotary_pos_emb._qeff_patched = True
    hf_deepseek_v4.apply_rotary_pos_emb = qeff_apply_deepseek_v4_rotary_pos_emb


def qeff_deepseek_v4_cache_update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
    if not self.is_initialized:
        self.lazy_initialization(key_states, value_states)
        self.values = self.keys

    self.cumulative_length += key_states.shape[-2]
    full = torch.cat([self.keys, key_states], dim=-2)
    if self.keys.dim() == key_states.dim() and self.keys.shape[-2] >= key_states.shape[-2]:
        self.keys = torch.cat([self.keys[:, :, key_states.shape[-2] :, :], key_states], dim=-2)
    else:
        start = max(0, full.shape[-2] - self.sliding_window + 1)
        self.keys = full[:, :, start:, :]
    self.values = self.keys
    return full, full


def patch_deepseek_v4_cache_for_qeff_export() -> None:
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as hf_deepseek_v4

    if getattr(hf_deepseek_v4.DeepseekV4HCACache.update, "_qeff_patched", False):
        return
    qeff_deepseek_v4_cache_update._qeff_patched = True
    hf_deepseek_v4.DeepseekV4HCACache.update = qeff_deepseek_v4_cache_update
    hf_deepseek_v4.DeepseekV4CSACache.update = qeff_deepseek_v4_cache_update


def build_deepseek_v4_cache(config, legacy_cache=None) -> Cache:
    """Build DeepSeek-V4 CSA/HCA cache layers from native Transformers classes.

    DeepSeek-V4 attention layers need cache layers with compressor/indexer state
    methods such as ``store_compression_weights``. Generic ``DynamicCache`` still
    instantiates plain ``DynamicLayer`` for V4 layer types in Transformers 5.10.1,
    so QEff builds the native V4 cache layers explicitly. During ONNX export,
    QEff passes legacy tensor tuples; ``legacy_cache`` seeds the sliding KV rows
    into those native cache layers before forward runs.
    """
    patch_deepseek_v4_rotary_for_qeff_export()
    patch_deepseek_v4_cache_for_qeff_export()

    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4CSACache, DeepseekV4HCACache

    legacy_cache = tuple(legacy_cache or ())
    layers = []
    for layer_idx, layer_type in enumerate(config.layer_types):
        if layer_type == "compressed_sparse_attention":
            layer = DeepseekV4CSACache(config)
        else:
            layer = DeepseekV4HCACache(config)

        if layer_idx < len(legacy_cache) and len(legacy_cache[layer_idx]) >= 2:
            key_states, value_states = legacy_cache[layer_idx][0], legacy_cache[layer_idx][1]
            layer.keys = key_states
            layer.values = key_states if value_states is None else value_states
            layer.dtype = key_states.dtype
            layer.device = key_states.device
            layer.is_initialized = True
            layer.cumulative_length = int(key_states.shape[-2])
        layers.append(layer)
    return Cache(layers=layers)


class QEffDeepseekV4RMSNorm(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from QEfficient.customop.rms_norm import CustomRMSNormFunc

        return CustomRMSNormFunc.apply(hidden_states, self.weight, self.variance_epsilon)


class QEffDeepseekV4Experts(nn.Module):
    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        tokens = hidden_states.shape[0]
        expanded_hidden_states = hidden_states.unsqueeze(0).expand(self.num_experts, tokens, self.hidden_dim)
        gate_up = torch.bmm(expanded_hidden_states, self.gate_up_proj.transpose(1, 2))
        expert_hidden_states = self._apply_gate(gate_up)
        expert_outputs = torch.bmm(expert_hidden_states, self.down_proj.transpose(1, 2))

        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).to(top_k_weights.dtype)
        expert_weights = torch.einsum("tke,tk->te", expert_mask, top_k_weights)
        return torch.einsum("eth,te->th", expert_outputs, expert_weights).to(hidden_states.dtype)


class QEffDeepseekV4ForCausalLM(nn.Module):
    def get_submodules_for_export(self) -> Type[nn.Module]:
        return set()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        patch_deepseek_v4_rotary_for_qeff_export()
        patch_deepseek_v4_cache_for_qeff_export()

        if past_key_values is None:
            past_key_values = build_deepseek_v4_cache(self.config)
        elif not isinstance(past_key_values, Cache):
            past_key_values = build_deepseek_v4_cache(self.config, past_key_values)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        if position_ids is not None:
            hidden_states = hidden_states[:, -1:, :]
        elif isinstance(logits_to_keep, int) and logits_to_keep:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        elif not isinstance(logits_to_keep, int):
            hidden_states = hidden_states[:, logits_to_keep, :]

        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_dummy_pkv_cache(self, config, batch_size, seq_len):
        return build_deepseek_v4_cache(config)
