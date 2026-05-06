# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
    Glm4MoeLiteAttention,
    Glm4MoeLiteConfig,
    Glm4MoeLiteDecoderLayer,
    Glm4MoeLiteForCausalLM,
    Glm4MoeLiteMoE,
    Glm4MoeLiteModel,
    Glm4MoeLiteRotaryEmbedding,
    repeat_kv,
    rotate_half,
)

from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


class QEffGlm4MoeLiteRotaryEmbedding(Glm4MoeLiteRotaryEmbedding):
    def __init__(self, config: Glm4MoeLiteConfig, device=None):
        super().__init__(config=config, device=device)
        self._set_cos_sin_cache(
            seq_len=self.original_max_seq_len,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", (emb.cos() * self.attention_scaling).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self.attention_scaling).to(dtype), persistent=False)


def qeff_apply_rotary_pos_emb_interleave(q_rot, k_rot, cos_cached, sin_cached, position_ids):
    """Interleaved RoPE: reorders elements before applying standard rotation."""
    cos = cos_cached[position_ids].unsqueeze(1)  # [B, 1, S, head_dim]
    sin = sin_cached[position_ids].unsqueeze(1)

    b, h, s, d = q_rot.shape
    q_rot = q_rot.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    b, h, s, d = k_rot.shape
    k_rot = k_rot.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return q_embed.to(q_rot.dtype), k_embed.to(k_rot.dtype)


def qeff_apply_rotary_pos_emb(q_rot, k_rot, cos_cached, sin_cached, position_ids):
    cos = cos_cached[position_ids].unsqueeze(1)  # [B, 1, S, head_dim]
    sin = sin_cached[position_ids].unsqueeze(1)
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return q_embed.to(q_rot.dtype), k_embed.to(k_rot.dtype)


def eager_attention_forward(module, query, key, value, attention_mask, scaling):
    key_states = repeat_kv(key.to(query.dtype), module.num_key_value_groups)
    value_states = repeat_kv(value.to(query.dtype), module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = torch.where(
            attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), attn_weights.float()
        )
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class QEffGlm4MoeLiteAttention(Glm4MoeLiteAttention):
    """MLA attention adapted for QEfficient: static RoPE cache, QEffDynamicCache, no flash attn."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cos_cached: Optional[torch.Tensor] = None,
        sin_cached: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_length = hidden_states.shape[:2]

        # Q projection (with optional LoRA decomposition)
        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(batch_size, seq_length, -1, self.qk_head_dim).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV low-rank compression (MLA): project → split → decompress
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass_compressed, k_rot = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_full = self.kv_b_proj(self.kv_a_layernorm(k_pass_compressed))
        kv_full = kv_full.view(batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_pass, value_states = torch.split(kv_full, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # k_rot starts as [B, S, rope_dim] → reshape to [B, 1, S, rope_dim] for broadcasting
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        if self.config.rope_interleave:
            q_rot, k_rot = qeff_apply_rotary_pos_emb_interleave(
                q_rot, k_rot, cos_cached, sin_cached, position_ids
            )
        else:
            q_rot, k_rot = qeff_apply_rotary_pos_emb(q_rot, k_rot, cos_cached, sin_cached, position_ids)

        # Expand single-head k_rot to all attention heads
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is not None:
            cache_kwargs = {"batch_index": batch_index, "position_ids": position_ids}
            if comp_ctx_lengths is not None:
                attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
                cache_kwargs["CCL"] = attention_mask.shape[-1]
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output, _ = eager_attention_forward(
            self, query_states, key_states, value_states, attention_mask, self.scaling
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class QEffGlm4MoeLiteMoE(Glm4MoeLiteMoE):
    """MoE block with explicit expert loop replacing nonzero()-based dynamic dispatch."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        B, S, H = hidden_states.shape
        T = B * S
        x = hidden_states.view(T, H)

        # Routing: gate flattens hidden_states internally, returns [T, E]
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)

        # Build dense routing matrix [T, n_routed_experts] via scatter
        routing_matrix = torch.zeros(T, self.n_routed_experts, device=hidden_states.device)
        routing_matrix.scatter_(1, topk_indices, topk_weights.float())

        # Static expert loop — trace-compatible (no nonzero/dynamic indexing)
        expert_out = torch.zeros(T, H, device=hidden_states.device)
        for expert_idx in range(self.n_routed_experts):
            routing_weight = routing_matrix[:, expert_idx].unsqueeze(-1)  # [T, 1]
            # gate_up_proj[e]: [2*I, H],  down_proj[e]: [H, I]
            gate_up = F.linear(x, self.experts.gate_up_proj[expert_idx])  # [T, 2*I]
            gate, up = gate_up.chunk(2, dim=-1)
            expert_hidden = self.experts.act_fn(gate) * up  # [T, I]
            out = F.linear(expert_hidden, self.experts.down_proj[expert_idx])  # [T, H]
            expert_out = expert_out + out * routing_weight

        expert_out = expert_out.view(*orig_shape)
        return expert_out.to(residuals.dtype) + self.shared_experts(residuals)


class QEffGlm4MoeLiteDecoderLayer(Glm4MoeLiteDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        cos_cached: Optional[torch.Tensor] = None,
        sin_cached: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            cache_position=cache_position,
            cos_cached=cos_cached,
            sin_cached=sin_cached,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class QEffGlm4MoeLiteModel(Glm4MoeLiteModel):
    def __qeff_init__(self):
        self.rotary_emb = QEffGlm4MoeLiteRotaryEmbedding(config=self.config)
        self.sin_cached = torch.nn.Parameter(self.rotary_emb.sin_cached)
        self.cos_cached = torch.nn.Parameter(self.rotary_emb.cos_cached)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> BaseModelOutputWithPast:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=past_key_values_length)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                use_cache=use_cache,
                cache_position=cache_position,
                cos_cached=self.cos_cached,
                sin_cached=self.sin_cached,
            )

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_values = past_key_values.to_legacy_cache()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


class QEffGlm4MoeLiteForCausalLM(Glm4MoeLiteForCausalLM):
    def __qeff_init__(self):
        self.to(torch.float16)

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffGlm4MoeLiteDecoderLayer}

    def get_dummy_pkv_cache(self, config, batch_size, seq_len):
        n_heads = config.num_key_value_heads
        k_head_dim = config.qk_head_dim
        v_head_dim = config.v_head_dim
        past_key_values = []
        for _ in range(config.num_hidden_layers):
            key_cache = torch.zeros([batch_size, n_heads, seq_len, k_head_dim], dtype=torch.float16)
            value_cache = torch.zeros([batch_size, n_heads, seq_len, v_head_dim], dtype=torch.float16)
            past_key_values.append((key_cache, value_cache))
        return past_key_values

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            inputs_embeds=inputs_embeds,
            batch_index=batch_index,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )

        logit_idx = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[torch.arange(position_ids.shape[0]).view(-1, 1), logit_idx]
        logits = self.lm_head(hidden_states).float()

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )
