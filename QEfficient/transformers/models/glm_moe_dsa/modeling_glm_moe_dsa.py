# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""QEfficient modeling for ``glm_moe_dsa`` (zai-org/GLM-5.1).

MLA attention (DeepSeek-V3 family) + grouped top-k MoE (256 routed × top-8,
n_group=1 collapses to global top-k). The DSA indexer is omitted: with
``index_topk=2048`` and a compiled ``ctx_len <= 2048`` every key is retained,
so the sparse mask is all-zeros and attention is identical to dense MLA. Any
``ctx_len > 2048`` requires reintroducing ``GlmMoeDsaIndexer``.

Reference siblings: ``glm4_moe`` (grouped-topk MoE + chunked-prefill kernel),
``deepseek_v3`` (MLA).
"""

from typing import List, Optional, Tuple, Type, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaAttention,
    GlmMoeDsaDecoderLayer,
    GlmMoeDsaForCausalLM,
    GlmMoeDsaModel,
    GlmMoeDsaMoE,
    GlmMoeDsaRotaryEmbedding,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from QEfficient.blocking.attention_blocking import past_key_value_update
from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.transformers.models.glm4_moe.modeling_glm4_moe import (
    _cumsum_scatter_gather_update_expert_blocked,
    eager_attention_forward,
    qeff_apply_precomputed_rotary_pos_emb,
)


class QEffGlmMoeDsaRotaryEmbedding(GlmMoeDsaRotaryEmbedding):
    """Static cos/sin cache; only the rope slice (``qk_rope_head_dim``) is rotated."""

    def __init__(self, config, device=None):
        super().__init__(config=config)
        self._set_cos_sin_cache(
            seq_len=config.max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype) * self.attention_scaling,
            self.sin_cached[:seq_len].to(dtype=x.dtype) * self.attention_scaling,
        )


class QEffGlmMoeDsaAttention(GlmMoeDsaAttention):
    """MLA attention with precomputed RoPE + QEffDynamicCache update.

    Caches fully-expanded per-head K/V (matching the HF reference), so the
    standard cache works unchanged. The DSA indexer is intentionally not
    invoked — see module docstring.
    """

    def __qeff_init__(self):
        self.rotary_emb = QEffGlmMoeDsaRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        sin_cached: Optional[torch.Tensor] = None,
        cos_cached: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_compressed, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        kv = self.kv_b_proj(self.kv_a_layernorm(k_compressed))
        kv = kv.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        q_pe, k_pe = qeff_apply_precomputed_rotary_pos_emb(q_pe, k_pe, cos_cached, sin_cached, self.qk_rope_head_dim)

        query_states = torch.cat([q_nope, q_pe], dim=-1)
        k_pe = k_pe.expand(-1, self.num_heads, -1, -1)
        key_states = torch.cat([k_nope, k_pe], dim=-1)

        if past_key_value is not None:
            key_states, value_states, attention_mask, _ = past_key_value_update(
                module=self,
                key=key_states,
                value=value_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                position_ids=position_ids,
            )

        attn_output, attn_weights = eager_attention_forward(
            self, query_states, key_states, value_states, attention_mask, scaling=self.scaling, **kwargs
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QEffGlmMoeDsaMoE(GlmMoeDsaMoE):
    """Grouped top-k MoE (n_group=1 → global top-k); batched-BMM aggregation."""

    def __qeff_init__(self):
        gate_proj, up_proj = self.experts.gate_up_proj.chunk(2, dim=1)
        self.all_gate_proj = nn.Parameter(gate_proj.transpose(1, 2).contiguous())
        self.all_up_proj = nn.Parameter(up_proj.transpose(1, 2).contiguous())
        self.all_down_proj = nn.Parameter(self.experts.down_proj.transpose(1, 2).contiguous())
        self.act_fn = self.experts.act_fn
        self.num_experts = self.experts.num_experts

    def route_tokens_to_experts(self, router_logits):
        router_scores = router_logits.sigmoid()
        scores_for_choice = router_scores + self.gate.e_score_correction_bias.unsqueeze(0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = torch.einsum("ab->a", topk_weights).unsqueeze(-1) + 1e-20
            topk_weights = topk_weights / denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        bs_seq, _ = hidden_states.shape
        gate_proj = self.all_gate_proj[topk_indices.flatten()]
        up_proj = self.all_up_proj[topk_indices.flatten()]
        down_proj = self.all_down_proj[topk_indices.flatten()]
        expert_in = (
            hidden_states.unsqueeze(1).expand(-1, self.top_k, -1).contiguous().view(-1, 1, self.config.hidden_size)
        )
        gate_out = torch.bmm(expert_in, gate_proj)
        up_out = torch.bmm(expert_in, up_proj)
        hidden = self.act_fn(gate_out) * up_out
        expert_output = torch.bmm(hidden, down_proj)
        experts_out = expert_output.view(bs_seq, self.top_k, self.config.hidden_size)
        experts_out = experts_out * topk_weights.unsqueeze(-1)
        final_hidden_states = torch.einsum("abc->ac", experts_out)
        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class QEffGlmMoeDsaDecoderLayer(GlmMoeDsaDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        sin_cached: Optional[torch.Tensor] = None,
        cos_cached: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            sin_cached=sin_cached,
            cos_cached=cos_cached,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class QEffGlmMoeDsaModel(GlmMoeDsaModel):
    """Single model-level precomputed RoPE indexed by position_ids."""

    def __qeff_init__(self):
        self.rotary_emb = QEffGlmMoeDsaRotaryEmbedding(config=self.config)
        self.sin_cached = nn.Parameter(self.rotary_emb.sin_cached * self.rotary_emb.attention_scaling)
        self.cos_cached = nn.Parameter(self.rotary_emb.cos_cached * self.rotary_emb.attention_scaling)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
            position_ids = cache_position.unsqueeze(0)

        attention_mask = _create_causal_mask(position_ids=position_ids, target_length=past_seen_tokens)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        sin = self.sin_cached[position_ids].unsqueeze(1)
        cos = self.cos_cached[position_ids].unsqueeze(1)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                use_cache=use_cache,
                sin_cached=sin,
                cos_cached=cos,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if past_key_values is not None and hasattr(past_key_values, "to_legacy_cache"):
            past_key_values = past_key_values.to_legacy_cache()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


class QEffGlmMoeDsaForCausalLM(GlmMoeDsaForCausalLM):
    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffGlmMoeDsaDecoderLayer}

    def get_dummy_pkv_cache(self, config, batch_size, seq_len):
        # MLA: K = qk_nope_head_dim + qk_rope_head_dim, V = v_head_dim (asymmetric).
        k_shape = (batch_size, config.num_attention_heads, seq_len, config.qk_nope_head_dim + config.qk_rope_head_dim)
        v_shape = (batch_size, config.num_attention_heads, seq_len, config.v_head_dim)
        return [
            [torch.zeros(k_shape, dtype=config.torch_dtype), torch.zeros(v_shape, dtype=config.torch_dtype)]
            for _ in range(config.num_hidden_layers)
        ]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = hidden_states[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states).to(hidden_states.dtype)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )


class QEffPrefillChunkedGlmMoeDsaMoE(QEffGlmMoeDsaMoE):
    """Prefill-only chunked MoE. Reuses ``glm4_moe``'s ``_cumsum_scatter_gather_update_expert_blocked``
    kernel (same expert weight layout)."""

    supports_moe_prefill_blocking = True

    def _forward_expert_blocked(self, hidden_states, topk_indices, topk_weights):
        T, H = hidden_states.shape
        num_experts = self.num_experts
        num_nsp = self.expert_blocking_num_nsp
        if num_experts % num_nsp != 0:
            raise ValueError(f"num_experts ({num_experts}) must be divisible by expert_blocking_num_nsp ({num_nsp})")

        routing_weights = hidden_states.new_zeros((T, num_experts))
        routing_weights.scatter_(1, topk_indices, topk_weights)

        local_experts = num_experts // num_nsp
        rw = routing_weights.transpose(0, 1).contiguous().view(local_experts, num_nsp, T).transpose(0, 1).contiguous()
        W_g = self.all_gate_proj.view(local_experts, num_nsp, H, -1).transpose(0, 1).contiguous()
        W_u = self.all_up_proj.view(local_experts, num_nsp, H, -1).transpose(0, 1).contiguous()
        W_d = self.all_down_proj.view(local_experts, num_nsp, -1, H).transpose(0, 1).contiguous()
        expert_out = hidden_states.new_zeros((num_nsp, T, H))
        rw_unsq = rw.unsqueeze(-1)

        for slot in range(local_experts):
            expert_out = _cumsum_scatter_gather_update_expert_blocked(
                x=hidden_states,
                T2Ei=rw[:, slot, :] > 0,
                W_g=W_g[:, slot],
                W_u=W_u[:, slot],
                W_d=W_d[:, slot],
                routing_weight=rw_unsq[:, slot],
                expert_out=expert_out,
                act_fn=self.act_fn,
                packed_chunk_size=self.expert_blocking_packed_chunk_size,
            )
        return expert_out.sum(dim=0)

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self._forward_expert_blocked(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states
