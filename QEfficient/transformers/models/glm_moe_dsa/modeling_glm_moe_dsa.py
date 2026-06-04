# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
QEfficient modeling for GLM-5.1 (``glm_moe_dsa`` / ``GlmMoeDsaForCausalLM``).

Architecture (native HF, transformers >= 5.4.0):
  - Multi-head Latent Attention (MLA), DeepSeek-V3 style:
        x -> q_a_proj -> RMSNorm -> q_b_proj -> split(q_nope, q_pe) -> RoPE(q_pe)
        x -> kv_a_proj_with_mqa -> split(kv_compressed, k_pe) -> RMSNorm(kv_compressed)
                                                              -> kv_b_proj -> (k_nope, value)
    Cache holds the fully-expanded key/value (per-head qk_head_dim / v_head_dim) — the same
    convention the HF reference uses, so it plugs straight into QEffDynamicCache.
  - Grouped top-k MoE (256 routed experts, top-8, 1 shared expert). With ``n_group == 1`` the
    group-masking collapses to a plain global top-k, so we route globally and aggregate the
    chosen experts with a batched BMM (M3) + einsum reduction (M1).
  - DSA (Disentangled Sparse Attention) indexer: OMITTED for this integration. The indexer keeps
    the ``index_topk`` highest-scoring keys per query; with ``index_topk == 2048`` and a compiled
    ``ctx_len <= 2048`` every key is always retained, so the sparse mask is all-zeros and attention
    is identical to dense causal MLA. See modeling-summary.md for the ctx bound. If a future compile
    uses ctx_len > index_topk, the indexer must be added back (see HF GlmMoeDsaIndexer).

Reference QEff files: deepseek_v3 (MLA) and glm4_moe (grouped-topk MoE + precomputed RoPE).
"""

from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaAttention,
    GlmMoeDsaConfig,
    GlmMoeDsaDecoderLayer,
    GlmMoeDsaForCausalLM,
    GlmMoeDsaModel,
    GlmMoeDsaMoE,
    repeat_kv,
    rotate_half,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from QEfficient.blocking.attention_blocking import past_key_value_update
from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


class QEffGlmMoeDsaRotaryEmbedding(nn.Module):
    """Precomputed (static) RoPE cache for GLM-MoE-DSA.

    ``glm_moe_dsa`` uses the default split-half (NeoX/Llama) RoPE applied only to the rope slice
    (``qk_rope_head_dim`` dims). ``rope_theta`` and the rotary dim are read from the config.
    """

    def __init__(self, config: GlmMoeDsaConfig, device=None):
        super().__init__()
        self.config = config
        self.dim = config.qk_rope_head_dim
        self.base = config.rope_parameters["rope_theta"]
        self.max_position_embeddings = config.max_position_embeddings
        self.attention_scaling = 1.0  # rope_type == "default"

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=self.max_position_embeddings, device=inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
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


def qeff_apply_precomputed_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
):
    """Split-half RoPE on the first ``rotary_dim`` channels; pass the rest through unchanged.

    Here q/k are the rope slices (q_pe / k_pe) whose head dim equals ``rotary_dim`` == qk_rope_head_dim,
    so ``q_pass`` / ``k_pass`` are empty — but we keep the general form to mirror glm4_moe.
    """
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    half_dim = rotary_dim // 2

    q_half = torch.cat((-q_rot[..., half_dim:], q_rot[..., :half_dim]), dim=-1)
    k_half = torch.cat((-k_rot[..., half_dim:], k_rot[..., :half_dim]), dim=-1)

    q_embed = (q_rot * cos) + (q_half * sin)
    k_embed = (k_rot * cos) + (k_half * sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        # U1 (masked value via fork constant) + U3 (torch.where, no boolean indexing)
        attn_weights = torch.where(
            attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=attn_weights.dtype), attn_weights
        )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class QEffGlmMoeDsaAttention(GlmMoeDsaAttention):
    """MLA attention with precomputed RoPE, KV-cache update and torch.where masking.

    Caches the fully-expanded key/value (matching the HF reference), so QEffDynamicCache works
    unchanged. The DSA indexer is intentionally not invoked — see module docstring.
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

        # ===== Query (LoRA) path =====
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.qk_head_dim).transpose(1, 2)  # [B, H, S, qk_head_dim]
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # ===== KV (MLA compressed) path =====
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # [B, S, kv_lora_rank + rope]
        k_compressed, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)  # [B, 1, S, rope]

        kv = self.kv_b_proj(self.kv_a_layernorm(k_compressed))
        kv = kv.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # ===== RoPE (precomputed, split-half over the rope slice) =====
        rotary_dim = self.qk_rope_head_dim
        q_pe, k_pe = qeff_apply_precomputed_rotary_pos_emb(q_pe, k_pe, cos_cached, sin_cached, rotary_dim)

        query_states = torch.cat([q_nope, q_pe], dim=-1)  # [B, H, S, qk_head_dim]
        k_pe = k_pe.expand(-1, self.num_heads, -1, -1)
        key_states = torch.cat([k_nope, k_pe], dim=-1)  # [B, H, S, qk_head_dim]

        # ===== KV cache update (handles CCL mask slice — C1) =====
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
    """Grouped top-k MoE. With n_group == 1 the grouping is a no-op, so we route globally and
    aggregate the chosen experts via batched BMM (M3) and einsum reduction (M1)."""

    def __qeff_init__(self):
        # GlmMoeDsaNaiveMoe stores experts as 3D tensors: gate_up_proj [E, 2*I, H], down_proj [E, H, I].
        gate_proj, up_proj = self.experts.gate_up_proj.chunk(2, dim=1)  # each [E, I, H]
        self.all_gate_proj = torch.nn.Parameter(gate_proj.transpose(1, 2).contiguous())  # [E, H, I]
        self.all_up_proj = torch.nn.Parameter(up_proj.transpose(1, 2).contiguous())  # [E, H, I]
        self.all_down_proj = torch.nn.Parameter(self.experts.down_proj.transpose(1, 2).contiguous())  # [E, I, H]
        self.act_fn = self.experts.act_fn
        self.num_experts = self.experts.num_experts

    def route_tokens_to_experts(self, router_logits):
        # router_logits: [T, E] (raw linear output from the gate)
        router_scores = router_logits.sigmoid()
        scores_for_choice = router_scores + self.gate.e_score_correction_bias.unsqueeze(0)
        # n_group == 1 / topk_group == 1 → the single group is always selected, so grouped top-k
        # is identical to a global top-k over all experts.
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]  # [T, top_k]
        topk_weights = router_scores.gather(1, topk_indices)  # weights from un-biased scores
        if self.norm_topk_prob:
            # M1/M5: einsum reduction instead of .sum (subfunction-safe)
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
        # M1: einsum aggregation over the top-k axis (NOT .sum(dim=1))
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
    """Pass-through layer threading the QEff attention signature + precomputed RoPE.

    ``self.mlp`` is either the dense GlmMoeDsaMLP (first ``first_k_dense_replace`` layers) or the
    transformed QEffGlmMoeDsaMoE — both take a single hidden_states argument.
    """

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
    """L1: single precomputed RoPE at model level, indexed by position_ids and passed to layers."""

    def __qeff_init__(self):
        self.rotary_emb = QEffGlmMoeDsaRotaryEmbedding(config=self.config)
        self.sin_cached = torch.nn.Parameter(self.rotary_emb.sin_cached * self.rotary_emb.attention_scaling)
        self.cos_cached = torch.nn.Parameter(self.rotary_emb.cos_cached * self.rotary_emb.attention_scaling)

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
        sin = self.sin_cached[position_ids].unsqueeze(1)  # [B, 1, S, rope]
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

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


class QEffGlmMoeDsaForCausalLM(GlmMoeDsaForCausalLM):
    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffGlmMoeDsaDecoderLayer}

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
        # U2: INT32 logit gather — keep only the last valid position per sequence.
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = hidden_states[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states).to(hidden_states.dtype)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )


# Reuse the canonical chunked-blocked expert kernel from the sibling MoE arch — same
# expert weight layout (`all_{gate,up,down}_proj` in `[E,H,I]` / `[E,I,H]`).
from QEfficient.transformers.models.glm4_moe.modeling_glm4_moe import (
    _cumsum_scatter_gather_update_expert_blocked,
)


class QEffPrefillChunkedGlmMoeDsaMoE(QEffGlmMoeDsaMoE):
    """Prefill-only chunked variant of the GLM-MoE-DSA MoE block.

    Activated by ``PrefillOnlyChunkedTransform`` for the prefill QPC of a disaggregated
    compile. Splits the routed experts across ``expert_blocking_num_nsp`` virtual slots and
    streams tokens through them in packed chunks of ``expert_blocking_packed_chunk_size``.
    Both attributes are injected by ``modeling_auto`` after the transform applies.

    The expert-side math is identical to ``QEffGlm4MoeMoE``'s prefill-chunked path because
    the routed-expert weight layout is the same (3D fused gate_up_proj / down_proj). The
    only GLM-MoE-DSA specifics are kept in the parent class: the sigmoid-gated router with
    ``e_score_correction_bias`` for selection (un-biased scores for weighting), and the
    shared-experts add at the end of ``forward``.
    """

    supports_moe_prefill_blocking = True

    def _forward_expert_blocked(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        T, H = hidden_states.shape
        num_experts = self.num_experts
        num_nsp = self.expert_blocking_num_nsp
        if num_experts % num_nsp != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by expert_blocking_num_nsp ({num_nsp})"
            )

        routing_weights = hidden_states.new_zeros((T, num_experts))
        routing_weights.scatter_(1, topk_indices, topk_weights)

        local_experts = num_experts // num_nsp
        rw = (
            routing_weights.transpose(0, 1)
            .contiguous()
            .view(local_experts, num_nsp, T)
            .transpose(0, 1)
            .contiguous()
        )
        W_g = self.all_gate_proj.view(local_experts, num_nsp, H, -1).transpose(0, 1).contiguous()
        W_u = self.all_up_proj.view(local_experts, num_nsp, H, -1).transpose(0, 1).contiguous()
        W_d = self.all_down_proj.view(local_experts, num_nsp, -1, H).transpose(0, 1).contiguous()
        expert_out = hidden_states.new_zeros((num_nsp, T, H))
        routing_weights_unsqueezed = rw.unsqueeze(-1)

        for slot in range(local_experts):
            expert_out = _cumsum_scatter_gather_update_expert_blocked(
                x=hidden_states,
                T2Ei=rw[:, slot, :] > 0,
                W_g=W_g[:, slot],
                W_u=W_u[:, slot],
                W_d=W_d[:, slot],
                routing_weight=routing_weights_unsqueezed[:, slot],
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
