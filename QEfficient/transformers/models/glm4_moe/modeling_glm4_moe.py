# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Any, Dict, List, Optional, Type, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.glm4_moe.modeling_glm4_moe import (
    Glm4MoeAttention,
    Glm4MoeConfig,
    Glm4MoeDecoderLayer,
    Glm4MoeForCausalLM,
    Glm4MoeModel,
    Glm4MoeMoE,
    Glm4MoeRotaryEmbedding,
    repeat_kv,
    rotate_half,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from QEfficient.blocking.attention_blocking import (
    AttentionBlockingConfig,
    BlockingMode,
    generic_blocked_attention_interface,
    past_key_value_update,
)
from QEfficient.customop.ctx_scatter_gather import (
    CtxGatherFunc3DGeneralized,
    CtxScatterFunc3DGeneralized,
    CtxScatterFunc3DInt,
)
from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


class QEffGlm4MoeRotaryEmbedding(Glm4MoeRotaryEmbedding):
    """
    Copied from Glm4MoeForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm4_moe/modeling_glm4_moe.py
    The only differences are:
    - Add static sin/cos computations.
    """

    def __init__(self, config: Glm4MoeConfig, device=None):
        super().__init__(config=config)

        self._set_cos_sin_cache(
            seq_len=self.original_max_seq_len, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype) * self.attention_scaling,
            self.sin_cached[:seq_len].to(dtype=x.dtype) * self.attention_scaling,
        )


def qeff_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 1,
):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)

    # Cast back to original dtype
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def qeff_apply_precomputed_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
):
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


def eager_attention_forward_blocked_kv(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_kv_blocks: Optional[torch.Tensor] = None,
    cache_kwargs: Optional[Dict[str, Any]] = None,
    layer_idx: int = None,
    past_key_value: Optional[Cache] = None,
    **kwargs,
):
    # Initialize result tensor
    output = torch.zeros_like(query)

    # Initialize Running Maximum
    batch_size, num_heads, seq_len, _ = query.shape
    current_max = torch.full((batch_size, num_heads, seq_len), (MIN_MASKED_ATTENTION_VALUE).to(query.dtype))

    # Initialize Denominator
    current_denominator = torch.zeros(batch_size, num_heads, seq_len)

    past_seen_tokens = cache_kwargs.get("past_seen_tokens")
    position_ids = cache_kwargs.get("position_ids")
    block_size = -(-past_seen_tokens // num_kv_blocks)
    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=query.dtype)

    for j in range(num_kv_blocks):
        start_index = j * block_size
        end_index = (j + 1) * block_size
        K_block, V_block = past_key_value.read_only_blockedKV(start_index, end_index, layer_idx, cache_kwargs)
        K_block_states = repeat_kv(K_block, module.num_key_value_groups)
        V_block_states = repeat_kv(V_block, module.num_key_value_groups)
        past_seen_tokens_start = start_index
        past_seen_tokens_end = torch.where(
            torch.tensor(past_seen_tokens, dtype=torch.int) < torch.tensor(end_index, dtype=torch.int),
            past_seen_tokens,
            end_index,
        )
        causal_mask_block = _create_causal_mask(
            position_ids=position_ids, target_length=past_seen_tokens_end, start_index=past_seen_tokens_start
        )

        # Compute attention scores for the block
        attn_weights_block = torch.matmul(query, K_block_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights_block = torch.where(causal_mask_block, masked_tensor, attn_weights_block)

        # Update Running row maximum
        prev_max = current_max
        current_max = torch.max(prev_max, attn_weights_block.max(dim=-1).values)
        delta_max = prev_max - current_max

        current_exp = torch.exp(
            attn_weights_block - current_max.unsqueeze(-1)
        )  # Subract current_max from each column of attn_weights_block

        # update running denominator
        prev_denominator = current_denominator
        # Replace .sum() to fix the ReduceSum Issuse in subfunction
        curr_exp_sum = torch.einsum("bhqk->bhq", current_exp)
        current_denominator = prev_denominator * torch.exp(delta_max) + curr_exp_sum

        prob = current_exp / current_denominator.unsqueeze(-1)

        prev_output = output
        output = ((prev_denominator / current_denominator).unsqueeze(-1)) * prev_output * torch.exp(
            delta_max.unsqueeze(-1)
        ) + torch.matmul(prob, V_block_states)
    attn_output = output.transpose(1, 2).contiguous()
    attn_weights = None

    return attn_output, attn_weights


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = torch.where(
            attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=key_states.dtype), attn_weights
        )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=key_states.dtype).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def _build_matched_idx_from_cumsum(T2Ei: torch.Tensor) -> torch.Tensor:
    """Build a packed-row to original-token index table for active expert rows."""
    batch_size, seq_len = T2Ei.shape
    int32_max = torch.iinfo(torch.int32).max
    int32_max_scalar = torch.tensor(int32_max, dtype=torch.int32, device=T2Ei.device)
    token_idx = torch.arange(seq_len, dtype=torch.int32, device=T2Ei.device).unsqueeze(0).expand(batch_size, -1)
    valid_prefix = torch.cumsum(T2Ei.to(torch.int32), dim=1)
    valid_dest = valid_prefix - 1
    scatter_pos = torch.where(T2Ei, valid_dest, int32_max_scalar)
    matched_idx = torch.full_like(token_idx, int32_max)
    matched_idx = CtxScatterFunc3DInt.apply(
        matched_idx.unsqueeze(-1),
        scatter_pos,
        token_idx.unsqueeze(-1),
    ).squeeze(-1)
    return matched_idx


def _cumsum_scatter_gather_update_expert_blocked(
    x: torch.Tensor,
    T2Ei: torch.Tensor,
    W_g: torch.Tensor,
    W_u: torch.Tensor,
    W_d: torch.Tensor,
    routing_weight: torch.Tensor,
    expert_out: torch.Tensor,
    act_fn,
    packed_chunk_size: int,
) -> torch.Tensor:
    batch_size, seq_len = T2Ei.shape
    packed_chunk_size = max(1, min(packed_chunk_size, seq_len))

    matched_idx = _build_matched_idx_from_cumsum(T2Ei)
    valid_rows = T2Ei.to(torch.int32).sum(dim=1, keepdim=True)
    row_range = torch.arange(packed_chunk_size, dtype=torch.int32, device=x.device).unsqueeze(0)
    x_expanded = x.unsqueeze(0).expand(batch_size, -1, -1)

    for packed_start in range(0, seq_len, packed_chunk_size):
        packed_stop = packed_start + packed_chunk_size
        chunk_matched_idx = matched_idx[:, packed_start:packed_stop]

        x_chunk = CtxGatherFunc3DGeneralized.apply(x_expanded, chunk_matched_idx)
        gate_prime = x_chunk @ W_g
        up_prime = x_chunk @ W_u
        down_chunk = (up_prime * act_fn(gate_prime)) @ W_d

        rw_chunk = CtxGatherFunc3DGeneralized.apply(routing_weight, chunk_matched_idx)
        down_chunk = down_chunk * rw_chunk

        expert_out_chunk = CtxGatherFunc3DGeneralized.apply(expert_out, chunk_matched_idx)
        updated_chunk = expert_out_chunk + down_chunk

        chunk_valid_rows = torch.clamp(
            valid_rows - packed_start,
            min=torch.zeros_like(valid_rows),
            max=torch.full_like(valid_rows, packed_chunk_size),
        )
        updated_chunk = torch.where(
            (row_range < chunk_valid_rows).unsqueeze(-1), updated_chunk, torch.zeros_like(updated_chunk)
        )
        expert_out = CtxScatterFunc3DGeneralized.apply(expert_out, chunk_matched_idx, updated_chunk)

    return expert_out


class QEffGlm4MoeAttention(Glm4MoeAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __qeff_init__(self):
        self.rotary_emb = QEffGlm4MoeRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        sin_cached: Optional[torch.Tensor] = None,
        cos_cached: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        if self.use_qk_norm:  # main diff from Llama
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if sin_cached is not None and cos_cached is not None:
            sin, cos = sin_cached, cos_cached
            rotary_dim = int(self.rotary_emb.cos_cached.shape[-1])
            query_states, key_states = qeff_apply_precomputed_rotary_pos_emb(
                query_states, key_states, cos, sin, rotary_dim
            )
        else:
            kv_seq_len = (
                past_key_value.get_seq_length(self.layer_idx, cache_position)
                if past_key_value is not None
                else key_states.shape[-2]
            )
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            # cache_kwargs = {
            #     "sin": sin,
            #     "cos": cos,
            #     "cache_position": cache_position,
            #     "batch_index": batch_index,
            #     "position_ids": position_ids,
            # }
            past_seen_tokens = past_key_value.get_seq_length(self.layer_idx) if past_key_value is not None else 0
            blocking_config = getattr(self, "attn_blocking_config", AttentionBlockingConfig())
            use_blocking = blocking_config is not None and (blocking_config.mode != BlockingMode.NONE)
            if use_blocking:
                attn_output, attn_weights = generic_blocked_attention_interface(
                    module=self,
                    query=query_states,
                    key=key_states,
                    value=value_states,
                    attention_mask=attention_mask,
                    scaling=self.scaling,
                    layer_idx=self.layer_idx,
                    past_key_value=past_key_value,
                    blocking_config=blocking_config,
                    comp_ctx_lengths=comp_ctx_lengths,
                    batch_index=batch_index,
                    position_ids=position_ids,
                    past_seen_tokens=past_seen_tokens,
                )
            else:
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
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    **kwargs,
                )
        else:
            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QEffGlm4MoeDecoderLayer(Glm4MoeDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        sin_cached: Optional[torch.Tensor] = None,
        cos_cached: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            sin_cached=sin_cached,
            cos_cached=cos_cached,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class QEffGlm4MoeModel(Glm4MoeModel):
    def __qeff_init__(self):
        self.rotary_emb = QEffGlm4MoeRotaryEmbedding(config=self.config)
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
        cache_position: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        attention_mask = _create_causal_mask(position_ids=position_ids, target_length=past_seen_tokens)

        hidden_states = inputs_embeds

        # decoder layers
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
                cache_position=cache_position,
                sin_cached=sin,
                cos_cached=cos,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


class QEffGlm4MoeTopkRouter(nn.Module):
    @torch.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices

    def orig_forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = torch.nn.functional.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            # denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            denominator = torch.einsum("ab->a", topk_weights).unsqueeze(-1) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states):
        # orig_i, orig_w = self.orig_forward(hidden_states)
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        # router_logits = torch.nn.functional.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        router_logits = torch.nn.functional.linear(hidden_states, self.weight)

        # router_logits: [T, E] where E=160
        router_scores = router_logits.sigmoid()  # (0,1), [T, 160]

        # Only used for choosing which experts win
        scores_for_choice = router_scores + self.e_score_correction_bias.unsqueeze(0)  # [T, 160]

        # Choose top_k experts globally (top_k == num_experts_per_tok == 8)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]  # [T, 8]

        # Weights come from router_scores (NOT bias-corrected)
        topk_weights = router_scores.gather(1, topk_indices)  # [T, 8]

        if self.norm_topk_prob:
            # denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            denominator = torch.einsum("ab->a", topk_weights).unsqueeze(-1) + 1e-20
            topk_weights /= denominator

        topk_weights = topk_weights * self.routed_scaling_factor  # *2.5
        return topk_indices, topk_weights


class QEffGlm4MoeMoE(Glm4MoeMoE):
    """
    MoE Block
    """

    def __qeff_init__(
        self,
    ):
        if hasattr(self.experts, "gate_up_proj"):
            gate_proj, up_proj = self.experts.gate_up_proj.chunk(2, dim=1)
            self.all_gate_proj = torch.nn.Parameter(gate_proj.transpose(1, 2).contiguous())
            self.all_up_proj = torch.nn.Parameter(up_proj.transpose(1, 2).contiguous())
            self.all_down_proj = torch.nn.Parameter(self.experts.down_proj.transpose(1, 2).contiguous())
            self.act_fn = self.experts.act_fn
            self.num_experts = self.experts.num_experts
            return

        self.all_gate_proj = torch.nn.Parameter(
            torch.cat([exp.gate_proj.weight.T.unsqueeze(0) for exp in self.experts], dim=0)
        )
        self.all_up_proj = torch.nn.Parameter(
            torch.cat([exp.up_proj.weight.T.unsqueeze(0) for exp in self.experts], dim=0)
        )
        self.all_down_proj = torch.nn.Parameter(
            torch.cat([exp.down_proj.weight.T.unsqueeze(0) for exp in self.experts], dim=0)
        )
        self.act_fn = self.experts[0].act_fn
        self.num_experts = len(self.experts)

    def orig_moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        r"""
        CALL FOR CONTRIBUTION! I don't have time to optimise this right now, but expert weights need to be fused
        to not have to do a loop here (deepseek has 256 experts soooo yeah).
        """
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        if hasattr(self.experts, "gate_up_proj"):
            return self.experts(hidden_states, topk_indices, topk_weights)

        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output)

        # in original deepseek, the output of the experts are gathered once we leave this module
        # thus the moe module is itelsf an IsolatedParallel module
        # and all expert are "local" meaning we shard but we don't gather
        return final_hidden_states.type(hidden_states.dtype)

    def moe(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        bs, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        gate_proj = self.all_gate_proj[topk_indices.flatten()]
        up_proj = self.all_up_proj[topk_indices.flatten()]
        down_proj = self.all_down_proj[topk_indices.flatten()]
        expert_in = (
            hidden_states.unsqueeze(1).expand(-1, self.gate.top_k, -1).contiguous().view(-1, 1, self.config.hidden_size)
        )
        gate_out = torch.bmm(expert_in, gate_proj)
        up_out = torch.bmm(expert_in, up_proj)
        hidden = self.act_fn(gate_out) * up_out
        expert_output = torch.bmm(hidden, down_proj)
        experts_out = expert_output.view(bs * seq_len, self.gate.top_k, self.config.hidden_size)
        experts_out = experts_out * topk_weights.unsqueeze(-1)
        # final_hidden_states = experts_out.sum(dim=1)
        final_hidden_states = torch.einsum("abc->ac", experts_out)

        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states):
        """
        Forward pass of MoE block.
        """
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_output = self.gate(hidden_states)
        if isinstance(router_output, tuple):
            topk_indices, topk_weights = router_output
        else:
            topk_indices, topk_weights = self.route_tokens_to_experts(router_output)
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class QEffPrefillChunkedGlm4MoeMoE(QEffGlm4MoeMoE):
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
            raise ValueError(f"num_experts ({num_experts}) must be divisible by expert_blocking_num_nsp ({num_nsp})")

        routing_weights = hidden_states.new_zeros((T, num_experts))
        routing_weights.scatter_(1, topk_indices, topk_weights)

        local_experts = num_experts // num_nsp
        rw = routing_weights.transpose(0, 1).contiguous().view(local_experts, num_nsp, T).transpose(0, 1).contiguous()
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

        return torch.einsum("nth->th", expert_out)

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_output = self.gate(hidden_states)
        if isinstance(router_output, tuple):
            topk_indices, topk_weights = router_output
        else:
            topk_indices, topk_weights = self.route_tokens_to_experts(router_output)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        hidden_states = self._forward_expert_blocked(hidden_states, topk_indices, topk_weights).view(*orig_shape)

        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class QEffGlm4MoeForCausalLM(Glm4MoeForCausalLM):
    """
    Copied from Glm4MoeForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm4_moe/modeling_glm4_moe.py
    """

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffGlm4MoeDecoderLayer}

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
        cache_position: Optional[torch.LongTensor] = None,
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
            cache_position=cache_position,
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
            attentions=outputs.attentions,
        )
