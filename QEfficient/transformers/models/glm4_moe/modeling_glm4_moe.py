# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import math
import os
from typing import Any, Callable, Dict, List, Optional, Union

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
    current_max = torch.full((batch_size, num_heads, seq_len), float(MIN_MASKED_ATTENTION_VALUE))

    # Initialize Denominator
    current_denominator = torch.zeros(batch_size, num_heads, seq_len)

    past_seen_tokens = cache_kwargs.get("past_seen_tokens")
    position_ids = cache_kwargs.get("position_ids")
    block_size = -(-past_seen_tokens // num_kv_blocks)
    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32)

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
            attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), attn_weights
        )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class QEffGlm4MoePrefillOnlyAttention(Glm4MoeAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __qeff_init__(self):
        self.rotary_emb = QEffGlm4MoeRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
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

        kv_seq_len = past_key_value.get_seq_length(self.layer_idx, cache_position)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        num_kv_blocks = int(os.environ.get("NUM_KV_BLOCKS", 0))
        past_seen_tokens = past_key_value.get_seq_length() if past_key_value is not None else 0
        if past_key_value is not None:
            if num_kv_blocks > 0:
                cache_kwargs = {
                    "batch_index": batch_index,
                    "position_ids": position_ids,
                    "past_seen_tokens": past_seen_tokens,
                }
                past_key_value.write_only(key_states, value_states, self.layer_idx, cache_kwargs)
                attention_interface = eager_attention_forward_blocked_kv
                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    num_kv_blocks=num_kv_blocks,
                    cache_kwargs=cache_kwargs,
                    layer_idx=self.layer_idx,
                    past_key_value=past_key_value,
                    **kwargs,
                )
            else:
                # sin and cos are specific to RoPE models; position_ids needed for the static cache
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                    "batch_index": batch_index,
                    "position_ids": position_ids,
                }
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

                attention_interface: Callable = eager_attention_forward
                attn_output, attn_weights = attention_interface(
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


class QEffGlm4MoeAttention(Glm4MoeAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __qeff_init__(self):
        self.rotary_emb = QEffGlm4MoeRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
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

        kv_seq_len = past_key_value.get_seq_length(self.layer_idx, cache_position)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "batch_index": batch_index,
                "position_ids": position_ids,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            attention_interface: Callable = eager_attention_forward
            attn_output, attn_weights = attention_interface(
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
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
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
            batch_index=batch_index,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
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
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
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

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        attention_mask = _create_causal_mask(position_ids=position_ids, target_length=past_seen_tokens)

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                batch_index=batch_index,
                cache_position=cache_position,
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


class QEffGlm4MoeMoE(Glm4MoeMoE):
    """
    MoE Block
    """

    def __qeff_init__(
        self,
    ):
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

    def orig_moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        r"""
        CALL FOR CONTRIBUTION! I don't have time to optimise this right now, but expert weights need to be fused
        to not have to do a loop here (deepseek has 256 experts soooo yeah).
        """
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx in range(len(self.experts)):
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
        final_hidden_states = experts_out.sum(dim=1)

        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states):
        """
        Forward pass of MoE block.
        """
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class QEffPrefillOnlyGlm4MoeMoE(Glm4MoeMoE):
    """
    MoE Block
    """

    def moe_blocked_forward(
        self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, expert_mask: torch.Tensor, num_experts: int
    ):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        target_blocks = int(os.environ.get("NUM_FFN_BLOCKS", 1))
        block_positions = []
        T = hidden_states.shape[0]
        for j in range(target_blocks):
            block_positions.append(j * (T // target_blocks))
        for expert_idx in range(num_experts):
            expert = self.experts[expert_idx]
            block_count = 0
            outs = []
            for block_idx in range(target_blocks):
                block_count += 1
                qi = block_positions[block_idx]

                # Calculate block size (last block should be handled with remainder)
                if block_idx == target_blocks - 1:
                    real_q_len = T - qi
                else:
                    real_q_len = block_positions[block_idx + 1] - qi

                tgb = hidden_states[qi : qi + real_q_len, :]
                gate_out = expert.gate_proj(tgb)
                up_out = expert.up_proj(tgb)
                hidden = expert.act_fn(gate_out) * up_out
                down_out = expert.down_proj(hidden)
                outs.append(down_out)
            expert_output = torch.cat(outs, dim=0)
            current_hidden_states = expert_output * expert_mask[:, expert_idx].unsqueeze(-1)
            final_hidden_states += current_hidden_states

        return final_hidden_states.type(hidden_states.dtype)

    def moe_blocked_weights_forward(
        self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, expert_mask: torch.Tensor, num_experts: int
    ):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        target_blocks = int(os.environ.get("NUM_FFN_BLOCKS", 1))
        block_positions = []
        T = hidden_states.shape[0]
        for j in range(target_blocks):
            block_positions.append(j * (T // target_blocks))
        for expert_idx in range(num_experts):
            expert = self.experts[expert_idx]
            block_count = 0
            outs = []
            for block_idx in range(target_blocks):
                block_count += 1
                qi = block_positions[block_idx]

                # Calculate block size (last block should be handled with remainder)
                if block_idx == target_blocks - 1:
                    real_q_len = T - qi
                else:
                    real_q_len = block_positions[block_idx + 1] - qi

                tgb = hidden_states[qi : qi + real_q_len, :]
                wg_col_shape = self.config.moe_intermediate_size
                w_block_size = int(os.environ.get("FFN_W_BLOCK_SIZE", wg_col_shape))
                wg_num_blocks = math.ceil(wg_col_shape / w_block_size)
                if w_block_size < wg_col_shape:
                    last_block_size = wg_col_shape % w_block_size if wg_col_shape % w_block_size != 0 else w_block_size
                else:
                    last_block_size = wg_col_shape

                intermediates = []
                for i in range(wg_num_blocks):
                    if i == wg_num_blocks - 1:
                        cur_gate = tgb @ expert.gate_proj.weight.T[:, -last_block_size:]
                        cur_up = tgb @ expert.up_proj.weight.T[:, -last_block_size:]
                    else:
                        cur_gate = tgb @ expert.gate_proj.weight.T[:, i * w_block_size : (i + 1) * w_block_size]
                        cur_up = tgb @ expert.up_proj.weight.T[:, i * w_block_size : (i + 1) * w_block_size]

                    cur_intermediate = expert.act_fn(cur_gate) * cur_up
                    intermediates.append(cur_intermediate)

                intermediate = torch.cat(intermediates, dim=-1)

                wd_col_shape = self.config.hidden_size
                wd_block_size = int(os.environ.get("FFN_W_BLOCK_SIZE", wd_col_shape))
                wd_num_blocks = math.ceil(wd_col_shape / wd_block_size)
                if wd_block_size < wd_col_shape:
                    last_block_size = (
                        wd_col_shape % wd_block_size if wd_col_shape % wd_block_size != 0 else wd_block_size
                    )
                else:
                    last_block_size = wd_col_shape
                downs = []
                for i in range(wd_num_blocks):
                    if i == wd_num_blocks - 1:
                        downs.append((intermediate @ expert.down_proj.weight.T[:, -last_block_size:]))
                    else:
                        downs.append(
                            (intermediate @ expert.down_proj.weight.T[:, i * wd_block_size : (i + 1) * wd_block_size])
                        )

                down_out_block = torch.cat(downs, dim=1)
                outs.append(down_out_block)

            expert_output = torch.cat(outs, dim=0)
            current_hidden_states = expert_output * expert_mask[:, expert_idx].unsqueeze(-1)
            final_hidden_states += current_hidden_states

        return final_hidden_states.type(hidden_states.dtype)

    def moe(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, expert_mask: torch.Tensor, num_experts: int):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        for expert_idx in range(num_experts):
            expert = self.experts[expert_idx]
            gate_out = expert.gate_proj(hidden_states)
            up_out = expert.up_proj(hidden_states)
            hidden = expert.act_fn(gate_out) * up_out
            expert_output = expert.down_proj(hidden)
            current_hidden_states = expert_output * expert_mask[:, expert_idx].unsqueeze(-1)
            final_hidden_states += current_hidden_states

        return final_hidden_states.type(hidden_states.dtype)

    def orig_moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        r"""
        CALL FOR CONTRIBUTION! I don't have time to optimise this right now, but expert weights need to be fused
        to not have to do a loop here (deepseek has 256 experts soooo yeah).
        """
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = expert_mask.permute(2, 0, 1)
        for expert_idx in range(len(self.experts)):
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

    def forward(self, hidden_states):
        """
        Forward pass of MoE block.
        """
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        # orig_out = self.orig_moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        mask = torch.zeros(hidden_states.shape[0], self.config.n_routed_experts)
        mask.scatter_(1, topk_indices, topk_weights)
        if os.environ.get("NUM_FFN_BLOCKS", None) is not None and os.environ.get("FFN_W_BLOCK_SIZE", None) is not None:
            hidden_states = self.moe_blocked_weights_forward(
                hidden_states, topk_weights, mask, self.config.n_routed_experts
            ).view(*orig_shape)
        elif os.environ.get("NUM_FFN_BLOCKS", None) is not None:
            hidden_states = self.moe_blocked_forward(
                hidden_states, topk_weights, mask, self.config.n_routed_experts
            ).view(*orig_shape)
        else:
            hidden_states = self.moe(hidden_states, topk_weights, mask, self.config.n_routed_experts).view(*orig_shape)

        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class QEffGlm4MoeForCausalLM(Glm4MoeForCausalLM):
    """
    Copied from Glm4MoeForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm4_moe/modeling_glm4_moe.py
    """

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
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
        logits = self.lm_head(hidden_states).float()

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
