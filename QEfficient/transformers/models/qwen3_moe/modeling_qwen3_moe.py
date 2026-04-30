# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from typing import List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeConfig,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeSparseMoeBlock,
    repeat_kv,
    rotate_half,
)

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


class QEffQwen3MoeRotaryEmbedding(Qwen3MoeRotaryEmbedding):
    def __init__(self, config: Qwen3MoeConfig, device=None):
        super().__init__(config=config)

        # Build here to make `torch.jit.trace` work.
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


def qeff_apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    # Cast back to original dtype
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
):
    key_states = repeat_kv(key, module.num_key_value_groups)

    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = torch.where(
            attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=module.config.torch_dtype), attn_weights
        )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


EXPERT_BLOCKING_NUM_NSP = int(os.environ.get("EXPERT_BLOCKING_NUM_NSP", "16"))
EXPERT_BLOCKING_PACKED_CHUNK_SIZE = int(os.environ.get("EXPERT_BLOCKING_PACKED_CHUNK_SIZE", "256"))


def _build_matched_idx_from_cumsum(T2Ei: torch.Tensor) -> torch.Tensor:
    """Build packed->original token index table for an NSP-sliced expert mask.

    Given ``T2Ei`` of shape ``[num_nsp, T]`` marking which tokens are routed to
    an expert, produces an index tensor where ``matched_idx[b, j]`` is the
    original token position in ``x`` that lands at packed position ``j`` for
    NSP lane ``b`` (or ``INT32_MAX`` when ``j`` is past the last valid row).
    """
    batch_size, seq_len = T2Ei.shape
    int32_max = torch.iinfo(torch.int32).max
    int32_max_scalar = torch.tensor(int32_max, dtype=torch.int32, device=T2Ei.device)
    token_idx = torch.arange(seq_len, dtype=torch.int32, device=T2Ei.device).unsqueeze(0).expand(batch_size, -1)
    valid_prefix = torch.cumsum(T2Ei.to(torch.int32), dim=1)
    valid_dest = valid_prefix - 1
    scatter_pos = torch.where(T2Ei, valid_dest, int32_max_scalar)
    # Once the compiler fix for ConstantOfShape(INT32_MAX) is available, this
    # can be switched back to ``torch.full_like(token_idx, int32_max)``.
    matched_idx = int32_max_scalar.expand_as(token_idx)
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
    T: int,
    packed_chunk_size: int,
) -> torch.Tensor:
    """Cumsum-scatter-gather-update expert helper for NSP-blocked dispatch.

    Accumulates one local expert's contribution in-place onto ``expert_out``.
    Uses a packed/cumsum layout so the MLP runs only over active rows, then
    scatters the weighted output back to original token positions.

    Shapes:
        x               : [T, H]
        T2Ei            : [num_nsp, T]            (bool)
        W_g, W_u        : [num_nsp, H, I]
        W_d             : [num_nsp, I, H]
        routing_weight  : [num_nsp, T]
        expert_out      : [num_nsp, T, H]         (accumulator, in-out)
    """
    batch_size, seq_len = T2Ei.shape
    packed_chunk_size = max(1, min(packed_chunk_size, seq_len))

    matched_idx = _build_matched_idx_from_cumsum(T2Ei)
    valid_rows = T2Ei.to(torch.int32).sum(dim=1, keepdim=True)
    row_range = torch.arange(packed_chunk_size, dtype=torch.int32, device=x.device).unsqueeze(0)
    x_expanded = x.unsqueeze(0).expand(batch_size, -1, -1)
    rw_expanded = routing_weight.unsqueeze(-1)

    for packed_start in range(0, seq_len, packed_chunk_size):
        packed_stop = packed_start + packed_chunk_size
        chunk_matched_idx = matched_idx[:, packed_start:packed_stop]

        x_chunk = CtxGatherFunc3DGeneralized.apply(x_expanded, chunk_matched_idx)

        gate_prime = x_chunk @ W_g
        up_prime = x_chunk @ W_u
        down_chunk = (up_prime * act_fn(gate_prime)) @ W_d

        rw_chunk = CtxGatherFunc3DGeneralized.apply(rw_expanded, chunk_matched_idx)
        down_chunk = down_chunk * rw_chunk

        expert_out_chunk = CtxGatherFunc3DGeneralized.apply(expert_out, chunk_matched_idx)
        updated_chunk = expert_out_chunk + down_chunk

        chunk_valid_rows = torch.clamp(valid_rows - packed_start, min=0, max=packed_chunk_size)
        updated_chunk = torch.where(
            (row_range < chunk_valid_rows).unsqueeze(-1), updated_chunk, torch.zeros_like(updated_chunk)
        )
        expert_out = CtxScatterFunc3DGeneralized.apply(expert_out, chunk_matched_idx, updated_chunk)

    return expert_out


class QEffPrefillChunkedQwen3MoeSparseMoeBlock(Qwen3MoeSparseMoeBlock):
    def __qeff_init__(self):
        self.gate_proj_w = []
        self.up_proj_w = []
        self.down_proj_w = []
        with torch.no_grad():
            for e in range(self.num_experts):
                self.gate_proj_w.append(self.experts[e].gate_proj.weight.T)
                self.up_proj_w.append(self.experts[e].up_proj.weight.T)
                self.down_proj_w.append(self.experts[e].down_proj.weight.T)
            self.gate_proj_w = torch.stack(self.gate_proj_w)  # [E, H, I]
            self.up_proj_w = torch.stack(self.up_proj_w)  # [E, H, I]
            self.down_proj_w = torch.stack(self.down_proj_w)  # [E, I, H]

    def _forward_expert_blocked(self, x: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        T, H = x.shape
        num_nsp = EXPERT_BLOCKING_NUM_NSP
        if self.num_experts % num_nsp != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by EXPERT_BLOCKING_NUM_NSP ({num_nsp})"
            )
        local_experts = self.num_experts // num_nsp
        rw = routing_weights.transpose(0, 1).contiguous().view(local_experts, num_nsp, T).transpose(0, 1).contiguous()
        W_g = self.gate_proj_w.view(local_experts, num_nsp, H, -1).transpose(0, 1).contiguous()
        W_u = self.up_proj_w.view(local_experts, num_nsp, H, -1).transpose(0, 1).contiguous()
        W_d = self.down_proj_w.view(local_experts, num_nsp, -1, H).transpose(0, 1).contiguous()
        expert_out = x.new_zeros((num_nsp, T, H))
        for slot in range(local_experts):
            routing_weight = rw[:, slot, :]
            T2Ei = routing_weight > 0
            expert_out = _cumsum_scatter_gather_update_expert_blocked(
                x=x,
                T2Ei=T2Ei,
                W_g=W_g[:, slot],
                W_u=W_u[:, slot],
                W_d=W_d[:, slot],
                routing_weight=routing_weight,
                expert_out=expert_out,
                act_fn=self.experts[0].act_fn,
                T=T,
                packed_chunk_size=EXPERT_BLOCKING_PACKED_CHUNK_SIZE,
            )
        return expert_out.sum(dim=0)

    def orig_forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, H = hidden_states.shape
        T = B * S
        x = hidden_states.view(T, H)
        router_logits = self.gate(x)  # [T, E]
        prob = F.softmax(router_logits, -1, dtype=torch.float)
        top_w, top_i = torch.topk(prob, self.top_k, -1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            top_w /= top_w.sum(-1, keepdim=True)
        top_w = top_w.to(hidden_states.dtype)
        masked_logits = torch.zeros_like(router_logits)
        masked_logits.scatter_(1, top_i, top_w)
        routing_weights = masked_logits
        expert_out = x.new_zeros((T, H))
        for e in range(self.num_experts):
            routing_weight = routing_weights[:, e].unsqueeze(-1)
            W_g, W_u = self.experts[e].gate_proj.weight.T, self.experts[e].up_proj.weight.T
            W_d = self.experts[e].down_proj.weight.T
            gate = x @ W_g
            up = x @ W_u
            down = (up * self.experts[e].act_fn(gate)) @ W_d
            expert_out += down * routing_weight
        return expert_out.view(B, S, H), router_logits

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, H = hidden_states.shape
        T = B * S
        x = hidden_states.view(T, H)
        router_logits = self.gate(x)
        prob = F.softmax(router_logits, -1, dtype=torch.float)
        top_w, top_i = torch.topk(prob, self.top_k, -1)
        if self.norm_topk_prob:
            top_w /= top_w.sum(-1, keepdim=True)
        top_w = top_w.to(hidden_states.dtype)
        routing_weights = torch.zeros_like(router_logits)
        routing_weights.scatter_(1, top_i, top_w)

        if self.num_experts % EXPERT_BLOCKING_NUM_NSP == 0:
            expert_out = self._forward_expert_blocked(x=x, routing_weights=routing_weights)
            return expert_out.view(B, S, H), router_logits

        expert_out = x.new_zeros((T, H))
        for e in range(self.num_experts):
            routing_weight = routing_weights[:, e].unsqueeze(-1)
            W_g, W_u = self.experts[e].gate_proj.weight.T, self.experts[e].up_proj.weight.T
            W_d = self.experts[e].down_proj.weight.T
            gate = x @ W_g
            up = x @ W_u
            down = (up * self.experts[e].act_fn(gate)) @ W_d
            expert_out += down * routing_weight
        return expert_out.view(B, S, H), router_logits


class QEffQwen3MoeSparseMoeBlock(Qwen3MoeSparseMoeBlock):
    def __qeff_init__(self):
        self.gate_proj_w = []
        self.up_proj_w = []
        self.down_proj_w = []
        with torch.no_grad():
            for e in range(self.num_experts):
                self.gate_proj_w.append(self.experts[e].gate_proj.weight.T)
                self.up_proj_w.append(self.experts[e].up_proj.weight.T)
                self.down_proj_w.append(self.experts[e].down_proj.weight.T)
            self.gate_proj_w = torch.stack(self.gate_proj_w)
            self.up_proj_w = torch.stack(self.up_proj_w)
            self.down_proj_w = torch.stack(self.down_proj_w)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, H = hidden_states.shape
        T = B * S
        hidden_states = hidden_states.view(T, H)
        router_logits = self.gate(hidden_states)  # [T, E]
        prob = F.softmax(router_logits, -1, dtype=torch.float)
        top_w, top_i = torch.topk(prob, self.top_k, -1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            top_w = top_w / torch.einsum("bi->b", top_w)[:, None]
        top_w = top_w.to(hidden_states.dtype)

        gate_proj_w = self.gate_proj_w[top_i.flatten()]
        up_proj_w = self.up_proj_w[top_i.flatten()]
        down_proj_w = self.down_proj_w[top_i.flatten()]

        expert_in = hidden_states.unsqueeze(1).expand(-1, self.top_k, -1).contiguous().view(-1, 1, H)
        gate = torch.bmm(expert_in, gate_proj_w)
        up = torch.bmm(expert_in, up_proj_w)
        intermediate = up * self.experts[0].act_fn(gate)
        experts_out = torch.bmm(intermediate, down_proj_w)
        experts_out = experts_out.view(B * S, self.top_k, H)
        experts_out = experts_out * top_w.unsqueeze(-1)
        experts_out = torch.einsum("bnd->bd", experts_out)
        return experts_out.view(B, S, H), router_logits


class QEffQwen3MoeAttention(Qwen3MoeAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cos_cached: Optional[torch.Tensor] = None,
        sin_cached: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # kv_seq_len = past_key_value.get_seq_length(self.layer_idx, cache_position)
        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos_cached, sin_cached)

        past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
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
                past_key_value=past_key_values,
                blocking_config=blocking_config,
                comp_ctx_length=comp_ctx_lengths,
                batch_index=batch_index,
                position_ids=position_ids,
                past_seen_tokens=past_seen_tokens,
            )
        else:
            key_states, value_states, _ = past_key_value_update(
                module=self,
                key=key_states,
                value=value_states,
                attention_mask=attention_mask,
                past_key_value=past_key_values,
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
                scaling=self.scaling,
            )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class QEffQwen3MoeDecoderLayer(Qwen3MoeDecoderLayer):
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
        sin_cached=None,
        cos_cached=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=use_cache,
            cache_position=cache_position,
            sin_cached=sin_cached,
            cos_cached=cos_cached,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states

        hidden_states = residual + hidden_states

        return hidden_states


class QEffQwen3MoeModel(Qwen3MoeModel):
    def __qeff_init__(self):
        self.rotary_emb = QEffQwen3MoeRotaryEmbedding(config=self.config)
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
    ) -> MoeModelOutputWithPast:
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

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        sin = self.sin_cached[position_ids].unsqueeze(1)
        cos = self.cos_cached[position_ids].unsqueeze(1)

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
                sin_cached=sin,
                cos_cached=cos,
            )

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_values = past_key_values.to_legacy_cache()

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


class QEffQwen3MoeForCausalLM(Qwen3MoeForCausalLM):
    def get_submodules_for_export(self) -> Type[nn.Module]:
        """
        Return the set of class used as the repeated layer across the model for subfunction extraction.
        Notes:
            This method should return the *class object* (not an instance).
            Downstream code can use this to find/build subfunctions for repeated blocks.
        """
        return {QEffQwen3MoeDecoderLayer}

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
    ) -> MoeCausalLMOutputWithPast:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logit_idx = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[torch.arange(position_ids.shape[0]).view(-1, 1), logit_idx]
        logits = self.lm_head(hidden_states).float()

        return MoeCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )
