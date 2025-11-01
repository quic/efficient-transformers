# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
from typing import Callable, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
    GptOssConfig,
    GptOssDecoderLayer,
    GptOssExperts,
    GptOssForCausalLM,
    GptOssMLP,
    GptOssModel,
    GptOssRotaryEmbedding,
    repeat_kv,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from QEfficient.transformers.cache_utils import QEffHybridCacheForGPTOSS
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


class QEffGptOssExperts(GptOssExperts):
    def __qeff_init__(self):
        self.gate_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.expert_dim))
        self.up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.expert_dim))
        self.gate_proj_bias = nn.Parameter(torch.empty(self.num_experts, self.expert_dim))
        self.up_proj_bias = nn.Parameter(torch.empty(self.num_experts, self.expert_dim))


class QEffGptOssMLP(GptOssMLP):
    def alt_forward(self, hidden: torch.Tensor):
        B, S, H = hidden.shape
        T = B * S
        hidden = hidden.view(T, H)

        # Router computation
        router_logits = F.linear(hidden, self.router.weight, self.router.bias)

        # Top-k selection
        top_w, top_i = torch.topk(router_logits, self.router.top_k, dim=-1)  # both [T, K]
        top_w = torch.nn.functional.softmax(top_w, dim=1, dtype=top_w.dtype)

        masked_logits = torch.zeros_like(router_logits)
        masked_logits.scatter_(1, top_i, top_w)

        # Routing weights for each expert [T, E]
        routing_weights = masked_logits

        # ────────────────── allocate the output tensor ─────
        expert_out = hidden.new_zeros((T, H))  # accumulation buffer

        # ───────────────────────── Expert computation loop ─────────────────────────────
        for e in range(self.experts.num_experts):
            routing_weight = routing_weights[:, e].unsqueeze(-1)  # [T, 1]

            W_g, W_u = self.experts.gate_proj[e], self.experts.up_proj[e]  # [H, I], [H, I]
            b_g, b_u = self.experts.gate_proj_bias[e], self.experts.up_proj_bias[e]  # [I], [I]
            W_d = self.experts.down_proj[e]  # [I, H]
            b_d = self.experts.down_proj_bias[e]  # [H]

            # Gate and Up projections
            gate = (hidden @ W_g) + b_g  # [T, I]
            up = (hidden @ W_u) + b_u  # [T, I]

            # Apply GptOss activation with clamping
            gate = gate.clamp(min=None, max=self.experts.limit)
            up = up.clamp(min=-self.experts.limit, max=self.experts.limit)

            # GLU activation
            glu = gate * torch.sigmoid(gate * self.experts.alpha)
            intermediate = (up + 1) * glu  # [T, I]

            # Down projection
            down_out = (intermediate @ W_d) + b_d  # [T, H]

            # Apply routing weights and accumulate
            masked_down = torch.where(routing_weight > 0, down_out * routing_weight, torch.zeros_like(expert_out))
            expert_out += masked_down

        # original shape [B, S, H]
        return expert_out.view(B, S, H), router_logits

    # ------------------- Gather based, weights as activation approach ---------------
    def forward_weights_as_activation(self, hidden_states):
        bs, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(bs * seq_len, self.experts.hidden_size)

        # Router computation
        router_logits = F.linear(hidden_states, self.router.weight, self.router.bias)
        router_top_value, router_indices = torch.topk(router_logits, self.router.top_k, dim=-1)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)

        # GATHER - collect weights for selected experts
        gate_up_proj = self.experts.gate_up_proj[router_indices.flatten()]
        gate_up_proj_bias = self.experts.gate_up_proj_bias[router_indices.flatten()]
        down_proj = self.experts.down_proj[router_indices.flatten()]
        down_proj_bias = self.experts.down_proj_bias[router_indices.flatten()]

        # Apply Chosen Experts (without routing weights first)
        # expert_in = hidden_states.repeat_interleave(self.router.top_k, dim=0)
        # expert_in = expert_in.view(-1, 1, self.experts.hidden_size)
        # Reshape for bmm: (bs*seq_len*top_k, 1, hidden_size)
        expert_in = (
            hidden_states.unsqueeze(1)
            .expand(-1, self.router.top_k, -1)
            .contiguous()
            .view(-1, 1, self.experts.hidden_size)
        )

        gate_up = torch.bmm(expert_in, gate_up_proj) + gate_up_proj_bias.unsqueeze(1)
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]

        # Apply activation with clamping
        gate = gate.clamp(min=None, max=self.experts.limit)
        up = up.clamp(min=-self.experts.limit, max=self.experts.limit)
        glu = gate * torch.sigmoid(gate * self.experts.alpha)
        gated_output = (up + 1) * glu

        experts_out = torch.bmm(gated_output, down_proj) + down_proj_bias.unsqueeze(1)
        experts_out = experts_out.view(bs * seq_len, self.router.top_k, self.experts.hidden_size)

        # Apply routing weights AFTER expert computation (This is before on Llama4)
        experts_out = experts_out * router_top_value.unsqueeze(-1)
        experts_out = experts_out.sum(dim=1)

        return experts_out, router_logits

    # ------------------- Gather based, weights as activation approach, With Seperate Gate, up Projections ---------------
    def forward(self, hidden_states):
        # print("Seperate Split, Up, Gate Projections")
        bs, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(bs * seq_len, self.experts.hidden_size)

        # Router computation
        router_logits = F.linear(hidden_states, self.router.weight, self.router.bias)
        router_top_value, router_indices = torch.topk(router_logits, self.router.top_k, dim=-1)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)

        # GATHER - collect weights for selected experts (separate gate and up projections)
        gate_proj = self.experts.gate_proj[router_indices.flatten()]
        gate_proj_bias = self.experts.gate_proj_bias[router_indices.flatten()]
        up_proj = self.experts.up_proj[router_indices.flatten()]
        up_proj_bias = self.experts.up_proj_bias[router_indices.flatten()]
        down_proj = self.experts.down_proj[router_indices.flatten()]
        down_proj_bias = self.experts.down_proj_bias[router_indices.flatten()]

        # Reshape for bmm: (bs*seq_len*top_k, 1, hidden_size)
        expert_in = (
            hidden_states.unsqueeze(1)
            .expand(-1, self.router.top_k, -1)
            .contiguous()
            .view(-1, 1, self.experts.hidden_size)
        )

        # Apply gate and up projections separately using bmm
        gate = torch.bmm(expert_in, gate_proj) + gate_proj_bias.unsqueeze(1)
        up = torch.bmm(expert_in, up_proj) + up_proj_bias.unsqueeze(1)

        # Apply activation with clamping
        gate = gate.clamp(min=None, max=self.experts.limit)
        up = up.clamp(min=-self.experts.limit, max=self.experts.limit)

        # GLU activation
        glu = gate * torch.sigmoid(gate * self.experts.alpha)
        gated_output = (up + 1) * glu

        # Down projection
        experts_out = torch.bmm(gated_output, down_proj) + down_proj_bias.unsqueeze(1)
        experts_out = experts_out.view(bs * seq_len, self.router.top_k, self.experts.hidden_size)

        # Apply routing weights AFTER expert computation
        experts_out = experts_out * router_top_value.unsqueeze(-1)
        experts_out = experts_out.sum(dim=1)

        return experts_out, router_logits

    def optimized_moe_forward(self, hidden_states: torch.Tensor):
        B, S, H = hidden_states.shape
        T = B * S
        hidden_states = hidden_states.view(T, H)

        # Router computation
        router_logits = F.linear(hidden_states, self.router.weight, self.router.bias)

        # Top-k selection
        top_w, selected_experts = torch.topk(router_logits, self.router.top_k, dim=-1)  # both [T, K]
        top_w = torch.nn.functional.softmax(top_w, dim=1, dtype=top_w.dtype)

        # Creating experts mask and routing weights masked
        awesome_experts_mask_1 = (
            torch.nn.functional.one_hot(selected_experts[:, 0], num_classes=self.experts.num_experts)
            .bool()
            .T.unsqueeze(-1)
        )
        awesome_experts_mask_2 = (
            torch.nn.functional.one_hot(selected_experts[:, 1], num_classes=self.experts.num_experts)
            .bool()
            .T.unsqueeze(-1)
        )
        awesome_experts_mask_3 = (
            torch.nn.functional.one_hot(selected_experts[:, 2], num_classes=self.experts.num_experts)
            .bool()
            .T.unsqueeze(-1)
        )
        awesome_experts_mask_4 = (
            torch.nn.functional.one_hot(selected_experts[:, 3], num_classes=self.experts.num_experts)
            .bool()
            .T.unsqueeze(-1)
        )

        gateupout1 = torch.zeros(hidden_states.shape[0], self.experts.intermediate_size)  # T, hs
        gateupout2 = torch.zeros(hidden_states.shape[0], self.experts.intermediate_size)  # T, hs
        gateupout3 = torch.zeros(hidden_states.shape[0], self.experts.intermediate_size)  # T, hs
        gateupout4 = torch.zeros(hidden_states.shape[0], self.experts.intermediate_size)  # T, hs

        # ───────────────────────── Expert computation loop ─────────────────────────────
        for e in range(self.experts.num_experts):
            W_g, W_u = self.experts.gate_proj[e], self.experts.up_proj[e]  # [H, I], [H, I]
            b_g, b_u = self.experts.gate_proj_bias[e], self.experts.up_proj_bias[e]  # [I], [I]

            # Gate and Up projections
            gate = (hidden_states @ W_g) + b_g  # [T, I]
            up = (hidden_states @ W_u) + b_u  # [T, I]

            # Apply GptOss activation with clamping
            gate = gate.clamp(min=None, max=self.experts.limit)
            up = up.clamp(min=-self.experts.limit, max=self.experts.limit)

            # GLU activation
            glu = gate * torch.sigmoid(gate * self.experts.alpha)
            intermediate = (up + 1) * glu  # [T, I]

            gateupout1 += torch.where(awesome_experts_mask_1[e], intermediate, torch.zeros_like(gateupout1))
            gateupout2 += torch.where(awesome_experts_mask_2[e], intermediate, torch.zeros_like(gateupout2))
            gateupout3 += torch.where(awesome_experts_mask_3[e], intermediate, torch.zeros_like(gateupout3))
            gateupout4 += torch.where(awesome_experts_mask_4[e], intermediate, torch.zeros_like(gateupout4))

        concat_down = torch.zeros((self.router.top_k, T, H))
        concat_mask = torch.cat(
            (
                awesome_experts_mask_1.unsqueeze(0),
                awesome_experts_mask_2.unsqueeze(0),
                awesome_experts_mask_3.unsqueeze(0),
                awesome_experts_mask_4.unsqueeze(0),
            ),
            dim=0,
        )

        concat_gateout = torch.cat(
            (gateupout1.unsqueeze(0), gateupout2.unsqueeze(0), gateupout3.unsqueeze(0), gateupout4.unsqueeze(0)), dim=0
        )

        for e in range(self.experts.num_experts):
            W_d = self.experts.down_proj[e]  # [I, H]
            b_d = self.experts.down_proj_bias[e]  # [H]

            # Down projection
            down_out = (concat_gateout @ W_d) + b_d  # [T, H]

            concat_down += torch.where(concat_mask[:, e, :], down_out, torch.zeros_like(concat_down))

        downout1, downout2, downout3, downout4 = concat_down[0], concat_down[1], concat_down[2], concat_down[3]
        hidden_states = (
            downout1 * top_w[:, 0].unsqueeze(-1)
            + downout2 * top_w[:, 1].unsqueeze(-1)
            + downout3 * top_w[:, 2].unsqueeze(-1)
            + downout4 * top_w[:, 3].unsqueeze(-1)
        ).reshape(B, S, H)

        # original shape [B, S, H]
        return hidden_states, router_logits


#  Can be replaced with llama/modeling_llama.py::QEffLlamaRotaryEmbedding but keeping it following transformers ideology
class QEffGptOssRotaryEmbedding(GptOssRotaryEmbedding):
    """
    Copied from LlamaForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    The only differences are:
    - Add static sin/cos computations.
    """

    def __init__(self, config: GptOssConfig, device=None):
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

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype) * self.attention_scaling,
            self.sin_cached[:seq_len].to(dtype=x.dtype) * self.attention_scaling,
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def qeff_apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension seperately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
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

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = torch.where(
            attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), attn_weights
        )

    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)

    # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
    # when training with bsz>1 we clamp max values.
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # we drop the sink here
    attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class QEffGptOssAttention(GptOssAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __qeff_init__(self):
        self.rotary_emb = QEffGptOssRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        sliding_mask=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # kv_seq_len = key_states.shape[-2]

        # kv_seq_len = past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=32 * 1024)
        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "batch_index": batch_index,
                "position_ids": position_ids,
                "config": self.config,
                "is_sliding": self.sliding_window is not None,
                "sliding_window": past_key_value.sliding_window_len,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.sliding_window is not None:
            attention_mask = sliding_mask
        else:
            attention_mask = attention_mask

        attention_interface: Callable = eager_attention_forward
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value


class QEffGptOssDecoderLayer(GptOssDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        sliding_mask=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            batch_index=batch_index,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            sliding_mask=sliding_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states)  # diff with llama: router scores
        # alth, _ = self.mlp.alt_forward(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class QEffGptOssModel(GptOssModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = QEffHybridCacheForGPTOSS.from_legacy_cache(self.config, past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens
        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=past_key_values.max_cache_len)
        sliding_mask = _create_causal_mask(
            position_ids=position_ids,
            target_length=past_key_values.sliding_window_len,
            sliding_window=past_key_values.sliding_window_len,
        )

        hidden_states = inputs_embeds
        # position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                batch_index=batch_index,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position,
                sliding_mask=sliding_mask,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class QEffGptOssForCausalLM(GptOssForCausalLM):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, GptOssForCausalLM

        >>> model = GptOssForCausalLM.from_pretrained("mistralai/GptOss-8x7B-v0.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/GptOss-8x7B-v0.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs[0][torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return MoeCausalLMOutputWithPast(
            loss=None,
            aux_loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def get_pkv_dynamic_axes(
        self,
    ):
        pkv_dynamic_axes = []
        for layer_type in self.config.layer_types:
            if layer_type == "sliding_attention":
                pkv_dynamic_axes.append({0: "batch_size", 2: "sliding_window"})
            elif layer_type == "full_attention":
                pkv_dynamic_axes.append({0: "batch_size", 2: "ctx_len"})
        return pkv_dynamic_axes
