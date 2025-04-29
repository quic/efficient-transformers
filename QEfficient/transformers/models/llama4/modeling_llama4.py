# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.llama4.modeling_llama4 import (
    Llama4ForCausalLM,
    Llama4ForConditionalGeneration,
    Llama4TextAttention,
    Llama4TextConfig,
    Llama4TextDecoderLayer,
    Llama4TextModel,
    Llama4TextMoe,
    logger,
    repeat_kv,
)

from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo, get_padding_shape_from_config


class QEffLlama4TextRotaryEmbedding(nn.Module):
    def __init__(self, config: Llama4TextConfig, device=None):
        super().__init__()
        self.config = config
        self.rope_type = "llama3" if config.rope_scaling is not None else "default"
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        self.max_seq_len_cached = config.max_position_embeddings

        # Get inverse frequency and scaling function (handles yarn/etc)
        inv_freq, self.attention_scaling = self.rope_init_fn(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute static cache
        self._set_freqs_cis_cache(self.max_seq_len_cached, device)

    def _set_freqs_cis_cache(self, seq_len, device):
        # Compute frequencies
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)  # [seq_len]
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim/2]

        # Convert to [real, imag] = [cos, sin]
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        freqs_cis = torch.stack([cos, sin], dim=-1)  # [seq_len, dim/2, 2]

        self.register_buffer("freqs_cis_cached", freqs_cis * self.attention_scaling, persistent=False)

    def forward(self, seq_len: Optional[int] = None, position_ids: Optional[torch.LongTensor] = None):
        """
        Returns: freqs_cis: [batch, seq_len, dim/2, 2] if position_ids given,
                           [seq_len, dim/2, 2] if only seq_len is given.
        """
        if position_ids is not None:
            # position_ids: [batch, seq_len]
            return self.freqs_cis_cached[position_ids]  # shape: [batch, seq_len, dim/2, 2]
        else:
            assert seq_len is not None, "Either seq_len or position_ids must be provided."
            return self.freqs_cis_cached[:seq_len]  # shape: [seq_len, dim/2, 2]


def qeff_apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    # freqs_cis is already in [..., 2] form (real, imag)
    freqs_cis_exp = freqs_cis[:, :, None, :]  # expand to match shape for broadcast

    def complex_mul(a, b):
        real = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
        imag = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
        return torch.stack([real, imag], dim=-1)

    xq_out = complex_mul(xq_, freqs_cis_exp)
    xk_out = complex_mul(xk_, freqs_cis_exp)
    xq_out = xq_out.reshape(*xq_out.shape[:-2], -1)
    xk_out = xk_out.reshape(*xk_out.shape[:-2], -1)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = torch.where(attention_mask, torch.tensor(-10000.0, dtype=torch.float32), attn_weights)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class QEffLlama4TextMoe(Llama4TextMoe):
    # ------------------- Gather based, weights as activation approach ---------------
    def forward(self, hidden_states):
        bs, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(bs * seq_len, self.hidden_dim)
        router_logits = self.router(hidden_states).transpose(0, 1)
        router_top_value, router_indices = torch.topk(router_logits.transpose(0, 1), self.top_k, dim=1)

        # GATHER
        gate_up_proj = self.experts.gate_up_proj[router_indices.flatten()]
        down_proj = self.experts.down_proj[router_indices.flatten()]

        # apply router top value
        expert_in = hidden_states * torch.sigmoid(router_top_value)
        ######################
        # Apply Chosen Experts
        ######################
        expert_in = expert_in.view(bs * seq_len, 1, self.hidden_dim)
        gateup = torch.bmm(expert_in, gate_up_proj)
        gate, up = gateup.chunk(2, dim=-1)
        experts_out = torch.bmm((up * self.experts.act_fn(gate)), down_proj).view(bs * seq_len, self.hidden_dim)
        shared_out = self.shared_expert(hidden_states)
        out = shared_out + experts_out
        return out, router_logits

    # --------------------- New Method, Avoid IO Communication -------------
    # def forward(self, hidden: torch.Tensor):
    #     B, S, H = hidden.shape
    #     T = B * S
    #     x = hidden.view(T, H)

    #     router_logits = self.router(x)

    #     # *top-k = 1*  → LLama4
    #     top_w, top_i = torch.topk(router_logits, self.top_k, dim=-1)  # both [T, K]
    #     masked_logits = torch.full_like(router_logits, float("-inf"))
    #     masked_logits.scatter_(1, top_i, top_w)

    #     # ── Book-keeping: create one boolean mask per expert once  ───────────────
    #     # mask_e[e]  ==  True where token routed to that expert. Shape [E, T]
    #     routing_weights = torch.sigmoid(masked_logits.float()).to(hidden.dtype)

    #     # ────────────────── allocate the two big tensors ─────
    #     I = self.experts.intermediate_size                # = 8/3 · H
    #     upgate     = x.new_zeros((T, I))
    #     expert_out = x.new_zeros((T, H))                  # accum-out buffer

    #     # ───────────────────────── Stage-1 : Up-Gate ─────────────────────────────
    #     # Loop over experts — weight matrices are already sharded per-expert
    #     for e in range(self.num_experts):
    #         W_g, W_u = self.experts.weights_for(e)
    #         upgate = torch.where(routing_weights[:, e].unsqueeze(-1) > 0,
    #                              (self.experts.act_fn(x @ W_g)) *  (x @ W_u),
    #                              upgate)

    #     # At this point  upgate[t]  holds   UpGate(x_t)   for that token’s expert,
    #     # and arbitrary (zeros) data for tokens not routed to that expert.
    #     # ───────────────────────── Stage-2 : Down ────────────────────────────────
    #     for e in range(self.num_experts):
    #         routing_weight = routing_weights[:, e].unsqueeze(-1)
    #         # Predicated accumulate into expert_out
    #         masked_out = torch.where(routing_weight > 0,
    #                                 (upgate @ self.experts.down_proj[e]) * routing_weight,
    #                                 torch.zeros_like(expert_out))
    #         expert_out  += masked_out

    #     # ───────────────────────── Stage-3 : Shared expert ───────────────────────
    #     shared_out = self.shared_expert(x)                # [T, H]
    #     final = shared_out + expert_out  # restore [B,S,H]
    #     return final.view(B, S, H), router_logits

    # def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Implements expert routing and computation as in original LLaMA 4 MoE.

    #     Stages:
    #     - Input: [T, d_model]
    #     - Router: top-k logits masking + sigmoid (not softmax)
    #     - Routing mask: [T, E] with weights in (0, 1) (sparse but dense-shaped)
    #     - Expert MLP: run for all tokens, zero-out unrouted via `torch.where`
    #     - Shared MLP: runs for all tokens
    #     - Final: expert_output + shared_output, reshaped to [B, S, H]

    #     Returns:
    #         final: final output of MoE block → [B, S, H]
    #         router_logits: raw logits before sigmoid → [T, E] for analysis/debug
    #     """

    #     # -----------------------------------------------------------
    #     # : Flatten input from [B, S, H] to [T, H]
    #     # -----------------------------------------------------------
    #     batch, seq_len, hidden_dim = hidden_states.shape         # B = batch, S = seq_len, H = hidden
    #     T = batch * seq_len                                      # T = total number of tokens
    #     hidden_states_flat = hidden_states.view(T, hidden_dim)   # [T, H] = token-wise input

    #     # -----------------------------------------------------------
    #     # : Compute router logits → [T, E]
    #     # Route each token to its top-k preferred experts
    #     # -----------------------------------------------------------
    #     router_logits = self.router(hidden_states_flat)          # [T, E] where E = num_experts

    #     # -----------------------------------------------------------
    #     # Top-K selection → get topk expert scores and indices
    #     # Select k experts per token. This reduces compute and bandwidth.
    #     # -----------------------------------------------------------
    #     topk_values, topk_indices = torch.topk(router_logits, self.top_k, dim=1)  # both [T, K]

    #     # -----------------------------------------------------------
    #     # : Build sparse routing matrix with -inf mask
    #     # Initialize as -inf everywhere (which → sigmoid ≈ 0 later)
    #     # Then scatter the top-k values in — others remain -inf
    #     # -----------------------------------------------------------
    #     masked_logits = torch.full_like(router_logits, float("-inf"))  # [T, E]
    #     masked_logits.scatter_(1, topk_indices, topk_values)           # Set only top-k logits

    #     # -----------------------------------------------------------
    #     # : Sigmoid routing weights [T, E]
    #     # This gives per-token-per-expert gating signal ∈ (0, 1)
    #     # Unlike softmax, this does not normalize — each expert acts independently.
    #     # -----------------------------------------------------------
    #     routing_weights = torch.sigmoid(masked_logits.float()).to(hidden_states.dtype)  # [T, E]

    #     # -----------------------------------------------------------
    #     # : Zero tensor for accumulating expert outputs
    #     # Each expert contributes selectively into this output
    #     # -----------------------------------------------------------
    #     expert_output = torch.zeros_like(hidden_states_flat)  # [T, H]

    #     # -----------------------------------------------------------
    #     # : Per-expert routing and computation loop
    #     # For each expert e ∈ [0, E-1]:
    #     #   - Run MLP
    #     #   - Scale output with routing weight
    #     #   - Apply `torch.where` to zero-out tokens not routed to e
    #     #   - Accumulate into final expert output
    #     # -----------------------------------------------------------
    #     for expert_idx in range(self.num_experts):
    #         # 1. Apply expert-specific gate and up projections (W_e): [T, 2I]
    #         gate_up = hidden_states_flat @ self.experts.gate_up_proj[expert_idx]  # [T, 2I]

    #         # 2. Split into gate and up branches (standard LLaMA4 MLP design)
    #         gate, up = gate_up.chunk(2, dim=-1)                                   # [T, I], [T, I]
    #         activated = self.experts.act_fn(gate) * up                            # [T, I]

    #         # 3. Down projection: convert expert activation to output dim
    #         expert_out = activated @ self.experts.down_proj[expert_idx]           # [T, H]

    #         # 4. Get routing weights for this expert: [T, 1]
    #         routing_weight = routing_weights[:, expert_idx].unsqueeze(-1)         # [T, 1]

    #         # 5. Conditionally apply expert output
    #         #    : cond ? X * w : 0 → ONNX-safe execution
    #         #    Only routed tokens contribute non-zero values
    #         masked_out = torch.where(
    #             routing_weight > 0,                     # condition
    #             expert_out * routing_weight,            # scaled expert output
    #             torch.zeros_like(expert_out)            # fallback = 0
    #         )

    #         # 6. Accumulate into total expert output
    #         expert_output += masked_out  # [T, H]

    #     # -----------------------------------------------------------
    #     # : Shared expert path (non-sparse MLP)
    #     # Always applied to every token — acts as a residual path
    #     # -----------------------------------------------------------
    #     shared_output = self.shared_expert(hidden_states_flat)  # [T, H]

    #     # -----------------------------------------------------------
    #     # : Final output = shared + expert → reshape back to [B, S, H]
    #     # -----------------------------------------------------------
    #     final = expert_output + shared_output                   # [T, H]
    #     final = final.view(batch, seq_len, hidden_dim)          # [B, S, H]

    #     return final, router_logits


class QEffLlama4TextAttention(Llama4TextAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(*input_shape, -1, self.head_dim)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]

        kv_seq_len = past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        ##
        if self.use_rope:  # the 16E model skips rope for long context on certain layers
            query_states, key_states = qeff_apply_rotary_emb(
                query_states, key_states, position_embeddings.to(query_states.device)
            )

        if hasattr(self, "qk_norm"):  # the 128E model does not use qk_norm
            query_states = self.qk_norm(query_states)
            key_states = self.qk_norm(key_states)

        # Use temperature tuning from https://arxiv.org/abs/2501.19399) to NoROPE layers
        if self.attn_temperature_tuning and not self.use_rope:
            attn_scales = (
                torch.log(torch.floor((cache_position.float() + 1.0) / self.floor_scale) + 1.0) * self.attn_scale + 1.0
            )
            attn_scales = attn_scales.view((*input_shape, 1, 1))
            query_states = (query_states * attn_scales).to(query_states.dtype)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"batch_index": batch_index, "position_ids": position_ids}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value


class QEffLlama4TextDecoderLayer(Llama4TextDecoderLayer):
    """
    Copied from LlamaForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama.py
    The only differences are:
    - add new args batch idx for the CB models
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        chunk_causal_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        # use local attention mask for ROPE layers
        if self.use_chunked_attention and chunk_causal_mask is not None:
            attention_mask = chunk_causal_mask

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_value=past_key_value,
            batch_index=batch_index,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        if self.is_moe_layer:
            # Change by VB
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        hidden_states = residual + hidden_states.view(residual.shape)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class QEffLlama4TextModel(Llama4TextModel):
    """
    Copied from LlamaForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama.py
    The only differences are:
    - add new args cache idx for the kv retention
    """

    def __init__(self, config: Llama4TextModel):
        super().__init__(config)
        # Define the general __qeff_init__() for any changes in the init calls
        # Set the init in the module mapping pytorch transforms
        self.config = config
        self.__qeff_init__()

    def __qeff_init__(self):
        self.rotary_emb = QEffLlama4TextRotaryEmbedding(config=self.config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=past_seen_tokens)

        _, chunk_causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        freq_cis = self.rotary_emb(hidden_states, position_ids=position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                chunk_causal_mask=chunk_causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                batch_index=batch_index,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=freq_cis,
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

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


class QEffLlama4ForCausalLM(Llama4ForCausalLM):
    """
    Copied from Llama4ForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama.py
    The only differences are:
    - add new args cache idx for the kv retention
    """

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        # Cast to INT32 to avoid issue while running in ONNXRT
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs[0][torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class QEffLlama4EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        vision_outputs = self.model.vision_model(pixel_values=pixel_values)
        return vision_outputs.last_hidden_state


class QEffLlama4DecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.language_model = self.model.language_model

    def forward(self, input_ids, vit_embeds, position_ids, past_key_values):
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        image_input_embeds = input_embeds.reshape(B * N, C)
        image_input_ids = input_ids.reshape(B * N)
        selected = image_input_ids == 200092  # Llama4 Image Token Index
        indices1 = selected.unsqueeze(0).to(torch.int64).cumsum(1) - 1
        indices0 = torch.arange(selected.unsqueeze(0).shape[0]).view(-1, 1)
        image_features_expanded = vit_embeds.reshape(-1, C).unsqueeze(0)[indices0, indices1]
        image_input_embeds = torch.where(selected.unsqueeze(0).unsqueeze(-1), image_features_expanded, input_embeds)
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), input_embeds, image_input_embeds)
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds, position_ids=position_ids, past_key_values=past_key_values, use_cache=True
        )
        return outputs.logits, vit_embeds, outputs.past_key_values


class QEffLlama4ForConditionalGeneration(Llama4ForConditionalGeneration):
    def get_qeff_vision_encoder(self):
        return QEffLlama4EncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEffLlama4DecoderWrapper(self)

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        img_size: int,
        kv_offload: bool = False,
        **compiler_options,
    ):
        # TODO: check if this should be named num_patches or something else
        batch_size_times_num_tiles = compiler_options.pop("batch_size_times_num_tiles", None)
        if batch_size_times_num_tiles is None:
            logger.warning(
                "User should pass `batch_size_times_num_tiles` to compile API to fix the dynamic axes `pixel_values`, you can get more info by calling get_inputs_info function!, Since its not found setting its value to 17"
            )
            batch_size_times_num_tiles = 17

        prefill_seq_len = 3952  # FIXME based on the feature size
        ctx_len = ctx_len if ctx_len else constants.INTERN_CTX_LEN
        if img_size is None and hasattr(self.config.vision_config, "image_size"):
            img_size = getattr(self.config.vision_config, "image_size")
        elif img_size is None:
            img_size = 336  # FIXME based on llama4 Image size
            logger.warning("Setting img_size to be 336, as it was neither passed nor found in vision_config")

        vision = [
            {
                "batch_size_times_num_tiles": batch_size_times_num_tiles,
                "img_size": img_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
            }
        ]
        lang = [
            {
                "batch_size": batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "batch_size_times_num_tiles": batch_size_times_num_tiles,
                "img_size": img_size,
            },
            {
                "batch_size": batch_size,
                "seq_len": "1",
                "ctx_len": ctx_len,
                "batch_size_times_num_tiles": batch_size_times_num_tiles,
                "img_size": img_size,
            },
        ]

        specializations = {}

        if kv_offload:
            specializations["vision"] = vision
            specializations["lang"] = lang
            return specializations, compiler_options
        else:
            return lang, compiler_options

    def get_onnx_dynamic_axes(self, kv_offload: bool = False):
        # Define dynamic axes
        vision_dynamic_axes = {}
        lang_dynamic_axes = {}
        lang_dynamic_axes["input_ids"] = {0: "batch_size", 1: "seq_len"}
        lang_dynamic_axes["position_ids"] = {0: "batch_size", 1: "seq_len"}
        lang_dynamic_axes["vit_embeds"] = {0: "batch_size_times_num_tiles"}
        vision_dynamic_axes["pixel_values"] = {0: "batch_size_times_num_tiles", 2: "img_size", 3: "img_size"}

        pkv_dynamic_axes = {0: "batch_size", 2: "ctx_len"}
        for i in range(self.language_model.config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_dynamic_axes[f"past_{kv}.{i}"] = pkv_dynamic_axes

        dynamic_axes = {}
        if kv_offload:
            dynamic_axes["vision"] = vision_dynamic_axes
            dynamic_axes["lang"] = lang_dynamic_axes
        else:
            dynamic_axes = {**vision_dynamic_axes, **lang_dynamic_axes}
        return dynamic_axes

    def get_output_names(self, kv_offload: bool = False):
        vision_output_names = ["vit_embeds"]
        lang_output_names = ["logits"]
        for i in range(self.language_model.config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_output_names.append(f"past_{kv}.{i}_RetainedState")

        output_names = {}
        if kv_offload:
            lang_output_names.insert(1, "vit_embeds_RetainedState")
            output_names["vision"] = vision_output_names
            output_names["lang"] = lang_output_names
        else:
            lang_output_names.insert(1, "pixel_values_RetainedState")
            return lang_output_names
        return output_names

    def get_dummy_inputs(self, kv_offload: bool = False):
        if vis_cfg := getattr(self.config, "vision_config", None):
            img_size = getattr(vis_cfg, "image_size", 336)
        else:
            img_size = 336
        # if img_size != constants.INTERN_IMG_SIZE and kv_offload:
        #     raise NotImplementedError("Image Size other than 448 is not supported for Intern models yet.")

        # patch_size = getattr(self.config.vision_config, "patch_size", None)
        # downsample_ratio = getattr(self.config, "downsample_ratio", None)
        # if patch_size and downsample_ratio:
        #     computed_feature_size = int(((img_size / patch_size) * downsample_ratio) ** 2)
        #     if computed_feature_size != constants.INTERN_FEATURE_SIZE:
        #         logger.warning(
        #             "Discrepancy detected between estimated and actual feature sizes. Could impact on functionality or accuracy"
        #         )

        # Define shapes
        inputs_shapes = {}
        inputs_shapes["input_ids"] = (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)
        inputs_shapes["vit_embeds"] = (
            17,  # constants.INTERN_NUM_PATCHES,
            14,  # constants.INTERN_FEATURE_SIZE,
            self.language_model.config.hidden_size,  # 5120
        )
        inputs_shapes["position_ids"] = (
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN,
        )
        inputs_shapes["pixel_values"] = (
            17,  # constants.INTERN_NUM_PATCHES,
            constants.INTERN_NUM_CHANNELS,
            img_size,
            img_size,
        )

        # Define inputs
        vision_inputs = {}
        lang_inputs = {}
        vision_inputs["pixel_values"] = torch.zeros((inputs_shapes["pixel_values"]), dtype=torch.float32)
        lang_inputs["input_ids"] = torch.zeros((inputs_shapes["input_ids"]), dtype=torch.int64)
        lang_inputs["vit_embeds"] = torch.zeros((inputs_shapes["vit_embeds"]), dtype=torch.float32)
        lang_inputs["position_ids"] = (
            torch.arange(constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN, dtype=torch.int64)
            .view(1, constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)
            .repeat(constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, 1)
        )

        # Add data for KV
        kv_cache_shape = get_padding_shape_from_config(
            config=self.language_model.config,
            batch_size=constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            seq_len=constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN,
        )

        lang_inputs["past_key_values"] = [[] for _ in range(self.language_model.config.num_hidden_layers)]
        for i in range(self.language_model.config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_inputs["past_key_values"][i].append(torch.zeros(kv_cache_shape, dtype=torch.float32))

        inputs = {}
        if kv_offload:
            inputs["vision"] = vision_inputs
            inputs["lang"] = lang_inputs
        else:
            lang_inputs.pop("vit_embeds")
            inputs = {**vision_inputs, **lang_inputs}

        return inputs

    def get_inputs_info(self):
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="attention_mask", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(
                name="pixel_values",
                datatype=torch.float32,
                shape=("batch_size_times_num_tiles", 3, "img_size", "img_size"),
            ),
        ]
