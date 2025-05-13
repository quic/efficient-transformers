# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.models.llama.modeling_llama import repeat_kv

from QEfficient.customop.rms_norm import CustomRMSNormFunc
from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.transformers.models.llama.modeling_llama import qeff_apply_rotary_pos_emb


class QEFFGrok1CustomRMSNormAIC(nn.Module):
    """
    RMSNorm module that works by replacing the current module with compiler known custom-op.
    """

    def forward(self, hidden_states):
        return CustomRMSNormFunc.apply(
            hidden_states, self.scale, self.variance_epsilon if hasattr(self, "variance_epsilon") else self.eps
        )


class QEffGrok1MultiHeadAttention(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        layers_start_end_indices: Optional[List[int]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        layer_idx = (
            layer_idx % (layers_start_end_indices[1] - layers_start_end_indices[0])
            if layers_start_end_indices is not None
            else layer_idx
        )

        if past_key_value is not None:
            kv_seq_len = past_key_value.get_usable_length(kv_seq_len, layer_idx)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "batch_index": batch_index,
                "position_ids": position_ids,
            }  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)).to(torch.float)
        attn_weights = attn_weights * self.attn_output_multiplier
        attn_weights = self.max_attn_val * F.tanh(attn_weights / self.max_attn_val)

        if attention_mask is not None:
            attn_weights = torch.where(attention_mask, torch.tensor(-10000.0, dtype=torch.float32), attn_weights)

        attn_weights = F.softmax(attn_weights, dim=-1).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class QEffGrok1MoeBlock(nn.Module):
    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, H = hidden.shape
        T = B * S
        x = hidden.view(T, H)

        router_logits = self.gate(x)  # [T, E]
        prob = F.softmax(router_logits, -1, dtype=torch.float)
        top_w, top_i = torch.topk(prob, self.top_k, -1)
        top_w = top_w.to(x.dtype)

        expert1_idx, expert2_idx = top_i[:, 0], top_i[:, 1]  # [T]
        weight1, weight2 = top_w[:, 0], top_w[:, 1]  # [T]

        # I = self.config.ffn_dim
        I = 32768  # TODO: Find a way to identify from config
        upgate1 = x.new_zeros((T, I))
        upgate2 = x.new_zeros((T, I))
        expert_out1 = x.new_zeros((T, H))
        expert_out2 = x.new_zeros((T, H))

        for e in range(self.num_experts):
            exp = self.experts[e]
            mask1 = (expert1_idx == e).unsqueeze(1)  # [T, 1]
            mask2 = (expert2_idx == e).unsqueeze(1)  # [T, 1]

            hidden_gate = (exp.act_fn(exp.linear(x))) * exp.linear_v(x)
            # Accumulate weighted contributions
            upgate1 += torch.where(mask1, hidden_gate, torch.zeros_like(upgate1))
            upgate2 += torch.where(mask2, hidden_gate, torch.zeros_like(upgate2))

        for e in range(self.num_experts):
            exp = self.experts[e]
            mask1 = (expert1_idx == e).unsqueeze(1)
            mask2 = (expert2_idx == e).unsqueeze(1)

            expert_out1 += torch.where(
                mask1, exp.linear_1(upgate1) * weight1.unsqueeze(1), torch.zeros_like(expert_out1)
            )
            expert_out2 += torch.where(
                mask2, exp.linear_1(upgate2) * weight2.unsqueeze(1), torch.zeros_like(expert_out2)
            )

        expert_out = expert_out1 + expert_out2
        return expert_out.view(B, S, H), router_logits


class QEffGrok1DecoderLayer(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        layers_start_end_indices: Optional[List[int]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.pre_attn_norm(hidden_states)
        hidden_states, attention_weights, present_key_value = self.attn(
            hidden_states,
            layer_idx=self.layer_idx,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            batch_index=batch_index,
            output_attentions=output_attentions,
            use_cache=use_cache,
            layers_start_end_indices=layers_start_end_indices,
        )
        hidden_states = self.post_attn_norm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_moe_norm(hidden_states)
        hidden_states, router_logits = self.moe_block(hidden_states)
        hidden_states = self.post_moe_norm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attention_weights,)
        if use_cache:
            outputs += (present_key_value,)
        if output_router_logits:
            outputs += (router_logits,)
        return outputs


class QEffGrok1Model(nn.Module):
    def __qeff_init__(self):
        for idx, layer in enumerate(self.layers):
            layer.layer_idx = idx

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        layers_start_end_indices: Optional[List[int]] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if layers_start_end_indices is not None:
            if layers_start_end_indices[0] != 0:
                assert inputs_embeds is not None and input_ids is None, (
                    "got input_ids or did not get inputs_embeds, expected input_ids to be None or inputs_embeds to be not None"
                )
            else:
                assert input_ids is not None and inputs_embeds is None, (
                    "got input_embeds or did not get inputs_ids, expected input_ids to be not None or inputs_embeds to be None"
                )

        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = inputs_embeds * self.embedding_multiplier_scale

        attention_mask = _create_causal_mask(position_ids=position_ids, target_length=past_key_values_length)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = () if use_cache else None

        start_idx = layers_start_end_indices[0] if layers_start_end_indices is not None else 0
        end_idx = layers_start_end_indices[1] if layers_start_end_indices is not None else len(self.layers)
        for idx, decoder_layer in enumerate(self.layers[start_idx:end_idx]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                batch_index=batch_index,
                output_attentions=output_attentions,
                use_cache=use_cache,
                layers_start_end_indices=layers_start_end_indices,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        if layers_start_end_indices is not None and layers_start_end_indices[1] != len(self.layers):
            pass
        else:
            hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_values = past_key_values.to_legacy_cache()

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


class QEffGrok1ModelForCausalLM(nn.Module):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        batch_index: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        layers_start_end_indices = getattr(self, "layers_start_end_indices", None)
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
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            layers_start_end_indices=layers_start_end_indices,
            **kwargs,
        )
        if layers_start_end_indices is not None and layers_start_end_indices[1] != len(self.model.layers):
            return outputs[0], outputs.past_key_values

        # Cast to int32 to avoid ONNXRT issue
        logit_idx = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs[0][torch.arange(position_ids.shape[0]).view(-1, 1), logit_idx]
        logits = self.lm_head(hidden_states)
        logits = logits * self.output_multiplier_scale
        logits = logits.float()

        return MoeCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )
