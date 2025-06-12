# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

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
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


class QEFFGrok1CustomRMSNormAIC(nn.Module):
    """
    RMSNorm module that works by replacing the current module with compiler known custom-op.
    """

    def forward(self, hidden_states):
        """
        Forward pass of the RMSNorm module.

        Args:
            hidden_states (torch.Tensor): Input tensor to be normalized.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        return CustomRMSNormFunc.apply(
            hidden_states, self.scale, self.variance_epsilon if hasattr(self, "variance_epsilon") else self.eps
        )


class QEffGrok1MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    """

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
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass of the multi-head attention module.

        Args:
            hidden_states (torch.Tensor): Input tensor.
            layer_idx (int): Layer index.
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): Position ids. Defaults to None.
            past_key_value (Optional[Tuple[torch.Tensor]], optional): Past key value. Defaults to None.
            batch_index (Optional[torch.LongTensor], optional): Batch index. Defaults to None.
            output_attentions (bool, optional): Whether to output attentions. Defaults to False.
            use_cache (bool, optional): Whether to use cache. Defaults to False.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]: Attention output, attention weights, and past key value.
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
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
            attn_weights = torch.where(attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), attn_weights)

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

        return attn_output, attn_weights, past_key_value


class QEffGrok1MoeBlock(nn.Module):
    """
    Mixture of experts (MoE) block.
    """

    def forward(self, hidden_states: torch.Tensor):
        """
        Forward pass of the MoE block.

        Args:
            hidden_states (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: MoE output.
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        # Creating experts mask and routing weights masked
        awesome_experts_mask_1 = (
            torch.nn.functional.one_hot(selected_experts[:, 0], num_classes=self.num_experts).bool().T.unsqueeze(-1)
        )
        awesome_experts_mask_2 = (
            torch.nn.functional.one_hot(selected_experts[:, 1], num_classes=self.num_experts).bool().T.unsqueeze(-1)
        )

        gateupout1 = torch.zeros(hidden_states.shape[0], self.ffn_dim)  # T, hs
        gateupout2 = torch.zeros(hidden_states.shape[0], self.ffn_dim)  # T, hs
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            current_expert_output = expert_layer.act_fn(expert_layer.linear(hidden_states)) * expert_layer.linear_v(
                hidden_states
            )
            gateupout1 += torch.where(
                awesome_experts_mask_1[expert_idx], current_expert_output, torch.zeros_like(gateupout1)
            )
            gateupout2 += torch.where(
                awesome_experts_mask_2[expert_idx], current_expert_output, torch.zeros_like(gateupout2)
            )

        downout1 = torch.zeros_like(hidden_states)
        downout2 = torch.zeros_like(hidden_states)
        concat_mask = torch.cat((awesome_experts_mask_1.unsqueeze(0), awesome_experts_mask_2.unsqueeze(0)), dim=0)
        concat_down = torch.cat((downout1.unsqueeze(0), downout2.unsqueeze(0)), dim=0)
        concat_gateout = torch.cat((gateupout1.unsqueeze(0), gateupout2.unsqueeze(0)), dim=0)
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            concat_down += torch.where(
                concat_mask[:, expert_idx, :], expert_layer.linear_1(concat_gateout), torch.zeros_like(concat_down)
            )

        downout1, downout2 = concat_down[0], concat_down[1]
        hidden_states = (
            downout1 * routing_weights[:, 0].unsqueeze(-1) + downout2 * routing_weights[:, 1].unsqueeze(-1)
        ).reshape(batch_size, sequence_length, hidden_dim)

        return hidden_states, router_logits


class QEffGrok1DecoderLayer(nn.Module):
    """
    Decoder block of Grok1 model.
    """

    def __qeff_init__(self):
        """
        Assigning extra args to Moe block of decoder.
        """
        self.moe_block.ffn_dim = self.config.intermediate_size

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
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Initialize the decoder layer.

        Args:
            hidden_states (torch.Tensor): Input tensor.
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): Position ids. Defaults to None.
            past_key_value (Optional[Tuple[torch.Tensor]], optional): Past key value. Defaults to None.
            batch_index (Optional[torch.LongTensor], optional): Batch index. Defaults to None.
            output_attentions (Optional[bool], optional): Whether to output attentions. Defaults to False.
            output_router_logits (Optional[bool], optional): Whether to output router logits. Defaults to False.
            use_cache (Optional[bool], optional): Whether to use cache. Defaults to False.

        Returns:
            Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]: Decoder output, attention weights, and past key value.
        """
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
    """
    Grok1 model
    """

    def __qeff_init__(self):
        """
        Initialize the extra args to model.
        """
        for idx, layer in enumerate(self.layers):
            layer.layer_idx = idx
            layer.config = self.config

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        """
        Forward pass of the Grok1 model.
        Args:
            input_ids (torch.LongTensor, optional): Input ids. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): Position ids. Defaults to None.
            past_key_values (Optional[List[torch.FloatTensor]], optional): Past key values. Defaults to None.
            batch_index (Optional[torch.LongTensor], optional): Batch index. Defaults to None.
            inputs_embeds (Optional[torch.FloatTensor], optional): Input embeddings. Defaults to None.
            use_cache (Optional[bool], optional): Whether to use cache. Defaults to None.
            output_attentions (Optional[bool], optional): Whether to output attentions. Defaults to None.
            output_hidden_states (Optional[bool], optional): Whether to output hidden states. Defaults to None.
            output_router_logits (Optional[bool], optional): Whether to output router logits. Defaults to None.
            return_dict (Optional[bool], optional): Whether to return a dictionary. Defaults to None.

        Returns:
            Union[Tuple, MoeModelOutputWithPast]: Model output.
        """
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

        seq_length_with_past = seq_length
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

        for idx, decoder_layer in enumerate(self.layers):
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
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

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
    """
    Grok model for causal language modeling.
    """

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """
        Forward pass for Grok model for causal language modeling

        Args:
            input_ids (torch.LongTensor, optional): Input ids. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): Position ids. Defaults to None.
            past_key_values (Optional[List[torch.FloatTensor]], optional): Past key values. Defaults to None.
            batch_index (Optional[torch.LongTensor], optional): Batch index. Defaults to None.
            inputs_embeds (Optional[torch.FloatTensor], optional): Input embeddings. Defaults to None.
            labels (Optional[torch.LongTensor], optional): Labels. Defaults to None.
            use_cache (Optional[bool], optional): Whether to use cache. Defaults to None.
            output_attentions (Optional[bool], optional): Whether to output attentions. Defaults to None.
            output_hidden_states (Optional[bool], optional): Whether to output hidden states. Defaults to None.
            output_router_logits (Optional[bool], optional): Whether to output router logits. Defaults to None.
            return_dict (Optional[bool], optional): Whether to return a dictionary. Defaults to None.

        Returns:
            MoeCausalLMOutputWithPast: Model output.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

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
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            **kwargs,
        )

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
