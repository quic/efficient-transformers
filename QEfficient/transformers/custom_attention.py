# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import math
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertSelfAttention

from QEfficient.utils.constants import BLOCK_SIZE


class QEffBertSelfAttention(BertSelfAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        block_size: int = BLOCK_SIZE,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.

        bsz, tgt_len, _ = hidden_states.size()
        num_blocks = max(1, (tgt_len // block_size))
        context_layers = torch.zeros(bsz, self.num_attention_heads, tgt_len, self.attention_head_size)
        for iteration in range(num_blocks):
            attention_score_current = torch.matmul(
                query_layer[:, :, iteration * block_size : (iteration + 1) * block_size, :], key_layer.transpose(2, 3)
            ) / math.sqrt(self.attention_head_size)
            if attention_mask is not None:  # no matter the length, we just slice it
                attention_score_current = (
                    attention_score_current
                    + attention_mask[:, :, iteration * block_size : (iteration + 1) * block_size, :]
                )
            # upcast attention to fp32
            attention_probs = nn.functional.softmax(attention_score_current, dim=-1, dtype=torch.float32).to(
                query_layer.dtype
            )
            attention_probs = self.dropout(attention_probs)
            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layers[:, :, iteration * block_size : (iteration + 1) * block_size, :] = context_layer

        context_layers = context_layers.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layers.size()[:-2] + (self.all_head_size,)
        context_layers = context_layers.view(new_context_layer_shape)

        outputs = (context_layers, attention_probs) if output_attentions else (context_layers,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
