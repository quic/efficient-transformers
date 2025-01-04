from typing import Optional, Tuple

import torch
from transformers.models.bert.modeling_bert import BERT_SELF_ATTENTION_CLASSES, BertSelfAttention

from QEfficient.utils.constants import BLOCK_SIZE


class QEffBlockBertSelfAttention(BertSelfAttention):
    # Adapted from BertSelfAttention
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        block_size: int = BLOCK_SIZE,
    ) -> Tuple[torch.Tensor]:
        if self.position_embedding_type != "absolute" or output_attentions or head_mask is not None:
            return super().forward(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
        bsz, tgt_len, _ = hidden_states.size()

        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # If this is instantiated as a cross-attention module, the keys and values come from an encoder; the attention
        # mask needs to be such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        attention_mask = encoder_attention_mask if is_cross_attention else attention_mask

        # Check `seq_length` of `past_key_value` == `len(current_states)` to support prefix tuning
        if is_cross_attention and past_key_value and past_key_value[0].shape[2] == current_states.shape[1]:
            key_layer, value_layer = past_key_value
        else:
            key_layer = self.transpose_for_scores(self.key(current_states))
            value_layer = self.transpose_for_scores(self.value(current_states))
            if past_key_value is not None and not is_cross_attention:
                key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The tgt_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create
        # a causal mask in case tgt_len == 1.
        is_causal = (
            True if self.is_decoder and not is_cross_attention and attention_mask is None and tgt_len > 1 else False
        )
        num_blocks = max(1, (tgt_len // block_size))
        attn_scale = torch.rsqrt(torch.tensor(query_layer.size(-1)))
        seq_len = query_layer.size(2)
        attn_output = []
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, seq_len)
            query_layer_block = query_layer[:, :, start_idx:end_idx, :]

            attn_output_ = torch.nn.functional.scaled_dot_product_attention(
                query_layer_block,
                key_layer,
                value_layer,
                attn_mask=attention_mask,
                dropout_p=self.dropout_prob if self.training else 0.0,
                is_causal=is_causal,
                scale=attn_scale,
            )
            attn_output.append(attn_output_)
        if i == 0:
            attn_output = attn_output_.clone()
        else:
            attn_output = torch.cat([attn_output, attn_output_.clone()], dim=2)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)

        outputs = (attn_output,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

BERT_SELF_ATTENTION_CLASSES.update(
    {
        "custom": QEffBlockBertSelfAttention,
    }
)
