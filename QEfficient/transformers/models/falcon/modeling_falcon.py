# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""PyTorch Falcon model."""

import math
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.models.falcon.modeling_falcon import (
    FalconAttention,
    FalconDecoderLayer,
    FalconForCausalLM,
    FalconModel,
    _prepare_4d_causal_attention_mask_for_sdpa,
    apply_rotary_pos_emb,
    build_alibi_tensor,
    dropout_add,
    logger,
)

from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask


class QEffFalconAttention(FalconAttention):
    """
    Copied from FalconAttention: https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon/modeling_falcon.py
    The only differences are:
    - add new args position idx for the cache_kwargs for kv retention
    """

    # Copied from
    # transformers.models.llama.modeling_llama.LlamaAttention._init_rope with
    # Llama->Falcon

    def _split_heads(self, fused_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim), results share same memory storage as `fused_qkv`
        Args:
            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]
        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        if self.new_decoder_architecture:
            batch, seq_len, _ = fused_qkv.shape
            qkv = fused_qkv.view(batch, seq_len, -1, self.num_heads // self.num_kv_heads + 2, self.head_dim)
            query = qkv[:, :, :, :-2]
            key = qkv[:, :, :, [-2]]
            value = qkv[:, :, :, [-1]]

            # Calculate the required broadcasting dimensions
            broadcast_shape = query.shape

            # Expand 'key' and 'value' along the broadcast dimensions
            key = key.expand(broadcast_shape)
            value = value.expand(broadcast_shape)

            query, key, value = [x.flatten(2, 3) for x in (query, key, value)]
            return query, key, value
        elif not self.multi_query:
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
            return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
        else:
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads + 2, self.head_dim)
            return fused_qkv[..., :-2, :], fused_qkv[..., [-2], :], fused_qkv[..., [-1], :]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        alibi: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
        num_kv_heads = self.num_heads if self.new_decoder_architecture else self.num_kv_heads
        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, query_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size, self.num_heads, query_length, self.head_dim)
        key_layer = key_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, self.head_dim)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, self.head_dim)

        kv_seq_len = key_layer.shape[-2]
        if layer_past is not None:
            kv_seq_len = layer_past[0].shape[-2]
        if alibi is None:
            cos, sin = self.rotary_emb(value_layer, seq_len=kv_seq_len)
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, position_ids)

        if layer_past is not None:
            past_key_value = layer_past
            cache_kwargs = {
                "position_ids": position_ids,
                "batch_index": batch_index,
            }
            pkv = DynamicCache()
            pkv.key_cache.append(past_key_value[0])
            pkv.value_cache.append(past_key_value[1])
            key_layer, value_layer = pkv.update(key_layer, value_layer, 0, cache_kwargs)

        kv_length = key_layer.shape[-2]
        if use_cache:
            present = (pkv.key_cache[0], pkv.value_cache[0])
        else:
            present = None

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_layer.device.type == "cuda" and attention_mask is not None:
            query_layer = query_layer.contiguous()
            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()

        if alibi is None:
            # todo: (vbaddi) Comment the below, not required for Cloud AI 100
            attention_scores = query_layer @ key_layer.transpose(-1, -2)
            attention_scores /= math.sqrt(self.head_dim)
            # attention_mask = attention_mask[:, :, None, :, :]
            attention_scores = torch.where(
                attention_mask, torch.tensor(-10000.0, dtype=torch.float32), attention_scores
            )
            # attention_scores = F.softmax(attention_scores + attention_mask, dim=-1, dtype=hidden_states.dtype)
            attention_scores = F.softmax(attention_scores, dim=-1, dtype=hidden_states.dtype)
            # It is unclear why neither dropout nor head_mask is applied here (while it is with alibi).
            attn_output = attention_scores @ value_layer

            attn_output = attn_output.view(batch_size, self.num_heads, query_length, self.head_dim)
            attn_output = attn_output.permute(0, 2, 1, 3)
            attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)

            attn_output = self.dense(attn_output)

            if output_attentions:
                return attn_output, present, attention_scores
            else:
                return attn_output, present

        else:
            if self._use_sdpa and not output_attentions and head_mask is None:
                attn_output = F.scaled_dot_product_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    attn_mask=attention_mask,
                    dropout_p=self.attention_dropout.p if self.training else 0.0,
                    is_causal=self.is_causal and attention_mask is None and query_length > 1,
                )
                attn_output = attn_output.transpose(1, 2)
                attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)

                attn_output = self.dense(attn_output)
            else:
                matmul_result = query_layer @ key_layer.transpose(-1, -2)

                # change view to [batch_size, num_heads, q_length, kv_length]
                attention_scores = matmul_result.view(batch_size, self.num_heads, query_length, kv_length)

                # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
                input_dtype = attention_scores.dtype
                # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
                if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
                    attention_scores = attention_scores.to(torch.float32)

                attention_logits = attention_scores + alibi.view(batch_size, self.num_heads, 1, -1)
                attention_logits *= self.inv_norm_factor
                attention_probs = F.softmax(attention_logits + attention_mask, dim=-1, dtype=hidden_states.dtype)
                # [batch_size, num_heads, q_length, kv_length]
                attention_probs = self.attention_dropout(attention_probs)

                if head_mask is not None:
                    attention_probs = attention_probs * head_mask

                # change view [batch_size, num_heads, q_length, kv_length]
                attention_probs_reshaped = attention_probs.view(batch_size, self.num_heads, query_length, kv_length)

                # matmul: [batch_size * num_heads, q_length, head_dim]
                attn_output = (attention_probs_reshaped @ value_layer).flatten(0, 1)

                # change view [batch_size, q_length, num_heads * head_dim]
                attn_output = self._merge_heads(attn_output)

                attn_output = self.dense(attn_output)

            if output_attentions:
                return attn_output, present, attention_probs
            else:
                return attn_output, present


FALCON_ATTENTION_CLASSES = {"eager": FalconAttention, "sdpa": FalconAttention}


class QEffFalconModel(FalconModel):
    """
    Copied from FalconModel: https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon/modeling_falcon.py
    The only differences are:
    - add new args position idx for the cache_kwargs for kv retention
    - update causal attention mask
    """

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[-2]

        if self.use_alibi:
            mask = (
                torch.ones(
                    (batch_size, seq_length + past_key_values_length), device=inputs_embeds.device, dtype=torch.long
                )
                if attention_mask is None
                else attention_mask
            )
            alibi = build_alibi_tensor(mask, self.num_heads, dtype=hidden_states.dtype)
        else:
            alibi = None
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all
            # cases.
            if alibi is None:
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                )
            elif head_mask is None:
                alibi = alibi.reshape(batch_size, -1, *alibi.shape[1:])

                attention_mask_2d = attention_mask
                # We don't call _prepare_4d_causal_attention_mask_for_sdpa as
                # we need to mask alibi using the 4D attention_mask untouched.
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                )

                # We take care to integrate alibi bias in the attention_mask
                # here.
                if attention_mask_2d is None:
                    attention_mask = alibi / math.sqrt(self.config.hidden_size // self.num_heads)
                else:
                    attention_mask = torch.masked_fill(
                        alibi / math.sqrt(self.config.hidden_size // self.num_heads),
                        attention_mask < -1,
                        torch.finfo(alibi.dtype).min,
                    )

                    # From PyTorch 2.1 onwards, F.scaled_dot_product_attention with the memory-efficient attention backend
                    # produces nans if sequences are completely unattended in
                    # the attention mask. Details:
                    # https://github.com/pytorch/pytorch/issues/110213
                    if seq_length > 1:
                        attention_mask = AttentionMaskConverter._unmask_unattended(
                            attention_mask, attention_mask_2d, unmasked_value=0.0
                        )
            else:
                # PyTorch SDPA does not support head_mask, we fall back on the
                # eager implementation in this case.
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                )
        else:
            # 4d mask is passed through the layers
            attention_mask = _create_causal_mask(position_ids=position_ids, target_length=past_key_values_length)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    alibi,
                    attention_mask,
                    position_ids,
                    head_mask[i],
                    layer_past,
                    use_cache,
                    output_attentions,
                )
            elif batch_index is not None:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    batch_index=batch_index,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class QEffFalconForCausalLM(FalconForCausalLM):
    """
    Copied from FalconForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon/modeling_falcon.py
    The only differences are:
    - add new args position idx for the cache_kwargs for kv retention
    - update the hidden_states, and fix for onnx model
    """

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            batch_index=batch_index,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # lm_logits = self.lm_head(hidden_states)
        # Cast to INT32 to avoid issue while running in ONNXRT
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = transformer_outputs[0][torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class QEffFalconDecoderLayer(FalconDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        layer_past: Optional[Union[Cache, Tuple[torch.Tensor, torch.Tensor]]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        residual = hidden_states

        if self.config.new_decoder_architecture and self.config.num_ln_in_parallel_attn == 2:
            attention_layernorm_out = self.ln_attn(hidden_states)
            mlp_layernorm_out = self.ln_mlp(hidden_states)
        else:
            attention_layernorm_out = self.input_layernorm(hidden_states)

        # Self attention.
        attn_outputs = self.self_attention(
            attention_layernorm_out,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            batch_index=batch_index,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )

        attention_output = attn_outputs[0]

        if not self.config.new_decoder_architecture:
            if self.config.parallel_attn:
                mlp_layernorm_out = attention_layernorm_out
            else:
                residual = dropout_add(
                    attention_output, residual, self.config.attention_dropout, training=self.training
                )
                mlp_layernorm_out = self.post_attention_layernorm(residual)

        if (
            self.config.new_decoder_architecture
            and self.config.parallel_attn
            and self.config.num_ln_in_parallel_attn == 1
        ):
            mlp_layernorm_out = attention_layernorm_out

        outputs = attn_outputs[1:]

        # MLP.
        mlp_output = self.mlp(mlp_layernorm_out)

        if self.config.new_decoder_architecture or self.config.parallel_attn:
            mlp_output += attention_output

        output = dropout_add(mlp_output, residual, self.config.hidden_dropout, training=self.training)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, past_kv, attentions
