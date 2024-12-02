# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""PyTorch Falcon model."""

import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.falcon.modeling_falcon import (
    FalconAttention,
    FalconConfig,
    FalconDecoderLayer,
    FalconForCausalLM,
    FalconModel,
    FalconRotaryEmbedding,
    apply_rotary_pos_emb,
    build_alibi_tensor,
    dropout_add,
    logger,
    rotate_half,
)

from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask


class QEffFalconRotaryEmbedding(FalconRotaryEmbedding):
    """
    Copied from LlamaForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    The only differences are:
    - Add static sin/cos computations.
    """

    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[FalconConfig] = None,
    ):
        super(FalconRotaryEmbedding, self).__init__()  # Initialize nn.Module
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`LlamaRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.45"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

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


def qeff_apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
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

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    # Cast back to original dtype
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


class QEffFalconAttention(FalconAttention):
    """
    Copied from FalconAttention: https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon/modeling_falcon.py
    The only differences are:
    - add new args position idx for the cache_kwargs for kv retention
    """

    # Copied from
    # transformers.models.llama.modeling_llama.LlamaAttention._init_rope with
    # Llama->Falcon

    def __init__(self, config: FalconConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        # Define the general __qeff_init__() for any changes in the init calls
        # Set the init in the module mapping pytorch transforms
        self.config = config
        self.__qeff_init__()

    def __qeff_init__(self):
        self.rotary_emb = QEffFalconRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        layer_past: Optional[Cache] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ):
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
        num_kv_heads = self.num_heads if self.new_decoder_architecture else self.num_kv_heads
        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, query_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size, self.num_heads, query_length, self.head_dim)
        key_layer = key_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, self.head_dim)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, self.head_dim)

        # kv_seq_len = key_layer.shape[-2]
        # if layer_past is not None:
        #     if self.layer_idx is None:
        #         raise ValueError(
        #             f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
        #             "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
        #             "with a layer index."
        #         )
        #     kv_seq_len = layer_past.get_usable_length(kv_seq_len, self.layer_idx)

        if alibi is None:
            if position_embeddings is None:
                logger.warning_once(
                    "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                    "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                    "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                    "removed and `position_embeddings` will be mandatory."
                )
                cos, sin = self.rotary_emb(value_layer, position_ids)
            else:
                cos, sin = position_embeddings
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin)

        if layer_past is not None:
            cache_kwargs = {"cache_position": cache_position}
            if alibi is None:
                cache_kwargs.update({"sin": sin, "cos": cos, "batch_index": batch_index, "position_ids": position_ids})
            key_layer, value_layer = layer_past.update(key_layer, value_layer, self.layer_idx, cache_kwargs)

        kv_length = key_layer.shape[-2]
        if self._use_sdpa and query_layer.device.type == "cuda" and attention_mask is not None:
            # For torch<=2.1.2, SDPA with memory-efficient backend is bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            query_layer = query_layer.contiguous()
            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, : key_layer.shape[-2]]

        if alibi is None:
            if self._use_sdpa and not output_attentions:
                # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
                # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
                # The query_length > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not
                # create a causal mask in case query_length == 1.
                is_causal = True if self.is_causal and attention_mask is None and query_length > 1 else False
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=is_causal,
                )
                attention_scores = None
            else:
                attention_scores = query_layer @ key_layer.transpose(-1, -2)
                attention_scores /= math.sqrt(self.head_dim)
                attention_scores = torch.where(
                    attention_mask, torch.tensor(-10000.0, dtype=torch.float32), attention_scores
                )
                attention_scores = F.softmax(attention_scores + attention_mask, dim=-1, dtype=hidden_states.dtype)
                # It is unclear why neither dropout nor head_mask is applied here (while it is with alibi).
                attn_output = attention_scores @ value_layer

            attn_output = attn_output.view(batch_size, self.num_heads, query_length, self.head_dim)
            attn_output = attn_output.permute(0, 2, 1, 3)
            attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)

            attn_output = self.dense(attn_output)

            if output_attentions:
                return attn_output, layer_past, attention_scores
            else:
                return attn_output, layer_past

        else:
            if self._use_sdpa and not output_attentions and head_mask is None:
                # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
                # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
                is_causal = True if self.is_causal and attention_mask is None and query_length > 1 else False
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    attn_mask=attention_mask,
                    dropout_p=self.attention_dropout.p if self.training else 0.0,
                    is_causal=is_causal,
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
                return attn_output, layer_past, attention_probs
            else:
                return attn_output, layer_past


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
        past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        # Compute alibi tensor: check build_alibi_tensor documentation
        alibi = None
        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        batch_size, seq_length, _ = inputs_embeds.shape
        if self.use_alibi:
            mask = (
                torch.ones(
                    (batch_size, seq_length + past_key_values_length), device=inputs_embeds.device, dtype=torch.long
                )
                if attention_mask is None
                else attention_mask
            )
            alibi = build_alibi_tensor(mask, self.num_heads, dtype=inputs_embeds.dtype)

        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            position_ids,
            past_key_values,
            output_attentions,
            head_mask,
            alibi,
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        next_decoder_cache = None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    alibi,
                    causal_mask,
                    position_ids,
                    head_mask[i],
                    past_key_values,
                    use_cache,
                    output_attentions,
                    cache_position,
                    position_embeddings,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=past_key_values,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    batch_index=batch_index,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                next_decoder_cache = outputs[1]

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attentions] if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
        head_mask: torch.Tensor,
        alibi: torch.Tensor,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not using_static_cache
            and not output_attentions
            and head_mask is None
            and alibi is None
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, _ = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        batch_size, sequence_length, _ = input_tensor.shape
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=target_length)

        # We take care to integrate alibi bias in the causal_mask here
        if head_mask is None and alibi is not None:
            alibi = alibi.reshape(batch_size, -1, *alibi.shape[1:])
            causal_mask = torch.masked_fill(
                alibi / math.sqrt(self.config.hidden_size // self.num_heads),
                causal_mask < -1,
                min_dtype,
            )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


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
        cache_position: Optional[torch.LongTensor] = None,
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
            cache_position=cache_position,
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
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
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
            position_embeddings=position_embeddings,
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
