# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
# This file is adapted from vllm implementation by snowflake here: https://github.com/Snowflake-Labs/vllm/blob/swiftkv/vllm/model_executor/models/llama_swiftkv.py
# The Modules are updated as required by Cloud AI 100 HW requirements.


"""Inference-only LLaMA model compatible with HuggingFace weights."""

import math
from typing import List, Optional, Tuple, Type, Union

import torch
from torch import nn
from transformers import LlamaConfig
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm, logger, repeat_kv

from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.transformers.models.llama.modeling_llama import (
    QEffLlamaDecoderLayer,
    QEffLlamaRotaryEmbedding,
    qeff_apply_rotary_pos_emb,
)
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


class QEffLlamaSwiftKVConfig(LlamaConfig):
    """
    Args:
        num_key_value_layers (int, optional):
            The number of layers, from the first layer, that have keys and
            values. If None, all layers have keys and values.
        last_key_value_heads (int, optional):
            The number of heads in the last layer that have keys and values.
            If None, the number of heads in the last key-value layer is equal
            to the number of heads in all the other key-value layers.
    """

    model_type = "llama_swiftkv"

    def __init__(
        self,
        swiftkv: bool = False,
        num_key_value_layers: Optional[int] = None,
        key_value_group_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.swiftkv = swiftkv
        self.num_key_value_layers = num_key_value_layers or self.num_hidden_layers
        self.key_value_group_size = key_value_group_size or 1
        assert (self.num_hidden_layers - self.num_key_value_layers) % self.key_value_group_size == 0


class QEffLlamaSwiftKVAttention(nn.Module):
    def __init__(self, config: QEffLlamaSwiftKVConfig, layer_idx) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.layer_idx = layer_idx
        self.q_proj_swiftkv = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj_swiftkv = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj_swiftkv = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = QEffLlamaRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        attention_mask: torch.Tensor = None,
        batch_index: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()
        q_len = 1  # as we always run this for single token
        query = self.q_proj_swiftkv(hidden_states)
        # Reshape the query, key, and value tensors.
        query_states = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        cache_kwargs = {"position_ids": position_ids, "batch_index": batch_index}
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            if comp_ctx_lengths is not None:
                attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
                cache_kwargs["CCL"] = attention_mask.shape[-1]
            kv_seq_len = past_key_value.get_seq_length(self.layer_idx)
        key_states, value_states = past_key_value.read_only(self.layer_idx, cache_kwargs=cache_kwargs)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        position_ids = position_ids[torch.arange(bsz), position_ids.to(torch.int32).argmax(1)].unsqueeze(1)
        query_states, _ = qeff_apply_rotary_pos_emb(
            query_states, torch.empty_like(query_states), cos, sin, position_ids
        )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = torch.where(
                attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), attn_weights
            )
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


class QEffLlamaSwiftKVDecoderLayer(nn.Module):
    def __init__(self, config: QEffLlamaSwiftKVConfig, layer_idx) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = config.num_key_value_heads
        self.self_attn = QEffLlamaSwiftKVAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values,
        comp_ctx_lengths,
        causal_mask,
        batch_index: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, past_key_values = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_value=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            attention_mask=causal_mask,
            batch_index=batch_index,
        )

        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class QEffLlamaSwiftKVModel(nn.Module):
    config_class = QEffLlamaSwiftKVConfig

    def __init__(self, config: QEffLlamaSwiftKVConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.config = config

        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, None)
        self.layers = torch.nn.ModuleList(
            [
                QEffLlamaDecoderLayer(config=config, layer_idx=idx)
                if idx < config.num_key_value_layers
                else QEffLlamaSwiftKVDecoderLayer(config=config, layer_idx=idx)
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_swiftkv = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _run_swiftkv_layers(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values,
        comp_ctx_lengths,
        causal_mask,
        batch_index,
    ) -> torch.Tensor:
        for layer_idx in range(self.config.num_key_value_layers, self.config.num_hidden_layers):
            layer = self.layers[layer_idx]
            hidden_states = layer(
                hidden_states, position_ids, past_key_values, comp_ctx_lengths, causal_mask, batch_index
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        self.config._attn_implementation = "eager"
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
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
            else:
                causal_mask = _create_causal_mask(position_ids=position_ids, target_length=target_length)

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

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        past_key_values: List[torch.Tensor],
        comp_ctx_lengths: Optional[torch.LongTensor],
        batch_index: Optional[torch.LongTensor] = None,
    ):
        inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        use_cache = True

        if use_cache and not isinstance(past_key_values, Cache):
            if past_key_values is None:
                past_key_values = QEffDynamicCache()
            else:
                past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            None, inputs_embeds, cache_position, position_ids, past_key_values, False
        )
        hidden_states = inputs_embeds

        next_decoder_cache = None

        for layer_idx in range(self.config.num_key_value_layers):
            layer = self.layers[layer_idx]
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                output_attentions=False,
                use_cache=True,
            )

        bsz, q_len, _ = hidden_states.size()
        swiftkv_hidden_states = self.norm_swiftkv(hidden_states)
        ####################################
        ## THE MAGIC OF SWIFT KV BEGINS HERE
        ####################################
        for layer_idx in range(self.config.num_key_value_layers, self.config.num_hidden_layers):
            self_attn = self.layers[layer_idx].self_attn
            key_states = self_attn.k_proj_swiftkv(swiftkv_hidden_states)
            value_states = self_attn.v_proj_swiftkv(swiftkv_hidden_states)
            key_states = key_states.view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(
                1, 2
            )

            if past_key_values is not None:
                if self_attn.layer_idx is None:
                    raise ValueError(
                        f"The cache structure has changed since version v4.36. If you are using {self_attn.__class__.__name__} "
                        "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                        "with a layer index."
                    )
                kv_seq_len = past_key_values.get_seq_length(self_attn.layer_idx)

            cos, sin = self_attn.rotary_emb(value_states, seq_len=kv_seq_len)
            _, key_states = qeff_apply_rotary_pos_emb(torch.empty_like(key_states), key_states, cos, sin, position_ids)
            cache_kwargs = {"position_ids": position_ids, "batch_index": batch_index}
            past_key_values.write_only(key_states, value_states, self_attn.layer_idx, cache_kwargs)

        last_pos_id = position_ids.to(torch.int32).argmax(1, keepdim=True)
        orig_hidden_states = hidden_states

        # Extracting only the last valid position id to be processed by self-attn of half of the layers, as KV cache is already filled.
        if batch_index is not None:
            hidden_states = orig_hidden_states[torch.arange(orig_hidden_states.shape[0]).reshape(-1, 1), last_pos_id, :]
            causal_mask = causal_mask[torch.arange(orig_hidden_states.shape[0]).reshape(-1, 1), :, last_pos_id, :]
        else:
            hidden_states = orig_hidden_states[torch.arange(bsz).reshape(-1, 1), last_pos_id, :]
            causal_mask = causal_mask[torch.arange(bsz).reshape(-1, 1), :, last_pos_id, :]

        hidden_states, next_decoder_cache = self._run_swiftkv_layers(
            hidden_states, position_ids, past_key_values, comp_ctx_lengths, causal_mask, batch_index
        )
        # We can fill the orig_hidden_states with the processed hidden_states here but it's not needed as for next token prediction
        # we only need the last valid pos_indices hidden_states.
        # Here the shape of hiden_states is [batch_size, 1, hidden_dim] instead of [batch_size, seq_len, hidden_dim]
        # This saves un-necessary data movement on devices.
        ####################################
        ## THE MAGIC OF SWIFT KV ENDS HERE
        ####################################

        next_cache = next_decoder_cache.to_legacy_cache()
        return hidden_states, next_cache


class QEffLlamaSwiftKVForCausalLM(PreTrainedModel):
    config_class = QEffLlamaSwiftKVConfig

    def __init__(self, config: QEffLlamaSwiftKVConfig):
        super().__init__(config=config)

        self.model = QEffLlamaSwiftKVModel(
            config=config,
        )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config

    def get_repeated_layer_class(self) -> Type[nn.Module]:
        """
        Return the set of class used as the repeated layer across the model for subfunction extraction.

        Notes:
            This method should return the *class object* (not an instance).
            Downstream code can use this to find/build subfunctions for repeated blocks.
        """
        return {QEffLlamaSwiftKVDecoderLayer}

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Optional[Union[List[torch.FloatTensor]]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
    ):
        hidden_states, output_past_key_values = self.model(
            input_ids, position_ids, past_key_values, comp_ctx_lengths, batch_index
        )
        logits = self.lm_head(hidden_states)
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=output_past_key_values,
            hidden_states=None,
            attentions=None,
        )
