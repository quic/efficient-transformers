# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.cache_utils import Cache, HybridCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Attention,
    Gemma3Config,
    Gemma3DecoderLayer,
    Gemma3ForCausalLM,
    Gemma3TextModel,
    logger,
    repeat_kv,
    rotate_half,
)

from QEfficient.customop.rms_norm import CustomRMSNorm
from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask


class GemmaRMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float):
        hidden_states = hidden_states.to(torch.float32)
        div_first = hidden_states * torch.rsqrt(torch.tensor(hidden_states.shape[-1], dtype=torch.float32))
        variance = div_first.pow(2).sum(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + epsilon)
        return weight * hidden_states

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, hidden_states: torch.Value, weight: torch.Value, epsilon: torch.Value) -> torch.Value:
        return g.onnxscript_op(CustomRMSNorm, hidden_states, weight, epsilon_f=epsilon).setTypeAs(hidden_states)


class QEffGemma3CustomRMSNormAIC(nn.Module):
    """
    RMSNorm module that works by replacing the current module with compiler known custom-op.
    """

    def forward(self, hidden_states):
        return GemmaRMSNormFunc.apply(
            hidden_states,
            self.weight.float() + 1.0,
            self.variance_epsilon if hasattr(self, "variance_epsilon") else self.eps,
        )


class QEffGemma3RotaryEmbedding(nn.Module):
    """
    Copied from Gemma2RotaryEmbedding: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modeling_gemma2.py
    The only differences are:
    - Add static sin/cos computations.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
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
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
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


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap

    if attention_mask is not None:
        attn_weights = torch.where(attention_mask, torch.tensor(-10000.0, dtype=torch.float32), attn_weights)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class QEffGemma3Attention(Gemma3Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Gemma3Config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        # Define the general __qeff_init__() for any changes in the init calls
        # Set the init in the module mapping pytorch transforms
        self.__qeff_init__()

    def __qeff_init__(self):
        self.rotary_emb = QEffGemma3RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.config.max_position_embeddings,
            base=self.config.rope_theta,
        )

        config = copy.deepcopy(self.config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}

        self.rotary_emb_local = QEffGemma3RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len = past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        if self.is_sliding:
            cos, sin = self.rotary_emb_local(value_states, seq_len=kv_seq_len)
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "batch_index": batch_index, "position_ids": position_ids}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        # import ipdb; ipdb.set_trace()
        if self.config.attn_logit_softcapping is not None:
            attn_weights = attn_weights / self.config.attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.config.attn_logit_softcapping

        if attention_mask is not None:  # no matter the length, we just slice it
            attn_weights = torch.where(attention_mask.bool(), torch.tensor(-10000.0, dtype=torch.float32), attn_weights)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.config.num_attention_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.config.num_attention_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QEffGemma3DecoderLayer(Gemma3DecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        last_cache_position: int = 0,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # apply global RoPE to non-sliding layer only
        if self.self_attn.is_sliding:
            position_embeddings = position_embeddings_local
        else:
            position_embeddings = position_embeddings_global

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            batch_index=batch_index,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class QEffGemma3TextModel(Gemma3TextModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Gemma3TextDecoderLayer`]

    Args:
        config: Gemma3TextConfig
    """

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        last_cache_position: Optional[int] = None,
        **flash_attn_kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            # return_legacy_cache = True
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = _create_causal_mask(
            position_ids=position_ids, target_length=past_seen_tokens, sliding_window=self.config.sliding_window
        )

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    position_embeddings_global,
                    position_embeddings_local,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    last_cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings_global=position_embeddings_global,
                    position_embeddings_local=position_embeddings_local,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    batch_index=batch_index,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    last_cache_position=last_cache_position,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if use_cache:
            next_cache = past_key_values.to_legacy_cache()

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


class QEffGemma3ForCausalLMModel(Gemma3ForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Gemma3ForCausalLM

        >>> model = Gemma3ForCausalLM.from_pretrained("google/gemma-2-9b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""

        if self.training and self.config._attn_implementation != "eager":
            logger.warning_once(
                "It is strongly recommended to train Gemma3 models with the `eager` attention implementation "
                f"instead of `{self.config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **loss_kwargs,
        )

        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs[0][torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
