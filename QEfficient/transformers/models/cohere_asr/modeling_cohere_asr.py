# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
QEff CohereAsr wrapper.

CohereAsr is a Whisper-style encoder-decoder ASR model: a `ParakeetEncoder`
(fast-conformer, bidirectional, no KV cache) feeding a causal, KV-cached text
decoder that cross-attends to the fixed encoder output.

The only differences from the upstream `CohereAsrDecoder`/`CohereAsrCrossAttention`
are:
- `create_causal_mask`/`create_bidirectional_mask` (from `transformers.masking_utils`)
  crash during ONNX export tracing (`IndexError: tuple index out of range` in
  `sdpa_mask`, since both dispatch through it). Replaced with QEfficient's
  ONNX-safe `_create_causal_mask` and `_prepare_4d_attention_mask`.
- self-attention cache update passes `position_ids` via `cache_kwargs`, as
  required by `QEffDynamicLayer.update()`.
- cross-attention KV reuse is rewritten branch-free with `torch.where`/
  `torch.index_put`, gated on `input_features.shape[2] == 1` (the same
  dummy-shape trick Whisper uses to select the "Decode" specialization),
  instead of the Python-bool `past_key_values.is_updated` dict lookup that
  cannot be traced into a single ONNX graph shared by both specializations.
- masking uses QEfficient's `torch.where`-based fp16-safe convention instead
  of HF's additive `attn_weights + attention_mask`.
"""

from typing import Optional, Tuple, Type, Union

import torch
from torch import nn
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput
from transformers.models.cohere_asr.modeling_cohere_asr import (
    CohereAsrCrossAttention,
    CohereAsrDecoder,
    CohereAsrDecoderLayer,
    CohereAsrForConditionalGeneration,
    CohereAsrSelfAttention,
    repeat_kv,
)

from QEfficient.transformers.cache_utils import QEffEncoderDecoderCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils._utils import IOInfo
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE, ONNX_EXPORT_EXAMPLE_SEQ_LEN


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

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class QEffCohereAsrSelfAttention(CohereAsrSelfAttention):
    """
    Copied from CohereAsrSelfAttention: https://github.com/huggingface/transformers/blob/main/src/transformers/models/cohere_asr/modeling_cohere_asr.py
    The only differences are:
    - pass `position_ids` via `cache_kwargs` to `past_key_values.update`, as required by `QEffDynamicLayer`
    - use QEfficient's `eager_attention_forward` (torch.where-based fp16-safe masking)
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        if past_key_values is not None:
            self_attention_cache = past_key_values.self_attention_cache
            cache_kwargs = {"position_ids": position_ids}
            key_states, value_states = self_attention_cache.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QEffCohereAsrCrossAttention(CohereAsrCrossAttention):
    """
    Copied from CohereAsrCrossAttention: https://github.com/huggingface/transformers/blob/main/src/transformers/models/cohere_asr/modeling_cohere_asr.py
    The only differences are:
    - the encoder K/V cache-or-recompute decision is rewritten branch-free with
      torch.where/torch.index_put (gated on `input_features.shape[2] == 1`) instead
      of the Python-bool `past_key_values.is_updated` dict lookup, so both the
      "Encoder" and "Decode" compile specializations trace to the same ONNX graph
    - use QEfficient's `eager_attention_forward` (torch.where-based fp16-safe masking)
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        input_features: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        bsz, tgt_len = hidden_states.shape[:-1]
        src_len = encoder_hidden_states.shape[1]

        q_input_shape = (bsz, tgt_len, -1, self.head_dim)
        kv_input_shape = (bsz, src_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(*q_input_shape).transpose(1, 2)

        if past_key_values is not None:
            cross_attention_cache = past_key_values.cross_attention_cache
            key_states_old = cross_attention_cache.layers[self.layer_idx].keys
            value_states_old = cross_attention_cache.layers[self.layer_idx].values

            key_states_new = self.k_proj(encoder_hidden_states).view(*kv_input_shape).transpose(1, 2)
            value_states_new = self.v_proj(encoder_hidden_states).view(*kv_input_shape).transpose(1, 2)

            # Select freshly computed or cached cross-attention K/V based on the
            # dummy input_features shape used by the "Decode" specialization.
            key_states = torch.where(input_features.shape[2] == torch.tensor(1), key_states_old, key_states_new)
            value_states = torch.where(input_features.shape[2] == torch.tensor(1), value_states_old, value_states_new)

            cross_attention_cache.layers[self.layer_idx].keys = key_states
            cross_attention_cache.layers[self.layer_idx].values = value_states
        else:
            key_states = self.k_proj(encoder_hidden_states).view(*kv_input_shape).transpose(1, 2)
            value_states = self.v_proj(encoder_hidden_states).view(*kv_input_shape).transpose(1, 2)

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QEffCohereAsrDecoderLayer(CohereAsrDecoderLayer):
    """
    Copied from CohereAsrDecoderLayer: https://github.com/huggingface/transformers/blob/main/src/transformers/models/cohere_asr/modeling_cohere_asr.py
    The only difference is threading `input_features` through to the cross-attention block.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        input_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states, _ = self.encoder_attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                input_features=input_features,
            )
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class QEffCohereAsrDecoder(CohereAsrDecoder):
    """
    Copied from CohereAsrDecoder: https://github.com/huggingface/transformers/blob/main/src/transformers/models/cohere_asr/modeling_cohere_asr.py
    The only differences are:
    - `create_causal_mask` -> QEfficient's ONNX-traceable `_create_causal_mask`
    - `create_bidirectional_mask` -> `_prepare_4d_attention_mask`, both of which use
      only standard tensor ops, unlike `create_causal_mask`/`create_bidirectional_mask`
      which dispatch through `sdpa_mask` and crash with
      `IndexError: tuple index out of range` while tracing to ONNX
    - threads `input_features` through to the decoder layers for cross-attention KV reuse
    """

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        encoder_hidden_states = self.proj(encoder_hidden_states)
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if (use_cache or past_key_values is not None) and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = QEffEncoderDecoderCache.from_legacy_cache(past_key_values)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        pos_emb = self.pos_emb(position_ids.squeeze(0))
        inputs_embeds = self.embedding_layernorm(inputs_embeds + pos_emb)

        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=past_seen_tokens)
        encoder_attention_mask = (
            _prepare_4d_attention_mask(encoder_attention_mask, inputs_embeds.dtype)
            if encoder_attention_mask is not None
            else None
        )

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                input_features=input_features,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        next_cache = past_key_values if (use_cache or past_key_values is not None) else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
        )


class QEffCohereAsrForConditionalGeneration(CohereAsrForConditionalGeneration):
    """
    Copied from CohereAsrForConditionalGeneration: https://github.com/huggingface/transformers/blob/main/src/transformers/models/cohere_asr/modeling_cohere_asr.py
    The only differences are:
    - added get_dummy_inputs, get_onnx_dynamic_axes, get_output_names, get_specializations,
      get_inputs_info, get_submodules_for_export for AutoModel export
    - changed forward inputs decoder_input_ids/decoder_position_ids to input_ids/position_ids
    - transposes `input_features` from Parakeet's native `(batch, seq_len, num_mel_bins)`
      to the mel-bins-major `(batch, num_mel_bins, feature_len)` contract that
      `QEFFAutoModelForSpeechSeq2Seq` (Whisper-derived) hard-codes at the runtime boundary
    - sets `config.num_mel_bins` (absent on `CohereAsrConfig`) from `config.encoder_config`,
      since `QEFFAutoModelForSpeechSeq2Seq.generate()` reads `self.model.config.num_mel_bins`
    """

    def __qeff_init__(self):
        self.config.num_mel_bins = self.config.encoder_config.num_mel_bins

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {self.model.encoder.layers[0].__class__, QEffCohereAsrDecoderLayer}

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Union[EncoderDecoderCache, Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        # Parakeet's native input_features is (batch, seq_len, num_mel_bins); the runtime
        # boundary (QEFFAutoModelForSpeechSeq2Seq) presents mel-bins-major
        # (batch, num_mel_bins, feature_len), matching Whisper's Conv1d convention.
        encoder_input_features = input_features.transpose(1, 2) if input_features is not None else None

        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(encoder_input_features, attention_mask=attention_mask)

        decoder_outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=getattr(encoder_outputs, "attention_mask", None),
            past_key_values=past_key_values,
            use_cache=use_cache,
            input_features=input_features,
        )
        logits = self.proj_out(decoder_outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
        )

    def get_dummy_inputs(self, **kwargs):
        bs = 1
        seq_len = int(kwargs.get("prefill_seq_len", ONNX_EXPORT_EXAMPLE_SEQ_LEN))
        encoder_seq_len = self.config.max_position_embeddings
        encoder_feature_count = self.config.num_mel_bins
        num_key_value_heads = self.config.num_key_value_heads
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        num_layers = self.config.num_hidden_layers

        inputs = {
            "input_features": torch.zeros((bs, encoder_feature_count, 1), dtype=torch.float32),
            "input_ids": torch.zeros((bs, seq_len), dtype=torch.int64),
            "position_ids": torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(bs, 1),
            "past_key_values": [[] for _ in range(num_layers)],
        }

        kv_cache_shape = (bs, num_key_value_heads, seq_len, head_dim)
        kv_cross_cache_shape = (bs, num_key_value_heads, encoder_seq_len, head_dim)

        for i in range(num_layers):
            for self_cross in ["self", "cross"]:
                for kv in ["key", "value"]:
                    inputs["past_key_values"][i].append(
                        torch.zeros(
                            kv_cache_shape if self_cross == "self" else kv_cross_cache_shape, dtype=torch.float32
                        )
                    )

        return inputs

    def get_specializations(self, batch_size: int, encoder_ctx_len, ctx_len, **compiler_options):
        if encoder_ctx_len is None:
            encoder_ctx_len = self.config.max_position_embeddings
        feature_len = encoder_ctx_len * 2

        encoder_specializations = {
            "_graph_name": "Encoder",
            "batch_size": batch_size,
            "seq_len": 1,
            "encoder_ctx_len": encoder_ctx_len,
            "decoder_ctx_len": ctx_len,
            "feature_len": feature_len,
        }

        decoder_specializations = {
            "_graph_name": "Decode",
            "batch_size": batch_size,
            "seq_len": 1,
            "encoder_ctx_len": encoder_ctx_len,
            "decoder_ctx_len": ctx_len,
            "feature_len": 1,  # dummy feature so torch.where knows whether to run cross attention or not
        }

        specializations = [encoder_specializations, decoder_specializations]

        return specializations, compiler_options

    def get_onnx_dynamic_axes(self):
        num_layers = self.config.num_hidden_layers

        dynamic_axes = {
            "input_features": {0: "batch_size", 2: "feature_len"},
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
        }
        pkv_self_dynamic_axes = {
            0: "batch_size",
            2: "decoder_ctx_len",
        }
        pkv_cross_dynamic_axes = {
            0: "batch_size",
            2: "encoder_ctx_len",
        }
        for i in range(num_layers):
            for self_cross in ["self", "cross"]:
                for kv in ["key", "value"]:
                    dynamic_axes[f"past_{kv}_{self_cross}.{i}"] = (
                        pkv_self_dynamic_axes if self_cross == "self" else pkv_cross_dynamic_axes
                    )

        return dynamic_axes

    def get_output_names(self):
        output_names = ["logits"]
        for i in range(self.config.num_hidden_layers):
            for self_cross in ["self", "cross"]:
                for kv in ["key", "value"]:
                    output_names.append(f"past_{kv}_{self_cross}.{i}_RetainedState")
        return output_names

    def get_inputs_info(self):
        return [
            IOInfo(name="input_features", datatype=torch.float32, shape=("batch_size", "num_mel_bins", "feature_len")),
        ]
