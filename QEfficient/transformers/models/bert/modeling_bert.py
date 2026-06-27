# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
QEff wrappers for BERT-family encoder models — rebased for Transformers v5.5.

Fixes:

1. Attention mask (TF v5.5):
   BertModel / RobertaModel / XLMRobertaModel gained `_create_attention_masks`
   which calls `create_bidirectional_mask` → `sdpa_mask`/`eager_mask`.  That
   reads `inputs_embeds.shape[1]` as a 0-dim symbolic tensor during ONNX
   tracing and crashes with IndexError.  We replace it with
   `_prepare_4d_attention_mask` which uses only standard tensor ops.

2. Non-persistent buffers — works for ALL export modes:
   `BertEmbeddings` registers `position_ids` and `token_type_ids` as
   non-persistent buffers (persistent=False).  `accelerate.init_empty_weights`
   only moves parameters and persistent buffers to the meta device, so these
   two buffers remain on CPU.  During weight-free dynamo tracing the embedding
   weight is on meta while the index tensor is on CPU, raising
   FakeTensorDeviceMismatchError.

   Fix: `QEffBertEmbeddings` removes both buffers and overrides `forward` to
   create them from `input_ids.device`.  This resolves the issue regardless of
   export mode:
     - weight-free + dynamo : input_ids.device = meta  → tensors on meta  ✓
     - dynamo (no wf)       : input_ids.device = cpu   → tensors on cpu   ✓
     - classic torch.onnx   : input_ids.device = cpu   → tensors on cpu   ✓
     - runtime inference    : input_ids.device = cpu   → tensors on cpu   ✓

   The same issue exists for RoBERTa / XLM-RoBERTa which have their own
   non-persistent `position_ids` / `token_type_ids` buffers.  `QEffRobertaEmbeddings`
   and `QEffXLMRobertaEmbeddings` apply the same fix, but use RoBERTa's
   `create_position_ids_from_input_ids` which accounts for the padding_idx offset
   (positions start at padding_idx+1, not 0).

   Wiring: `_swap_bert_embeddings` / `_swap_roberta_embeddings` are called from
   BOTH `__qeff_init__` (the hook fired by ModuleMappingTransform after the
   class-swap, which does NOT call __init__) AND `__init__` (for direct-
   construction paths), with isinstance guards so the swaps are idempotent.
"""

import torch
import torch.nn as nn
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.models.bert.modeling_bert import BertEmbeddings, BertModel
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings, RobertaModel
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaEmbeddings, XLMRobertaModel

_BERT_EMBEDDING_PARAM_MODULES = (
    "word_embeddings",
    "position_embeddings",
    "token_type_embeddings",
    "LayerNorm",
    "dropout",
)


class QEffBertEmbeddings(BertEmbeddings):
    """
    BertEmbeddings without non-persistent CPU buffers.

    Removes `position_ids` and `token_type_ids` buffers and recreates them
    on `input_ids.device` in forward(), so the class works correctly for:
      - weight-free dynamo export  (device = meta)
      - regular dynamo / classic   (device = cpu)
      - runtime inference          (device = cpu)
    """

    def __init__(self, config):
        super().__init__(config)
        for name in ("position_ids", "token_type_ids"):
            self._buffers.pop(name, None)
            self._non_persistent_buffers_set.discard(name)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        return super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )


def _swap_bert_embeddings(model: nn.Module) -> None:
    """
    Replace model.embeddings with QEffBertEmbeddings, preserving all trained
    parameter submodules.  Idempotent — no-op if already swapped.

    Only applies to pure BertEmbeddings — NOT to RobertaEmbeddings /
    XLMRobertaEmbeddings, which use a different position-ID scheme
    (padding_idx+1 offset via create_position_ids_from_input_ids).
    Swapping those would break every attention layer's position encoding.
    """
    if not hasattr(model, "embeddings") or isinstance(model.embeddings, QEffBertEmbeddings):
        return
    from transformers.models.bert.modeling_bert import BertEmbeddings as _HFBertEmbeddings
    if not isinstance(model.embeddings, _HFBertEmbeddings):
        return
    old_emb: nn.Module = model.embeddings
    new_emb = QEffBertEmbeddings(model.config)
    for name in _BERT_EMBEDDING_PARAM_MODULES:
        if hasattr(old_emb, name):
            setattr(new_emb, name, getattr(old_emb, name))
    model.embeddings = new_emb


# ── RoBERTa-family embeddings fix ─────────────────────────────────────────────

class _QEffRobertaEmbeddingsMixin:
    """
    Mixin for RoBERTa / XLM-RoBERTa embeddings.

    Removes non-persistent `position_ids` and `token_type_ids` buffers and
    recomputes them in `forward()` using RoBERTa's padding-aware
    `create_position_ids_from_input_ids` (positions start at padding_idx+1).
    Works correctly for meta (weight-free), cpu (dynamo/classic), and runtime.
    """

    def __init__(self, config):
        super().__init__(config)
        for name in ("position_ids", "token_type_ids"):
            self._buffers.pop(name, None)
            self._non_persistent_buffers_set.discard(name)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        if position_ids is None:
            if input_ids is not None:
                position_ids = self.create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                )
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(
                    inputs_embeds, self.padding_idx
                )

        if token_type_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            input_shape = input_ids.size() if input_ids is not None else inputs_embeds.size()[:-1]
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        return super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )


class QEffRobertaEmbeddings(_QEffRobertaEmbeddingsMixin, RobertaEmbeddings):
    pass


class QEffXLMRobertaEmbeddings(_QEffRobertaEmbeddingsMixin, XLMRobertaEmbeddings):
    pass


def _swap_roberta_embeddings(model: nn.Module) -> None:
    """
    Replace model.embeddings with the appropriate QEff RoBERTa embeddings class.
    Idempotent — no-op if already swapped or not a RoBERTa-family model.
    """
    if not hasattr(model, "embeddings"):
        return
    old_emb = model.embeddings
    if isinstance(old_emb, (QEffRobertaEmbeddings, QEffXLMRobertaEmbeddings)):
        return

    if isinstance(old_emb, XLMRobertaEmbeddings):
        new_emb = QEffXLMRobertaEmbeddings(model.config)
    elif isinstance(old_emb, RobertaEmbeddings):
        new_emb = QEffRobertaEmbeddings(model.config)
    else:
        return

    for name in _BERT_EMBEDDING_PARAM_MODULES:
        if hasattr(old_emb, name):
            setattr(new_emb, name, getattr(old_emb, name))
    model.embeddings = new_emb

class _QEffBertFamilyMixin:
    """
    Mixin applied to BertModel / RobertaModel / XLMRobertaModel.

    Wires in both fixes (attention mask + embeddings) across all construction
    paths:
      - __init__       : direct construction (QEFFAutoModel(hf_model, ...))
      - __qeff_init__  : called by ModuleMappingTransform after class-swap
                         (no __init__ is called in that path)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _swap_bert_embeddings(self)
        _swap_roberta_embeddings(self)

    def __qeff_init__(self):
        _swap_bert_embeddings(self)
        _swap_roberta_embeddings(self)

    def _create_attention_masks(
        self,
        attention_mask,
        encoder_attention_mask,
        embedding_output,
        encoder_hidden_states,
        past_key_values,
    ):
        if self.config.is_decoder:
            return super()._create_attention_masks(
                attention_mask,
                encoder_attention_mask,
                embedding_output,
                encoder_hidden_states,
                past_key_values,
            )

        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, embedding_output.dtype)
        else:
            attention_mask = None

        if encoder_attention_mask is not None:
            encoder_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask, embedding_output.dtype, tgt_len=embedding_output.shape[1]
            )

        return attention_mask, encoder_attention_mask


class QEffBertModel(_QEffBertFamilyMixin, BertModel):
    pass


class QEffRobertaModel(_QEffBertFamilyMixin, RobertaModel):
    pass


class QEffXLMRobertaModel(_QEffBertFamilyMixin, XLMRobertaModel):
    pass
