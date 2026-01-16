# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from QEfficient.transformers.models.deberta_v2.modeling_deberta_v2 import (
    QEffDebertaV2ForSequenceClassification,
    QEffDisentangledSelfAttention,
)

__all__ = [
    "QEffDebertaV2ForSequenceClassification",
    "QEffDisentangledSelfAttention",
]
