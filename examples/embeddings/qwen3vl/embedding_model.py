# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Example-facing wrapper for Qwen3-VL embedding runtime helpers."""

from QEfficient.transformers.models.qwen3_vl._embedding_utils import (
    QEffQwen3VLEmbedder,
    resolve_model_source,
)

__all__ = ["QEffQwen3VLEmbedder", "resolve_model_source"]
