# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Optional

from QEfficient.transformers.attention_blocking import AttentionBlockingConfig


def derive_blocking_config(model_config, device_info: Optional[dict] = None, compile_params: Optional[dict] = None):
    """
    Dummy policy for auto blocking config selection.
    This can be replaced by a real model + device + compile param heuristic.
    """
    max_seq_len = getattr(model_config, "max_position_embeddings", None)
    num_kv_blocks = 2
    if isinstance(max_seq_len, int) and max_seq_len > 0:
        num_kv_blocks = max(1, min(8, max_seq_len // 128))
    return AttentionBlockingConfig(mode="kv", num_kv_blocks=num_kv_blocks)
