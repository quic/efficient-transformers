# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""Shared QAIC SDK imports and dtype mapping utilities."""

from __future__ import annotations

__all__ = ["AIC_TO_NP", "aicapi", "is_qaicrt_imported", "qaicrt"]

import platform
import sys

import numpy as np

try:
    import qaicrt

    is_qaicrt_imported = True
except ImportError:
    try:
        sys.path.append(f"/opt/qti-aic/dev/lib/{platform.machine()}")
        import qaicrt

        is_qaicrt_imported = True
    except ImportError:
        is_qaicrt_imported = False

try:
    import QAicApi_pb2 as aicapi
except ImportError:
    sys.path.append("/opt/qti-aic/dev/python")
    import QAicApi_pb2 as aicapi

# ── dtype mapping ─────────────────────────────────────────────────────────────
AIC_TO_NP: dict[int, np.dtype] = {
    getattr(aicapi, "BFLOAT16_TYPE", 11): np.dtype(np.float16),
    aicapi.FLOAT_TYPE: np.dtype(np.float32),
    aicapi.FLOAT_16_TYPE: np.dtype(np.float16),
    aicapi.INT8_Q_TYPE: np.dtype(np.int8),
    aicapi.UINT8_Q_TYPE: np.dtype(np.uint8),
    aicapi.INT16_Q_TYPE: np.dtype(np.int16),
    aicapi.INT32_Q_TYPE: np.dtype(np.int32),
    aicapi.INT32_I_TYPE: np.dtype(np.int32),
    aicapi.INT64_I_TYPE: np.dtype(np.int64),
    aicapi.INT8_TYPE: np.dtype(np.int8),
}
