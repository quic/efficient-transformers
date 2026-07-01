# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""Stateless buffer utility functions for QAIC inference sessions."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def build_completion_error(bindings, allowed_shapes: list, buf_dims_for_idx) -> str:
    """Build a descriptive error message when waitForCompletion fails."""
    msg = "Failed to run"
    if not allowed_shapes:
        return msg
    msg += '\n\n(Only if "No matching dimension found" error is present above)'
    msg += "\nAllowed shapes:"
    for i, allowed_shape in enumerate(allowed_shapes):
        msg += f"\n{i}\n"
        for binding, (elemsize, shape), (_, passed_shape) in zip(bindings, allowed_shape, buf_dims_for_idx):
            if passed_shape[0] == 0:
                if not binding.is_partial_buf_allowed:
                    logger.warning("Partial buffer not allowed for: %s", binding.name)
                continue
            msg += f"{binding.name}:\t{elemsize}\t{shape}\n"
    msg += "\n\nPassed shapes:\n"
    for binding, (elemsize, shape) in zip(bindings, buf_dims_for_idx):
        if shape[0] == 0:
            continue
        msg += f"{binding.name}:\t{elemsize}\t{shape}\n"
    return msg


def make_inputs_contiguous(inputs: dict[str, Any]) -> None:
    """Ensure all input arrays are C-contiguous in-place."""
    for k, v in inputs.items():
        inputs[k] = np.ascontiguousarray(v)


def inputs_to_tuple_list(
    inputs: dict[str, Any],
    binding_index_map: dict[str, int],
) -> list[tuple[int, np.ndarray]]:
    """
    Convert {name: array} -> [(binding_index, array)] tuple list.

    This is the format expected by execObj.setData() and
    execObj.setDataWithSlices() -- avoids a full qbuffer/buf_dims update
    and is the zero-copy path for inputs that are already contiguous.
    """
    result: list[tuple[int, np.ndarray]] = []
    for name, buf in inputs.items():
        if name not in binding_index_map:
            logger.warning("Buffer: %s not found", name)
            continue
        if buf is None:
            continue
        result.append((binding_index_map[name], buf))
    return result
