# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
TorchScript symbolic wrappers and onnxscript translation functions for the
three FP8 DequantizeLinear granularities.

* ``FP8DequantizePerTensor`` / ``FP8DequantizePerTensorFunc``
      Scalar scale, DequantizeLinear (no axis).
      Compiled for both opset-17 (legacy) and opset-21 (dynamo) via qeff_custom_op.

* ``FP8DequantizePerAxis`` / ``FP8DequantizePerAxisFunc``
      (out_features,) scale, DequantizeLinear(axis=0).
      Compiled for both opset-17 and opset-21 via qeff_custom_op.

* ``FP8DequantizeBlocked`` / ``FP8DequantizeBlockedFunc``
      Compact (out//R, in//C) scale, DequantizeLinear(axis=-1, block_size=C).
      block_size was introduced in opset 21 — compiled for opset-21 only.
      TorchScript symbolic emits Unsqueeze+Tile+Flatten+DequantizeLinear.
"""

import onnxscript
import torch

from QEfficient.customop.onnxscript_utils import _DYNAMO_FUNC_ATTR, qeff_custom_op
from QEfficient.utils import constants

legacy_ops = getattr(onnxscript, "opset" + str(constants.ONNX_LEGACY_EXPORT_OPSET))
dynamo_ops = getattr(onnxscript, "opset" + str(constants.ONNX_DYNAMO_EXPORT_OPSET))
ops21 = onnxscript.opset21


# ── onnxscript translation functions ─────────────────────────────────────────
# Per-tensor and per-axis: valid from opset 13 — compiled for both opsets.
# Blocked: block_size requires opset 21 — compiled for opset-21 only.


# FP8 DequantizeLinear with FP8 input requires opset 21 for ORT to accept it.
# The legacy variant is compiled at ONNX_LEGACY_EXPORT_OPSET (21) via qeff_custom_op.
# The dynamo variant must also be at opset 21 — we compile it explicitly and
# attach it with _DYNAMO_FUNC_ATTR so get_dynamo_onnxscript_func() finds it.


# Legacy (TorchScript path) variants compiled at ONNX_LEGACY_EXPORT_OPSET.
# Stored under _legacy names so the clean public names can be used for the
# dynamo variants, giving consistent ONNX function names across all three
# FP8 granularities: FP8DequantizePerTensor, FP8DequantizePerAxis, FP8DequantizeBlocked.
@qeff_custom_op("", constants.ONNX_LEGACY_EXPORT_OPSET)
def _FP8DequantizePerTensor_legacy(
    weight: onnxscript.FLOAT8E4M3FN,
    scale: onnxscript.FLOAT,
) -> onnxscript.FLOAT:
    return legacy_ops.DequantizeLinear(weight, scale)


@qeff_custom_op("", constants.ONNX_LEGACY_EXPORT_OPSET)
def _FP8DequantizePerAxis_legacy(
    weight: onnxscript.FLOAT8E4M3FN,
    scale: onnxscript.FLOAT,
) -> onnxscript.FLOAT:
    return legacy_ops.DequantizeLinear(weight, scale, axis=0)


# Dynamo variants compiled at ONNX_DYNAMO_EXPORT_OPSET (18).
# Named FP8DequantizePerTensor / FP8DequantizePerAxis so the ONNX function
# name matches the clean convention used by FP8DequantizeBlocked.
_fp8_custom_opset = onnxscript.values.Opset(domain="com.qualcomm.cloud", version=1)


@onnxscript.script(_fp8_custom_opset)
def FP8DequantizePerTensor(
    weight: onnxscript.FLOAT8E4M3FN,
    scale: onnxscript.FLOAT,
) -> onnxscript.FLOAT:
    """Per-tensor: DequantizeLinear(weight, scale), scalar scale, no axis."""
    return dynamo_ops.DequantizeLinear(weight, scale)


@onnxscript.script(_fp8_custom_opset)
def FP8DequantizePerAxis(
    weight: onnxscript.FLOAT8E4M3FN,
    scale: onnxscript.FLOAT,
) -> onnxscript.FLOAT:
    """Per-axis: DequantizeLinear(weight, scale, axis=0), scale: (out_features,)."""
    return dynamo_ops.DequantizeLinear(weight, scale, axis=0)


setattr(_FP8DequantizePerTensor_legacy, _DYNAMO_FUNC_ATTR, FP8DequantizePerTensor)
setattr(_FP8DequantizePerAxis_legacy, _DYNAMO_FUNC_ATTR, FP8DequantizePerAxis)

# Public aliases: the rest of the codebase imports these names.
# CustomOpTransform uses the legacy variants (TorchScript path).
# DYNAMO_CUSTOM_OP_TABLE uses get_dynamo_onnxscript_func() on the legacy variants
# which returns the cleanly-named dynamo variants above.
FP8DequantizePerTensorLegacy = _FP8DequantizePerTensor_legacy
FP8DequantizePerAxisLegacy = _FP8DequantizePerAxis_legacy


@onnxscript.script(_fp8_custom_opset)
def FP8DequantizeBlocked(
    weight: onnxscript.FLOAT8E4M3FN,
    scale: onnxscript.FLOAT,
) -> onnxscript.FLOAT:
    """Blocked: Unsqueeze+Tile+Flatten(scale) + DequantizeLinear(axis=-1, block_size=col_bs).

    NOTE: This is a placeholder used only for the dynamo custom_translation_table.
    The actual block sizes are model-specific; the TorchScript symbolic path
    (FP8DequantizeBlockedFunc) handles the concrete Tile+Flatten+DequantizeLinear
    emission for the legacy export path.
    """
    axes = ops21.Constant(value_ints=[1])
    scale_unsq = ops21.Unsqueeze(scale, axes)
    repeats = ops21.Constant(value_ints=[1, 128, 1])
    tiled = ops21.Tile(scale_unsq, repeats)
    scale_row = ops21.Flatten(tiled, axis=2)
    return ops21.DequantizeLinear(weight, scale_row, axis=-1, block_size=128)


# ── TorchScript-path symbolic wrappers ───────────────────────────────────────


class FP8DequantizePerTensorFunc(torch.autograd.Function):
    """TorchScript symbolic for qefficient::fp8_dequantize_per_tensor."""

    @staticmethod
    def forward(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return weight.to(scale.dtype) * scale

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, weight: torch.Value, scale: torch.Value) -> torch.Value:
        return g.op("DequantizeLinear", weight, scale)


class FP8DequantizePerAxisFunc(torch.autograd.Function):
    """TorchScript symbolic for qefficient::fp8_dequantize_per_axis."""

    @staticmethod
    def forward(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if scale.ndim == 1:
            scale = scale.unsqueeze(-1)
        return weight.to(scale.dtype) * scale

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, weight: torch.Value, scale: torch.Value) -> torch.Value:
        return g.op("DequantizeLinear", weight, scale, axis_i=0)


class FP8DequantizeBlockedFunc(torch.autograd.Function):
    """TorchScript symbolic for qefficient::fp8_dequantize_blocked.

    Emits Unsqueeze(scale) + Tile([1, row_bs, 1]) + Flatten(axis=2) +
    DequantizeLinear(axis=-1, block_size=col_bs).
    """

    @staticmethod
    def forward(weight: torch.Tensor, scale: torch.Tensor, row_block_size: int, col_block_size: int) -> torch.Tensor:
        scale_row = scale.repeat_interleave(row_block_size, dim=0)
        scale_full = scale_row.repeat_interleave(col_block_size, dim=-1)
        return weight.to(scale_full.dtype) * scale_full

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(
        g: torch.Graph,
        weight: torch.Value,
        scale: torch.Value,
        row_block_size,
        col_block_size,
    ) -> torch.Value:
        import torch.onnx.symbolic_helper as sym_help

        row_bs = sym_help._maybe_get_const(row_block_size, "i")
        col_bs = sym_help._maybe_get_const(col_block_size, "i")
        axes = g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64))
        scale_unsq = g.op("Unsqueeze", scale, axes)
        repeats = g.op("Constant", value_t=torch.tensor([1, row_bs, 1], dtype=torch.int64))
        tiled = g.op("Tile", scale_unsq, repeats)
        scale_row = g.op("Flatten", tiled, axis_i=2)
        return g.op("DequantizeLinear", weight, scale_row, axis_i=-1, block_size_i=col_bs)


# ── Register symbolics for the legacy TorchScript export path ────────────────
torch.onnx.register_custom_op_symbolic(
    "qefficient::fp8_dequantize_per_tensor",
    FP8DequantizePerTensorFunc.symbolic,
    constants.ONNX_LEGACY_EXPORT_OPSET,
)
torch.onnx.register_custom_op_symbolic(
    "qefficient::fp8_dequantize_per_axis",
    FP8DequantizePerAxisFunc.symbolic,
    constants.ONNX_LEGACY_EXPORT_OPSET,
)
torch.onnx.register_custom_op_symbolic(
    "qefficient::fp8_dequantize_blocked",
    FP8DequantizeBlockedFunc.symbolic,
    constants.ONNX_LEGACY_EXPORT_OPSET,
)
