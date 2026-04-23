# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import onnxscript
import torch
from onnx import TensorProto

from QEfficient.utils import constants

ops = getattr(onnxscript, "opset" + str(constants.ONNX_EXPORT_OPSET))


@onnxscript.script(onnxscript.values.Opset("com.qti.aisw.onnx", 1))
def CastToUInt4(weight_packed: onnxscript.UINT8) -> onnxscript.UINT8:
    """
    Unpack packed uint8 weights into uint4 values and cast output to UINT4.
    Supports N-D input: all leading dimensions are preserved; only the last
    dimension (in_features // 2) is doubled to (in_features).

    Input:  (..., in_features // 2) UINT8
            Each byte holds two nibbles: byte = (w_y << 4) | (w_x & 0x0F)
    Output: (..., in_features) UINT4, values in [0, 15]

    Operations:
      w_x          = weight_packed % 16          (lower nibble)
      w_y          = (weight_packed >> 4) % 16   (upper nibble)
      stacked      = concat([w_x, w_y], axis=-1) after unsqueeze
                     → (..., in//2, 2)
      leading_dims = shape[:-1]
      new_shape    = [...leading_dims, last_dim * 2]
      reshaped     = reshape(stacked, new_shape)
      output       = Cast(reshaped, to=UINT4)
    """
    sixteen = ops.CastLike(ops.Constant(value_ints=[16]), weight_packed)

    # Lower nibble: weight_packed & 0x0F  =  weight_packed % 16
    w_x = ops.Mod(weight_packed, sixteen)

    # Upper nibble: (weight_packed >> 4) & 0x0F
    shift = ops.CastLike(ops.Constant(value_ints=[4]), weight_packed)
    w_shifted = ops.BitShift(weight_packed, shift, direction="RIGHT")
    w_y = ops.Mod(w_shifted, sixteen)

    # Stack along a new last dim → (..., in_features//2, 2)
    w_x_unsq = ops.Unsqueeze(w_x, [-1])
    w_y_unsq = ops.Unsqueeze(w_y, [-1])
    stacked = ops.Concat(w_x_unsq, w_y_unsq, axis=-1)

    # N-D aware reshape: preserve all leading dims, double the last dim.
    # packed_shape = [d0, d1, ..., last_dim]
    packed_shape = ops.Shape(weight_packed)
    # All dims except the last: [d0, d1, ...]
    leading_dims = ops.Slice(packed_shape, starts=[0], ends=[-1], axes=[0])
    # Last dim only: [last_dim]
    last_dim = ops.Slice(packed_shape, starts=[-1], ends=[2147483647], axes=[0])
    # Double the last dim: [last_dim * 2]
    last_dim_doubled = ops.Mul(last_dim, ops.Constant(value_ints=[2]))
    # New shape: [d0, d1, ..., last_dim * 2]
    new_shape = ops.Concat(leading_dims, last_dim_doubled, axis=0)
    reshaped = ops.Reshape(stacked, new_shape)

    # Cast to UINT4 — data_type value is version-dependent (21 in ONNX 1.18, 23 in newer)
    return ops.Cast(reshaped, to=int(TensorProto.UINT4))


class CastToUInt4Func(torch.autograd.Function):
    """
    Custom op: unpacks packed uint8 → uint8 (values 0-15) in PyTorch.
    In ONNX the custom op subgraph includes a Cast → UINT4 as its last step.
    Supports N-D input: all leading dimensions are preserved.

    PyTorch forward  : packed uint8 (..., in//2) → uint8 (..., in), values [0, 15]
    ONNX symbolic    : emits CastToUInt4 node (com.qti.aisw.onnx)
                       The subgraph ends with Cast → UINT4.
    """

    @staticmethod
    def forward(weight_packed: torch.Tensor) -> torch.Tensor:
        w_x = weight_packed & 0x0F  # lower nibble, (..., in//2), range [0, 15]
        w_y = (weight_packed >> 4) & 0x0F  # upper nibble, (..., in//2), range [0, 15]
        # New shape: all leading dims unchanged, last dim doubled
        new_shape = list(weight_packed.shape[:-1]) + [weight_packed.shape[-1] * 2]
        return torch.stack(
            [w_x, w_y], dim=-1
        ).reshape(
            new_shape
        )  # Can't add a cast operation to uint4 here, as its not supported in pytorch; The ONNX export will handle the cast to IINT4 in the symbolic method.

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, weight_packed: torch.Value) -> torch.Value:
        output = g.onnxscript_op(CastToUInt4, weight_packed)
        return output


class DequantizeLinearFunc(torch.autograd.Function):
    """
    Emits a standard ONNX DequantizeLinear node (ai.onnx domain, not custom).

    Symmetric blockwise quantization — no zero_point:
      output = x * scale   (per block along the last axis)

    Supports N-D input:
      weight_unpacked : (..., in_features)   — quantized values
      scale           : (..., num_blocks)    — per-block scales
      block_size      : int                  — elements per block

    PyTorch forward  : expand blockwise scale along last dim, multiply
    ONNX symbolic    : DequantizeLinear(weight_unpacked, scale,
                                        axis=2, block_size=block_size)
                       axis=2 for 3D input (2, out_features, in_features).
                       No zero_point input (symmetric).
    """

    @staticmethod
    def forward(
        weight_unpacked: torch.Tensor, scale: torch.Tensor, zeros: torch.Tensor, block_size: int
    ) -> torch.Tensor:
        # Expand per-block scale → per-element scale along last dim
        scale_expanded = scale.repeat_interleave(block_size, dim=-1)
        zeros_expanded = zeros.repeat_interleave(block_size, dim=-1)
        return (weight_unpacked.to(torch.int8) - zeros_expanded.to(torch.int8)) * scale_expanded

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(
        g: torch.Graph, weight_unpacked: torch.Value, scale: torch.Value, zeros: torch.Value, block_size: int
    ) -> torch.Value:
        # Standard DequantizeLinear: symmetric (no zero_point), blockwise.
        # Input is 3D: (2, out_features, in_features) → axis=2 (last dim).
        # DequantizeLinear natively supports batch dimensions.
        return g.op(
            "DequantizeLinear",
            weight_unpacked,
            scale,
            zeros,
            axis_i=2,
            block_size_i=block_size,
        )
