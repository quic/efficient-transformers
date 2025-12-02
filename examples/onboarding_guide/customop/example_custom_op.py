# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Custom op template - shows the 3-layer pattern we use for all custom ops.
"""

import onnxscript
import torch
from torch import nn

from QEfficient.utils import constants

ops = getattr(onnxscript, "opset" + str(constants.ONNX_EXPORT_OPSET))


# Layer 1: ONNX Script
# This is what the compiler sees when it compiles your model


@onnxscript.script(onnxscript.values.Opset(domain="com.qti.aisw.onnx", version=1))
def CustomOpBluePrint(input: onnxscript.FLOAT):
    """
    ONNX implementation of your operation.
    Important: Only use tensor inputs - no strings or other types!
    """
    sqrt_2 = ops.Constant(value_floats=[1.4142135623730951])
    half = ops.Constant(value_floats=[0.5])
    one = ops.Constant(value_floats=[1.0])

    x_scaled = ops.Div(input, sqrt_2)
    erf_x = ops.Erf(x_scaled)
    result = ops.Mul(input, ops.Mul(half, ops.Add(one, erf_x)))

    return result


# Layer 2: PyTorch Autograd Function
# Connects PyTorch execution to ONNX export
# Pytorch forward function is called during PyTorch execution (CPU/GPU).
# When running on ONNX runtime, the CustomOpBluePrint function (Layer 1) is called instead.


class CustomOpBluePrintFunc(torch.autograd.Function):
    @staticmethod
    def forward(input: torch.Tensor, mode: str = "default"):
        """PyTorch implementation - can use any PyTorch ops"""
        if mode == "approximate":
            return 0.5 * input * (1.0 + torch.tanh(0.7978845608028654 * (input + 0.044715 * input**3)))
        else:
            return input * 0.5 * (1.0 + torch.erf(input / 1.4142135623730951))

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Value, mode: str = "default") -> torch.Value:
        """Called during ONNX export - don't pass string params!"""
        return g.onnxscript_op(CustomOpBluePrint, input).setTypeAs(input)


# Layer 3: Module Wrapper
# What users actually interact with


class CustomOpBluePrintAIC(nn.Module):
    def __init__(self, mode: str = "default"):
        super().__init__()
        if mode not in ["default", "approximate"]:
            raise ValueError(f"mode must be 'default' or 'approximate', got {mode}")
        self._mode_str = mode

    @property
    def mode(self) -> str:
        return self._mode_str if hasattr(self, "_mode_str") else "default"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return CustomOpBluePrintFunc.apply(input, self.mode)

    def extra_repr(self) -> str:
        return f"mode={self.mode}"
