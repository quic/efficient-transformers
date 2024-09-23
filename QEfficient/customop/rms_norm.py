# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import onnxscript
import torch
from torch import nn

ops = onnxscript.opset14


@onnxscript.script(onnxscript.values.Opset(domain="com.qti.aisw.onnx", version=1))
def CustomRMSNorm(hidden_states: onnxscript.FLOAT, weight: onnxscript.FLOAT, epsilon: float):
    weight = ops.Cast(weight, to=1)
    variance = ops.ReduceMean(ops.Pow(hidden_states, 2), axes=[-1], keepdims=1)
    epsilon = ops.Expand(epsilon, ops.Shape(variance))
    hidden_states = hidden_states * ops.Reciprocal(ops.Sqrt(variance + epsilon))
    return weight * hidden_states


class CustomRMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + epsilon)
        return weight * hidden_states

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, hidden_states: torch.Value, weight: torch.Value, epsilon: torch.Value) -> torch.Value:
        return g.onnxscript_op(CustomRMSNorm, hidden_states, weight, epsilon_f=epsilon).setTypeAs(hidden_states)


class CustomRMSNormAIC(nn.Module):
    """
    RMSNorm module that works by replacing the current module with compiler known custom-op.
    """

    def __init__(self, hidden_size, eps=1e-05):
        super(CustomRMSNormAIC, self).__init__()
        self.variance_epsilon = eps
        self.eps = eps  # Added to support GemmaRMSNorm
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states):
        return CustomRMSNormFunc.apply(
            hidden_states, self.weight, self.variance_epsilon if hasattr(self, "variance_epsilon") else self.eps
        )


class GemmaCustomRMSNormAIC(CustomRMSNormAIC):
    """
    Modify the init function to add +1 to the weights
    """

    def __init__(self, hidden_size, eps=1e-05):
        super().__init__(hidden_size, eps)
        # QEff Init
        self.__qeff_init__()

    def __qeff_init__(self):
        with torch.no_grad():
            self.weight.copy_(self.weight + 1.0)
