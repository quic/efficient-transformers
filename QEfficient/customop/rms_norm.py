# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
RMS Norm CustomOp Node in QAic Domain for Cloud AI 100
This is to handle the FP16 Overflow seen in RMS Norm for LLMs
"""

import torch
from torch.onnx.symbolic_helper import parse_args

op_source = """
#include <torch/script.h>

torch::Tensor custom_rms_norm(torch::Tensor hidden_states, torch::Tensor weight, double eps) {
  torch::Tensor output;
  torch::Tensor variance;
  bool keepdim;
  // double eps = 1e-5;
  variance = hidden_states.pow(2).mean(-1, keepdim=true);
  output = hidden_states * torch::rsqrt(variance + eps);
  output = output * weight;
  return output;
}

TORCH_LIBRARY(QAic, m) {
  m.def("QEffCustomRMSNorm", &custom_rms_norm);
}
"""

# Compile and load the custom op
torch.utils.cpp_extension.load_inline(
    name="custom_rms_norm",
    cpp_sources=op_source,
    is_python_module=False,
    verbose=True,
)


# Wrapper module for custom relu C++ op
class QEffCustomRMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states):
        return torch.ops.QAic.QEffCustomRMSNorm(hidden_states, self.weight, self.eps)


# ONNX export symbolic helper
@parse_args("v", "v", "f")
def custom_rms_norm(g, hidden_states, weight, eps):
    return g.op("QAic::QEffCustomRMSNorm", hidden_states, weight, eps_f=eps).setTypeAs(hidden_states)


torch.onnx.register_custom_op_symbolic("QAic::QEffCustomRMSNorm", custom_rms_norm, 1)
