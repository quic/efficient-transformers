# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn

from QEfficient.transformers.quantizers.quantizer_utils import dequantize_gemm


class WQLinear_GEMM(nn.Module):
    def __init__(self, bits, group_size, in_features, out_features, bias):
        super().__init__()

        if bits != 4:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size if group_size != -1 else in_features

        # quick sanity check (make sure alignment)
        if self.in_features % self.group_size != 0:
            raise ValueError(f"in_features should be perfectly divisible by group_size, got in_features = {self.in_features}, group_size = {self.group_size} while initializing WQLinear_GEMM module")
        if out_features % (32 // self.bits) != 0:
            raise ValueError(f"out_features must be perfectly divisible by number of weights packed into int32 value i.e. 8, got out_features={self.out_features}")

        # For compatibility with QuantLinearORT
        self.g_idx = torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32)
        self.register_buffer(
            "qweight",
            torch.zeros(
                (in_features, out_features // (32 // self.bits)),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (in_features // self.group_size, out_features // (32 // self.bits)),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (in_features // self.group_size, out_features),
                dtype=torch.float16,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (out_features),
                    dtype=torch.float16,
                ),
            )
        else:
            self.bias = None

    def forward(self, x):
        # Only Inference supported
        with torch.no_grad():
            out_shape = x.shape[:-1] + (self.out_features,)

            out = dequantize_gemm(self.qweight, self.qzeros, self.scales, self.bits, self.group_size)
            out = torch.matmul(x.float(), out.float())

            out = out + self.bias if self.bias is not None else out
            out = out.reshape(out_shape)

        return out
