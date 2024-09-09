# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn

from QEfficient.transformers.quantizers.qunatizer_utils import dequantize_gemm


class WQLinear_GEMM(nn.Module):
    def __init__(self, bits, groupsize, infeatures, outfeatures, bias):
        super().__init__()

        if bits != 4:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.groupsize = groupsize if groupsize != -1 else infeatures

        # quick sanity check (make sure alignment)
        if self.infeatures % self.groupsize != 0:
            raise ValueError(
                f"infeatures should be perfectly divisible by groupsize, got infeatures = {self.infeatures}, groupsize = {self.groupsize} while initializing WQLinear_GEMM module"
            )
        if outfeatures % (32 // self.bits) != 0:
            raise ValueError(
                f"outfeatures must be perfectly divisible by number of weights packed into int32 value i.e. 8, got outfeatures={self.outfeatures}"
            )

        # For compatibility with QuantLinearORT
        self.g_idx = torch.tensor([i // groupsize for i in range(infeatures)], dtype=torch.int32)
        self.register_buffer(
            "qweight",
            torch.zeros(
                (infeatures, outfeatures // (32 // self.bits)),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (infeatures // self.groupsize, outfeatures // (32 // self.bits)),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (infeatures // self.groupsize, outfeatures),
                dtype=torch.float16,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (outfeatures),
                    dtype=torch.float16,
                ),
            )
        else:
            self.bias = None

    def forward(self, x):
        # Only Inference supported
        with torch.no_grad():
            out_shape = x.shape[:-1] + (self.outfeatures,)

            out = dequantize_gemm(self.qweight, self.qzeros, self.scales, self.bits, self.groupsize)
            out = torch.matmul(x.float(), out.float())

            out = out + self.bias if self.bias is not None else out
            out = out.reshape(out_shape)

        return out
