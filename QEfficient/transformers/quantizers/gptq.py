# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import math

import torch
from torch import nn

from QEfficient.transformers.quantizers.quantizer_utils import dequantize_gptq


class QuantLinearGPTQ(nn.Module):
    """
    A quantized linear layer using GPTQ (Generalized Post-Training Quantization).
    This class supports only 4-bit quantization and is compatible with QuantLinearORT.

    Research paper link- GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (https://arxiv.org/abs/2210.17323)

    Attributes:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        bits (int): The number of bits used for quantization (must be 4).
        act_order (None or bool): The activation order.
        orig_fp_weight (None or torch.Tensor): The original floating-point weights.
        maxq (int): The maximum quantization value.
        group_size (int): The group size for quantization.
        pack_mode (str): The packing mode, set to "GPTQ".
        qweight (torch.Tensor): The quantized weight tensor.
        qzeros (torch.Tensor): The quantized zeros tensor.
        scales (torch.Tensor): The scales tensor.
        g_idx (torch.Tensor): The group index tensor.
        bias (torch.Tensor or None): The bias tensor, if applicable.
    """

    def __init__(self, bits, group_size, in_features, out_features, bias):
        super().__init__()
        if bits != 4:
            raise NotImplementedError("Only 4 bits are supported.")
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.act_order = None
        self.orig_fp_weight = None
        self.maxq = 2**self.bits - 1
        self.group_size = group_size if group_size != -1 else in_features
        self.pack_mode = "GPTQ"

        # For compatibility with QuantLinearORT
        self.register_buffer(
            "qweight",
            torch.zeros((in_features // 32 * self.bits, out_features), dtype=torch.int32),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros((math.ceil(in_features / self.group_size), out_features // 32 * self.bits), dtype=torch.int32),
        )
        self.register_buffer(
            "scales",
            torch.zeros((math.ceil(in_features / self.group_size), out_features), dtype=torch.float16),
        )
        self.g_idx = torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32)
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros((out_features), dtype=torch.float16),
            )
        else:
            self.bias = None

    def forward(self, x):
        # Only Inference supported
        out, _, _ = dequantize_gptq(self.qweight.T, self.qzeros, self.scales, self.bits, self.g_idx)
        out = torch.matmul(x.float(), out.float())
        out = out + self.bias if self.bias is not None else out

        return out
