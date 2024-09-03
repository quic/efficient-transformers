# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn


class WQLinear_GEMM(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias):
        super().__init__()

        if w_bit != 4:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features

        # quick sanity check (make sure alignment)
        if self.in_features % self.group_size != 0:
            raise ValueError(
                f"in_features should be perfectly divisible by group_size, got in_features = {self.in_features}, group_size = {self.group_size} while initializing WQLinear_GEMM module"
            )
        if out_features % (32 // self.w_bit) != 0:
            raise ValueError(
                f"out_features must be perfectly divisible by number of weights packed into int32 value i.e. 8, got out_features={self.out_features}"
            )

        # For compatibility with QuantLinearORT
        self.g_idx = torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32)
        self.register_buffer(
            "qweight",
            torch.zeros(
                (in_features, out_features // (32 // self.w_bit)),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (in_features // self.group_size, out_features // (32 // self.w_bit)),
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

            out = dequantize_gemm(self.qweight, self.qzeros, self.scales, self.w_bit, self.group_size)
            out = torch.matmul(x.float(), out.float())

            out = out + self.bias if self.bias is not None else out
            out = out.reshape(out_shape)

        return out


def unpack_and_reverse_weights_and_zeros(qweight: torch.Tensor, qzeros: torch.Tensor, bits: int):
    shifts = torch.arange(0, 32, bits)

    # unpacking weights column-wise
    int_weights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(
        torch.int8  # smallest dtype available
    )
    int_weights = int_weights.view(int_weights.shape[0], -1)

    # unpacking zeros column-wise
    int_zeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(
        torch.int8  # smallest dtype available
    )
    int_zeros = int_zeros.view(int_zeros.shape[0], -1)

    reverse_order_tensor = torch.arange(
        int_weights.shape[-1],
        dtype=torch.int32,
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, [0, 4, 1, 5, 2, 6, 3, 7]]
    reverse_order_tensor = reverse_order_tensor.view(-1)

    int_zeros = int_zeros[:, reverse_order_tensor]
    int_weights = int_weights[:, reverse_order_tensor]

    return int_weights, int_zeros


def unpack_awq_weights(qweight, qzeros, scales, bits):
    int_weight, int_zeros = unpack_and_reverse_weights_and_zeros(qweight, qzeros, bits)

    # overflow checks
    int_weight = torch.bitwise_and(int_weight, (2**bits) - 1)
    int_zeros = torch.bitwise_and(int_zeros, (2**bits) - 1)

    return scales, int_weight, int_zeros


def dequantize_gemm(qweight, qzeros, scales, bits, group_size):
    # Unpack the qweight and qzeros tensors
    scales, int_weight, int_zeros = unpack_awq_weights(qweight, qzeros, scales, bits)

    # fp16 weights
    scales = scales.repeat_interleave(group_size, dim=0)
    int_zeros = int_zeros.repeat_interleave(group_size, dim=0)

    int_weight = (int_weight - int_zeros) * scales

    return int_weight
