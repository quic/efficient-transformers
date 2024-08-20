# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
from torch import nn

from QEfficient.base.pytorch_transforms import ModuleMutatorTransform
from QEfficient.customop.matmulnbits import QuantLinearORT
from QEfficient.transformers.quantizers.awq import WQLinear_GEMM, unpack_awq_weights


class AwqToOnnxTransform(ModuleMutatorTransform):
    _match_class = WQLinear_GEMM

    @staticmethod
    def unpack_and_dequantize_awq(qweight, qzeros, scales, bits, group_size):
        # Unpack the qweight and qzeros tensors
        scales, iweight, izeros = unpack_awq_weights(qweight, qzeros, scales, bits, group_size)
        # fp16 weights
        scales_expand = scales.repeat_interleave(group_size, dim=0)
        izeros_expand = izeros.repeat_interleave(group_size, dim=0)

        iweight = (iweight - izeros_expand) * scales_expand

        return iweight.T, scales, izeros.to(torch.int32)

    @classmethod
    def mutate(cls, original_module: nn.Module, parent_module: nn.Module):
        fp16_weight, scales, zeros = cls.unpack_and_dequantize_awq(
            original_module.qweight,
            original_module.qzeros,
            original_module.scales,
            original_module.w_bit,
            original_module.group_size,
        )

        original_module.weight = fp16_weight
        new_module = QuantLinearORT(
            original_module.w_bit,
            original_module.group_size,
            original_module.in_features,
            original_module.out_features,
            original_module.bias is not None,
        )
        new_module.bias = original_module.bias if original_module.bias is not None else None
        new_module.pack(original_module, scales.T, zeros.T, original_module.g_idx)
        return new_module
