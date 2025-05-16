# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
from torch import nn

from QEfficient.base.pytorch_transforms import ModuleMutatorTransform
from QEfficient.customop.matmulnbits import QuantLinearORT
from QEfficient.transformers.quantizers.awq import WQLinear_GEMM
from QEfficient.transformers.quantizers.gptq import QuantLinearGPTQ
from QEfficient.transformers.quantizers.quantizer_compressed_tensors import FP8DeQuantLinear
from QEfficient.transformers.quantizers.quantizer_utils import dequantize_gptq, unpack_weights


class AwqToMatmulNbitsTransform(ModuleMutatorTransform):
    _match_class = WQLinear_GEMM

    @staticmethod
    def unpack_and_dequantize_awq(qweight, qzeros, scales, bits, group_size):
        # Unpack the qweight and qzeros tensors
        scales, int_weight, int_zeros = unpack_weights(qweight, qzeros, scales, bits, "awq")

        # fp16 weights
        scales_expand = scales.repeat_interleave(group_size, dim=0)
        int_zeros_expand = int_zeros.repeat_interleave(group_size, dim=0)
        int_weight = (int_weight - int_zeros_expand) * scales_expand

        return int_weight.T, scales, int_zeros.to(torch.int32)

    @classmethod
    def mutate(cls, original_module: nn.Module, parent_module: nn.Module):
        fp16_weight, scales, zeros = cls.unpack_and_dequantize_awq(
            original_module.qweight,
            original_module.qzeros,
            original_module.scales,
            original_module.bits,
            original_module.group_size,
        )

        original_module.weight = fp16_weight
        new_module = QuantLinearORT(
            original_module.bits,
            original_module.group_size,
            original_module.in_features,
            original_module.out_features,
            original_module.bias is not None,
        )
        new_module.bias = original_module.bias if original_module.bias is not None else None
        new_module.pack(original_module, scales.T, zeros.T, original_module.g_idx)
        return new_module


class GPTQToMatmulNbitsTransform(ModuleMutatorTransform):
    """
    A transformation class that mutates a ``QuantLinearGPTQ`` module to a ``QuantLinearORT``
    module by unpacking and dequantizing the quantized weights.
    """

    _match_class = QuantLinearGPTQ

    @staticmethod
    def unpack_and_dequantize_gptq(qweight, qzeros, scales, bits, g_idx):
        # Unpack the qweight and qzeros tensors
        int_weight, scales, int_zeros = dequantize_gptq(qweight.T, qzeros, scales, bits, g_idx)
        return int_weight, scales, int_zeros.to(torch.int32)

    @classmethod
    def mutate(cls, original_module: nn.Module, parent_module: nn.Module):
        """
        ``Mutates`` the original ``QuantLinearGPTQ`` module to a ``QuantLinearORT`` module.

        Args:
            original_module (nn.Module): The original ``QuantLinearGPTQ`` module.
            parent_module (nn.Module): The parent module containing the original module.

        Returns:
            :nn.Module: The new ``QuantLinearORT`` module with unpacked and de-quantized weights.
        """

        fp16_weight, scales, zeros = cls.unpack_and_dequantize_gptq(
            original_module.qweight,
            original_module.qzeros,
            original_module.scales,
            original_module.bits,
            original_module.g_idx,
        )
        original_module.weight = fp16_weight.T
        new_module = QuantLinearORT(
            original_module.bits,
            original_module.group_size,
            original_module.in_features,
            original_module.out_features,
            original_module.bias is not None,
        )
        new_module.bias = original_module.bias if original_module.bias is not None else None
        new_module.pack(original_module, scales.T, zeros.T, original_module.g_idx)
        return new_module


class FP8DeQuantLinearToLinearTransform(ModuleMutatorTransform):
    _match_class = FP8DeQuantLinear

    @classmethod
    def mutate(cls, original_module, parent_module):
        #  -- de-quantizing the weights --
        dequant_weights = original_module.weight.to(torch.float32) * original_module.weight_scale
        dequant_linear_layer = nn.Linear(
            original_module.in_features, original_module.out_features, bias=original_module.bias is not None
        )
        dequant_linear_layer.weight = torch.nn.Parameter(dequant_weights)
        if original_module.bias is not None:
            dequant_linear_layer.bias = torch.nn.Parameter(original_module.bias.float())
        return dequant_linear_layer
