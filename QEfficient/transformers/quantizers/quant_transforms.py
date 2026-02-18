# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
from torch import nn
from transformers import AutoConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts

from QEfficient.base.pytorch_transforms import ModuleMutatorTransform
from QEfficient.customop.matmulnbits import QuantLinearORT
from QEfficient.transformers.quantizers.awq import WQLinear_GEMM
from QEfficient.transformers.quantizers.gptq import QuantLinearGPTQ
from QEfficient.transformers.quantizers.quantizer_compressed_tensors import (
    FP8BlockWiseDequantLinear,
    FP8BlockWiseDequantQwen3VLMoeTextExperts,
    FP8DeQuantLinear,
)
from QEfficient.transformers.quantizers.quantizer_mxfp4 import QEffMxfp4GptOssExperts
from QEfficient.transformers.quantizers.quantizer_utils import (
    blockwise_dequantize,
    convert_moe_packed_tensors,
    dequantize_gptq,
    unpack_weights,
)


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


class Mxfp4GptOssExpertDequantizeTransform(ModuleMutatorTransform):
    """
    Used to dequantize the weights of an Mxfp4GptOssExpert module and replace with transformers GptOssExperts with dequantized weights
    """

    _match_class = QEffMxfp4GptOssExperts

    @classmethod
    def mutate(cls, original_module, parent_module):
        dequant_module = GptOssExperts(original_module.config)
        dequant_module.gate_up_proj = torch.nn.Parameter(
            convert_moe_packed_tensors(
                original_module.gate_up_proj_blocks, original_module.gate_up_proj_scales, dtype=torch.float32
            )
        )
        dequant_module.down_proj = torch.nn.Parameter(
            convert_moe_packed_tensors(
                original_module.down_proj_blocks, original_module.down_proj_scales, dtype=torch.float32
            )
        )
        dequant_module.gate_up_proj_bias = original_module.gate_up_proj_bias
        dequant_module.down_proj_bias = original_module.down_proj_bias
        return dequant_module


class FP8BlockWiseDequantLinearToLinearTransform(ModuleMutatorTransform):
    """
    Used to dequantize the weights of an FP8BlockWiseDequantLinear module and replace with a regular Linear layer
    """

    _match_class = FP8BlockWiseDequantLinear

    @classmethod
    def mutate(cls, original_module, parent_module):
        #  -- de-quantizing the weights --
        dequant_weights = blockwise_dequantize(
            original_module.weight, original_module.weight_scale_inv, original_module.weight_block_size
        )
        dequant_linear_layer = nn.Linear(
            original_module.in_features, original_module.out_features, bias=original_module.bias is not None
        )
        dequant_linear_layer.weight = torch.nn.Parameter(dequant_weights)
        if original_module.bias is not None:
            dequant_linear_layer.bias = torch.nn.Parameter(original_module.bias.float())
        return dequant_linear_layer


class FP8BlockWiseDequantQwen3VLMoeTextExpertsToQwen3VLMoeTextExpertsTransform(ModuleMutatorTransform):
    _match_class = FP8BlockWiseDequantQwen3VLMoeTextExperts
    _model_type = "qwen3_vl_moe"

    @classmethod
    def mutate(cls, original_module, parent_module):
        config = AutoConfig.for_model(cls._model_type).text_config
        config.num_experts = original_module.num_experts
        config.intermediate_size = original_module.intermediate_size
        config.hidden_size = original_module.hidden_size
        assert original_module.act_fn.__class__.__name__ == "SiLUActivation", (
            "Only SiLU activation is supported for now."
        )
        assert config.hidden_act == "silu", "expected silu act fn, something changed in transformers code"
        dequant_module = Qwen3VLMoeTextExperts(config)
        dequant_module.gate_up_proj = torch.nn.Parameter(
            blockwise_dequantize(
                original_module.gate_up_proj,
                original_module.gate_up_proj_scale_inv,
                original_module.weights_block_size,
            )
        )
        dequant_module.down_proj = torch.nn.Parameter(
            blockwise_dequantize(
                original_module.down_proj,
                original_module.down_proj_scale_inv,
                original_module.weights_block_size,
            )
        )
        return dequant_module
