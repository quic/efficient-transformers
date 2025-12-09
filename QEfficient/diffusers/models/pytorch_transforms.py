# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle, RMSNorm
from diffusers.models.transformers.transformer_flux import (
    FluxAttention,
    FluxAttnProcessor,
    FluxSingleTransformerBlock,
    FluxTransformer2DModel,
    FluxTransformerBlock,
)
from torch import nn

from QEfficient.base.pytorch_transforms import ModuleMappingTransform
from QEfficient.customop.rms_norm import CustomRMSNormAIC
from QEfficient.diffusers.models.normalization import (
    QEffAdaLayerNormContinuous,
    QEffAdaLayerNormZero,
    QEffAdaLayerNormZeroSingle,
)
from QEfficient.diffusers.models.transformers.transformer_flux import (
    QEffFluxAttention,
    QEffFluxAttnProcessor,
    QEffFluxSingleTransformerBlock,
    QEffFluxTransformer2DModel,
    QEffFluxTransformerBlock,
)


class CustomOpsTransform(ModuleMappingTransform):
    _module_mapping = {
        RMSNorm: CustomRMSNormAIC,
        nn.RMSNorm: CustomRMSNormAIC,  #  for torch.nn.RMSNorm
    }


class AttentionTransform(ModuleMappingTransform):
    _module_mapping = {
        FluxSingleTransformerBlock: QEffFluxSingleTransformerBlock,
        FluxTransformerBlock: QEffFluxTransformerBlock,
        FluxTransformer2DModel: QEffFluxTransformer2DModel,
        FluxAttention: QEffFluxAttention,
        FluxAttnProcessor: QEffFluxAttnProcessor,
    }


class NormalizationTransform(ModuleMappingTransform):
    _module_mapping = {
        AdaLayerNormZero: QEffAdaLayerNormZero,
        AdaLayerNormZeroSingle: QEffAdaLayerNormZeroSingle,
        AdaLayerNormContinuous: QEffAdaLayerNormContinuous,
    }
