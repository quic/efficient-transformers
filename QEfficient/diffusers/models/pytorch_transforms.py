# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from diffusers.models.attention_processor import Attention
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    WanDecoder3d,
    WanEncoder3d,
    WanResample,
    WanResidualBlock,
)
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle, RMSNorm
from diffusers.models.transformers.transformer_flux import (
    FluxAttention,
    FluxAttnProcessor,
    FluxSingleTransformerBlock,
    FluxTransformer2DModel,
    FluxTransformerBlock,
)
from diffusers.models.transformers.transformer_qwenimage import (
    QwenDoubleStreamAttnProcessor2_0,
    QwenImageTransformer2DModel,
)
from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor, WanTransformer3DModel
from torch import nn

from QEfficient.base.pytorch_transforms import ModuleMappingTransform
from QEfficient.customop.rms_norm import CustomRMSNormAIC
from QEfficient.diffusers.models.autoencoders.autoencoder_kl_wan import (
    QEffWanDecoder3d,
    QEffWanEncoder3d,
    QEffWanResample,
    QEffWanResidualBlock,
)
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
from QEfficient.diffusers.models.transformers.transformer_qwenimage import (
    QEffQwenDoubleStreamAttnProcessor2_0,
    QEffQwenImageAttention,
    QEffQwenImageTransformer2DModel,
)
from QEfficient.diffusers.models.transformers.transformer_wan import (
    QEffWanAttention,
    QEffWanAttnProcessor,
    QEffWanTransformer3DModel,
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
        WanAttnProcessor: QEffWanAttnProcessor,
        WanAttention: QEffWanAttention,
        WanTransformer3DModel: QEffWanTransformer3DModel,
        QwenImageTransformer2DModel: QEffQwenImageTransformer2DModel,
        QwenDoubleStreamAttnProcessor2_0: QEffQwenDoubleStreamAttnProcessor2_0,
        Attention: QEffQwenImageAttention,
        WanDecoder3d: QEffWanDecoder3d,
        WanEncoder3d: QEffWanEncoder3d,
        WanResidualBlock: QEffWanResidualBlock,
        WanResample: QEffWanResample,
    }


class NormalizationTransform(ModuleMappingTransform):
    _module_mapping = {
        AdaLayerNormZero: QEffAdaLayerNormZero,
        AdaLayerNormZeroSingle: QEffAdaLayerNormZeroSingle,
        AdaLayerNormContinuous: QEffAdaLayerNormContinuous,
    }
