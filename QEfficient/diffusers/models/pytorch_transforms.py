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
from typing import Tuple
import torch
from torch import nn
from diffusers.models.attention import JointTransformerBlock
from diffusers.models.attention_processor import Attention, JointAttnProcessor2_0
from diffusers.models.normalization import RMSNorm, AdaLayerNormZero, AdaLayerNormZeroSingle, AdaLayerNormContinuous
from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock, FluxTransformerBlock, FluxTransformer2DModel 

from QEfficient.base.pytorch_transforms import ModuleMappingTransform
from QEfficient.customop.rms_norm import CustomRMSNormAIC
from QEfficient.diffusers.models.attention import QEffJointTransformerBlock
from QEfficient.diffusers.models.attention_processor import (
    QEffAttention,
    QEffJointAttnProcessor2_0,
)
from QEfficient.diffusers.models.transformers.transformer_flux import  QEffFluxSingleTransformerBlock, QEffFluxTransformerBlock, QEffFluxTransformer2DModel
from QEfficient.diffusers.models.normalization import QEffAdaLayerNormZero, QEffAdaLayerNormZeroSingle, QEffAdaLayerNormContinuous

class CustomOpsTransform(ModuleMappingTransform):
    _module_mapping = {
        RMSNorm: CustomRMSNormAIC,
        nn.RMSNorm: CustomRMSNormAIC #  for torch.nn.RMSNorm
      }

    @classmethod
    def apply(cls, model: nn.Module) -> Tuple[nn.Module, bool]:
        model, transformed = super().apply(model)
        return model, transformed


class AttentionTransform(ModuleMappingTransform):
    _module_mapping = {
        Attention: QEffAttention,
        JointAttnProcessor2_0: QEffJointAttnProcessor2_0,
        JointTransformerBlock: QEffJointTransformerBlock,
        FluxSingleTransformerBlock: QEffFluxSingleTransformerBlock,
        FluxTransformerBlock: QEffFluxTransformerBlock,
        FluxTransformer2DModel: QEffFluxTransformer2DModel,
    }

    @classmethod
    def apply(cls, model: nn.Module) -> Tuple[nn.Module, bool]:
        model, transformed = super().apply(model)
        return model, transformed

class NormalizationTransform(ModuleMappingTransform):
    _module_mapping = {
        AdaLayerNormZero: QEffAdaLayerNormZero,
        AdaLayerNormZeroSingle: QEffAdaLayerNormZeroSingle,
        AdaLayerNormContinuous: QEffAdaLayerNormContinuous,
    }
    @classmethod
    def apply(cls, model: nn.Module) -> Tuple[nn.Module, bool]:
        model, transformed = super().apply(model)
        return model, transformed
