# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
from typing import Tuple

from diffusers.models.attention import JointTransformerBlock
from diffusers.models.attention_processor import Attention, JointAttnProcessor2_0
from torch import nn

from QEfficient.base.pytorch_transforms import ModuleMappingTransform
from QEfficient.diffusers.models.attention import QEffJointTransformerBlock
from QEfficient.diffusers.models.attention_processor import (
    QEffAttention,
    QEffJointAttnProcessor2_0,
)


class CustomOpsTransform(ModuleMappingTransform):
    _module_mapping = {}


class AttentionTransform(ModuleMappingTransform):
    _module_mapping = {
        Attention: QEffAttention,
        JointAttnProcessor2_0: QEffJointAttnProcessor2_0,
        JointTransformerBlock: QEffJointTransformerBlock,
    }

    @classmethod
    def apply(cls, model: nn.Module) -> Tuple[nn.Module, bool]:
        model, transformed = super().apply(model)
        return model, transformed
