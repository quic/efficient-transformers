# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
from typing import Tuple
from diffusers import AutoencoderKL
from QEfficient.diffusers.models.autoencoders.autoencoder_kl import QEffAutoencoderKL 
from QEfficient.base.pytorch_transforms import ModuleMappingTransform
from torch import nn


class AutoencoderKLTransform(ModuleMappingTransform):
    """Transforms a Diffusers AutoencoderKL model to a QEfficientAutoencoderKL model."""

    _module_mapping = {
            AutoencoderKL: QEffAutoencoderKL,
        }
    @classmethod
    def apply(cls, model: nn.Module) -> Tuple[nn.Module, bool]:
        model, transformed = super().apply(model)
        return model, transformed    