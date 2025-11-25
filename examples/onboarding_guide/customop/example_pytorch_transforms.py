# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Example pytorch_transforms.py showing how to register custom operations.

This file demonstrates how to add custom operation mappings to the transform
system so they are automatically applied when loading models.

For the actual production transforms, see:
- QEfficient/transformers/models/pytorch_transforms.py
"""

from transformers.activations import (
    NewGELUActivation,
)

from QEfficient.base.pytorch_transforms import ModuleMappingTransform
from QEfficient.customop import CustomGELUAIC


class CustomOpsTransform(ModuleMappingTransform):
    """
    Maps standard PyTorch operations to custom Cloud AI 100 implementations.

    How it works:
    1. When a model is loaded, this transform scans all modules
    2. For each module type in _module_mapping, it replaces the module
       with the corresponding custom implementation
    3. The replacement happens automatically before ONNX export
    """

    _module_mapping = {
        # ACTIVATION FUNCTIONS
        # GELU
        NewGELUActivation: CustomGELUAIC,
        # TODO: Add other activation functions
        # nn.SiLU: CustomSiLUAIC,
        # nn.Mish: CustomMishAIC,
        # NORMALIZATION LAYERS
        # RMSNorm - Used by Llama, Mistral, Mixtral, etc.
        # from transformers.models.llama.modeling_llama import LlamaRMSNorm
        # LlamaRMSNorm: CustomRMSNormAIC,
        # TODO: Add your model's normalization layers
        # YourModelRMSNorm: CustomRMSNormAIC,
        # OTHER OPERATIONS
        # TODO: Add other custom operations
        # nn.Linear: CustomLinearAIC,  # If you have a custom linear layer
        # nn.Embedding: CustomEmbeddingAIC,  # If you have custom embeddings
    }
