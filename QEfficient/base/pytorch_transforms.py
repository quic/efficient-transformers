# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from typing import Dict, Tuple, Type

from torch import nn


class PytorchTransform:
    """
    PytorchTransform is the base class that can do any transformation to a given PyTorch module by overriding apply method.
    """

    def __init__(self):
        raise TypeError("Transform classes are not to be instantiated. Directly use the `apply` method.")

    @classmethod
    def apply(cls, model: nn.Module) -> Tuple[nn.Module, bool]:
        """
        Override this class method to apply a transformation.
        :param model: The torch module to transform, this module may be transformed in-place

        :returns: Torch module after applying the transform
        :returns: Boolean indicating whether transform was applied
        """
        raise NotImplementedError("Use subclasses for Pytorch transform")


class ModuleMapping(PytorchTransform):
    """
    Replaces the PyTorch modules based on the _module_mapping class variable.
    """

    _module_mapping: Dict[Type[nn.Module], Type[nn.Module]]

    @classmethod
    def apply(cls, model: nn.Module) -> Tuple[nn.Module, bool]:
        transformed = False
        for module in model.modules():
            if repl_module := cls._module_mapping.get(type(module)):
                module.__class__ = repl_module
                transformed = True
        return model, transformed

    @classmethod
    def register(cls, from_module: Type[nn.Module], to_module: Type[nn.Module]):
        """
        Add a new module type in the module mapping for this transform. ::
            FlashAttention.register(LLamaAttention, LlamaFlashAttention)
        """
        cls._module_mapping[from_module] = to_module
