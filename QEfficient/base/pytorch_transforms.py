# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
from types import MethodType
from typing import Callable, Dict, Tuple, Type

from torch import nn

from QEfficient.utils.logging_utils import logger


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


class ModuleMappingTransform(PytorchTransform):
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
                # Handling the __init__ calls in the models
                if hasattr(module, "__qeff_init__"):
                    module.__qeff_init__()
                transformed = True
        return model, transformed

    @classmethod
    def register(cls, from_module: Type[nn.Module], to_module: Type[nn.Module]):
        """
        Add a new module type in the module mapping for this transform. ::
            FlashAttention.register(LLamaAttention, LlamaFlashAttention)
        """
        cls._module_mapping[from_module] = to_module


class ModuleMutatorTransform(PytorchTransform):
    """Serves as base class for any transform that mutates pytorch module in any way.
    Mutate here mean, we initialize a new pytorch module object using info from original module and
    replace original module with new module.

    Raises:
        NotImplementedError: Not supposed to use directly, Create a subclass and implement mutate method and assign a valid nn.Module class to _match_class variable.
    """

    _match_class: nn.Module

    @classmethod
    def apply(cls, model: nn.Module) -> Tuple[nn.Module, bool]:
        transformed = False
        for name, module in model.named_children():
            if isinstance(module, cls._match_class):
                setattr(model, name, cls.mutate(module, model))
                transformed = True
            else:
                cls.apply(module)

        if isinstance(model, cls._match_class):
            model = cls.mutate(model, None)
            transformed = True

        return model, transformed

    @classmethod
    def mutate(cls, original_module: nn.Module, parent_module: nn.Module):
        raise NotImplementedError("Please implement your own method by inheriting this class")


class ExternalModuleMapperTransform(PytorchTransform):
    """
    Serves as base class for any transform that want to map a particular method of a class to a new method implementation.
    """

    _match_class_replace_method: Dict[nn.Module, Dict[str, Callable]]
    _match_string_replace_method: Dict[str, Dict[str, Callable]]

    @classmethod
    def apply(cls, model: nn.Module) -> Tuple[nn.Module, bool]:
        transformed = False
        for module in model.modules():
            if (repl_method_map := cls._match_class_replace_method.get(type(module))) or (
                repl_method_map := cls._match_string_replace_method.get(module.__class__.__name__)
            ):
                for orig_method_name, mapped_method in repl_method_map.items():
                    setattr(module, orig_method_name, MethodType(mapped_method, module))

                    if hasattr(module, "__qeff_init__"):
                        module.__qeff_init__()

                    transformed = True

        return model, transformed


class SplitGateUpWeightsTransform(PytorchTransform):
    """
    split fused Gate+Up weights and copy into the model

    For every transformer layer inside `model`:
      • expects   <PREFIX>.experts.gate_up_proj   in the *source* `sd`
      • copies halves into
            <PREFIX>.experts.gate_proj     <-- Gate   [E,H,I]
            <PREFIX>.experts.up_proj       <-- Up     [E,H,I]
    """

    @classmethod
    def apply(cls, model: nn.Module) -> Tuple[nn.Module, bool]:
        transformed = False
        model_class = model.__class__.__name__ if hasattr(model, "model") else model.__class__.__name__

        if model_class not in VLM_SPLIT_GATE_UP_WEIGHTS:
            return model, transformed

        model_tmp = model.language_model if hasattr(model, "language_model") else model

        num_layers = len(model_tmp.model.layers)
        delete_fused_key = True
        sd = model_tmp.state_dict()
        for layer_idx in range(num_layers):
            # ---- build the textual prefix once per layer ----------
            prefix = f"model.layers.{layer_idx}.feed_forward.experts."

            fused_key = prefix + "gate_up_proj"
            gate_key = prefix + "gate_proj"
            up_key = prefix + "up_proj"

            # ---- split  [E,H,2I] → two  [E,H,I]  tensors ----------------------
            fused = sd[fused_key]  # [E, H, 2I]  (no .weight here)
            E, H, two_I = fused.shape
            ffn_dim = two_I // 2
            gate, up = fused.split(ffn_dim, dim=-1)  # views – no copy

            experts = model_tmp.model.layers[layer_idx].feed_forward.experts
            experts.gate_proj.data.copy_(gate)
            experts.up_proj.data.copy_(up)

            # ---- update the state-dict so load_state_dict sees the right keys
            sd[gate_key] = gate
            sd[up_key] = up

            if delete_fused_key:
                del sd[fused_key]

            logger.info(f"[layer {layer_idx:02d}] loaded gate_proj & up_proj from fused tensor  (shape {fused.shape})")
            transformed = True

        if hasattr(model, "language_model"):
            model.language_model = model_tmp
        else:
            model = model_tmp
        return model, transformed


class SplitGateUpWeightsTransformGPTOSS(PytorchTransform):
    """
    split fused Gate+Up weights and copy into the model

    For every transformer layer inside `model`:
      • expects   <PREFIX>.experts.gate_up_proj   in the *source* `sd`
      • copies halves into
            <PREFIX>.experts.gate_proj     <-- Gate   [E,H,I]
            <PREFIX>.experts.up_proj       <-- Up     [E,H,I]
    """

    @classmethod
    def apply(cls, model: nn.Module) -> Tuple[nn.Module, bool]:
        transformed = False
        model_class = model.__class__.__name__ if hasattr(model, "model") else model.__class__.__name__

        if model_class not in VLM_SPLIT_GATE_UP_WEIGHTS:
            return model, transformed

        model_tmp = model.language_model if hasattr(model, "language_model") else model
        num_layers = len(model_tmp.model.layers)
        delete_fused_key = True
        sd = model_tmp.state_dict()

        for layer_idx in range(num_layers):
            # ---- build the textual prefix once per layer ----------
            prefix = f"model.layers.{layer_idx}.mlp.experts."
            fused_key = prefix + "gate_up_proj"
            fused_bias_key = prefix + "gate_up_proj_bias"
            gate_key = prefix + "gate_proj"
            up_key = prefix + "up_proj"
            gate_bias_key = prefix + "gate_proj_bias"
            up_bias_key = prefix + "up_proj_bias"

            # ---- split [E,H,2I] → two [E,H,I] tensors ----------------------
            fused = sd[fused_key]  # [E, H, 2I]
            fused_bias = sd[fused_bias_key]  # [E, 2I]
            E, H, two_I = fused.shape
            # ffn_dim = two_I // 2

            # For GptOss, gate/up are interleaved: [gate0, up0, gate1, up1, ...]
            gate = fused[..., ::2]  # [E, H, I] - even indices
            up = fused[..., 1::2]  # [E, H, I] - odd indices
            gate_bias = fused_bias[..., ::2]  # [E, I] - even indices
            up_bias = fused_bias[..., 1::2]  # [E, I] - odd indices

            experts = model_tmp.model.layers[layer_idx].mlp.experts
            experts.gate_proj.data.copy_(gate)
            experts.up_proj.data.copy_(up)
            experts.gate_proj_bias.data.copy_(gate_bias)
            experts.up_proj_bias.data.copy_(up_bias)

            # ---- update the state-dict so load_state_dict sees the right keys
            sd[gate_key] = gate
            sd[up_key] = up
            sd[gate_bias_key] = gate_bias
            sd[up_bias_key] = up_bias

            if delete_fused_key:
                del sd[fused_key]
                del sd[fused_bias_key]

            logger.info(f"[layer {layer_idx:02d}] loaded gate_proj & up_proj from fused tensor (shape {fused.shape})")
            transformed = True

        if hasattr(model, "language_model"):
            model.language_model = model_tmp
        else:
            model = model_tmp

        return model, transformed


VLM_SPLIT_GATE_UP_WEIGHTS = {"QEffLlama4ForConditionalGeneration", "QEffLlama4ForCausalLM", "QEffGptOssForCausalLM"}
