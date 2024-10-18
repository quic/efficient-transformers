# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from typing import Dict, Optional, Tuple

from torch import nn

from QEfficient.base.pytorch_transforms import ModuleMappingTransform
from QEfficient.lora.layers import LinearBase, LinearMultiLoRA
from QEfficient.lora.lora_model import QEffLoraModelLlamaForCausalLM, QEffLoraModelMistralForCausalLM
from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaForCausalLM
from QEfficient.transformers.models.mistral.modeling_mistral import QEffMistralForCausalLM


class LoraModelInputsTransform(ModuleMappingTransform):
    _module_mapping = {
        QEffMistralForCausalLM: QEffLoraModelMistralForCausalLM,
        QEffLlamaForCausalLM: QEffLoraModelLlamaForCausalLM,
    }


class TargetModulesTransform(ModuleMappingTransform):
    _module_mapping = {nn.Linear: LinearMultiLoRA}

    _module_mapping_nontarget = {nn.Linear: LinearBase}

    # whole set of supported target modules for now (make sure **kwargs are passed in on modeling file)
    all_modules = {"q_proj", "k_proj", "v_proj", "o_proj"}

    # a class method that deals with target module names
    @classmethod
    def apply(
        cls, model: nn.Module, target_modules: Optional[Dict], lora_rank: int, max_num_adapters: int
    ) -> Tuple[nn.Module, bool]:
        transformed = False
        nontarget_modules = {key for key in cls.all_modules if key not in target_modules}

        for name, module in model.named_modules():
            if repl_module := cls._module_mapping.get(type(module)):
                if name.split(".")[-1] in target_modules:
                    module.__class__ = repl_module
                    if hasattr(module, "multilora_init"):
                        module.multilora_init(lora_rank, max_num_adapters)
                    transformed = True
                elif name.split(".")[-1] in nontarget_modules:
                    module.__class__ = cls._module_mapping_nontarget.get(type(module))
                    transformed = True

        return model, transformed
