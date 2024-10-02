# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from typing import Dict, Tuple, Type, Optional


from QEfficient.base.pytorch_transforms import ModuleMappingTransform
from QEfficient.lora.layers import LinearMultiLoRA

import transformers
from torch import nn

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.lora.lora_model import QEffLoraModelMistralForCausalLM
from QEfficient.lora.lora_model import QEffLoraModelLlamaForCausalLM

from QEfficient.transformers.models.mistral.modeling_mistral import QEffMistralForCausalLM
from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaForCausalLM

class LoraModelInputsTransform(ModuleMappingTransform):
    _module_mapping = {QEffMistralForCausalLM: QEffLoraModelMistralForCausalLM, 
                       QEffLlamaForCausalLM: QEffLoraModelLlamaForCausalLM
    }

class TargetModulesTransform(ModuleMappingTransform):
    _module_mapping = {
        nn.Linear: LinearMultiLoRA
    }

    # a class method that deals with target module names
    @classmethod
    def apply(cls, model: nn.Module, target_modules: Optional[Dict], lora_rank: int, max_num_adapters: int) -> Tuple[nn.Module, bool]:
        transformed = False
        for name, module in model.named_modules():
            if repl_module := cls._module_mapping.get(type(module)):
                if name.split('.')[-1] in target_modules:
                    module.__class__ = repl_module

                    if hasattr(module, "multilora_init"):
                        module.multilora_init(lora_rank, max_num_adapters)

                    transformed = True
        return model, transformed