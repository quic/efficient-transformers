# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import hashlib
import os
from typing import Any, List, Optional, Union

from peft import load_peft_weights, PeftConfig
import torch.nn as nn
import torch

import QEfficient
from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.lora.pytorch_transforms import TargetModulesTransform, LoraModelInputsTransform
from QEfficient.transformers.pytorch_transforms import CBTransform, CustomOpsTransform, KVCacheTransform
from QEfficient.utils import get_qpc_dir_path, load_hf_tokenizer
from QEfficient.utils.constants import QEFF_MODELS_DIR

from QEfficient.utils.logging_utils import logger


class QEffAutoLoraModelForCausalLM(QEFFAutoModelForCausalLM):
    """
    The QEFF class is designed for manipulating any causal language model from the HuggingFace hub.
    Although it is possible to initialize the class directly, we highly recommend using the ``from_pretrained`` method for initialization.
    Please note that the QEFF class is also a part of the ``QEfficient`` module.

    ``Mandatory`` Args:
        :model (nn.Module):  PyTorch model
        :pretrained_model_name_or_path (str): We recommend passing name of the model as input here, as you are not using `from_pretrained` method. This name will be used for deciding path of the ``ONNX/qpc`` files generated during ``export``, ``compilation`` stages.

    .. code-block:: python

        from QEfficient import QEffAutoLoraModelForCausalLM

    """
    # inherit __init__() from QEFFAutoModelForCausalLM
    def __init__(self, model: nn.Module, pretrained_model_name_or_path: str, **kwargs) -> None:
        super().__init__(model, pretrained_model_name_or_path)
        self.base_model_name = pretrained_model_name_or_path
        self.adapter_weights = {}
        self.adapter_configs = {}
        self.active_adapters = set()
        self.max_num_adapters = 0
        self.active_adapter_to_id = {}

    def load_adapter(self, adapter_model_id: str, adapter_name: str):
        """Loads a new adapter from huggingface hub or local path into CPU cache

        Args:
            :adapter_model_id (str): Adapter model ID from huggingface hub or local path
            :adapter_name (str): Adapter name to be used to set this adapter as current
        """
        if (adapter_name in self.adapter_weights.keys()) and (adapter_name in self.adapter_configs.keys()):
            logger.warning(f"Overwrite weights and configs for adapter name {adapter_name}")
        
        self.adapter_weights[adapter_name] = {k: v.numpy().astype("float16") for k, v in load_peft_weights(adapter_model_id).items()}
        self.adapter_configs[adapter_name] = PeftConfig.from_pretrained(adapter_model_id)

    def set_adapter(self, adapter_name: str):
        "Sets active adapter from one of the loaded adapters"

        assert (adapter_name in self.adapter_weights.keys()) and (adapter_name in self.adapter_configs.keys()), f"Adapter name {adapter_name} has not been loaded yet"

        assert list(self.adapter_configs.values())[0] and self.adapter_configs[adapter_name].target_modules == list(self.adapter_configs.values())[0].target_modules, "Not all adapters have the same target modules"
        
        assert list(self.adapter_configs.values())[0] and self.adapter_configs[adapter_name].r == list(self.adapter_configs.values())[0].r, "Not all adapters have the same ranks"
        
        # set active adapter id to current max
        self.active_adapter_to_id[adapter_name] = self.max_num_adapters

        # add active adapter to set
        self.active_adapters.add(adapter_name)
        self.max_num_adapters = len(self.active_adapters)

        return self.active_adapter_to_id[adapter_name]
        
    def get_adapter_id(self, adapter_name):
        "get the adapter_id that maps to the adapter_name"

        return self.active_adapter_to_id[adapter_name]

    def load_adapter_weights_to_model(self):
        "Loads adapter weights to the model's multilora layer in a stacked format"

        num_hidden_layers = len(self.model.model.layers)
        for i in range(num_hidden_layers):

            for target_module in self.target_modules_for_all_adapters: 

                # stack all adapters weights
                a_tensor_list = []
                b_tensor_list = []
                c_tensor_list = []

                # TODO: turn this to be in accordance with active_adapter_to_id's
                for lora_id, lora_name in enumerate(self.adapter_weights):

                    if target_module == "q_proj" or target_module == "k_proj" or target_module == "v_proj" or target_module == "up_proj":
                        a_tensor_list.append(torch.from_numpy(self.adapter_weights[lora_name][f'base_model.model.model.layers.{i}.self_attn.{target_module}.lora_A.weight']))
                        b_tensor_list.append(torch.from_numpy(self.adapter_weights[lora_name][f'base_model.model.model.layers.{i}.self_attn.{target_module}.lora_B.weight']))
                    elif target_module == "up_proj" or target_module == "gate_proj" or target_module == "down_proj":
                        a_tensor_list.append(torch.from_numpy(self.adapter_weights[lora_name][f'base_model.model.model.layers.{i}.mlp.{target_module}.lora_A.weight']))
                        b_tensor_list.append(torch.from_numpy(self.adapter_weights[lora_name][f'base_model.model.model.layers.{i}.mlp.{target_module}.lora_B.weight']))
                    else:
                        raise NotImplementedError("Target module not supported!!")
                    
                    c_tensor_list.append(torch.tensor(self.adapter_configs[lora_name].lora_alpha / self.adapter_configs[lora_name].r, dtype=torch.float16))
            
                stacked_lora_A = torch.stack(a_tensor_list, dim=0).unsqueeze(1).transpose(2,3) # <num_adapters, 1, in_feature, r>
                stacked_lora_B = torch.stack(b_tensor_list, dim=0).unsqueeze(1).transpose(2,3) # <num_adapters, 1, r, out_feature>
                stacked_lora_C = torch.stack(c_tensor_list, dim=0).unsqueeze(1).unsqueeze(2).unsqueeze(3) # <num_loras, 1, 1, 1>

                # stored weight to corresponding ops
                if target_module == "q_proj":
                    module = self.model.model.layers[i].self_attn.q_proj
                elif target_module == "k_proj":
                    module = self.model.model.layers[i].self_attn.k_proj
                elif target_module == "v_proj":
                    module = self.model.model.layers[i].self_attn.v_proj
                elif target_module == "o_proj":
                    module = self.model.model.layers[i].self_attn.o_proj
                elif target_module == "up_proj":
                    module = self.model.model.layers[i].mlp.up_proj
                elif target_module == "gate_proj":
                    module = self.model.model.layers[i].mlp.gate_proj
                elif target_module == "down_proj":
                    module = self.model.model.layers[i].mlp.down_proj
                else:
                    raise NotImplementedError("Target module not supported!!")
                
                module.lora_weight_A.copy_(stacked_lora_A)
                module.lora_weight_B.copy_(stacked_lora_B)
                module.lora_weight_C.copy_(stacked_lora_C)
                    
    def init_adapter_model(self):
        "Initialize the fixed lora model with multiple adapter weigths standby"

        # assume all adapters have same target_modules and ranks
        assert self.max_num_adapters == len(self.active_adapters), "Inconsistent max_num_adapters and active_adapters"
        
        assert list(self.adapter_configs.values())[0] and all(list(self.adapter_configs.values())[i].target_modules == list(self.adapter_configs.values())[0].target_modules for i in range(self.max_num_adapters)), "Not all adapters have the same target modules"
        
        assert list(self.adapter_configs.values())[0] and all(list(self.adapter_configs.values())[i].r == list(self.adapter_configs.values())[0].r for i in range(self.max_num_adapters)), "Not all adapters have the same ranks"
        self.lora_rank = list(self.adapter_configs.values())[0].r

        # do the module replacement
        _, transformed = LoraModelInputsTransform.apply(self.model)

        self.target_modules_for_all_adapters = list(self.adapter_configs.values())[0].target_modules
        _, transformed = TargetModulesTransform.apply(self.model, 
                                                      self.target_modules_for_all_adapters,
                                                      self.lora_rank,
                                                      self.max_num_adapters)

        # load_weight to model
        self.load_adapter_weights_to_model()

    def export(self, **kwargs) -> str:
        """
        Exports the model to ``ONNX`` format using ``torch.onnx.export``.
        The model should already be transformed i.e. ``self.is_transformed`` should be ``True``.
        Otherwise, this will raise an ``AssertionError``.
        We currently don't support exporting non-transformed models. Please refer to the ``convert_to_cloud_bertstyle`` function in the **Low-Level API** for a legacy function that supports this."

        ``Optional`` Args:
            does not any arguments.

        Raises:
            :AttributeError: If ``pretrained_model_name_or_path`` is a path, this function needs model card name of the model so that it can distinguish between directories while saving the ``ONNX`` files generated. So, user needs to pass ``model_card_name`` as a valid ``string`` in that case, Otherwise this will raise the error.

        Returns:
            :str: Path of the generated ``ONNX`` graph.
        """

        self.full_batch_size = kwargs.get("full_batch_size", self.full_batch_size)

        # obtain all necessary information to initialize the model 
        self.init_adapter_model()

        assert self.is_transformed, "Please first run transform on the QEFFAutoModelForCausalLM object"

        # Export
        _, onnx_model_path = QEfficient.export(
            model_name=self.model_card_name,
            model_kv=self,
            tokenizer=self.tokenizer,
            full_batch_size=self.full_batch_size,
            max_num_adapters=self.max_num_adapters,
        )
        self.onnx_path = onnx_model_path

        return self.onnx_path