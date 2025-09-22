# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import hashlib
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn
from peft import PeftConfig, load_peft_weights
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

import QEfficient
from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.peft.lora.pytorch_transforms import LoraModelInputsTransform, TargetModulesTransform
from QEfficient.utils import constants, get_padding_shape_from_config
from QEfficient.utils.hash_utils import to_hashable
from QEfficient.utils.logging_utils import logger


class QEffAutoLoraModelForCausalLM(QEFFAutoModelForCausalLM):
    """
    QEff class for loading models with multiple LoRA adapters. Currently only Mistral and Llama model are supported.
    Once exported and compiled, the qpc can perform mixed batch inference with provided `prompt_to_adapter_mapping`.

    Args:
        :model (nn.Module): PyTorch model
        :continuous_batching (bool): Weather this model will be used for continuous batching in future. If this is not set True here, the model can not be exported/compiled for continuous batching later.

    .. code-block:: python

        from QEfficient.peft.lora import QEffAutoLoraModelForCausalLM

        m = QEffAutoPeftModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        m.load_adapter("predibase/gsm8k", "gsm8k")
        m.load_adapter("predibase/magicoder", "magicoder")
        m.compile(num_cores=16, device_group=[0])

        prompts=["code prompt", "math prompt", "generic"]
        m.generate(prompts, device_group=[0], prompt_to_adapter_mapping=["magicoder","gsm8k_id","base"])

    """

    def __init__(self, model: nn.Module, continuous_batching: bool = False, **kwargs) -> None:
        super().__init__(model, continuous_batching)
        if self.model.__class__.__name__ not in ["QEffMistralForCausalLM", "QEffLlamaForCausalLM"]:
            raise NotImplementedError(f"Only QEffMistralForCausalLM and QEffLlamaForCausalLM model are supported but get {self.model.__class__.__name__}")

        self.adapter_weights = {}
        self.adapter_configs = {}
        self.active_adapter_to_id = {}

        self.lora_rank = 0
        self.target_modules_for_all_adapters = []

    def __repr__(self) -> str:
        return self.__class__.__name__ + "\n" + self.model.__repr__()

    @property
    def model_hash(self) -> str:
        mhash = hashlib.sha256()

        # should use model config here
        mhash.update(to_hashable(self.model.model.config.to_diff_dict()))

        # create active adapter config dict
        active_adapter_configs = {}
        for adpt in self.active_adapter_to_id.keys():
            active_adapter_configs[adpt] = self.adapter_configs[adpt].to_dict()
        mhash.update(to_hashable(active_adapter_configs))

        # create active adapter weight dict
        active_adapter_weights = {}
        for adpt in self.active_adapter_to_id.keys():
            active_adapter_weights[adpt] = {key: value.tolist() for key, value in self.adapter_weights[adpt].items()}
        mhash.update(to_hashable(active_adapter_weights))

        # ensure model will be exported again if order of adapters changes
        mhash.update(to_hashable(self.active_adapter_to_id))

        # noncb & cb should have different onnx & qpc
        mhash.update(to_hashable({"continuous_batching": self.continuous_batching}))

        mhash = mhash.hexdigest()[:16]
        return mhash

    @property
    def get_model_config(self) -> dict:
        return self.model.model.config.__dict__

    def download_adapter(
        self,
        adapter_model_id: str,
        adapter_name: str,
        adapter_weight: Optional[dict] = None,
        adapter_config: Optional[PeftConfig] = None,
    ):
        """
        Loads a new adapter from huggingface hub or local path into CPU cache

        ``Mandatory`` Args:
            :adapter_model_id (str): Adapter model ID from huggingface hub or local path
            :adapter_name (str): Adapter name to be used to downloaded this adapter
        ``Optional`` Args:
            :adapter_weight (dict): Adapter weight tensors in dictionary format
            :adapter_config (PeftConfig): Adapter config in the format of PeftConfig
        """

        # check if adapter name already loaded
        if (adapter_name in self.adapter_weights.keys()) and (adapter_name in self.adapter_configs.keys()):
            logger.warning(f"{adapter_name} has been loaded. Skip download.")
        else:
            if adapter_weight and adapter_config:  # if sufficiently get adapter weight and adpater config
                self.adapter_weights[adapter_name] = adapter_weight
                self.adapter_configs[adapter_name] = adapter_config
            else:  # donwload with adapter_model_id
                self.adapter_weights[adapter_name] = {k: v.numpy().astype("float16") for k, v in load_peft_weights(adapter_model_id).items()}
                self.adapter_configs[adapter_name] = PeftConfig.from_pretrained(adapter_model_id)

    def load_adapter(
        self,
        adapter_model_id: str,
        adapter_name: str,
        adapter_weight: Optional[dict] = None,
        adapter_config: Optional[PeftConfig] = None,
    ):
        """
        Load adapter into CPU cache and set it as active

        ``Mandatory`` Args:
            :adapter_model_id (str): Adapter model ID from huggingface hub or local path
            :adapter_name (str): Adapter name to be used to load this adapter
        ``Optional`` Args:
            :adapter_weight (dict): Adapter weight tensors in dictionary format
            :adapter_config (PeftConfig): Adapter config in the format of PeftConfig
        """

        # check if adapter name already exist and activated
        if adapter_name in self.active_adapter_to_id.keys():
            logger.warning(f"{adapter_name} exists and activated. Please provide a different adapter_name.")
        else:
            self.download_adapter(adapter_model_id, adapter_name, adapter_weight, adapter_config)

            # starting from the second adapter_name, check if adapters has same target module and rank
            if list(self.adapter_configs.values())[0] and (self.adapter_configs[adapter_name].target_modules != list(self.adapter_configs.values())[0].target_modules):
                raise ValueError(f"{adapter_name} must have same target_modules as {list(self.adapter_configs.keys())[0]}")
            if list(self.adapter_configs.values())[0] and (self.adapter_configs[adapter_name].r != list(self.adapter_configs.values())[0].r):
                raise ValueError(f"{adapter_name} must have same rank as {list(self.adapter_configs.keys())[0]}")

            # set active adapter id to current max if adapter_name is new
            if adapter_name not in self.active_adapter_to_id.keys():
                self.active_adapter_to_id[adapter_name] = len(self.active_adapter_to_id) + 1  # reserve 0 for base

        return self.active_adapter_to_id[adapter_name]

    def unload_adapter(self, adapter_name: str):
        """
        Deactivate adpater and remove it from CPU cache

        ``Mandatory`` Args:
            :adapter_name (str): Adapter name to be unloaded
        """

        # step1: remove from active list if it's there
        if adapter_name not in self.active_adapter_to_id.keys():
            logger.info(f"Adapter name {adapter_name} is not set active yet")
            return False

        self.active_adapter_to_id.pop(adapter_name)

        # renumbering of active adapter id
        for index, (key, value) in enumerate(self.active_adapter_to_id.items()):
            self.active_adapter_to_id[key] = index + 1

        logger.warning(f"Deleting {adapter_name} from active adapters.")
        if self.onnx_path or self.qpc_path:
            logger.warning("Please redo compile_and_export() to reflect the active adapters changes.")
            self.onnx_path = None
            self.qpc_path = None

        # step2: delete from cache
        if adapter_name in self.adapter_weights.keys() and adapter_name in self.adapter_configs.keys():
            self.adapter_weights.pop(adapter_name)
            self.adapter_configs.pop(adapter_name)
            logger.warning(f"Unloading {adapter_name} from CPU cache.")

        return True

    def set_adapter(self, adapter_name: str):
        raise NotImplementedError("Set adapter is not supported in finite_adapters mode")

    def _load_adapter_weights_to_model(self):
        "Loads adapter weights to the model's multilora layer in a stacked format"

        num_hidden_layers = len(self.model.model.layers)
        for i in range(num_hidden_layers):
            for target_module in self.target_modules_for_all_adapters:
                # stack all adapters weights
                a_tensor_list = list(range(len(self.active_adapter_to_id) + 1))
                b_tensor_list = list(range(len(self.active_adapter_to_id) + 1))
                s_tensor_list = list(range(len(self.active_adapter_to_id) + 1))

                for lora_name, lora_id in self.active_adapter_to_id.items():
                    if target_module in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                        a_tensor_list[lora_id] = torch.from_numpy(self.adapter_weights[lora_name][f"base_model.model.model.layers.{i}.self_attn.{target_module}.lora_A.weight"])
                        b_tensor_list[lora_id] = torch.from_numpy(self.adapter_weights[lora_name][f"base_model.model.model.layers.{i}.self_attn.{target_module}.lora_B.weight"])
                    else:
                        raise NotImplementedError("Target module not supported!!")

                    s_tensor_list[lora_id] = torch.tensor(
                        self.adapter_configs[lora_name].lora_alpha / self.adapter_configs[lora_name].r,
                        dtype=torch.float16,
                    )

                # dummy zero tensor for base model
                a_tensor_list[0] = torch.zeros_like(a_tensor_list[1])
                b_tensor_list[0] = torch.zeros_like(b_tensor_list[1])
                s_tensor_list[0] = torch.zeros_like(s_tensor_list[1])

                # stack weight tensors
                stacked_lora_a = torch.stack(a_tensor_list, dim=0).unsqueeze(1).transpose(2, 3)  # <num_loras, 1, in_feature, r>
                stacked_lora_b = torch.stack(b_tensor_list, dim=0).unsqueeze(1).transpose(2, 3)  # <num_loras, 1, r, out_feature>
                stacked_lora_s = torch.stack(s_tensor_list, dim=0).unsqueeze(1).unsqueeze(2).unsqueeze(3)  # <num_loras, 1, 1, 1>

                # stored weight to corresponding ops
                if target_module == "q_proj":
                    module = self.model.model.layers[i].self_attn.q_proj
                elif target_module == "k_proj":
                    module = self.model.model.layers[i].self_attn.k_proj
                elif target_module == "v_proj":
                    module = self.model.model.layers[i].self_attn.v_proj
                elif target_module == "o_proj":
                    module = self.model.model.layers[i].self_attn.o_proj
                else:
                    raise NotImplementedError("Target module not supported!!")

                module.lora_a_weights.copy_(stacked_lora_a)
                module.lora_b_weights.copy_(stacked_lora_b)
                module.lora_scalings.copy_(stacked_lora_s)

    def _init_adapter_model(self):
        "Initialize the fixed lora model with multiple adapter weigths standby"

        # set lora rank
        self.lora_rank = list(self.adapter_configs.values())[0].r

        # do the module replacement
        _, transformed = LoraModelInputsTransform.apply(self.model)

        self.target_modules_for_all_adapters = list(self.adapter_configs.values())[0].target_modules
        _, transformed = TargetModulesTransform.apply(self.model, self.target_modules_for_all_adapters, self.lora_rank, len(self.active_adapter_to_id))

        # load_weight to model
        self._load_adapter_weights_to_model()

    def export(self, export_dir: Optional[str] = None) -> str:
        """
        Exports the model to ``ONNX`` format using ``torch.onnx.export``.
        We currently don't support exporting non-transformed models. Please refer to the ``convert_to_cloud_bertstyle`` function in the **Low-Level API** for a legacy function that supports this."

        ``Optional`` Args:
            does not any arguments.

        Returns:
            :str: Path of the generated ``ONNX`` graph.
        """

        # initialize the adapter model
        if len(self.active_adapter_to_id) == 0:
            raise ValueError("Please use load_adapter() to add at least one adapter; otherwise, refer to QEFFAutoModelForCausalLM for base model usage")

        self._init_adapter_model()

        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        seq_len = constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
        fbs = constants.ONNX_EXPORT_EXAMPLE_FBS
        kv_cache_shape = get_padding_shape_from_config(self.model.config, fbs if self.continuous_batching else bs, seq_len)
        example_inputs = {
            "input_ids": torch.zeros((bs, seq_len), dtype=torch.int64),
            "position_ids": torch.arange(seq_len, dtype=torch.int64).view(bs, seq_len),
            "past_key_values": [[] for _ in range(self.num_layers)],
            "lora_ids": torch.zeros(bs, dtype=torch.int64).view(bs, 1),
        }
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "lora_ids": {0: "batch_size"},
        }
        output_names = ["logits"]
        for i in range(self.num_layers):
            for kv in ["key", "value"]:
                example_inputs["past_key_values"][i].append(torch.zeros(kv_cache_shape, dtype=torch.float32))
                dynamic_axes[f"past_{kv}.{i}"] = {
                    0: "full_batch_size" if self.continuous_batching else "batch_size",
                    2: "ctx_len",
                }
                output_names.append(f"past_{kv}.{i}_RetainedState")

        if self.continuous_batching:
            example_inputs["batch_index"] = torch.arange(bs).view(bs, 1)
            dynamic_axes["batch_index"] = {0: "batch_size"}

        return self._export(
            example_inputs,
            output_names,
            dynamic_axes,
            export_dir=export_dir,
        )

    def generate(
        self,
        tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer],
        prompts: List[str],
        prompt_to_adapter_mapping: List[str] = None,
        device_id: Optional[List[int]] = None,
        runtime: Optional[str] = "AI_100",
        **kwargs,
    ):
        """
        This method generates output until ``eos`` or ``generation_len`` by executing the compiled ``qpc`` on ``Cloud AI 100`` Hardware cards.
        This is a sequential execution based on the ``batch_size`` of the compiled model and the number of prompts passed.
        If the number of prompts cannot be divided by the ``batch_size``, the last unfulfilled batch will be dropped.

        ``Mandatory`` Args:
            :tokenizer (PreTrainedTokenizerFast or PreTrainedTokenizer): The tokenizer used in the inference
            :prompts (List[str]): List of prompts to run the execution.
            :prompt_to_adapter_mapping (List[str]): The sequence of the adapter names will be matched with sequence of prompts and corresponding adapters will be used for the prompts."base" for base model (no adapter).
        ``optional`` Args:
            :device_id (List[int]): Device IDs to be used for execution. If ``len(device_id) > 1``, it enables multiple card setup. If ``None``, auto-device-picker will be used. ``Defaults to None``.
            :runtime (str, optional): Only ``AI_100`` runtime is supported as of now; ``ONNXRT`` and ``PyTorch`` coming soon. Defaults to "AI_100".

        """
        if runtime != "AI_100":
            raise ValueError("Only AI_100 runtime is supported right now via generate API")
        if not isinstance(self.qpc_path, Path):
            raise TypeError("Please run compile API first!")
        generation_len = kwargs.pop("generation_len", None)

        if not prompt_to_adapter_mapping:
            prompt_to_adapter_mapping = ["base" for _ in range(len(prompts))]

        if len(prompt_to_adapter_mapping) != len(prompts):
            raise RuntimeError(f"Number of prompts should match number of prompt_to_adapter_mapping, got len(prompts) = {len(prompts)}, len(prompt_to_adapter_mapping) = {len(prompt_to_adapter_mapping)}")

        return QEfficient.cloud_ai_100_exec_kv(
            tokenizer,
            self.qpc_path,
            prompt=prompts,
            device_id=device_id,
            generation_len=generation_len,
            prompt_to_lora_id_mapping=[self.active_adapter_to_id[name] if name != "base" else 0 for name in prompt_to_adapter_mapping],
        )
