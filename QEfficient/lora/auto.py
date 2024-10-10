# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import hashlib
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from peft import PeftConfig, load_peft_weights

import QEfficient
from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.lora.pytorch_transforms import LoraModelInputsTransform, TargetModulesTransform
from QEfficient.transformers.pytorch_transforms import CBTransform
from QEfficient.utils import get_qpc_dir_path, qpc_exists
from QEfficient.utils.cache import to_hashable
from QEfficient.utils.constants import QEFF_MODELS_DIR
from QEfficient.utils.logging_utils import logger

INTMAX = sys.maxsize


class QEffAutoLoraModelForCausalLM(QEFFAutoModelForCausalLM):
    """
    QEff class for loading models with mutltiple LoRA adapters.
    Once exported and compiled, the qpc can perform mixed batch inference with provided prompt_to_lora_id_mapping.

    Args:
        :model (nn.Module): PyTorch model
        :base_model_name (str): Model card name for base model
        :adapter_weights (Dict): A dictionary contains lora_name to lora_weight mapping
        :adapter_configs (Dict): A dictionary contains lora_name to lora_configs mapping
        :active_adapters (Set): A set of lora_names that are currently active
        :max_num_adapters (int): Total number of active adapters that to be exported and compiled
        :active_adapter_to_id (Dict): A dictionary contains active adapter's lora_name to lora_id mapping

    .. code-block:: python

        from QEfficient import QEffAutoLoraModelForCausalLM

        m = QEffAutoPeftModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        m.load_adapter("predibase/gsm8k", "gsm8k")
        m.load_adapter("predibase/magicoder", "magicoder")
        gsm8k_id = m.set_adapter("gsm8k")
        magicoder_id = m.set_adapter("magicoder")
        m.export(full_batch_size=3)
        m.compile(num_cores=16, device_group=[0])

        prompts=["code prompt", "math prompt", "generic"]
        m.generate(prompts, device_group=[0], prompt_to_lora_id_mapping=[magicoder_id,gsm8k_id,INTMAX])

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

        self.lora_rank = 0
        self.target_modules_for_all_adapters = []

    @property
    def model_hash(self) -> str:
        mhash = hashlib.sha256()

        # should use model config here
        mhash.update(to_hashable(self.model.model.config.to_diff_dict()))

        # create active adapter config dict
        active_adapter_configs = {}
        for adpt in self.active_adapters:
            active_adapter_configs[adpt] = self.adapter_configs[adpt].to_dict()
        mhash.update(to_hashable(active_adapter_configs))

        # ensure model will be exported again if order of adapters changes
        mhash.update(to_hashable(self.active_adapter_to_id))

        mhash = mhash.hexdigest()[:16]
        return mhash

    def load_adapter(self, adapter_model_id: str, adapter_name: str):
        """Loads a new adapter from huggingface hub or local path into CPU cache

        Args:
            :adapter_model_id (str): Adapter model ID from huggingface hub or local path
            :adapter_name (str): Adapter name to be used to set this adapter as current
        """
        if (adapter_name in self.adapter_weights.keys()) and (adapter_name in self.adapter_configs.keys()):
            logger.warning(f"Overwrite weights and configs for adapter name {adapter_name}")

        self.adapter_weights[adapter_name] = {
            k: v.numpy().astype("float16") for k, v in load_peft_weights(adapter_model_id).items()
        }
        self.adapter_configs[adapter_name] = PeftConfig.from_pretrained(adapter_model_id)

    def unload_adapter(self, adapter_name: str):
        if adapter_name not in self.adapter_weights.keys() and adapter_name not in self.adapter_configs.keys():
            print(f"Adapter name {adapter_name} is not loaded yet")
            return False

        if adapter_name in self.active_adapters:
            print(f"Adapter name {adapter_name} is stil in active list, do delete_adapter() before unloading")
            return False

        self.adapter_weights.pop(adapter_name)
        self.adapter_configs.pop(adapter_name)
        logger.warning(f"Unloading {adapter_name} from CPU cache.")
        return True

    def set_adapter(self, adapter_name: str):
        "Sets active adapter from one of the loaded adapters"

        assert (adapter_name in self.adapter_weights.keys()) and (
            adapter_name in self.adapter_configs.keys()
        ), f"Adapter name {adapter_name} has not been loaded yet"

        assert (
            list(self.adapter_configs.values())[0]
            and self.adapter_configs[adapter_name].target_modules
            == list(self.adapter_configs.values())[0].target_modules
        ), "Not all adapters have the same target modules"

        assert (
            list(self.adapter_configs.values())[0]
            and self.adapter_configs[adapter_name].r == list(self.adapter_configs.values())[0].r
        ), "Not all adapters have the same ranks"

        # set active adapter id to current max
        self.active_adapter_to_id[adapter_name] = self.max_num_adapters

        # add active adapter to set
        self.active_adapters.add(adapter_name)
        self.max_num_adapters = len(self.active_adapters)

        return self.active_adapter_to_id[adapter_name]

    def delete_adapter(self, adapter_name: str):
        if adapter_name not in self.active_adapters:
            print(f"Adapter name {adapter_name} is not set active yet")
            return False

        self.active_adapters.discard(adapter_name)
        self.max_num_adapters -= 1
        self.active_adapter_to_id.pop(adapter_name)

        # renumbering of active adapter id
        for index, (key, value) in enumerate(self.active_adapter_to_id.items()):
            self.active_adapter_to_id[key] = index

        logger.warning(f"Deleting {adapter_name} from active adapters.")
        if self.onnx_path or self.qpc_path:
            logger.warning("Please redo compile_and_export() to reflect the active adapters changes.")

        return True

    def get_adapter_id(self, adapter_name):
        "get the adapter_id that maps to the adapter_name"

        return self.active_adapter_to_id[adapter_name]

    def load_adapter_weights_to_model(self):
        "Loads adapter weights to the model's multilora layer in a stacked format"

        num_hidden_layers = len(self.model.model.layers)
        for i in range(num_hidden_layers):
            for target_module in self.target_modules_for_all_adapters:
                # stack all adapters weights
                a_tensor_list = list(range(self.max_num_adapters))
                b_tensor_list = list(range(self.max_num_adapters))
                c_tensor_list = list(range(self.max_num_adapters))

                for lora_name, lora_id in self.active_adapter_to_id.items():
                    if (
                        target_module == "q_proj"
                        or target_module == "k_proj"
                        or target_module == "v_proj"
                        or target_module == "o_proj"
                    ):
                        a_tensor_list[lora_id] = torch.from_numpy(
                            self.adapter_weights[lora_name][
                                f"base_model.model.model.layers.{i}.self_attn.{target_module}.lora_A.weight"
                            ]
                        )
                        b_tensor_list[lora_id] = torch.from_numpy(
                            self.adapter_weights[lora_name][
                                f"base_model.model.model.layers.{i}.self_attn.{target_module}.lora_B.weight"
                            ]
                        )
                    else:
                        raise NotImplementedError("Target module not supported!!")

                    c_tensor_list[lora_id] = torch.tensor(
                        self.adapter_configs[lora_name].lora_alpha / self.adapter_configs[lora_name].r,
                        dtype=torch.float16,
                    )

                stacked_lora_A = (
                    torch.stack(a_tensor_list, dim=0).unsqueeze(1).transpose(2, 3)
                )  # <num_adapters, 1, in_feature, r>
                stacked_lora_B = (
                    torch.stack(b_tensor_list, dim=0).unsqueeze(1).transpose(2, 3)
                )  # <num_adapters, 1, r, out_feature>
                stacked_lora_C = (
                    torch.stack(c_tensor_list, dim=0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                )  # <num_loras, 1, 1, 1>

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

                module.lora_weight_A.copy_(stacked_lora_A)
                module.lora_weight_B.copy_(stacked_lora_B)
                module.lora_weight_C.copy_(stacked_lora_C)

    def init_adapter_model(self):
        "Initialize the fixed lora model with multiple adapter weigths standby"

        # assume all adapters have same target_modules and ranks
        assert self.max_num_adapters == len(self.active_adapters), "Inconsistent max_num_adapters and active_adapters"

        assert list(self.adapter_configs.values())[0] and all(
            list(self.adapter_configs.values())[i].target_modules
            == list(self.adapter_configs.values())[0].target_modules
            for i in range(self.max_num_adapters)
        ), "Not all adapters have the same target modules"

        assert list(self.adapter_configs.values())[0] and all(
            list(self.adapter_configs.values())[i].r == list(self.adapter_configs.values())[0].r
            for i in range(self.max_num_adapters)
        ), "Not all adapters have the same ranks"
        self.lora_rank = list(self.adapter_configs.values())[0].r

        # do the module replacement
        _, transformed = LoraModelInputsTransform.apply(self.model)

        self.target_modules_for_all_adapters = list(self.adapter_configs.values())[0].target_modules
        _, transformed = TargetModulesTransform.apply(
            self.model, self.target_modules_for_all_adapters, self.lora_rank, self.max_num_adapters
        )

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
        export_dir = kwargs.get("export_dir", None)

        # obtain all necessary information to initialize the model
        self.init_adapter_model()

        assert self.is_transformed, "Please first run transform on the QEFFAutoModelForCausalLM object"

        # Caching export onnx
        if export_dir is None:
            model_card_dir = os.path.join(QEFF_MODELS_DIR, str(self.model_card_name))
            export_dir = Path(model_card_dir).with_name(str(self.model_card_name).split("/")[1] + "-" + self.model_hash)
        else:
            export_dir = Path(export_dir).with_name(export_dir.name + "-" + self.model_hash)
        onnx_dir_path = os.path.join(export_dir, "onnx")
        model_base_name = self.model_card_name.replace("/", "_") + "_kv"
        onnx_path = os.path.join(onnx_dir_path, f"{model_base_name}.onnx")

        if Path(onnx_path).is_file():
            self.onnx_path = onnx_path
            print(f"Using existing onnx path:-{self.onnx_path}")
            return self.onnx_path

        # Export
        os.makedirs(onnx_dir_path, exist_ok=True)
        _, onnx_model_path = QEfficient.export(
            model_name=self.model_card_name,
            model_kv=self,
            tokenizer=self.tokenizer,
            full_batch_size=self.full_batch_size,
            max_num_adapters=self.max_num_adapters,
            onnx_dir_path=onnx_dir_path,
        )
        self.onnx_path = onnx_model_path

        return self.onnx_path

    def export_and_compile(
        self,
        num_cores: int,
        device_group: List[int],
        batch_size: int = 1,
        prompt_len: int = 32,
        ctx_len: int = 128,
        mxfp6: bool = True,
        mxint8: bool = False,
        mos: int = -1,
        aic_enable_depth_first: bool = False,
        qpc_dir_suffix: Optional[str] = None,
        full_batch_size: Optional[int] = None,
    ) -> str:
        """
        This API is specific to Internal VLLM use-case and is not recommended to be used in your application unless your are using VLLM.
        """
        _, transformed = CBTransform.apply(self.model)
        if not transformed:
            raise RuntimeError("Could not apply Continuous batch transform on the model")
        if full_batch_size is not None:
            self.full_batch_size = full_batch_size

        self.export()

        qpc_base_dir_name = get_qpc_dir_path(
            model_card_name=self.model_card_name,
            num_cores=num_cores,
            mos=mos,
            batch_size=batch_size,
            prompt_len=prompt_len,
            ctx_len=ctx_len,
            mxfp6=mxfp6,
            mxint8=mxint8,
            device_group=device_group,
            full_batch_size=self.full_batch_size,
        )

        # Caching compiled qpc
        model_card_dir = os.path.join(QEFF_MODELS_DIR, str(self.model_card_name))
        export_dir = Path(model_card_dir).with_name(str(self.model_card_name).split("/")[1] + "-" + self.model_hash)
        qpc_dir_path = qpc_base_dir_name.replace(model_card_dir, str(export_dir))
        qpc_path = os.path.join(qpc_dir_path, "qpcs")

        if not qpc_exists(qpc_path):
            # Compile
            self.qpc_path = QEfficient.compile(
                onnx_path=self.onnx_path,
                qpc_path=qpc_dir_path,
                num_cores=num_cores,
                device_group=device_group,
                aic_enable_depth_first=aic_enable_depth_first,
                mos=mos,
                batch_size=batch_size,
                prompt_len=prompt_len,
                ctx_len=ctx_len,
                mxfp6=mxfp6,
                mxint8=mxint8,
                full_batch_size=full_batch_size,
            )
            print(f"Generated qpc:-{qpc_path}")
        else:
            self.qpc_path = qpc_path
            print(f"Using existing qpc path:-{self.qpc_path}")

        return self.qpc_path

    def run_cloud_ai_100(self, prompts: List[str], device_id: List[int] = None, **kwargs):
        assert isinstance(self.qpc_path, str), "Please run compile API first!"
        generation_len = kwargs.pop("generation_len", None)
        default_mapping = [INTMAX for _ in range(len(prompts))]
        prompt_to_lora_id_mapping = kwargs.pop("prompt_to_lora_id_mapping", default_mapping)
        return QEfficient.cloud_ai_100_exec_kv(
            self.tokenizer,
            self.qpc_path,
            prompt=prompts,
            device_id=device_id,
            generation_len=generation_len,
            full_batch_size=self.full_batch_size,
            prompt_to_lora_id_mapping=prompt_to_lora_id_mapping,
        )
