# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import json
import logging
import os
import shutil
import subprocess
import warnings
from typing import Dict, List, Optional

import numpy as np
import onnx
import torch
from onnxruntime import InferenceSession as ORTInferenceSession
from peft import AutoPeftModelForCausalLM
from torch import nn

from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.base.onnx_transforms import FP16ClipTransform, OnnxTransform, SplitTensorsTransform
from QEfficient.base.pytorch_transforms import PytorchTransform
from QEfficient.peft.onnx_transforms import AdaptersAsInputsTransform
from QEfficient.peft.pytorch_transforms import PeftModelInputsTransform
from QEfficient.transformers.pytorch_transforms import CustomOpsTransform, KVCacheTransform
from QEfficient.utils._utils import get_padding_shape_from_config
from QEfficient.utils.cache_dir import QEFF_HOME

logger = logging.getLogger(__name__)


class QEffAutoPeftModelForCausalLM(QEFFBaseModel):
    pytorch_transforms: List[PytorchTransform] = [CustomOpsTransform, KVCacheTransform, PeftModelInputsTransform]
    onnx_transforms: List[OnnxTransform] = [FP16ClipTransform, AdaptersAsInputsTransform, SplitTensorsTransform]
    _hf_auto_class = AutoPeftModelForCausalLM

    def __init__(self, model: nn.Module, card_name: Optional[str] = None):
        super().__init__(model)
        self.card_name = card_name
        self.num_layers = self.model.config.num_hidden_layers
        self.transform()

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str, **kwargs):
        # Base class
        card_name = kwargs.pop("card_name", None)
        if kwargs.get("use_cache") is False:
            warnings.warn("Overriding to use_cache=True")
        kwargs["use_cache"] = True
        model = cls._hf_auto_class.from_pretrained(pretrained_name_or_path, **kwargs)

        if card_name is None and (not os.path.exists(pretrained_name_or_path)):
            card_name = pretrained_name_or_path

        return cls(model, card_name)

    def transform(self, **kwargs):
        # Base class
        logger.info("Pytorch layers are transformed")
        for transform in self.pytorch_transforms:
            self.model, transformed = transform.apply(self.model)

    @property
    def sample_inputs(self) -> Dict[str, torch.Tensor]:
        kv_cache_shape = get_padding_shape_from_config(self.model.config, 1, 32)
        inputs = {
            "input_ids": torch.zeros((1, 32), dtype=torch.int64),
            "position_ids": torch.arange(32, dtype=torch.int64).view((1, 32)),
            "past_key_values": [
                (
                    torch.zeros(kv_cache_shape, dtype=torch.float32),
                    torch.zeros(kv_cache_shape, dtype=torch.float32),
                )
                for _ in range(self.num_layers)
            ],
        }
        return inputs

    @property
    def dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
        }
        for i in range(self.num_layers):
            dynamic_axes[f"past_key.{i}"] = {0: "batch_size", 2: "ctx_len"}
            dynamic_axes[f"past_value.{i}"] = {0: "batch_size", 2: "ctx_len"}
        return dynamic_axes

    @property
    def input_names(self) -> List[str]:
        return list(self.dynamic_axes.keys())

    @property
    def output_names(self) -> List[str]:
        outputs = ["logits"]
        for i in range(self.num_layers):
            outputs.append(f"past_key.{i}_RetainedState")
            outputs.append(f"past_value.{i}_RetainedState")
        return outputs

    def export(self, export_dir: Optional[str] = None) -> str:
        # Base class
        model_name = self.card_name.replace("/", "_")
        export_dir = export_dir or os.path.join(QEFF_HOME, model_name)
        self.onnx_path = os.path.join(export_dir, f"{model_name}.onnx")
        tmp_onnx_dir = os.path.join(export_dir, "onnx_tmp")
        tmp_onnx_path = os.path.join(tmp_onnx_dir, f"{model_name}.onnx")
        os.makedirs(tmp_onnx_dir, exist_ok=True)
        torch.onnx.export(
            self.model,
            (self.sample_inputs,),
            tmp_onnx_path,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes,
            opset_version=13,
        )
        logger.info("Pytorch export successful")

        model = onnx.load(tmp_onnx_path, load_external_data=False)
        transform_kwargs = {
            "onnx_base_dir": tmp_onnx_dir,
            "model_name": model_name,
            "adapter_name": self.model.active_adapter,
        }
        for transform in self.onnx_transforms:
            model, transformed = transform.apply(model, **transform_kwargs)
        logger.info("ONNX transforms applied")

        onnx.save(model, self.onnx_path)
        shutil.rmtree(tmp_onnx_dir)
        logger.info("Transformed onnx saved")

        return self.onnx_path

    def _compile(self, onnx_path, **kwargs):
        # Base class
        command = ["/opt/qti-aic/exec/qaic-exec", f"-m={onnx_path}", "-aic-hw", "-aic-hw-version=2.0"]
        for key, value in kwargs.items():
            option = "-" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(option)
                continue
            command.append(f"{option}={value}")
        logger.info(f"Running compiler: {command}")
        subprocess.run(command).check_returncode()

    def compile(
        self,
        *,
        batch_size: int = 1,
        prefill_seq_len: int,
        ctx_len: int,
        onnx_path: Optional[str] = None,
        binary_path: Optional[str] = None,
        num_devices: int = 1,
        num_cores: int = 16,
        mxfp6_matmul: bool = False,
        mxint8_kv_cache: bool = False,
        **compiler_options,
    ) -> str:
        model_name = self.card_name.replace("/", "_")
        model_dir = os.path.join(QEFF_HOME, model_name)
        onnx_path = onnx_path or self.onnx_path
        binary_path = binary_path or os.path.join(model_dir, "qpc")

        # Specializations
        specializations_json = os.path.join(model_dir, "specializations.json")
        with open(specializations_json, "w") as fp:
            json.dump(
                {
                    "specializations": [
                        {"batch_size": str(batch_size), "seq_len": str(prefill_seq_len), "ctx_len": str(ctx_len)},
                        {"batch_size": str(batch_size), "seq_len": "1", "ctx_len": str(ctx_len)},
                    ]
                },
                fp,
                indent=4,
            )

        # Custom IO
        kv_cache_dtype = "mxint8" if mxint8_kv_cache else "float16"
        custom_io_yaml = os.path.join(model_dir, f"custom_io_{kv_cache_dtype}.yaml")
        with open(custom_io_yaml, "w") as fp:
            for suffix in ["", "_RetainedState"]:
                for i in range(self.num_layers):
                    for kv in ["key", "value"]:
                        fp.write(f" - IOName: past_{kv}.{i}{suffix}\n   Precision: {kv_cache_dtype}\n\n")

        # MDP
        if num_devices > 1:
            mdp_ts_json = os.path.join(model_dir, f"mdp_ts_{num_devices}.json")
            with open(mdp_ts_json, "w") as fp:
                json.dump(
                    {
                        "connections": [{"devices": list(range(num_devices)), "type": "p2p"}],
                        "partitions": [
                            {
                                "name": "Partition0",
                                "devices": [{"deviceId": d, "numCores": num_cores} for d in range(num_devices)],
                            }
                        ],
                    },
                    fp,
                    indent=4,
                )
            compiler_options["mdp_load_partition_config"] = mdp_ts_json

        self._run_compiler(
            onnx_path,
            network_specialization_config=specializations_json,
            compile_only=True,
            aic_binary_dir=binary_path,
            convert_to_fp16=True,
            mxfp6_matmul=mxfp6_matmul,
            custom_IO_list_file=custom_io_yaml,
            retained_state=True,
            aic_num_cores=num_cores,
            **compiler_options,
        )

        self.binary_path = binary_path

    def run_pytorch(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Base class
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return dict(outputs)

    def run_ort(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Base class
        if not getattr(self, "ort_session", None):
            if self.onnx_path is None:
                self.export()
            self.ort_session = ORTInferenceSession(self.onnx_path)
        outputs = self.ort_session.run(None, inputs)
        return dict(zip(self.output_names, outputs))

    def run_cloud_ai_100(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Base class
        pass

    def generate(self, inputs: Dict[str, np.ndarray], streamer) -> np.ndarray:
        pass
