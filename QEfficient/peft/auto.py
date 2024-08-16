# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import hashlib
import json
import logging
import shutil
import subprocess
import warnings
from pathlib import Path
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
from QEfficient.peft.onnx_transforms import LoraWeightsToInputsTransform
from QEfficient.peft.pytorch_transforms import PeftModelInputsTransform
from QEfficient.transformers.pytorch_transforms import CustomOpsTransform, KVCacheTransform
from QEfficient.utils._utils import get_padding_shape_from_config
from QEfficient.utils.cache import QEFF_HOME, to_hashable

logger = logging.getLogger(__name__)


class QEffAutoPeftModelForCausalLM(QEFFBaseModel):
    pytorch_transforms: List[PytorchTransform] = [CustomOpsTransform, KVCacheTransform, PeftModelInputsTransform]
    onnx_transforms: List[OnnxTransform] = [FP16ClipTransform, LoraWeightsToInputsTransform, SplitTensorsTransform]
    _hf_auto_class = AutoPeftModelForCausalLM

    @classmethod
    def transform_names(cls) -> List[str]:
        # Base class
        return [x.__name__ for x in cls.pytorch_transforms + cls.onnx_transforms]

    def __init__(self, model: nn.Module):
        if model.active_peft_config.peft_type != "LORA":
            raise NotImplementedError("Only LoRA models are supported")

        base_model = model.get_base_model()
        self.model_name = type(base_model).__name__ + "-lora"

        # Compute the hash with: model_config, peft_config, transforms
        model_hash = hashlib.sha256()
        model_hash.update(to_hashable(base_model.config.to_diff_dict()))
        model_hash.update(to_hashable(model.active_peft_config.to_dict()))
        model_hash.update(to_hashable(self.transform_names()))
        self.model_hash = model_hash.hexdigest()[:16]
        self.model_dir = self.model_name + "-" + self.model_hash

        self.num_layers = model.config.num_hidden_layers
        super().__init__(model)
        self.transform()

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str, **kwargs):
        # Base class
        if kwargs.get("use_cache") is False:
            warnings.warn("Overriding to use_cache=True")
        kwargs["use_cache"] = True
        model = cls._hf_auto_class.from_pretrained(pretrained_name_or_path, **kwargs)
        return cls(model)

    def transform(self, **kwargs):
        # Base class
        for transform in self.pytorch_transforms:
            self.model, transformed = transform.apply(self.model)
        logger.info("Pytorch transforms applied")

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

    @property
    def export_kwargs(self) -> Dict[str, any]:
        return {"do_constant_folding": False}  # To avoid merging adapter weights with base weights

    @property
    def onnx_transform_kwargs(self) -> Dict[str, any]:
        return {"adapter_name": self.model.active_adapter}

    def export(self, export_dir: Optional[str] = None) -> str:
        # Base class
        export_dir = Path(export_dir or (QEFF_HOME / self.model_dir))
        onnx_path = export_dir / f"{self.model_name}.onnx"
        if onnx_path.is_file():
            self.onnx_path = onnx_path
            return onnx_path

        tmp_onnx_dir = export_dir / "onnx_tmp"
        tmp_onnx_path = tmp_onnx_dir / f"{self.model_name}.onnx"
        tmp_onnx_dir.mkdir(parents=True, exist_ok=True)

        try:
            torch.onnx.export(
                self.model,
                (self.sample_inputs,),
                str(tmp_onnx_path),
                input_names=self.input_names,
                output_names=self.output_names,
                dynamic_axes=self.dynamic_axes,
                opset_version=13,
                **self.export_kwargs,
            )
            logger.info("Pytorch export successful")

            model = onnx.load(tmp_onnx_path, load_external_data=False)
            onnx_transform_kwargs = {
                "onnx_base_dir": str(tmp_onnx_dir),
                "model_name": self.model_name,
            }
            onnx_transform_kwargs.update(self.onnx_transform_kwargs)
            for transform in self.onnx_transforms:
                model, transformed = transform.apply(model, **onnx_transform_kwargs)
            model.metadata_props.append(
                onnx.StringStringEntryProto(key="qeff_transforms", value=",".join(self.transform_names()))
            )
            logger.info("ONNX transforms applied")

            onnx.save(model, onnx_path)
            logger.info("Transformed onnx saved")

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise e

        finally:
            shutil.rmtree(tmp_onnx_dir, ignore_errors=True)

        self.onnx_path = onnx_path
        return onnx_path

    def _compile(self, onnx_path, **kwargs) -> str:
        # Base class
        command = ["/opt/qti-aic/exec/qaic-exec", f"-m={onnx_path}", "-aic-hw", "-aic-hw-version=2.0"]
        aic_binary_dir = Path(kwargs.pop("aic_binary_dir", None) or "qpc")
        for key, value in kwargs.items():
            option = "-" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(option)
                continue
            command.append(f"{option}={value}")

        # Compute hash for binary location
        compile_hash = hashlib.sha256(to_hashable(command)).hexdigest()[:16]
        aic_binary_dir = aic_binary_dir.with_name(aic_binary_dir.name + "-" + compile_hash)
        if aic_binary_dir.is_dir():
            if (aic_binary_dir / "programqpc.bin").is_file():
                return aic_binary_dir
            # Probably compilation failure last time, delete directory to start over
            shutil.rmtree(aic_binary_dir)

        command.append(f"-aic-binary-dir={aic_binary_dir}")
        logger.info(f"Running compiler: {' '.join(command)}")
        subprocess.run(command).check_returncode()
        return aic_binary_dir

    def compile(
        self,
        *,
        batch_size: int = 1,
        prefill_seq_len: int,
        ctx_len: int,
        onnx_path: Optional[str] = None,
        compile_dir: Optional[str] = None,
        num_devices: int = 1,
        num_cores: int = 16,
        mxfp6_matmul: bool = False,
        mxint8_kv_cache: bool = False,
        **compiler_options,
    ) -> str:
        compile_dir = Path(compile_dir or (QEFF_HOME / self.model_dir))
        onnx_path = Path(onnx_path or self.onnx_path)
        aic_binary_dir = compile_dir / "qpc"

        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX file not found at: {onnx_path}")

        # Specializations
        specializations_json = compile_dir / "specializations.json"
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
        custom_io_yaml = compile_dir / f"custom_io_{kv_cache_dtype}.yaml"
        with open(custom_io_yaml, "w") as fp:
            for suffix in ["", "_RetainedState"]:
                for i in range(self.num_layers):
                    for kv in ["key", "value"]:
                        fp.write(f" - IOName: past_{kv}.{i}{suffix}\n   Precision: {kv_cache_dtype}\n\n")

                for weight_name in self.adapter_weights[self.active_adapter]:
                    fp.write(f" - IOName: {weight_name}{suffix}\n   Precision: float16\n\n")

        # MDP
        if num_devices > 1:
            mdp_ts_json = compile_dir / f"mdp_ts_{num_devices}.json"
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

        self.qpc_path = self._compile(
            onnx_path,
            network_specialization_config=specializations_json,
            compile_only=True,
            aic_binary_dir=aic_binary_dir,
            convert_to_fp16=True,
            mxfp6_matmul=mxfp6_matmul,
            custom_IO_list_file=custom_io_yaml,
            retained_state=True,
            aic_num_cores=num_cores,
            **compiler_options,
        )
        return self.qpc_path

    def run_pytorch(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Base class
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return dict(outputs)

    def run_ort(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Base class

        # Initialize session
        if self.ort_session is None:
            if self.onnx_path is None:
                self.export()
            self.ort_session = ORTInferenceSession(self.onnx_path)
            self._retained_state = {}

        # Prepare inputs by adding retained state
        inputs = inputs.copy()
        for inp in self.ort_session.get_inputs():
            if inp.name not in inputs and inp.name in self._retained_state:
                inputs[inp.name] = self._retained_state[inp.name]

        # Run session
        outputs = self.ort_session.run(None, inputs)
        outputs = dict(zip(self.output_names, outputs))

        # Save retained state outputs
        for output_name, output in outputs.items():
            if output_name.endswith("_RetainedState"):
                self._retained_state[output_name[: -len("_RetainedState")]] = output

        # Remove retained state outputs
        outputs = {k: v for k, v in outputs if not k.endswith("_RetainedState")}
        return outputs

    def run_cloud_ai_100(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Base class
        pass

    def generate(self, inputs: Dict[str, np.ndarray], streamer) -> np.ndarray:
        pass
