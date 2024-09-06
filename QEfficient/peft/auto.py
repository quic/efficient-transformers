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
from typing import Dict, List, Optional, Union

import numpy as np
import onnx
import torch
from peft import AutoPeftModelForCausalLM, load_peft_weights
from torch import nn
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer

from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.base.onnx_transforms import FP16ClipTransform, OnnxTransform, SplitTensorsTransform
from QEfficient.base.pytorch_transforms import PytorchTransform
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.peft.onnx_transforms import AdapterWeightsToInputsTransform
from QEfficient.peft.pytorch_transforms import PeftModelInputsTransform
from QEfficient.transformers.pytorch_transforms import CustomOpsTransform, KVCacheTransform
from QEfficient.utils import constants
from QEfficient.utils._utils import get_padding_shape_from_config
from QEfficient.utils.cache import QEFF_HOME, to_hashable

logger = logging.getLogger(__name__)


class QEffAutoPeftModelForCausalLM(QEFFBaseModel):
    """
    QEff class for loading models with PEFT adapters (Only LoRA is supported currently).
    Once exported and compiled for an adapter, the same can be utilized for another adapter with same base model and adapter config.

    Args:
        :model (nn.Module): PyTorch model

    .. code-block:: python

        from QEfficient import QEffAutoPeftModelForCausalLM

        m = QEffAutoPeftModelForCausalLM.from_pretrained("predibase/magicoder", "magicoder")
        m.export()
        m.compile(prefill_seq_len=32, ctx_len=1024)

        inputs = ...  # A coding prompt
        outputs = m.generate(**inputs)

        inputs = ...  # A math prompt
        m.load_adapter("predibase/gsm8k", "gsm8k")
        m.set_adapter("gsm8k")
        outputs = m.generate(**inputs)
    """

    _pytorch_transforms: List[PytorchTransform] = [CustomOpsTransform, KVCacheTransform, PeftModelInputsTransform]
    _onnx_transforms: List[OnnxTransform] = [FP16ClipTransform, AdapterWeightsToInputsTransform, SplitTensorsTransform]
    _hf_auto_class = AutoPeftModelForCausalLM

    @classmethod
    def transform_names(cls) -> List[str]:
        # Base class
        return [x.__name__ for x in cls._pytorch_transforms + cls._onnx_transforms]

    def __init__(self, model: nn.Module):
        if model.active_peft_config.peft_type != "LORA":
            raise NotImplementedError("Only LoRA models are supported")

        super().__init__(model)

        self.num_layers = model.config.num_hidden_layers
        self.adapter_weights = {}
        self.transform()

    @property
    def model_name(self) -> str:
        return self.model.get_base_model().__class__.__name__ + "-lora"

    @property
    def model_hash(self) -> str:
        # NOTE: model_config.to_diff_dict() has "_name_or_path" attribute which is the model card name or path.
        # Using same card name will result in same hash. But, using a relative path for one run and
        # absolute path for another run will result in different hash.
        # The added complexity to resolve different paths to same location is not worth pursuing.
        # Instead, advise the user to always provide same relative paths or absolute paths for local models.

        # Compute the hash with: model_config, peft_config, transforms
        mhash = hashlib.sha256()
        mhash.update(to_hashable(self.model.get_base_model().config.to_diff_dict()))
        mhash.update(to_hashable(self.model.active_peft_config.to_dict()))
        mhash.update(to_hashable(self.transform_names()))
        mhash = mhash.hexdigest()[:16]
        return mhash

    def load_adapter(self, model_id: str, adapter_name: str):
        """Loads a new adapter from huggingface hub or local path

        Args:
            :model_id (str): Adapter model ID from huggingface hub or local path
            :adapter_name (str): Adapter name to be used to set this adapter as current
        """
        self.model.load_adapter(model_id, adapter_name)
        self.adapter_weights[adapter_name] = {
            k: v.numpy().astype("float16") for k, v in load_peft_weights(model_id).items()
        }

    @property
    def active_adapter(self) -> str:
        "Currently active adapter to be used for inference"
        return self.model.active_adapter

    def set_adapter(self, adapter_name: str):
        "Sets active adapter from one of the loaded adapters"
        self.model.set_adapter(adapter_name)

    def disable_adapter(self):
        # TODO: Set zero tensors as adapter weights
        raise NotImplementedError("Disabling adapters not supported currently")

    @classmethod
    def _from_pretrained(cls, pretrained_name_or_path: str, *args, **kwargs):
        # Base class
        model = cls._hf_auto_class.from_pretrained(pretrained_name_or_path, *args, **kwargs)
        return cls(model)

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str, *args, **kwargs):
        """
        Args:
            :pretrained_name_or_path (str): Model card name from huggingface or local path to model directory.
            :*args, **kwargs: Additional arguments to pass to peft.AutoPeftModelForCausalLM.
        """
        if kwargs.get("full_batch_size"):
            raise ValueError("Continuous batching currently not supported for PEFT models")
        if kwargs.get("use_cache") is False:
            warnings.warn("Overriding to use_cache=True")
        kwargs["use_cache"] = True
        obj = cls._from_pretrained(pretrained_name_or_path, *args, **kwargs)
        obj.load_adapter(pretrained_name_or_path, obj.active_adapter)
        return obj

    def transform(self, **kwargs):
        # Base class
        for transform in self._pytorch_transforms:
            self.model, transformed = transform.apply(self.model)
        logger.info("Pytorch transforms applied")

    def _export(
        self,
        example_inputs: Dict[str, torch.Tensor],
        input_names: List[str],
        output_names: List[str],
        dynamic_axes: Dict[str, Dict[int, str]],
        export_kwargs: Dict[str, any] = {},
        onnx_transform_kwargs: Dict[str, any] = {},
        export_dir: Optional[str] = None,
    ) -> str:
        # Base class
        export_dir = Path(export_dir or (QEFF_HOME / self.model_name))
        export_dir = export_dir.with_name(export_dir.name + "-" + self.model_hash)
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
                (example_inputs,),
                str(tmp_onnx_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=13,
                **export_kwargs,
            )
            logger.info("Pytorch export successful")

            model = onnx.load(tmp_onnx_path, load_external_data=False)
            onnx_transform_kwargs = {
                "onnx_base_dir": str(tmp_onnx_dir),
                "model_name": self.model_name,
                **onnx_transform_kwargs,
            }
            for transform in self._onnx_transforms:
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

    def export(self, export_dir: Optional[str] = None) -> str:
        example_shape = (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)
        kv_cache_shape = get_padding_shape_from_config(self.model.config, *example_shape)
        example_inputs = {
            "input_ids": torch.zeros(example_shape, dtype=torch.int64),
            "position_ids": torch.arange(example_shape[1], dtype=torch.int64).view(example_shape),
            "past_key_values": [[] for _ in range(self.num_layers)],
        }
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
        }
        output_names = ["logits"]
        for i in range(self.num_layers):
            for kv in ["key", "value"]:
                example_inputs["past_key_values"][i].append(torch.zeros(kv_cache_shape, dtype=torch.float32))
                dynamic_axes[f"past_{kv}.{i}"] = {0: "batch_size", 2: "ctx_len"}
                output_names.append(f"past_{kv}.{i}_RetainedState")

        input_names = list(dynamic_axes.keys())

        return self._export(
            example_inputs,
            input_names,
            output_names,
            dynamic_axes,
            export_kwargs={"do_constant_folding": False},  # To avoid merging adapter weights with base weights
            onnx_transform_kwargs={"adapter_name": self.model.active_adapter},
            export_dir=export_dir,
        )

    def _compile(
        self,
        onnx_path: Optional[str] = None,
        compile_dir: Optional[str] = None,
        *,
        specializations: Optional[List[Dict[str, int]]] = None,
        custom_io: Optional[Dict[str, str]] = None,
        mdp_ts_num_devices: int = 1,
        **compiler_options,
    ) -> str:
        # Base class
        onnx_path = Path(onnx_path or self.onnx_path)
        compile_dir = Path(compile_dir or onnx_path.parent)
        qpc_path = compile_dir / "qpc"
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX file not found at: {onnx_path}")

        command = ["/opt/qti-aic/exec/qaic-exec", f"-m={onnx_path}", "-aic-hw", "-aic-hw-version=2.0"]
        for key, value in compiler_options.items():
            option = "-" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(option)
                continue
            command.append(f"{option}={value}")
        compile_hash = hashlib.sha256(to_hashable(command))

        # Write specializations.json file
        if specializations is not None:
            specializations_json = compile_dir / "specializations.json"
            with open(specializations_json, "w") as fp:
                json.dump(
                    {"specializations": [{k: str(v) for k, v in spec.items()} for spec in specializations]},
                    fp,
                    indent=4,
                )
            command.append(f"-network-specialization-config={specializations_json}")
            compile_hash.update(to_hashable(specializations))

        # Write custom_io.yaml file
        if custom_io:
            custom_io_yaml = compile_dir / "custom_io.yaml"
            with open(custom_io_yaml, "w") as fp:
                for io_name, dtype in custom_io.items():
                    fp.write(f" - IOName: {io_name}\n   Precision: {dtype}\n\n")
            command.append(f"-custom-IO-list-file={custom_io_yaml}")
            compile_hash.update(to_hashable(custom_io))

        # Write mdp_config.json file
        if mdp_ts_num_devices > 1:
            num_cores = compiler_options.get("aic_num_cores", 16)
            mdp_ts_json = compile_dir / f"mdp_ts_{mdp_ts_num_devices}.json"
            with open(mdp_ts_json, "w") as fp:
                json.dump(
                    {
                        "connections": [{"devices": list(range(mdp_ts_num_devices)), "type": "p2p"}],
                        "partitions": [
                            {
                                "name": "Partition0",
                                "devices": [{"deviceId": d, "numCores": num_cores} for d in range(mdp_ts_num_devices)],
                            }
                        ],
                    },
                    fp,
                    indent=4,
                )
            command.append(f"-mdp-load-partition-config={mdp_ts_json}")
            compile_hash.update(to_hashable({"mdp_ts_num_devices": mdp_ts_num_devices}))

        # Check if already compiled
        compile_hash = compile_hash.hexdigest()[:16]
        qpc_path = qpc_path.with_name(qpc_path.name + "-" + compile_hash)
        if qpc_path.is_dir():
            if (qpc_path / "programqpc.bin").is_file():
                self.qpc_path = qpc_path
                return qpc_path
            # Probably compilation failure last time, delete directory to start over
            shutil.rmtree(qpc_path)

        command.append(f"-aic-binary-dir={qpc_path}")
        logger.info(f"Running compiler: {' '.join(command)}")
        subprocess.run(command).check_returncode()

        self.qpc_path = qpc_path
        return qpc_path

    def compile(
        self,
        onnx_path: Optional[str] = None,
        compile_dir: Optional[str] = None,
        *,
        batch_size: int = 1,
        prefill_seq_len: int,
        ctx_len: int,
        num_devices: int = 1,
        num_cores: int = 16,
        mxfp6_matmul: bool = False,
        mxint8_kv_cache: bool = False,
        **compiler_options,
    ) -> str:
        # Specializations
        specializations = [
            {"batch_size": batch_size, "seq_len": prefill_seq_len, "ctx_len": ctx_len},
            {"batch_size": batch_size, "seq_len": 1, "ctx_len": ctx_len},
        ]

        # Custom IO
        custom_io = {}
        kv_cache_dtype = "mxint8" if mxint8_kv_cache else "float16"
        for suffix in ["", "_RetainedState"]:
            for i in range(self.num_layers):
                for kv in ["key", "value"]:
                    custom_io[f"past_{kv}.{i}{suffix}"] = kv_cache_dtype
            for weight_name in self.adapter_weights[self.active_adapter]:
                custom_io[f"{weight_name}{suffix}"] = "float16"

        return self._compile(
            onnx_path,
            compile_dir,
            compile_only=True,
            retained_state=True,
            specializations=specializations,
            convert_to_fp16=True,
            mxfp6_matmul=mxfp6_matmul,
            custom_io=custom_io,
            mdp_ts_num_devices=num_devices,
            aic_num_cores=num_cores,
            **compiler_options,
        )

    def generate(
        self,
        inputs: Optional[Union[torch.Tensor, np.ndarray]] = None,
        generation_config: Optional[GenerationConfig] = None,
        stopping_criteria: Optional[StoppingCriteria] = None,
        streamer: Optional[BaseStreamer] = None,
        **kwargs,
    ) -> np.ndarray:
        # Initialize session
        if self.qpc_session is None:
            if self.qpc_path is None:
                raise FileNotFoundError("Please compile the model with `model.compile(...)`")
            self.qpc_session = QAICInferenceSession(str(self.qpc_path))

            # Skip buffers
            retained_buffers = [x for x in self.qpc_session.output_names if x.endswith("_RetainedState")]
            self.qpc_session.skip_buffers([x[: -len("_RetainedState")] for x in retained_buffers])
            self.qpc_session.skip_buffers(retained_buffers)

        generation_config = generation_config or self.model.generation_config
        generation_config, model_kwargs = self.model._prepare_generation_config(generation_config, **kwargs)
        if generation_config.do_sample:
            raise NotImplementedError("do_sample=True not supported currently")
        if generation_config.num_beams > 1:
            raise NotImplementedError("num_beams>1 not supported currently")
        if generation_config.max_new_tokens is None or generation_config.max_new_tokens <= 0:
            raise ValueError("Required max_new_tokens>0 value in generation_config")

        stopping_criteria = stopping_criteria or StoppingCriteriaList()
        stopping_criteria = self.model._get_stopping_criteria(generation_config, stopping_criteria)

        if inputs is not None:
            inputs = {"input_ids": inputs}
        else:
            inputs = {}
        inputs.update(model_kwargs)
        inputs = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        batch_size = max(
            [x[self.qpc_session.binding_index_map["input_ids"]][1][0] for x in self.qpc_session.allowed_shapes]
            + [self.qpc_session.bindings[self.qpc_session.binding_index_map["input_ids"]].dims[0]]
        )
        passed_batch_size = inputs["input_ids"].shape[0]
        if passed_batch_size != batch_size:
            raise ValueError(f"Model compiled for batch_size: {batch_size}, but passed batch_size: {passed_batch_size}")

        prefill_seq_len = max(
            [x[self.qpc_session.binding_index_map["input_ids"]][1][1] for x in self.qpc_session.allowed_shapes]
            + [self.qpc_session.bindings[self.qpc_session.binding_index_map["input_ids"]].dims[1]]
        )

        input_len = inputs["input_ids"].shape[1]
        num_chunks = -(input_len // -prefill_seq_len)  # Ceil divide without float
        padded_len = num_chunks * prefill_seq_len  # Convert to a multiple of prompt_len
        inputs["input_ids"] = np.concatenate(
            [inputs["input_ids"], np.zeros((batch_size, padded_len - input_len), dtype=inputs["input_ids"].dtype)], 1
        )
        next_position_ids = inputs.pop("attention_mask").sum(1, keepdims=True)
        inputs["position_ids"] = np.arange(padded_len).reshape(1, -1)
        inputs["position_ids"] = np.where(inputs["position_ids"] < next_position_ids, inputs["position_ids"], -1)
        generated_ids = np.zeros((batch_size, generation_config.max_new_tokens), dtype="int64")
        if streamer:
            streamer.put(inputs["input_ids"][:, :input_len])

        # Set adapter weights
        self.qpc_session.set_buffers(self.adapter_weights[self.active_adapter])

        # Run prefill
        for i in range(num_chunks):
            chunk_inputs = inputs.copy()
            chunk_inputs["input_ids"] = inputs["input_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
            chunk_inputs["position_ids"] = inputs["position_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
            outputs = self.qpc_session.run(chunk_inputs)

        # Get first token
        inputs["input_ids"] = outputs["logits"].argmax(2)
        inputs["position_ids"] = next_position_ids
        generated_ids[:, 0] = inputs["input_ids"].squeeze(1)
        if streamer:
            streamer.put(inputs["input_ids"])

        # Skip adapter weights
        self.qpc_session.skip_buffers(list(self.adapter_weights[self.active_adapter]))

        # Decode loop
        for num_token in range(1, generation_config.max_new_tokens):
            if stopping_criteria(torch.from_numpy(inputs["input_ids"]), torch.from_numpy(outputs["logits"])):
                break

            outputs = self.qpc_session.run(inputs)

            # Prepare inputs for next iteration
            inputs["input_ids"] = outputs["logits"].argmax(2)
            inputs["position_ids"] += 1
            generated_ids[:, num_token] = inputs["input_ids"].squeeze(1)
            if streamer:
                streamer.put(inputs["input_ids"])

        streamer.end()
        return generated_ids
