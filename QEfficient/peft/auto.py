# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import hashlib
import logging
import warnings
from typing import List, Optional, Union

import numpy as np
import torch
from peft import AutoPeftModelForCausalLM, PeftConfig, PeftModelForCausalLM, load_peft_weights
from torch import nn
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer

from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.base.onnx_transforms import BaseOnnxTransform, OnnxTransform
from QEfficient.base.pytorch_transforms import PytorchTransform
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.peft.lora import QEffAutoLoraModelForCausalLM
from QEfficient.peft.onnx_transforms import AdapterWeightsToInputsTransform
from QEfficient.peft.pytorch_transforms import PeftModelInputsTransform
from QEfficient.transformers.models.pytorch_transforms import CustomOpsTransform, KVCacheTransform
from QEfficient.utils import constants
from QEfficient.utils._utils import get_padding_shape_from_config
from QEfficient.utils.hash_utils import to_hashable

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
    _onnx_transforms: List[BaseOnnxTransform] = [OnnxTransform, AdapterWeightsToInputsTransform]
    _hf_auto_class = AutoPeftModelForCausalLM

    def __init__(self, model: nn.Module):
        if not isinstance(model, PeftModelForCausalLM):
            raise TypeError(f"Required pytorch module of type PeftModel, got {type(model)}")

        if model.active_peft_config.peft_type != "LORA":
            raise NotImplementedError("Only LoRA models are supported")

        super().__init__(model)

        self.num_layers = model.config.num_hidden_layers
        self.exported_peft_config = None
        self.adapter_weights = {
            adapter_name: {
                name.replace(f".{adapter_name}.weight", ".weight"): param.detach().numpy().astype("float16")
                for name, param in model.named_parameters()
                if name.endswith(f".{adapter_name}.weight")
            }
            for adapter_name in model.peft_config
        }

    def __repr__(self) -> str:
        return self.__class__.__name__ + "\n" + self.model.__repr__()

    @property
    def model_name(self) -> str:
        mname = self.model.get_base_model().__class__.__name__ + "-lora"
        if mname.startswith("QEff"):
            mname = mname[4:]
        return mname

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
        mhash.update(to_hashable(self._transform_names()))
        mhash = mhash.hexdigest()[:16]
        return mhash

    @property
    def get_model_config(self) -> dict:
        return self.model.get_base_model().config.__dict__

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
        if self.exported_peft_config is not None and self.exported_peft_config != self.model.peft_config[adapter_name]:
            raise ValueError(
                "Unable to activate incompatible adapter. "
                "Use an adapter compatible with export-time adapter "
                "or re-export with this adapter"
            )
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
            :finite_adapters (bool): set True to enable finite adapter mode with QEffAutoLoraModelForCausalLM class. Please refer to QEffAutoLoraModelForCausalLM for API specification.
            :adapter_name (str): Name used to identify loaded adapter.
            :args, kwargs: Additional arguments to pass to peft.AutoPeftModelForCausalLM.
        """
        if kwargs.get("full_batch_size"):
            raise NotImplementedError("Continuous batching currently not supported for PEFT models")
        if kwargs.get("use_cache") is False:
            warnings.warn("Overriding to use_cache=True")
        kwargs["use_cache"] = True

        if kwargs.pop("finite_adapters", False):  # initialize through finite_adapters class
            obj = QEffAutoLoraModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=PeftConfig.from_pretrained(
                    pretrained_name_or_path
                ).base_model_name_or_path,
                **kwargs,
            )
            if adapter_name := kwargs.pop("adapter_name", None):
                obj.load_adapter(pretrained_name_or_path, adapter_name=adapter_name)
                return obj
            if len(args) == 0 or not isinstance(list(args)[0], str):
                raise TypeError("Required adapter name argument in string format")
            obj.load_adapter(pretrained_name_or_path, list(args)[0])
        else:
            obj = cls._from_pretrained(pretrained_name_or_path, *args, **kwargs)
        return obj

    def export(self, export_dir: Optional[str] = None) -> str:
        self.exported_peft_config = self.model.active_peft_config

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

        return self._export(
            example_inputs,
            output_names,
            dynamic_axes,
            export_kwargs={"do_constant_folding": False},  # To avoid merging adapter weights with base weights
            onnx_transform_kwargs={"adapter_name": self.model.active_adapter},
            export_dir=export_dir,
        )

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
            mxint8_kv_cache=mxint8_kv_cache,
            **compiler_options,
        )

    def generate(
        self,
        inputs: Optional[Union[torch.Tensor, np.ndarray]] = None,
        device_ids: Optional[List[int]] = None,
        generation_config: Optional[GenerationConfig] = None,
        stopping_criteria: Optional[StoppingCriteria] = None,
        streamer: Optional[BaseStreamer] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate tokens from compiled binary. This method takes same parameters as HuggingFace transformers model.generate() method.

        Args:
            :inputs: input_ids
            :generation_config: Merge this generation_config with model-specific for the current generation.
            :stopping_criteria: Pass custom stopping_criteria to stop at a specific point in generation.
            :streamer: Streamer to put the generated tokens into.
            :kwargs: Additional parameters for generation_config or to be passed to the model while generating.
        """
        # Initialize session
        if self.qpc_session is None:
            if self.qpc_path is None:
                raise FileNotFoundError("Please compile the model with `model.compile(...)`")
            self.qpc_session = QAICInferenceSession(str(self.qpc_path), device_ids)

            # Skip buffers
            retained_buffers = [x for x in self.qpc_session.output_names if x.endswith("_RetainedState")]
            self.qpc_session.skip_buffers([x[: -len("_RetainedState")] for x in retained_buffers])
            self.qpc_session.skip_buffers(retained_buffers)

        generation_config = generation_config or self.model.generation_config
        generation_config, model_kwargs = self.model._prepare_generation_config(generation_config, **kwargs)
        self.model._prepare_special_tokens(generation_config, device="cpu")
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
            if stopping_criteria(torch.from_numpy(inputs["input_ids"]), torch.from_numpy(outputs["logits"])).all():
                break

            outputs = self.qpc_session.run(inputs)

            # Prepare inputs for next iteration
            inputs["input_ids"] = outputs["logits"].argmax(2)
            inputs["position_ids"] += 1
            generated_ids[:, num_token] = inputs["input_ids"].squeeze(1)
            if streamer:
                streamer.put(inputs["input_ids"])

        if streamer:
            streamer.end()
        return generated_ids
