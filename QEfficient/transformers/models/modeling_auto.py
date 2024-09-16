# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM

from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.base.onnx_transforms import FP16ClipTransform, SplitTensorsTransform
from QEfficient.transformers.pytorch_transforms import CBTransform, CustomOpsTransform, KVCacheTransform
from QEfficient.transformers.quantizers.auto import QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING, with_replaced_quantizers
from QEfficient.transformers.quantizers.quant_transforms import AwqToMatmulNbitsTransform, GPTQToMatmulNbitsTransform
from QEfficient.utils import constants, get_padding_shape_from_config

logger = logging.getLogger(__file__)


class QEFFTransformersBase(QEFFBaseModel):
    """
    Parent class for models QEFF provides from transformers i.e. (AutoModel, AutoModelForCausalLM, AutoModelForAudioClassification etc.) from transformers/models/modeling_auto.py file.
    """

    _hf_auto_class: type

    def __init__(self, model: nn.Module) -> None:
        if hasattr(model.config, "quantization_config") and not isinstance(
            model.config.quantization_config, tuple(QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING.values())
        ):
            raise AssertionError("Please use `from_pretrained` method to load quantized models")

        super().__init__(model)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "\n" + self.model.__repr__()

    @classmethod
    @with_replaced_quantizers
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        if kwargs.get("attn_implementation", None) not in {None, "eager"}:
            logger.warning('Updating attn_implementation="eager"')

        if kwargs.get("low_cpu_mem_usage", None):
            logger.warning("Updating low_cpu_mem_usage=False")

        kwargs.update({"attn_implementation": "eager", "low_cpu_mem_usage": False})

        model = cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return cls(model)


class QEFFAutoModelForCausalLM(QEFFTransformersBase):
    """
    The QEFF class is designed for manipulating any causal language model from the HuggingFace hub.
    Although it is possible to initialize the class directly, we highly recommend using the ``from_pretrained`` method for initialization.

    ``Mandatory`` Args:
        :model (nn.Module):  PyTorch model

    .. code-block:: python

        from QEfficient import QEFFAutoModelForCausalLM

        model = QEFFAutoModelForCausalLM.from_pretrained(model_name, num_hidden_layers=2)
        model.compile(prefill_seq_len=32, ctx_len=1024)

        inputs = ...
        outputs = model.generate(**inputs, max_new_tokens=128)
    """

    _hf_auto_class = AutoModelForCausalLM
    _pytorch_transforms = [AwqToMatmulNbitsTransform, GPTQToMatmulNbitsTransform, CustomOpsTransform, KVCacheTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(self, model: nn.Module):
        # Set use_cache=True to get KV values as output during ONNX export
        self.model.config.use_cache = True
        super().__init__(model)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        This method serves as the easiest entry point into using QEfficient. The interface is designed to be similar to transformers.AutoModelForCausalLM.
        Once the model is initialized, you can use other methods such as export, compile, and generate on the same object.

        Args:
            :pretrained_name_or_path (str): Model card name from huggingface or local path to model directory.
            :args, kwargs: Additional arguments to pass to transformers.AutoModelForCausalLM.
        """
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

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


class QEFFAutoModelForCausalLMwithCB(QEFFAutoModelForCausalLM):
    _pytorch_transforms = [AwqToMatmulNbitsTransform, GPTQToMatmulNbitsTransform, CustomOpsTransform, CBTransform]


class QEffAutoModel(QEFFTransformersBase):
    _hf_auto_class = AutoModel
    _pytorch_transforms = [AwqToMatmulNbitsTransform, GPTQToMatmulNbitsTransform, CustomOpsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def export(self):
        raise NotImplementedError("Reached too far!!")

    def compile(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Reached too far!!")
