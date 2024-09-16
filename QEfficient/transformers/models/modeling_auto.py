# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import os
from typing import Any, List

import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM

import QEfficient
from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.base.onnx_transforms import FP16ClipTransform, SplitTensorsTransform
from QEfficient.transformers.pytorch_transforms import CBTransform, CustomOpsTransform, KVCacheTransform
from QEfficient.transformers.quantizers.auto import QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING, with_replaced_quantizers
from QEfficient.transformers.quantizers.quant_transforms import AwqToMatmulNbitsTransform, GPTQToMatmulNbitsTransform
from QEfficient.utils import get_qpc_dir_path
from QEfficient.utils.logging_utils import logger


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
        self.model.config.use_cache = (
            True  # Always pass use_cache = True, to get KV values as output during ONNX export
        )

    def __repr__(self) -> str:
        return self.__class__.__name__ + "\n" + self.model.__repr__()

    @classmethod
    @with_replaced_quantizers
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        """
        This method serves as the easiest entry point into using QEfficient. The interface is designed to be similar to transformers.AutoModelForCausalLM.
        Once the model is initialized, you can use other methods such as export, compile, and generate on the same object.

        Accepts All the parameters that are acceptable by ``transformers.AutoModelForCausalLM``

        ``Mandatory`` Args:
            :pretrained_model_name_or_path (str): Model card name from huggingface or local path to model directory.

        .. code-block:: python

            from QEfficient import QEFFAutoModelForCausalLM

            # Initialize the model using from_pretrained similar to transformers.AutoModelForCausalLM
            model = QEFFAutoModelForCausalLM.from_pretrained("gpt2")

            # Now you can directly compile the model for Cloud AI 100
            model.compile(num_cores=14, device_group=[0])  # Considering you have a Cloud AI 100 Standard SKU

            # You can now execute the model
            model.generate(prompts=["Hi there!!"])

        """
        attn_implementation = kwargs.get("attn_implementation", None)
        if attn_implementation is not None and attn_implementation != "eager":
            logger.warning('Updating attn_implementation="eager"')
        kwargs.update({"attn_implementation": "eager"})

        if low_cpu_mem_usage := kwargs.get("low_cpu_mem_usage", None):
            logger.warning(f"Updating low_cpu_mem_usage to be 'False', got {low_cpu_mem_usage}")
        kwargs.update({"low_cpu_mem_usage": False})

        model = cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return cls(model)


class QEFFAutoModelForCausalLM(QEFFTransformersBase):
    """
    The QEFF class is designed for manipulating any causal language model from the HuggingFace hub.
    Although it is possible to initialize the class directly, we highly recommend using the ``from_pretrained`` method for initialization.
    Please note that the QEFF class is also a part of the ``QEfficient`` module.

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

    def export(self) -> str:
        """
        Exports the model to ``ONNX`` format using ``torch.onnx.export``.

        Returns:
            :str: Path of the generated ``ONNX`` graph.
        """
        # Export
        _, onnx_model_path = QEfficient.export(
            model_name=self.model_card_name,
            model_kv=self,
            tokenizer=self.tokenizer,
            full_batch_size=self.full_batch_size,
        )
        self.onnx_path = onnx_model_path

        return self.onnx_path

    def compile(
        self,
        num_cores: int,
        device_group: List[int] = None,
        batch_size: int = 1,
        prompt_len: int = 32,
        ctx_len: int = 128,
        mxfp6: bool = True,
        mxint8: bool = False,
        mos: int = -1,
        aic_enable_depth_first: bool = False,
    ) -> str:
        """
        This method compiles the exported ``ONNX`` model using the Cloud AI 100 Platform SDK compiler binary found at ``/opt/qti-aic/exec/qaic-exec`` and generates a ``qpc`` package.
        If the model has not been exported yet, this method will handle the export process.
        The generated ``qpc`` can be found under the directory ``efficient-transformers/qeff_models/{self.model_card_name}/qpc``.

        ``Mandatory`` Args:
            :num_cores (int): Number of cores used to compile the model.
            :device_group (List[int]): If this is a list of more that one integers, tensor-slicing is invoked, defaults to None, and automatically chooses suitable device.
        ``Optional`` Args:
            :model_card_name (Optional[str], optional): Name of the model, Mandatory if ``self.pretrained_model_name_or_path`` is a path. ``Defaults to None``.
            :batch_size (int, optional): Batch size. ``Defaults to 1``.
            :prompt_len (int, optional): The length of the Prefill prompt should be less that ``prompt_len``. ``Defaults to 32``.
            :ctx_len (int, optional): Maximum ``ctx`` that the compiled model can remember. ``Defaults to 128``.
            :mxfp6 (bool, optional): Whether to use ``mxfp6`` compression for weights. ``Defaults to True``.
            :mxint8 (bool, optional): Whether to use ``mxint8`` compression for KV cache. ``Defaults to False``.
            :mos (int, optional): Effort level to reduce on-chip memory. Defaults to -1, meaning no effort. ``Defaults to -1``.
            :aic_enable_depth_first (bool, optional): Enables DFS with default memory size. ``Defaults to False``.

        Returns:
            :str: Path of the compiled ``qpc`` package.
        """
        # Export first if self.ort_runtime_args are not populated
        if self.onnx_path is None:
            logger.info(f"Exporting the {self.model.__class__.__name__} model to ONNX for compilation!")
            self.export()

        # Prepare qpc dir path
        qpc_dir_path = get_qpc_dir_path(
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

        # Compile
        QEfficient.compile(
            onnx_path=self.onnx_path,
            qpc_path=os.path.dirname(qpc_dir_path),
            num_cores=num_cores,
            device_group=device_group,
            aic_enable_depth_first=aic_enable_depth_first,
            mos=mos,
            batch_size=batch_size,
            prompt_len=prompt_len,
            ctx_len=ctx_len,
            mxfp6=mxfp6,
            mxint8=mxint8,
            full_batch_size=self.full_batch_size,
        )
        self.qpc_path = qpc_dir_path
        return self.qpc_path

    def generate(self, prompts: List[str], device_id: List[int] = None, **kwargs):
        """
        This method generates output until ``eos`` or ``generation_len`` by executing the compiled ``qpc`` on ``Cloud AI 100`` Hardware cards.
        This is a sequential execution based on the ``batch_size`` of the compiled model and the number of prompts passed.
        If the number of prompts cannot be divided by the ``batch_size``, the last unfulfilled batch will be dropped.

        ``Mandatory`` Args:
            :prompts (List[str]): List of prompts to run the execution.
            :device_id (List[int]): Ids of devices for running the qpc pass as [0] in case of normal model / [0, 1, 2, 3] in case of tensor slicing model
        """
        self.run_cloud_ai_100(prompts=prompts, device_id=device_id, **kwargs)

    def run_cloud_ai_100(self, prompts: List[str], device_id: List[int] = None, **kwargs):
        if not isinstance(self.qpc_path, str):
            raise TypeError("Please run compile API first!")
        generation_len = kwargs.pop("generation_len", None)
        return QEfficient.cloud_ai_100_exec_kv(
            self.tokenizer,
            self.qpc_path,
            prompt=prompts,
            device_id=device_id,
            generation_len=generation_len,
            full_batch_size=self.full_batch_size,
        )


class QEFFAutoModelForCausalLMwithCB(QEFFAutoModelForCausalLM):
    _pytorch_transforms = [AwqToMatmulNbitsTransform, GPTQToMatmulNbitsTransform, CustomOpsTransform, CBTransform]


class QEffAutoModel(QEFFTransformersBase):
    _hf_auto_class = AutoModel
    _pytorch_transforms = [AwqToMatmulNbitsTransform, GPTQToMatmulNbitsTransform, CustomOpsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def execute(self, *args, **kwargs):  # type: ignore
        raise NotImplementedError("Reached too far!!")

    def export(self):
        raise NotImplementedError("Reached too far!!")

    def compile(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Reached too far!!")
