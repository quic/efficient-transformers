# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import hashlib
import os
from typing import Any, List, Optional, Union

import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedTokenizerFast

import QEfficient
from QEfficient.base.modeling_qeff import QEFFBaseModel, Runtime
from QEfficient.transformers.pytorch_transforms import CBTransform, CustomOpsTransform, KVCacheTransform
from QEfficient.transformers.quantizers.auto import QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING, with_replaced_quantizers
from QEfficient.transformers.quantizers.quant_transforms import AwqToMatmulNbitsTransform, GPTQToMatmulNbitsTransform
from QEfficient.transformers.quantizers.quantizer_awq import QEffAwqConfig
from QEfficient.transformers.quantizers.quantizer_gptq import QEffGPTQConfig
from QEfficient.utils import get_qpc_dir_path, load_hf_tokenizer
from QEfficient.utils.constants import QEFF_MODELS_DIR
from QEfficient.utils.logging_utils import logger

# Dictionary that defines the interface from transformers to be used underneath the QEFF interface
QEFFAutoModelToTransformersAutoModelMap = {
    "QEFFAutoModelForCausalLM": AutoModelForCausalLM,
    "QEFFAutoModel": AutoModel,
}


class QEFFTransformersBase(QEFFBaseModel):
    """
    Parent class for models QEFF provides from transformers i.e. (AutoModel, AutoModelForCausalLM, AutoModelForAudioClassification etc.) from transformers/models/modeling_auto.py file.
    """

    def __init__(self, model: nn.Module, pretrained_model_name_or_path: str, **kwargs) -> None:
        if hasattr(model.config, "quantization_config") and not isinstance(
            model.config.quantization_config, tuple(QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING.values())
        ):
            raise AssertionError("Please use `from_pretrained` method to load quantized models")

        super().__init__(model)
        self.model.config.use_cache = (
            True  # Always pass use_cache = True, to get KV values as output during ONNX export
        )
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        # Set model card name, which is used to decide ONNX, QPC files path during export and compile resp.
        if model_card_name := kwargs.pop("model_card_name", None):
            self.model_card_name = model_card_name
        elif os.path.isdir(self.pretrained_model_name_or_path):
            hash_object = hashlib.sha256()
            hash_object.update(self.pretrained_model_name_or_path.encode("utf-8"))
            self.model_card_name = hash_object.hexdigest()
        else:
            self.model_card_name = self.pretrained_model_name_or_path

        self.full_batch_size = kwargs.get("full_batch_size", None)
        self.kwargs = kwargs
        self._tokenizer = None
        self.is_transformed = False
        if kwargs.get("transform", True):
            self.transform(**kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}\n" + self.model.__repr__()

    @classmethod
    @with_replaced_quantizers
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        """
        This method serves as the easiest entry point into using QEfficient. The interface is designed to be similar to transformers.AutoModelForCausalLM.
        Once the model is initialized, you can use other methods such as export, compile, and generate on the same object.

        Accepts All the parameters that are acceptable by ``transformers.AutoModelForCausalLM``
        There are few additional parameters that this method can take.

        ``Mandatory`` Args:
            :transform (bool): Whether to optimize model for KV retention; default is ``True``. Pass ``False`` to get BertStyle model.
            :model_card_name (str): ``HuggingFace`` model card name or name of the model if custom, used for deciding directory name while saving ``ONNX/qpc`` files.
            :full_batch_size (int): Pass this if you want to execute model with continuous batching.
            Example usage:

        .. code-block:: python

            from QEfficient import QEFFAutoModelForCausalLM

            # Initialize the model using from_pretrained similar to transformers.AutoModelForCausalLM
            model = QEFFAutoModelForCausalLM.from_pretrained("gpt2")

            # Now you can directly compile the model for Cloud AI 100
            model.compile(num_cores=14, device_group=[0])  # Considering you have a Cloud AI 100 Standard SKU

            # You can now execute the model
            model.generate(prompts=["Hi there!!"])

        """
        model_card_name = kwargs.pop(
            "model_card_name", None
        )  # Remove model_card_name from kwargs for transformers APIs

        full_batch_size = kwargs.pop("full_batch_size", None)

        attn_implementation = kwargs.get("attn_implementation", None)
        if attn_implementation != "eager":
            logger.warning(f"Updating attn_implementation to be 'eager', got {attn_implementation}")
            kwargs.update({"attn_implementation": "eager"})

        if low_cpu_mem_usage := kwargs.get("low_cpu_mem_usage", None):
            logger.warning(f"Updating low_cpu_mem_usage to be 'False', got {low_cpu_mem_usage}")
        kwargs.update({"low_cpu_mem_usage": False})

        model = QEFFAutoModelToTransformersAutoModelMap[cls.__name__].from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        return cls(
            model,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            model_card_name=model_card_name,
            full_batch_size=full_batch_size,
            **kwargs,
        )

    @property
    def tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        """Returns the tokenizer for given model based on ``self.pretrained_model_name_or_path``.
        Loads the tokenizer if required.

        Returns:
            :Union[PreTrainedTokenizer, PreTrainedTokenizerFast]: Tokenizer from ``transformers`` for the given model.
        """
        if self._tokenizer is None:
            self._tokenizer = self.get_tokenizer()
        return self._tokenizer

    def get_tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=self.pretrained_model_name_or_path, **self.kwargs)
        return tokenizer


class QEFFAutoModelForCausalLM(QEFFTransformersBase):
    """
    The QEFF class is designed for manipulating any causal language model from the HuggingFace hub.
    Although it is possible to initialize the class directly, we highly recommend using the ``from_pretrained`` method for initialization.
    Please note that the QEFF class is also a part of the ``QEfficient`` module.

    ``Mandatory`` Args:
        :model (nn.Module):  PyTorch model
        :pretrained_model_name_or_path (str): We recommend passing name of the model as input here, as you are not using `from_pretrained` method. This name will be used for deciding path of the ``ONNX/qpc`` files generated during ``export``, ``compilation`` stages.

    .. code-block:: python

        from QEfficient import QEFFAutoModelForCausalLM

    """

    _pytorch_transforms = [CustomOpsTransform, KVCacheTransform]

    def transform(self, **kwargs):
        """
        This method applies all relevant optimization transforms on the model and toggles the ``self.is_transformed`` attribute to True. If the model is already transformed, the method will simply return.
        Please note that this method does not require any input arguments."

        Returns:
            :obj: Same object with transformed ``self.model``
        """
        if self.is_transformed:
            return

        if self.full_batch_size is not None:
            if KVCacheTransform in self._pytorch_transforms:
                self._pytorch_transforms[self._pytorch_transforms.index(KVCacheTransform)] = CBTransform
            if CBTransform not in self._pytorch_transforms:
                raise RuntimeError("please don't update _pytorch_transforms variable")
        else:
            if CBTransform in self._pytorch_transforms:
                self._pytorch_transforms[self._pytorch_transforms.index(CBTransform)] = KVCacheTransform
            if KVCacheTransform not in self._pytorch_transforms:
                raise RuntimeError("Please don't update _pytorch_transforms variable")

        # Update list of pytorch transforms if the model falls in AWQ/GPTQ category
        if hasattr(self.model.config, "quantization_config"):
            if isinstance(self.model.config.quantization_config, QEffAwqConfig):
                self._pytorch_transforms.insert(0, AwqToMatmulNbitsTransform)

            if isinstance(self.model.config.quantization_config, QEffGPTQConfig):
                self._pytorch_transforms.insert(0, GPTQToMatmulNbitsTransform)

        for transform in self._pytorch_transforms:
            transform.apply(self.model)
        self.is_transformed = True

    def execute(self, *args, **kwargs):  # type: ignore
        raise NotImplementedError("Reached too far!!")

    def export(self) -> str:
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
        qpc_base_dir_name = (
            os.path.dirname(qpc_base_dir_name) + "_" + qpc_dir_suffix if qpc_dir_suffix else qpc_base_dir_name
        )
        model_card_dir = os.path.join(QEFF_MODELS_DIR, str(self.model_card_name))
        os.makedirs(model_card_dir, exist_ok=True)
        qpc_dir_path = os.path.join(model_card_dir, qpc_base_dir_name)

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
        return self.qpc_path

    def generate(self, prompts: List[str], device_id: List[int] = None, runtime: str = "AI_100", **kwargs):
        """
        This method generates output until ``eos`` or ``generation_len`` by executing the compiled ``qpc`` on ``Cloud AI 100`` Hardware cards.
        This is a sequential execution based on the ``batch_size`` of the compiled model and the number of prompts passed.
        If the number of prompts cannot be divided by the ``batch_size``, the last unfulfilled batch will be dropped.

        ``Mandatory`` Args:
            :prompts (List[str]): List of prompts to run the execution.
            :device_id (List[int]): Ids of devices for running the qpc pass as [0] in case of normal model / [0, 1, 2, 3] in case of tensor slicing model
        ``optional`` Args:
            :runtime (str, optional): Only ``AI_100`` runtime is supported as of now; ``ONNXRT`` and ``PyTorch`` coming soon. Defaults to "AI_100".
        """
        if Runtime(runtime) != Runtime.AI_100:
            raise ValueError("Only AI_100 runtime is supported right now via generate API")
        self.run_cloud_ai_100(prompts=prompts, device_id=device_id, **kwargs)

    def run_cloud_ai_100(self, prompts: List[str],  device_id: List[int] = None, **kwargs):
        if not isinstance(self.qpc_path, str):
            raise TypeError("Please run compile API first!")
        if isinstance(self.device_id, type(None)):
            raise TypeError("please pass valid device_id as input argument")
        generation_len = kwargs.pop("generation_len", None)
        return QEfficient.cloud_ai_100_exec_kv(
            self.tokenizer,
            self.qpc_path,
            prompt=prompts,
            device_id=device_id,
            generation_len=generation_len,
            full_batch_size=self.full_batch_size,
        )


class QEffAutoModel(QEFFTransformersBase):
    def execute(self, *args, **kwargs):  # type: ignore
        raise NotImplementedError("Reached too far!!")

    def export(self):
        raise NotImplementedError("Reached too far!!")

    def compile(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Reached too far!!")
