# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import os
from typing import Any, List, Optional, Union

import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedTokenizerFast

import QEfficient
from QEfficient.base.modeling_qeff import QEFFBaseModel, Runtime
from QEfficient.transformers.pytorch_transforms import CBTransform, CustomOpsTransform, KVCacheTransform
from QEfficient.transformers.quantizers.quant_transforms import AwqToOnnxTransform
from QEfficient.transformers.quantizers.quantizer_awq import QEffAwqConfig, replace_transformers_quantizers
from QEfficient.utils import get_qpc_dir_path, load_hf_tokenizer
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
        super().__init__(model)
        self.model.config.use_cache = (
            True  # Always pass use_cache = True, to get KV values as output during ONNX export
        )
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        # Set model card name, which is used to decide ONNX, QPC files path during export and compile resp.
        model_card_name = kwargs.pop("model_card_name", None)
        self.model_card_name = (
            model_card_name
            if model_card_name
            else (self.pretrained_model_name_or_path if not os.path.isdir(self.pretrained_model_name_or_path) else None)
        )
        self.kwargs = kwargs
        self._tokenizer = None
        self.is_transformed = False
        if kwargs.get("transform", True):
            self.transform(**kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}\n" + self.model.__repr__()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        """
        This method serves as the easiest entry point into using QEfficient. The interface is designed to be similar to transformers.AutoModelForCausalLM.
        Once the model is initialized, you can use other methods such as export, compile, and generate on the same object.

        Accepts All the parameters that are acceptable by ``transformers.AutoModelForCausalLM``
        There are few additional parameters that this method can take.

        ``Mandatory`` Args:
            :transform (bool): Whether to optimize model for KV retention; default is ``True``. Pass ``False`` to get BertStyle model.
            :model_card_name (str): ``HuggingFace`` model card name or name of the model if custom, used for deciding directory name while saving ``ONNX/qpc`` files.

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

        low_cpu_mem_usage = kwargs.get("low_cpu_mem_usage", None)
        if low_cpu_mem_usage:
            logger.warning(f"Updating low_cpu_mem_usage to be 'False', got {low_cpu_mem_usage}")
        kwargs.update({"low_cpu_mem_usage": False})
        replace_transformers_quantizers()
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

        if kwargs.get("full_batch_size", None):
            self._pytorch_transforms.remove(KVCacheTransform)
            self._pytorch_transforms.append(CBTransform)
        if isinstance(self.model.config.quantization_config, QEffAwqConfig):
            self._pytorch_transforms.insert(0, AwqToOnnxTransform)

        for transform in self._pytorch_transforms:
            transform.apply(self.model)
        self.is_transformed = True

    def execute(self, *args, **kwargs):  # type: ignore
        raise NotImplementedError("Reached too far!!")

    def export(self, model_card_name: Optional[str] = None) -> str:
        """
        Exports the model to ``ONNX`` format using ``torch.onnx.export``.
        The model should already be transformed i.e. ``self.is_transformed`` should be ``True``.
        Otherwise, this will raise an ``AssertionError``.
        We currently don't support exporting non-transformed models. Please refer to the ``convert_to_cloud_bertstyle`` function in the **Low-Level API** for a legacy function that supports this."

        ``Optional`` Args:
            :model_card_name (Optional[str]): Name of the model card. Mandatory when model is initialized with path for ``pretrained_model_name_or_path`` argument during initialization. ``Defaults to None.``

        Raises:
            :AttributeError: If ``pretrained_model_name_or_path`` is a path, this function needs model card name of the model so that it can distinguish between directories while saving the ``ONNX`` files generated. So, user needs to pass ``model_card_name`` as a valid ``string`` in that case, Otherwise this will raise the error.

        Returns:
            :str: Path of the generated ``ONNX`` graph.
        """
        assert self.is_transformed, "Please first run transform on the QEFFAutoModelForCausalLM object"

        # Make sure model_card_name is available for export
        if self.model_card_name is None and model_card_name is None:
            raise AttributeError("Please pass model_card_name as valid string input")
        elif model_card_name is not None:
            self.model_card_name = model_card_name

        # Export
        _, onnx_model_path = QEfficient.export(model_name=self.model_card_name, model_kv=self, tokenizer=self.tokenizer)
        self.onnx_path = onnx_model_path

        return self.onnx_path

    def compile(
        self,
        num_cores: int,
        device_group: List[int],
        model_card_name: Optional[str] = None,
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
            :device_group (List[int]): If this is a list of more that one integers, tensor-slicing is invoked.
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
            self.export(model_card_name=model_card_name)

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
            full_batch_size=None,
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
        )
        self.qpc_path = qpc_dir_path
        self.device_id = device_group

        return self.qpc_path

    def generate(self, prompts: List[str], runtime: str = "AI_100", **kwargs):
        """
        This method generates output until ``eos`` or ``generation_len`` by executing the compiled ``qpc`` on ``Cloud AI 100`` Hardware cards.
        This is a sequential execution based on the ``batch_size`` of the compiled model and the number of prompts passed.
        If the number of prompts cannot be divided by the ``batch_size``, the last unfulfilled batch will be dropped.

        ``Mandatory`` Args:
            :prompts (List[str]): List of prompts to run the execution.
        ``optional`` Args:
            :runtime (str, optional): Only ``AI_100`` runtime is supported as of now; ``ONNXRT`` and ``PyTorch`` coming soon. Defaults to "AI_100".
        """
        assert Runtime(runtime) == Runtime.AI_100, "Only AI_100 runtime is supported right now via generate API"
        self.run_cloud_ai_100(prompts=prompts, **kwargs)

    def run_cloud_ai_100(self, prompts: List[str], **kwargs):
        assert isinstance(self.qpc_path, str), "Please run compile API first!"
        assert (
            self.device_id is not None
        ), "please pass valid device_id as input argument"  # FIXME: replace with isinstance
        generation_len = kwargs.pop("generation_len", None)
        return QEfficient.cloud_ai_100_exec_kv(
            self.tokenizer, self.qpc_path, prompt=prompts, device_id=self.device_id, generation_len=generation_len
        )


class QEffAutoModel(QEFFTransformersBase):
    def execute(self, *args, **kwargs):  # type: ignore
        raise NotImplementedError("Reached too far!!")

    def export(self):
        raise NotImplementedError("Reached too far!!")

    def compile(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Reached too far!!")
