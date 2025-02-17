# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import hashlib
import logging
import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedTokenizerFast

import QEfficient
from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.base.onnx_transforms import FP16ClipTransform, SplitTensorsTransform
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.models.pytorch_transforms import CustomOpsTransform, KVCacheTransform, SpDTransform
from QEfficient.transformers.quantizers.auto import QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING, with_replaced_quantizers
from QEfficient.transformers.quantizers.quant_transforms import AwqToMatmulNbitsTransform, GPTQToMatmulNbitsTransform
from QEfficient.utils import constants, get_padding_shape_from_config
from QEfficient.utils.cache import to_hashable

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
    def from_pretrained(cls, pretrained_model_name_or_path: str, is_tlm: bool = False, *args, **kwargs):
        if kwargs.get("attn_implementation", None) not in {None, "eager"}:
            logger.warning('Updating attn_implementation="eager"')

        if kwargs.get("low_cpu_mem_usage", None):
            logger.warning("Updating low_cpu_mem_usage=False")

        kwargs.update({"attn_implementation": "eager", "low_cpu_mem_usage": False})

        model = cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return cls(model, is_tlm=is_tlm)

    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname


class QEFFAutoModel(QEFFTransformersBase):
    """
    The QEFFAutoModel class is designed for manipulating any transformer model from the HuggingFace hub.
    Although it is possible to initialize the class directly, we highly recommend using the ``from_pretrained`` method for initialization.

    ``Mandatory`` Args:
        :model (nn.Module): PyTorch model

    .. code-block:: python

        from QEfficient import QEFFAutoModel
        from transformers import AutoTokenizer

        # Initialize the model using from_pretrained similar to transformers.AutoModel.
        model = QEFFAutoModel.from_pretrained("model_name")

        # Now you can directly compile the model for Cloud AI 100
        model.compile(num_cores=16)  # Considering you have a Cloud AI 100 SKU

        #prepare input
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        inputs = tokenizer("My name is", return_tensors="pt")

        # You can now execute the model
        model.generate(inputs)
    """

    _hf_auto_class = AutoModel
    _pytorch_transforms = [CustomOpsTransform, AwqToMatmulNbitsTransform, GPTQToMatmulNbitsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model)
        self.model.config.use_cache = True
        self.num_layers = model.config.num_hidden_layers

    @classmethod
    @with_replaced_quantizers
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        This method serves as the easiest entry point into using QEfficient. The interface is designed to be similar to transformers.AutoModel.
        Once the model is initialized, you can use other methods such as export, compile, and generate on the same object.

        Args:
            :pretrained_name_or_path (str): Model card name from HuggingFace or local path to model directory.
            :args, kwargs: Additional arguments to pass to transformers.AutoModel.

        .. code-block:: python

            from QEfficient import QEFFAutoModel
            from transformers import AutoTokenizer

            # Initialize the model using from_pretrained similar to transformers.AutoModel.
            model = QEFFAutoModel.from_pretrained("model_name")

            # Now you can directly compile the model for Cloud AI 100
            model.compile(num_cores=16)  # Considering you have a Cloud AI 100 SKU

            #prepare input
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            inputs = tokenizer("My name is", return_tensors="pt")

            # You can now execute the model
            model.generate(inputs)
        """
        if kwargs.get("attn_implementation", None) not in {None, "eager"}:
            logger.warning('Updating attn_implementation="eager"')

        if kwargs.get("low_cpu_mem_usage", None):
            logger.warning("Updating low_cpu_mem_usage=False")

        kwargs.update({"attn_implementation": "eager", "low_cpu_mem_usage": False, "add_pooling_layer": False})
        try:
            model = cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
            warnings.warn("Removing pooling layer from the model if exist")
        except TypeError:
            kwargs.pop("add_pooling_layer", None)
            model = cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return cls(model)

    @property
    def model_hash(self) -> str:
        # NOTE: model_config.to_diff_dict() has "_name_or_path" attribute which is the model card name or path.
        # Using same card name will result in same hash. But, using a relative path for one run and
        # absolute path for another run will result in different hash.
        # The added complexity to resolve different paths to same location is not worth pursuing.
        # Instead, advise the user to always provide same relative paths or absolute paths for local models.

        # Compute the hash with: model_config, transforms
        mhash = hashlib.sha256()
        mhash.update(to_hashable(self.model.config.to_diff_dict()))
        mhash.update(to_hashable(self._transform_names()))
        mhash = mhash.hexdigest()[:16]
        return mhash

    def export(self, export_dir: Optional[str] = None) -> str:
        """
        Exports the model to ``ONNX`` format using ``torch.onnx.export``.

        ``Optional`` Args:
           :export_dir (str, optional): The directory path to store ONNX-graph.

        Returns:
            :str: Path of the generated ``ONNX`` graph.
        """
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        seq_len = constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN

        example_inputs = {
            "input_ids": torch.zeros((bs, seq_len), dtype=torch.int64),
            "attention_mask": torch.ones((bs, seq_len), dtype=torch.int64),
        }

        dynamic_axes = {"input_ids": {0: "batch_size", 1: "seq_len"}, "attention_mask": {0: "batch_size", 1: "seq_len"}}

        output_names = ["output"]

        return self._export(
            example_inputs,
            output_names,
            dynamic_axes,
            export_dir=export_dir,
        )

    def compile(
        self,
        onnx_path: Optional[str] = None,
        compile_dir: Optional[str] = None,
        *,
        seq_len: int = 32,
        batch_size: int = 1,
        num_devices: int = 1,
        num_cores: int = 16,  # FIXME: Make this mandatory arg
        mxfp6_matmul: bool = False,
        **compiler_options,
    ) -> str:
        """
        This method compiles the exported ``ONNX`` model using the Cloud AI 100 Platform SDK compiler binary found at ``/opt/qti-aic/exec/qaic-exec`` and generates a ``qpc`` package.
        If the model has not been exported yet, this method will handle the export process.
        You can pass any other arguments that the `qaic-exec` takes as extra kwargs.

        ``Optional`` Args:
            :onnx_path (str, optional): Path to pre-exported onnx model.
            :compile_dir (str, optional): Path for saving the qpc generated.
            :seq_len (int, optional): The length of the prompt should be less that ``seq_len``. ``Defaults to 32``.
            :batch_size (int, optional): Batch size. ``Defaults to 1``.
            :num_devices (int): Number of devices the model needs to be compiled for. Defaults to 1.
            :num_cores (int): Number of cores used to compile the model.
            :mxfp6_matmul (bool, optional): Whether to use ``mxfp6`` compression for weights. ``Defaults to False``.
            :aic_enable_depth_first (bool, optional): Enables DFS with default memory size. ``Defaults to False``.
            :allow_mxint8_mdp_io (bool, optional): Allows MXINT8 compression of MDP IO traffic. ``Defaults to False.``
        Returns:
            :str: Path of the compiled ``qpc`` package.
        """

        specializations = [
            {"batch_size": batch_size, "seq_len": seq_len},
        ]

        return self._compile(
            onnx_path,
            compile_dir,
            compile_only=True,
            specializations=specializations,
            convert_to_fp16=True,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=num_devices,
            aic_num_cores=num_cores,
            **compiler_options,
        )

    def generate(
        self,
        inputs: torch.Tensor,
        device_ids: List[int] = None,
        runtime_ai100: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        This method generates output by executing PyTorch runtime or the compiled ``qpc`` on ``Cloud AI 100`` Hardware cards.
        ``Mandatory`` Args:
            :inputs (Union[torch.Tensor, np.ndarray]): inputs to run the execution.
        ``optional`` Args:
            :device_id (List[int]): Ids of devices for running the qpc pass as [0] in case of normal model / [0, 1, 2, 3] in case of tensor slicing model
            :runtime_ai100 (bool, optional): ``AI_100`` and ``PyTorch`` runtime is supported as of now. Defaults to ``True`` for ``AI_100`` runtime.
        Returns:
            :dict: Output from the ``AI_100`` or ``PyTorch`` runtime.
        """
        # AI_100 runtime
        if runtime_ai100:
            if not isinstance(self.qpc_path, Path):
                raise TypeError("Please run compile API first!")

            return self.cloud_ai_100_feature_generate(inputs=inputs, device_ids=device_ids)
        # PyTorch runtime
        else:
            return self.pytorch_feature_generate(model=self.model, inputs=inputs)

    def cloud_ai_100_feature_generate(
        self,
        inputs: torch.Tensor,
        device_ids: List[int] = [0],
    ) -> np.ndarray:
        """
        Generates features with list of prompts using AI 100 runtime.

        ``Mandatory`` Args:
            :inputs (Union[torch.Tensor, np.ndarray]): inputs to run the execution.
        ``Optional`` Args:
            device_ids (List[int], optional): A list of device IDs to use for the session. Defaults to [0].

        Returns:
           np.ndarray: A list of dictionaries containing the generated output features.
        """

        if self.qpc_session is None:
            self.qpc_session = QAICInferenceSession(str(self.qpc_path), device_ids)
            self.batch_size = self.qpc_session.bindings[0].dims[0]
            self.seq_len = self.qpc_session.bindings[0].dims[1]
        # Prepare input
        input_ids_len = inputs["input_ids"].shape[1]
        input_ids = np.array(
            torch.nn.functional.pad(inputs["input_ids"], (0, self.seq_len - inputs["input_ids"].size(1)), "constant", 0)
        )
        attention_mask = np.array(
            torch.nn.functional.pad(
                inputs["attention_mask"], (0, self.seq_len - inputs["attention_mask"].size(1)), "constant", 0
            )
        )

        inputs = dict(input_ids=input_ids, attention_mask=attention_mask)

        outputs = {
            "output": np.random.randn(self.batch_size, self.seq_len, self.qpc_session.bindings[2].dims[2]).astype(
                np.float32
            ),
        }
        self.qpc_session.set_buffers(outputs)
        outputs = self.qpc_session.run(inputs)
        outputs = outputs["output"][:, :input_ids_len, :]
        return outputs

    def pytorch_feature_generate(self, model, inputs: Union[torch.Tensor, np.ndarray]) -> List[torch.Tensor]:
        """
        Generates features from a list of text prompts using a PyTorch model.

        ``Mandatory`` Args:
            :model: The transformed PyTorch model used for generating features.
            :inputs (Union[torch.Tensor, np.ndarray]): inputs to run the execution.

        Returns:
            torch.Tensor: A list of output features generated by the model for each prompt.
        """
        return model(**inputs)
