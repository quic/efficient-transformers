# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import hashlib
import warnings
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSpeechSeq2Seq,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    TextStreamer,
)

import QEfficient
from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.base.onnx_transforms import FP16ClipTransform, SplitTensorsTransform
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.generation.text_generation_inference import (
    CloudAI100ExecInfoNew,
    PerfMetrics,
    calculate_latency,
    get_compilation_dims,
)
from QEfficient.transformers.models.pytorch_transforms import (
    CustomOpsTransform,
    KVCacheModuleMethodMapperTransform,
    KVCacheTransform,
    SpDTransform,
    VlmKVOffloadTransform,
    VlmNoKVOffloadTransform,
)
from QEfficient.transformers.quantizers.auto import QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING, with_replaced_quantizers
from QEfficient.transformers.quantizers.quant_transforms import (
    AwqToMatmulNbitsTransform,
    FP8DeQuantLinearToLinearTransform,
    GPTQToMatmulNbitsTransform,
)
from QEfficient.utils import constants, get_padding_shape_from_config
from QEfficient.utils.cache import to_hashable
from QEfficient.utils.logging_utils import logger


class QEFFTransformersBase(QEFFBaseModel):
    """
    Parent class for models QEFF provides from transformers i.e. (AutoModel, AutoModelForCausalLM, AutoModelForAudioClassification etc.) from transformers/models/modeling_auto.py file.
    """

    _hf_auto_class: type

    def __init__(self, model: nn.Module) -> None:
        if (
            hasattr(model, "config")
            and hasattr(model.config, "quantization_config")
            and not isinstance(model.config.quantization_config, tuple(QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING.values()))
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


class MultimodalUtilityMixin:
    def __new__(cls, *args, **kwargs):
        if cls is MultimodalUtilityMixin:
            raise TypeError(f"only children of '{cls.__name__}' may be instantiated")
        return object.__new__(cls)

    def auto_correct_inputs(self, inputs):
        checked = True
        inputs_info = self.model.get_inputs_info()
        for valid_input_info in inputs_info:
            if valid_input_info.name not in inputs:
                checked = False
                break
            if inputs[valid_input_info.name].dtype != valid_input_info.datatype:
                checked = False
                break

        if not checked:
            err_str: str = (
                "Expected following input names and shapes to be passed\n"
                + "\n".join([val.__repr__() for val in inputs_info])
                + "\ngot"
                + f"{[(k, v.shape, v.dtype) for k, v in inputs.items()]}"
            )

            raise RuntimeError(err_str)

        return {k: v for k, v in inputs.items() if k in [iinfo.name for iinfo in inputs_info]}


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

        This API can also be used as exception for VLM model since transformers support loading InternChatVL models via AutoModel API we support it via AutoModelForCausalLM API
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

        # This is support models that should be classified to in a different auto class but transformers load them via this class
        kv_offload = kwargs.pop("kv_offload", None)
        if model.__class__.__name__ in MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP:
            return MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP[model.__class__.__name__](
                model, kv_offload=kv_offload
            )

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

    @property
    def get_model_config(self) -> dict:
        return self.model.config.__dict__

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


class QEffVisionEncoderForTextImageToTextModel(QEFFBaseModel):
    _pytorch_transforms = [
        AwqToMatmulNbitsTransform,
        GPTQToMatmulNbitsTransform,
        CustomOpsTransform,
        KVCacheTransform,
        KVCacheModuleMethodMapperTransform,
    ]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(self, model: nn.modules):
        super().__init__(model)
        self.model = model.get_qeff_vision_encoder()

    def export(self, inputs, output_names, dynamic_axes, export_dir=None):
        return self._export(inputs, output_names, dynamic_axes, export_dir)

    def compile(
        self,
        compile_dir,
        compile_only,
        specializations,
        convert_to_fp16,
        mxfp6_matmul,
        mdp_ts_num_devices,
        aic_num_cores,
        custom_io,
        **compiler_options,
    ) -> str:
        return self._compile(
            compile_dir=compile_dir,
            compile_only=compile_only,
            specializations=specializations,
            convert_to_fp16=convert_to_fp16,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=mdp_ts_num_devices,
            aic_num_cores=aic_num_cores,
            custom_io=custom_io,
            **compiler_options,
        )

    @property
    def model_hash(self) -> str:
        # Compute the hash with: model_config, continuous_batching, transforms
        mhash = hashlib.sha256()
        mhash.update(to_hashable(self.model.model.config.to_diff_dict()))
        mhash.update(to_hashable(self._transform_names()))
        mhash.update(to_hashable({"QEffVisionEncoderForTextImageToTextModel": True}))
        mhash = mhash.hexdigest()[:16]
        return mhash

    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname

    @property
    def get_model_config(self) -> dict:
        return self.model.model.vision_model.config.__dict__


class QEffCausalLMForTextImageToTextModel(QEFFBaseModel):
    _pytorch_transforms = [
        AwqToMatmulNbitsTransform,
        GPTQToMatmulNbitsTransform,
        CustomOpsTransform,
        KVCacheTransform,
        VlmKVOffloadTransform,
    ]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(self, model):
        super().__init__(model)
        self.model = model.get_qeff_language_decoder()

    def export(self, inputs, output_names, dynamic_axes, export_dir=None):
        return self._export(inputs, output_names, dynamic_axes, export_dir)

    def compile(
        self,
        compile_dir,
        compile_only,
        specializations,
        convert_to_fp16,
        mxfp6_matmul,
        mdp_ts_num_devices,
        aic_num_cores,
        custom_io,
        **compiler_options,
    ) -> str:
        return self._compile(
            compile_dir=compile_dir,
            compile_only=compile_only,
            specializations=specializations,
            convert_to_fp16=convert_to_fp16,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=mdp_ts_num_devices,
            aic_num_cores=aic_num_cores,
            custom_io=custom_io,
            **compiler_options,
        )

    @property
    def model_hash(self) -> str:
        # Compute the hash with: model_config, continuous_batching, transforms
        mhash = hashlib.sha256()
        mhash.update(to_hashable(self.model.config.to_diff_dict()))
        mhash.update(to_hashable(self._transform_names()))
        mhash.update(to_hashable({"QEffCausalLMForTextImageToTextModel": True}))
        mhash = mhash.hexdigest()[:16]
        return mhash

    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname

    @property
    def get_model_config(self) -> dict:
        return self.model.language_model.config.__dict__


class _QEffAutoModelForImageTextToTextDualQPC:
    _hf_auto_class = AutoModelForImageTextToText

    def __init__(
        self,
        model: nn.Module,
        **kwargs,
    ):
        if kwargs.pop("full_batch_size", None):
            raise NotImplementedError("Continuous batching is not supported for image-text-to-text models yet.")
        self.model = model
        self.config = model.config
        self.vision_model = QEffVisionEncoderForTextImageToTextModel(model)
        self.lang_model = QEffCausalLMForTextImageToTextModel(model)

        self.input_shapes, self.output_names = None, None

    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        if kwargs.get("attn_implementation", None) not in {None, "eager"}:
            logger.warning('Updating attn_implementation="eager"')

        if kwargs.get("low_cpu_mem_usage", None):
            logger.warning("Updating low_cpu_mem_usage=False")

        kwargs.update({"attn_implementation": "eager", "low_cpu_mem_usage": False})
        model = cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(model, **kwargs)

    @property
    def onnx_path(self):
        return [self.vision_model.onnx_path, self.lang_model.onnx_path]

    @property
    def qpc_path(self):
        if self.vision_model.qpc_path and self.lang_model.qpc_path:
            return [self.vision_model.qpc_path, self.lang_model.qpc_path]
        elif self.vision_model.qpc_path:
            return self.vision_model.qpc_path
        else:
            return self.lang_model.qpc_path

    def export(
        self,
        export_dir: Optional[str] = None,
        **kwargs,
    ) -> str:
        inputs = self.model.get_dummy_inputs(kv_offload=True)
        dynamic_axes = self.model.get_onnx_dynamic_axes(kv_offload=True)
        output_names = self.model.get_output_names(kv_offload=True)
        self.vision_model.export(
            inputs["vision"],
            output_names["vision"],
            dynamic_axes["vision"],
            export_dir,
        )

        self.lang_model.export(inputs["lang"], output_names["lang"], dynamic_axes["lang"], export_dir)
        return self.onnx_path

    def compile(
        self,
        img_size: Optional[int] = None,
        vision_onnx_path: Optional[str] = None,
        lang_onnx_path: Optional[str] = None,
        compile_dir: Optional[str] = None,
        *,
        prefill_seq_len: Optional[int] = None,
        ctx_len: Optional[int] = None,
        batch_size: int = 1,
        full_batch_size: Optional[int] = None,
        kv_cache_batch_size: Optional[int] = None,
        num_devices: int = 1,
        num_cores: int = 16,  # FIXME: Make this mandatory arg
        mxfp6_matmul: bool = False,
        mxint8_kv_cache: bool = False,
        num_speculative_tokens: Optional[int] = None,
        skip_vision: Optional[bool] = False,
        skip_lang: Optional[bool] = False,
        **compiler_options,
    ) -> str:
        if any(param is not None for param in [full_batch_size, kv_cache_batch_size, num_speculative_tokens]):
            raise ValueError(
                f"Expected 'full_batch_size', 'kv_cache_batch_size', 'num_speculative_tokens' to be None but got: "
                f"full_batch_size={full_batch_size}, kv_cache_batch_size={kv_cache_batch_size}, num_speculative_tokens={num_speculative_tokens}, "
            )

        if skip_lang and skip_vision:
            raise ValueError("Expected at least one of 'skip_lang' or 'skip_vision' to be False")

        output_names = self.model.get_output_names(kv_offload=True)

        specializations, compiler_options = self.model.get_specializations(
            batch_size=batch_size,
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            img_size=img_size,
            kv_offload=True,
            **compiler_options,
        )

        custom_io_vision = {}
        kv_cache_dtype = "mxint8" if mxint8_kv_cache else "float16"
        custom_io_vision["pixel_values"] = "float16"
        for output_name in output_names["vision"]:
            if output_name.startswith("past_"):
                custom_io_vision[output_name] = kv_cache_dtype
            else:
                custom_io_vision[output_name] = "float16"

        if vision_onnx_path:
            self.vision_model.onnx_path = vision_onnx_path
        if lang_onnx_path:
            self.lang_model.onnx_path = lang_onnx_path

        if (self.vision_model.onnx_path is None and vision_onnx_path is None) or (
            self.lang_model.onnx_path is None and lang_onnx_path is None
        ):
            self.export()

        if not skip_vision:
            self.vision_model._compile(
                compile_dir,
                compile_only=True,
                specializations=specializations["vision"],
                convert_to_fp16=True,
                mxfp6_matmul=mxfp6_matmul,
                mdp_ts_num_devices=num_devices,
                aic_num_cores=num_cores,
                custom_io=custom_io_vision,
                mxint8_kv_cache=mxint8_kv_cache,
                **compiler_options,
            )

        if not skip_lang:
            custom_io_lang = {}
            # Inputs
            for output_name in output_names["lang"]:
                if output_name.endswith("_RetainedState"):
                    custom_io_lang[output_name[: -len("_RetainedState")]] = (
                        "float16" if "vision_embeds" in output_name else kv_cache_dtype
                    )

            # outputs
            for output_name in output_names["lang"]:
                if output_name.endswith("_RetainedState"):
                    custom_io_lang[output_name] = "float16" if "vision_embeds" in output_name else kv_cache_dtype

            self.lang_model._compile(
                compile_dir,
                compile_only=True,
                retained_state=True,
                specializations=specializations["lang"],
                convert_to_fp16=True,
                mxfp6_matmul=mxfp6_matmul,
                mdp_ts_num_devices=num_devices,
                aic_num_cores=num_cores,
                custom_io=custom_io_lang,
                mxint8_kv_cache=mxint8_kv_cache,
                **compiler_options,
            )
        return self.qpc_path

    def generate(
        self,
        inputs: torch.Tensor,
        streamer: Optional[TextStreamer] = None,
        device_ids: List[int] = None,
        runtime_ai100: bool = True,
        generation_len: Optional[int] = None,
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
        if not runtime_ai100:
            raise NotImplementedError("PyTorch execution is not supported yet for this model!")

        return self.kv_offload_generate(
            inputs=inputs, device_ids=device_ids, streamer=streamer, generation_len=generation_len
        )

    def kv_offload_generate(
        self,
        inputs: List[str] = None,
        streamer: Optional[TextStreamer] = None,
        device_ids: List[int] = None,
        generation_len: int = None,
    ):
        if not self.vision_model.qpc_path or not self.lang_model.qpc_path:
            raise TypeError("Please run compile API for vision and language model first!")

        lang_session = QAICInferenceSession(self.lang_model.qpc_path, device_ids, activate=False)

        vision_session = QAICInferenceSession(self.vision_model.qpc_path, device_ids)

        batch_size, ctx_len, fbs = get_compilation_dims(self.lang_model.qpc_path)

        pad_token_id = 1

        # Skip inputs/outputs
        lang_session.skip_buffers(
            [x for x in lang_session.input_names + lang_session.output_names if x.startswith("past_")]
        )

        # Read prompt and ctx len from session
        batch_size = max(
            [x[lang_session.binding_index_map["input_ids"]][1][0] for x in lang_session.allowed_shapes]
            + [lang_session.bindings[lang_session.binding_index_map["input_ids"]].dims[0]]
        )

        prefill_seq_len = max(
            [x[lang_session.binding_index_map["input_ids"]][1][1] for x in lang_session.allowed_shapes]
            + [lang_session.bindings[lang_session.binding_index_map["input_ids"]].dims[1]]
        )

        input_len = inputs["attention_mask"].sum(1, keepdims=True)
        input_ids_length = inputs["input_ids"].shape[1]
        num_chunks = -(input_ids_length // -prefill_seq_len)  # ceil divide without float
        # padded_len = num_chunks * prefill_seq_len  # Convert to a multiple of prompt_len
        padded_len = vision_session.bindings[vision_session.binding_index_map["input_ids"]].dims[1]
        if generation_len is None:
            generation_len = ctx_len - input_len.max()
        assert generation_len > 0, "generation length should be greater than zero"
        generated_ids = np.full((batch_size, generation_len + 1), pad_token_id)

        # Prepare inputs for prefill
        prefill_start = perf_counter()

        inputs["input_ids"] = torch.nn.functional.pad(
            inputs["input_ids"],
            (0, padded_len - input_ids_length),
            "constant",
            1,
        )
        inputs["attention_mask"] = torch.nn.functional.pad(
            inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
        )
        if "cross_attention_mask" in inputs:
            inputs["cross_attention_mask"] = torch.nn.functional.pad(
                inputs["cross_attention_mask"], (0, 0, 0, 0, 0, padded_len - input_ids_length)
            )

        for k, v in inputs.items():
            inputs[k] = np.array(v)

        vision_inputs = {
            k: v for k, v in inputs.items() if k in {"pixel_values", "aspect_ratio_ids", "aspect_ratio_mask"}
        }

        vision_inputs["pixel_values"] = vision_inputs["pixel_values"].astype("float16")
        vision_inputs["input_ids"] = inputs["input_ids"]
        vision_start = perf_counter()
        vision_outputs = vision_session.run(vision_inputs)
        vision_end = perf_counter()

        lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}
        lang_inputs["input_ids"] = inputs["input_ids"]
        lang_inputs["position_ids"] = np.where(
            lang_inputs.pop("attention_mask"), np.arange(padded_len), -1
        )  # Need to use -1 as position_ids for invalid tokens

        vision_session.deactivate()
        lang_session.activate()
        lang_inputs["vision_embeds"] = vision_outputs["vision_embeds"]
        # lang_session.set_buffers(vision_outputs)
        prefill_start = perf_counter()
        # Run prefill
        for i in range(num_chunks):
            chunk_inputs = lang_inputs.copy()
            chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
            chunk_inputs["position_ids"] = lang_inputs["position_ids"][
                :, i * prefill_seq_len : (i + 1) * prefill_seq_len
            ]
            chunk_inputs["vision_embeds"] = lang_inputs["vision_embeds"][
                :, i * prefill_seq_len : (i + 1) * prefill_seq_len
            ]
            outputs = lang_session.run(chunk_inputs)

        prefill_time = perf_counter() - prefill_start + vision_end - vision_start
        lang_inputs["vision_embeds"] = lang_inputs["vision_embeds"][:, :prefill_seq_len]
        # Skip inputs/outputs again
        lang_session.skip_buffers(
            [x for x in lang_session.input_names + lang_session.output_names if x.startswith("past_")]
        )

        # Get first token
        lang_inputs["input_ids"] = outputs["logits"].argmax(2)
        lang_inputs["position_ids"] = input_len.numpy()
        if "cross_attention_mask" in lang_inputs:
            bs, _, num_images, img_tiles = lang_inputs["cross_attention_mask"].shape
            lang_inputs["cross_attention_mask"] = torch.ones((bs, 1, num_images, img_tiles), dtype=torch.int64).numpy()
        generated_ids[:, 0] = lang_inputs["input_ids"].squeeze(1)

        if streamer:
            streamer.put(lang_inputs["input_ids"][0])

        # Decode loop
        decode_start = perf_counter()
        for num_token in range(1, generation_len):
            outputs = lang_session.run(lang_inputs)

            # Prepare inputs for next iteration
            lang_inputs["input_ids"] = outputs["logits"].argmax(2)
            lang_inputs["position_ids"] += 1
            generated_ids[:, num_token] = lang_inputs["input_ids"].squeeze(1)
            if streamer:
                streamer.put(lang_inputs["input_ids"][0])

        decode_end = perf_counter()
        if streamer:
            streamer.end()

        decode_perf = (num_token - 1) / (decode_end - decode_start)
        total_time = decode_end - decode_start + prefill_time
        total_perf = num_token / total_time

        return CloudAI100ExecInfoNew(
            batch_size=batch_size,
            generated_ids=generated_ids,
            perf_metrics=PerfMetrics(
                prefill_time=prefill_time, decode_perf=decode_perf, total_perf=total_perf, total_time=total_time
            ),
        )


class _QEFFAutoModelForImageTextToTextSingleQPC(QEFFTransformersBase, MultimodalUtilityMixin):
    _hf_auto_class = AutoModelForImageTextToText
    _pytorch_transforms = [
        AwqToMatmulNbitsTransform,
        GPTQToMatmulNbitsTransform,
        CustomOpsTransform,
        KVCacheTransform,
        KVCacheModuleMethodMapperTransform,
        VlmNoKVOffloadTransform,
    ]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(
        self,
        model: nn.Module,
        **kwargs,
    ):
        if kwargs.pop("full_batch_size", None):
            raise NotImplementedError("Continuous batching is not supported for image-text-to-text models yet.")
        super().__init__(model)

        # to handle internvl models
        if hasattr(self.model.config, "llm_config") and hasattr(self.model.config, "vision_config"):
            self.model.config.llm_config.use_cache = True
            self.model.config.llm_config._attn_implementation = "eager"
            self.model.config.vision_config.use_flash_attn = "false"
        else:
            self.model.config.text_config.use_cache = True

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        if kwargs.get("attn_implementation", None) not in {None, "eager"}:
            logger.warning('Updating attn_implementation="eager"')

        if kwargs.get("low_cpu_mem_usage", None):
            logger.warning("Updating low_cpu_mem_usage=False")

        kwargs.update({"attn_implementation": "eager", "low_cpu_mem_usage": False})
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        config._attn_implementation = "eager"
        config.vision_config.use_flash_attn = "false"
        model = cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path, config, *args, **kwargs)

        return cls(model, **kwargs)

    def export(
        self,
        export_dir: Optional[str] = None,
        **kwargs,
    ) -> str:
        inputs = self.model.get_dummy_inputs()
        dynamic_axes = self.model.get_onnx_dynamic_axes()
        output_names = self.model.get_output_names()
        return self._export(inputs, output_names, dynamic_axes, export_dir=export_dir)

    def compile(
        self,
        onnx_path: Optional[str] = None,
        img_size: Optional[int] = None,
        compile_dir: Optional[str] = None,
        *,
        prefill_seq_len: Optional[int] = None,
        ctx_len: Optional[int] = None,
        batch_size: int = 1,
        full_batch_size: Optional[int] = None,
        kv_cache_batch_size: Optional[int] = None,
        num_devices: int = 1,
        num_cores: int = 16,  # FIXME: Make this mandatory arg
        mxfp6_matmul: bool = False,
        mxint8_kv_cache: bool = False,
        num_speculative_tokens: Optional[int] = None,
        **compiler_options,
    ) -> str:
        if any(param is not None for param in [full_batch_size, kv_cache_batch_size, num_speculative_tokens]):
            raise ValueError(
                f"Expected 'full_batch_size', 'kv_cache_batch_size', 'num_speculative_tokens' to be None but got: "
                f"full_batch_size={full_batch_size}, kv_cache_batch_size={kv_cache_batch_size}, num_speculative_tokens={num_speculative_tokens}, "
            )

        output_names = self.model.get_output_names()

        # Get specializations from modelling file
        # TODO: expose this via the auto class as well
        specializations, compiler_options = self.model.get_specializations(
            batch_size=batch_size,
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            img_size=img_size,
            **compiler_options,
        )

        custom_io = {}
        kv_cache_dtype = "mxint8" if mxint8_kv_cache else "float16"
        # inputs
        for input_name in output_names:
            if input_name.endswith("_RetainedState"):
                custom_io[input_name[: -len("_RetainedState")]] = (
                    "float16" if "pixel_values" in input_name else kv_cache_dtype
                )

        # outputs
        for output_name in output_names:
            if output_name.endswith("_RetainedState"):
                custom_io[output_name] = "float16" if "pixel_values" in output_name else kv_cache_dtype

        self._compile(
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
        return self.qpc_path

    def get_onnx_dynamic_axes(self):
        return self.model.get_onnx_dynamic_axes()

    def generate(
        self,
        inputs: torch.Tensor,
        streamer: Optional[TextStreamer] = None,
        device_ids: List[int] = None,
        runtime_ai100: bool = True,
        generation_len: Optional[int] = None,
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
        if not runtime_ai100:
            raise NotImplementedError("PyTorch execution is not supported yet for this model!")

        return self.cloud_ai_100_generate(
            inputs=inputs, device_ids=device_ids, generation_len=generation_len, streamer=streamer
        )

    def cloud_ai_100_generate(
        self,
        inputs: torch.Tensor,
        device_ids: List[int],
        enable_debug_logs: bool = False,
        generation_len: int = None,
        streamer: Optional[TextStreamer] = None,
    ) -> np.ndarray:
        inputs = self.auto_correct_inputs(inputs)
        qpc_session = QAICInferenceSession(
            self.qpc_path, device_ids, enable_debug_logs=enable_debug_logs, activate=False
        )

        batch_size, ctx_len, fbs = get_compilation_dims(self.qpc_path)

        pad_token_id = 1

        # Skip inputs/outputs
        qpc_session.skip_buffers(
            [
                x
                for x in qpc_session.input_names + qpc_session.output_names
                if x.startswith("past_") or x.endswith("_RetainedState")
            ]
        )

        # Read prompt and ctx len from session
        batch_size = max(
            [x[qpc_session.binding_index_map["input_ids"]][1][0] for x in qpc_session.allowed_shapes]
            + [qpc_session.bindings[qpc_session.binding_index_map["input_ids"]].dims[0]]
        )

        prefill_seq_len = max(
            [x[qpc_session.binding_index_map["input_ids"]][1][1] for x in qpc_session.allowed_shapes]
            + [qpc_session.bindings[qpc_session.binding_index_map["input_ids"]].dims[1]]
        )

        input_len = inputs["attention_mask"].sum(1, keepdims=True)
        input_ids_length = inputs["input_ids"].shape[1]

        num_chunks = -(input_ids_length // -prefill_seq_len)  # ceil divide without float

        padded_len = num_chunks * prefill_seq_len  # Convert to a multiple of prompt_len
        if generation_len is None:
            generation_len = ctx_len - input_len.max()

        assert generation_len > 0, "generation length should be greater than zero"
        generated_ids = np.full((batch_size, generation_len + 1), pad_token_id)

        # Prepare inputs for prefill
        prefill_start = perf_counter()

        inputs["input_ids"] = torch.nn.functional.pad(
            inputs["input_ids"],
            (0, padded_len - input_ids_length),
            "constant",
            1,
        )
        inputs["attention_mask"] = torch.nn.functional.pad(
            inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
        )
        if "cross_attention_mask" in inputs:
            inputs["cross_attention_mask"] = torch.nn.functional.pad(
                inputs["cross_attention_mask"], (0, 0, 0, 0, 0, padded_len - input_ids_length)
            )
        for k, v in inputs.items():
            inputs[k] = np.array(v)

        if "pixel_values_RetainedState" in qpc_session.output_names:
            inputs["pixel_values"] = inputs["pixel_values"].astype("float16")

        inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)

        qpc_session.activate()

        # Run prefill

        for i in range(num_chunks):
            chunk_inputs = inputs.copy()
            chunk_inputs["input_ids"] = inputs["input_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
            chunk_inputs["position_ids"] = inputs["position_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
            outputs = qpc_session.run(chunk_inputs)

        prefill_time = perf_counter() - prefill_start
        # Get first token
        inputs["input_ids"] = outputs["logits"].argmax(2)
        inputs["position_ids"] = input_len.numpy()

        if "cross_attention_mask" in inputs:
            bs, _, num_images, img_tiles = inputs["cross_attention_mask"].shape
            inputs["cross_attention_mask"] = torch.ones((bs, 1, num_images, img_tiles), dtype=torch.int64).numpy()

        generated_ids[:, 0] = inputs["input_ids"].squeeze(1)
        if streamer:
            streamer.put(inputs["input_ids"][0])

        if "pixel_values_RetainedState" in qpc_session.output_names:
            qpc_session.skip_buffers(["pixel_values"])
            inputs.pop("pixel_values")

        # Decode loop
        decode_start = perf_counter()
        for num_token in range(1, generation_len):
            outputs = qpc_session.run(inputs)
            # Prepare inputs for next iteration
            inputs["input_ids"] = outputs["logits"].argmax(2)
            inputs["position_ids"] += 1
            generated_ids[:, num_token] = inputs["input_ids"].squeeze(1)
            if streamer:
                streamer.put(inputs["input_ids"][0])

        decode_end = perf_counter()
        if streamer:
            streamer.end()

        decode_perf = (num_token - 1) / (decode_end - decode_start)
        total_time = decode_end - prefill_start
        total_perf = num_token / total_time

        return CloudAI100ExecInfoNew(
            batch_size=batch_size,
            generated_ids=generated_ids,
            perf_metrics=PerfMetrics(
                prefill_time=prefill_time, decode_perf=decode_perf, total_perf=total_perf, total_time=total_time
            ),
        )

    @property
    def model_hash(self) -> str:
        mhash = hashlib.sha256()
        mhash.update(to_hashable(self.model.config.to_diff_dict()))
        mhash.update(to_hashable(self._transform_names()))
        mhash.update(to_hashable({"QEFFAutoModelForImageTextToText1QPC": True}))
        mhash = mhash.hexdigest()[:16]
        return mhash

    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname

    @property
    def get_model_config(self) -> dict:
        return self.model.config.__dict__


class QEFFAutoModelForImageTextToText:
    """
    The QEFFAutoModelForImageTextToText class is used to work with multimodal language models from the HuggingFace hub.
    While you can initialize the class directly, it's best to use the ``from_pretrained`` method for this purpose. This class supports both single and dual QPC approaches.
    Attributes:
        _hf_auto_class (class): The Hugging Face AutoModel class for ImageTextToText models.

    ``Mandatory`` Args:
        :pretrained_model_name_or_path (str): Model card name from HuggingFace or local path to model directory.

    ``Optional`` Args:
        :kv_offload (bool): Flag to toggle between single and dual QPC approaches. If set to False, the Single QPC approach will be used; otherwise, the dual QPC approach will be applied. Defaults to True.

    .. code-block:: python

        import requests
        from PIL import Image
        from transformers import AutoProcessor, TextStreamer

        from QEfficient import QEFFAutoModelForImageTextToText

        # Add HuggingFace Token to access the model
        HF_TOKEN = ""
        model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        query = "Describe this image."
        image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"

        ## STEP - 1 Load the Processor and Model, and kv_offload=True/False for dual and single qpc
        processor = AutoProcessor.from_pretrained(model_name, token=HF_TOKEN)
        model = QEFFAutoModelForImageTextToText.from_pretrained(model_name, token=HF_TOKEN, attn_implementation="eager", kv_offload=False)

        ## STEP - 2 Export & Compile the Model
        model.compile(
            prefill_seq_len=32,
            ctx_len=512,
            img_size=560,
            num_cores=16,
            num_devices=1,
            mxfp6_matmul=False,
        )

        ## STEP - 3 Load and process the inputs for Inference
        image = Image.open(requests.get(image_url, stream=True).raw)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": query},
                ],
            }
        ]
        input_text = [processor.apply_chat_template(messages, add_generation_prompt=True)]
        inputs = processor(
            text=input_text,
            images=image,
            return_tensors="pt",
            add_special_tokens=False,
            padding="max_length",
            max_length=32,
        )

        ## STEP - 4 Run Inference on the compiled model
        streamer = TextStreamer(processor.tokenizer)
        model.generate(inputs=inputs, streamer=streamer, generation_len=512)

    """

    _hf_auto_class = AutoModelForImageTextToText

    def __new__(self, model: nn.Module, kv_offload: Optional[bool] = True, **kwargs):
        if kv_offload:
            return _QEffAutoModelForImageTextToTextDualQPC(model, **kwargs)
        else:
            return _QEFFAutoModelForImageTextToTextSingleQPC(model, **kwargs)

    @classmethod
    @with_replaced_quantizers
    def from_pretrained(cls, pretrained_model_name_or_path: str, kv_offload: Optional[bool] = None, **kwargs):
        """Used to load models supported by transformers.AutoModelForImageTextToText for Cloud AI 100.

        Args:
            pretrained_model_name_or_path (str): Path or model card name on HuggingFace
            kv_offload (Optional[bool], optional): Should the KV of vision encoder be offloaded to CPU and use Two QPC. Defaults to None.

        Returns:
            _type_: _description_
        """
        # TODO: add a check to see if kv_offload is allowed for given model by loading the config and checking architecture or type of config here.
        if kwargs.get("attn_implementation", None) not in {None, "eager"}:
            logger.warning('Updating attn_implementation="eager"')

        if kwargs.get("low_cpu_mem_usage", None):
            logger.warning("Updating low_cpu_mem_usage=False")

        if kwargs.pop("continuous_batching", None):
            NotImplementedError("Continuous batching is not supported for image-text-to-text models yet.")

        kwargs.update({"attn_implementation": "eager", "low_cpu_mem_usage": False})
        model = cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(model, kv_offload=kv_offload, **kwargs)


MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP = {"InternVLChatModel": QEFFAutoModelForImageTextToText}


class QEFFAutoModelForCausalLM(QEFFBaseModel):
    """
    The QEFF class is designed for manipulating any causal language model from the HuggingFace hub.
    Although it is possible to initialize the class directly, we highly recommend using the ``from_pretrained`` method for initialization.

    ``Mandatory`` Args:
        :model (nn.Module):  PyTorch model
        :continuous_batching (bool): Weather this model will be used for continuous batching in future. If this is not set True here, the model can not be exported/compiled for continuous batching later.
        :is_tlm (bool): Whether this is a Speculative Decoding Target Language Model. If set to True, `num_logits_to_keep` input array will have to be fed to control the number of returned logits during prefill/decode.


    .. code-block:: python

        from QEfficient import QEFFAutoModelForCausalLM
        from transformers import AutoTokenizer

        model_name = "gpt2"
        model = QEFFAutoModelForCausalLM.from_pretrained(model_name, num_hidden_layers=2)
        model.compile(prefill_seq_len=128, ctx_len=256, num_cores=16, num_devices=1)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.generate(prompts=["Hi there!!"], tokenizer=tokenizer)
    """

    _hf_auto_class = AutoModelForCausalLM
    _pytorch_transforms = [
        AwqToMatmulNbitsTransform,
        GPTQToMatmulNbitsTransform,
        FP8DeQuantLinearToLinearTransform,
        CustomOpsTransform,
        KVCacheTransform,
    ]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(
        self,
        model: nn.Module,
        continuous_batching: bool = False,
        qaic_config: Optional[dict] = None,
        **kwargs,
    ):
        model_class_name = model.__class__.__name__
        if not (model_class_name.endswith("ForCausalLM") or model_class_name.endswith("LMHeadModel")):
            raise TypeError(f"Required pytorch module for CausalLM or LMHeadModel, got {model_class_name}")

        # TODO: remove from version 1.20
        if kwargs.pop("full_batch_size", None):
            continuous_batching = True
            warnings.warn(
                "full_batch_size argument is deprecated. Use continuous_batching=True instead.", DeprecationWarning, 2
            )
        if hasattr(model.config, "quantization_config") and not isinstance(
            model.config.quantization_config, tuple(QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING.values())
        ):
            logger.warning(
                "Please use `from_pretrained` method to load quantized models, might give unexpected results"
            )

        super().__init__(model)

        # Set use_cache=True to get KV values as output during ONNX export
        self.model.config.use_cache = True
        self.num_layers = model.config.num_hidden_layers
        self.continuous_batching = continuous_batching
        self.model, transformed = SpDTransform.apply(self.model, qaic_config, **kwargs)
        self.is_tlm = transformed

    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname

    def __repr__(self) -> str:
        return self.__class__.__name__ + "\n" + self.model.__repr__()

    @classmethod
    @with_replaced_quantizers
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        continuous_batching: bool = False,
        qaic_config: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        """
        This method serves as the easiest entry point into using QEfficient. The interface is designed to be similar to transformers.AutoModelForCausalLM.
        Once the model is initialized, you can use other methods such as export, compile, and generate on the same object.

        This API can also be used as exception for VLM model since transformers support loading InternChatVL models via AutoModel API we support it via AutoModelForCausalLM API
        Args:
            :pretrained_name_or_path (str): Model card name from HuggingFace or local path to model directory.
            :continuous_batching (bool): Whether this model will be used for continuous batching in future. If this is not set True here, the model can not be exported/compiled for continuous batching later.
            :is_tlm (bool): Whether this is a Speculative Decoding Target Language Model. If set to True, `num_logits_to_keep` input array will have to be fed to control the number of returned logits during prefill/decode.
            :args, kwargs: Additional arguments to pass to transformers.AutoModelForCausalLM.

        .. code-block:: python

            from QEfficient import QEFFAutoModelForCausalLM
            from transformers import AutoTokenizer

            # Initialize the model using from_pretrained similar to transformers.AutoModelForCausalLM
            model_name = "gpt2"
            model = QEFFAutoModelForCausalLM.from_pretrained(model_name)

            # Now you can directly compile the model for Cloud AI 100
            model.compile(num_cores=16) # Considering you have a Cloud AI 100 Standard SKU

            # You can now execute the model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model.generate(prompts=["Hi there!!"], tokenizer=tokenizer)
        """
        if kwargs.pop("full_batch_size", None):
            continuous_batching = True
            warnings.warn(
                "full_batch_size argument is deprecated. Use continuous_batching=True instead.", DeprecationWarning, 2
            )

        if kwargs.get("attn_implementation", None) not in {None, "eager"}:
            logger.warning('Updating attn_implementation="eager"')

        if kwargs.get("low_cpu_mem_usage", None):
            logger.warning("Updating low_cpu_mem_usage=False")

        kv_offload = kwargs.pop("kv_offload", None)

        kwargs.update({"attn_implementation": "eager", "low_cpu_mem_usage": False})
        model = cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if qaic_config is not None:
            qaic_config["pretrained_model_name_or_path"] = pretrained_model_name_or_path

        # This is support models that should be classified to in a different auto class but transformers load them via this class

        if model.__class__.__name__ in MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP:
            return MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP[model.__class__.__name__](
                model, kv_offload=kv_offload
            )

        return cls(
            model,
            continuous_batching=continuous_batching,
            qaic_config=qaic_config,
            **kwargs,
        )

    @property
    def model_hash(self) -> str:
        # Compute the hash with: model_config, continuous_batching, transforms
        mhash = hashlib.sha256()
        mhash.update(to_hashable(self.model.config.to_diff_dict()))
        mhash.update(to_hashable({"continuous_batching": self.continuous_batching}))
        mhash.update(to_hashable({"is_tlm": self.is_tlm}))
        mhash.update(to_hashable(self._transform_names()))
        mhash = mhash.hexdigest()[:16]
        return mhash

    @property
    def get_model_config(self) -> dict:
        return self.model.config.__dict__

    def export(self, export_dir: Optional[str] = None) -> str:
        """
        Exports the model to ``ONNX`` format using ``torch.onnx.export``.

        ``Optional`` Args:
            :export_dir (str, optional): The directory path to store ONNX-graph.

        Returns:
            :str: Path of the generated ``ONNX`` graph.
        """
        bs: int = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        seq_len: int = constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
        fbs = constants.ONNX_EXPORT_EXAMPLE_FBS
        kv_cache_shape = get_padding_shape_from_config(
            self.model.config, fbs if self.continuous_batching else bs, seq_len
        )
        example_inputs = {
            "input_ids": torch.zeros((bs, seq_len), dtype=torch.int64),
            "position_ids": torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(bs, 1),
            "past_key_values": [[] for _ in range(self.num_layers)],
        }
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
        }
        if len(kv_cache_shape) == 3:  # For GPTBigCode arch the pkv is 3d
            pkv_dynamic_axes = {
                0: "full_batch_size" if self.continuous_batching else "batch_size",
                1: "ctx_len",
            }
        else:  # pkv is 4d
            pkv_dynamic_axes = {
                0: "full_batch_size" if self.continuous_batching else "batch_size",
                2: "ctx_len",
            }
        output_names = ["logits"]

        for i in range(self.num_layers):
            for kv in ["key", "value"]:
                example_inputs["past_key_values"][i].append(torch.zeros(kv_cache_shape, dtype=torch.float32))
                dynamic_axes[f"past_{kv}.{i}"] = pkv_dynamic_axes
                output_names.append(f"past_{kv}.{i}_RetainedState")

        if self.continuous_batching:
            example_inputs["batch_index"] = torch.arange(bs).view(bs, 1)
            dynamic_axes["batch_index"] = {0: "batch_size"}

        if self.is_tlm:
            nlk = constants.ONNX_EXPORT_EXAMPLE_NLK  # Number of Logits to Keep
            example_inputs["num_logits_to_keep"] = torch.arange(nlk).view(nlk, 1)
            dynamic_axes["num_logits_to_keep"] = {0: "num_logits_to_keep"}

        return self._export(
            example_inputs,
            output_names,
            dynamic_axes,
            export_dir=export_dir,
        )

    def build_prefill_specialization(
        self,
        prefill_seq_len: int = 32,
        ctx_len: int = 128,
        batch_size: int = 1,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
    ):
        spec = {
            "batch_size": 1 if self.continuous_batching else batch_size,
            "seq_len": prefill_seq_len,
            "ctx_len": ctx_len,
            "num_logits_to_keep": 1 if self.is_tlm else None,
        }
        if self.continuous_batching:
            spec["full_batch_size"] = kv_cache_batch_size
        else:
            spec["batch_size"] = kv_cache_batch_size
        if full_batch_size:
            spec["full_batch_exec_size"] = full_batch_size
        return {k: v for k, v in spec.items() if v is not None}

    def build_decode_specialization(
        self,
        prefill_seq_len: int = 32,
        ctx_len: int = 128,
        batch_size: int = 1,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
        num_speculative_tokens: Optional[int] = None,
    ):
        if prefill_seq_len == 1 and not self.continuous_batching:
            return None  # Avoid duplication with prefill
        spec = {
            "batch_size": full_batch_size if self.continuous_batching else batch_size,
            "seq_len": (num_speculative_tokens + 1) if self.is_tlm else 1,
            "ctx_len": ctx_len,
            "num_logits_to_keep": (num_speculative_tokens + 1) if self.is_tlm else None,
        }
        if self.continuous_batching:
            spec["full_batch_size"] = kv_cache_batch_size
        else:
            spec["batch_size"] = kv_cache_batch_size
        return {k: v for k, v in spec.items() if v is not None}

    def compile(
        self,
        onnx_path: Optional[str] = None,
        compile_dir: Optional[str] = None,
        *,
        prefill_seq_len: int = 32,
        ctx_len: int = 128,
        batch_size: int = 1,
        full_batch_size: Optional[int] = None,
        kv_cache_batch_size: Optional[int] = None,
        num_devices: int = 1,
        num_cores: int = 16,  # FIXME: Make this mandatory arg
        mxfp6_matmul: bool = False,
        mxint8_kv_cache: bool = False,
        num_speculative_tokens: Optional[int] = None,
        prefill_only: Optional[bool] = None,
        **compiler_options,
    ) -> str:
        """
        This method compiles the exported ``ONNX`` model using the Cloud AI 100 Platform SDK compiler binary found at ``/opt/qti-aic/exec/qaic-exec`` and generates a ``qpc`` package.
        If the model has not been exported yet, this method will handle the export process.
        You can pass any other arguments that the `qaic-exec` takes as extra kwargs.

        ``Optional`` Args:
            :onnx_path (str, optional): Path to pre-exported onnx model.
            :compile_dir (str, optional): Path for saving the qpc generated.
            :num_cores (int): Number of cores used to compile the model.
            :num_devices (int): Number of devices the model needs to be compiled for. Defaults to 1.
            :batch_size (int, optional): Batch size. ``Defaults to 1``.
            :prefill_seq_len (int, optional): The length of the Prefill prompt should be less that ``prefill_seq_len``. ``Defaults to 32``.
            :ctx_len (int, optional): Maximum ``ctx`` that the compiled model can remember. ``Defaults to 128``.
            :full_batch_size (int, optional): Continuous batching batch size.
            :mxfp6_matmul (bool, optional): Whether to use ``mxfp6`` compression for weights. ``Defaults to False``.
            :mxint8_kv_cache (bool, optional): Whether to use ``mxint8`` compression for KV cache. ``Defaults to False``.
            :num_speculative_tokens (int, optional): Number of speculative tokens to take as input for Speculative Decoding Target Language Model.
            :mos (int, optional): Effort level to reduce on-chip memory. Defaults to -1, meaning no effort. ``Defaults to -1``.
            :aic_enable_depth_first (bool, optional): Enables DFS with default memory size. ``Defaults to False``.
            :prefill_only (bool): if ``True`` compile for prefill only and if ``False`` compile for decode only. Defaults to None, which compiles for both ``prefill and ``decode``.
            :compiler_options (dict, optional): Pass any compiler option as input. ``Defaults to None``.
            Following flag can be passed in compiler_options to enable QNN Compilation path.
                :enable_qnn (bool): Enables QNN Compilation. ``Defaults to False. if not passed.``
                :qnn_config (str): Path of QNN Config parameters file. ``Defaults to None. if not passed``
            for QAIC compilation path, any flag that is supported by ``qaic-exec`` can be passed. Params are converted to flags as below:
                - aic_num_cores=16 -> -aic-num-cores=16
                - convert_to_fp16=True -> -convert-to-fp16

        Returns:
            :str: Path of the compiled ``qpc`` package.
        """
        # --- Validation ---
        if prefill_only is not None and not isinstance(prefill_only, bool):
            raise TypeError("`prefill_only` must be a boolean.")

        if self.is_tlm:
            num_speculative_tokens = self.check_and_get_num_speculative_tokens(num_speculative_tokens, prefill_seq_len)

        if self.continuous_batching and full_batch_size is None:
            raise TypeError("`full_batch_size` is required when `continuous_batching=True`.")

        if kv_cache_batch_size and not full_batch_size:
            raise ValueError(
                "KV caching requires continuous batching. Please set `full_batch_size` and "
                "enable `continuous_batching=True` in `from_pretrained`."
            )

        # Infer kv_cache_batch_size if not provided
        kv_cache_batch_size = kv_cache_batch_size or full_batch_size or batch_size

        # --- Specializations ---
        specializations = []

        if prefill_only is None or prefill_only or prefill_seq_len == 1:
            specializations.append(
                self.build_prefill_specialization(
                    prefill_seq_len, ctx_len, batch_size, kv_cache_batch_size, full_batch_size
                )
            )
        if prefill_only is None or not prefill_only:
            decode_spec = self.build_decode_specialization(
                prefill_seq_len, ctx_len, batch_size, kv_cache_batch_size, full_batch_size, num_speculative_tokens
            )
            if decode_spec:
                specializations.append(decode_spec)

        # --- Compilation ---
        kv_cache_dtype = "mxint8" if mxint8_kv_cache else "float16"
        custom_io = {}

        for suffix in ["", "_RetainedState"]:
            for i in range(self.num_layers):
                for kv in ["key", "value"]:
                    custom_io[f"past_{kv}.{i}{suffix}"] = kv_cache_dtype

        qpc_path = self._compile(
            onnx_path=onnx_path,
            compile_dir=compile_dir,
            compile_only=True,
            retained_state=True,
            specializations=specializations,
            convert_to_fp16=True,
            mxfp6_matmul=mxfp6_matmul,
            custom_io=custom_io,
            mdp_ts_num_devices=num_devices,
            num_speculative_tokens=num_speculative_tokens,
            aic_num_cores=num_cores,
            mxint8_kv_cache=mxint8_kv_cache,
            **compiler_options,
        )

        return qpc_path

    # FIXME: Update this method to match with transformers AutoModelForCausalLM.generate
    def generate(
        self,
        tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer],
        prompts: List[str],
        device_id: List[int] = None,
        runtime_ai100: bool = True,
        **kwargs,
    ):
        """
        This method generates output until ``eos`` or ``generation_len`` by executing the compiled ``qpc`` on ``Cloud AI 100`` Hardware cards.
        This is a sequential execution based on the ``batch_size`` of the compiled model and the number of prompts passed.
        If the number of prompts cannot be divided by the ``batch_size``, the last unfulfilled batch will be dropped.

        ``Mandatory`` Args:
            :tokenizer (Union[PreTrainedTokenizerFast, PreTrainedTokenizer]): Pass tokenizer of the model.
            :prompts (List[str]): List of prompts to run the execution.

        ``optional`` Args:
            :device_id (List[int]): Ids of devices for running the qpc pass as [0] in case of normal model / [0, 1, 2, 3] in case of tensor slicing model
            :runtime_ai100 (bool, optional): ``AI_100`` and ``PyTorch`` runtime is supported as of now. Defaults to ``True`` for ``AI_100`` runtime.

        """
        if runtime_ai100:
            if not isinstance(self.qpc_path, Path):
                raise TypeError("Please run compile API first!")
            generation_len = kwargs.pop("generation_len", None)
            return QEfficient.cloud_ai_100_exec_kv(
                tokenizer,
                self.qpc_path,
                prompt=prompts,
                device_id=device_id,
                generation_len=generation_len,
                is_tlm=self.is_tlm,
            )
        else:
            raise NotImplementedError("Only AI_100 runtime is supported right now via generate API")

    def check_and_get_num_speculative_tokens(self, num_speculative_tokens: Optional[int], prefill_seq_len: int):
        if hasattr(self.model.config, "speculative_config"):
            num_speculative_tokens_ = self.model.config.speculative_config["num_speculative_tokens"]
            if num_speculative_tokens is not None:
                logger.warning(
                    f"arg `num_speculative_tokens` is a fixed value of {num_speculative_tokens_} for this model."
                    f" Passed value of {num_speculative_tokens} will be ignored."
                )
            num_speculative_tokens = num_speculative_tokens_
        elif num_speculative_tokens is None:
            raise TypeError("missing required argument `num_speculative_tokens` as `is_tlm` is True.")

        if not isinstance(num_speculative_tokens, int) and num_speculative_tokens < 2:
            ValueError(
                f"`num_speculative_tokens` arg should be an integer greater than 1, got {num_speculative_tokens}"
            )
        num_logits_to_keep = num_speculative_tokens + 1
        if prefill_seq_len < num_logits_to_keep:
            raise ValueError(
                f"sequence length ({prefill_seq_len}) must be at least `num_speculative_tokens+1` ({num_logits_to_keep})"
            )
        return num_speculative_tokens


class QEFFAutoModelForSpeechSeq2Seq(QEFFTransformersBase, MultimodalUtilityMixin):
    """
    The QEFFAutoModelForSpeechSeq2Seq class is designed for transformers models with a sequence-to-sequence speech-to-text modeling head, including Whisper and other Encoder-Decoder speech models.
    Although it is possible to initialize the class directly, we highly recommend using the ``from_pretrained`` method for initialization.

    ``Mandatory`` Args:
        :model (nn.Module): PyTorch model

    .. code-block:: python

        from QEfficient import QEFFAutoModelForSpeechSeq2Seq
        from processors import AutoProcessor

        # Initialize the model using from_pretrained similar to transformers.AutoModelForSpeechSeq2Seq.
        model = QEFFAutoModelForSpeechSeq2Seq.from_pretrained("model_name")

        # Now you can directly compile the model for Cloud AI 100
        model.compile(num_cores=16, device_group=[0])  # Considering you have a Cloud AI 100 SKU

        #prepare inputs
        processor = AutoProcessor.from_pretrained(model_name)
        input_audio, sample_rate = [...] # audio data loaded in via some external audio package, such as librosa or soundfile
        input_features = (
            processor(data, sampling_rate=sample_rate, return_tensors="pt").input_features.numpy().astype(np.float32)
        )
        decoder_input_ids = (
            torch.ones((batch_size, 1), dtype=torch.int64) * model.model.config.decoder_start_token_id
        ).numpy()
        decoder_position_ids = torch.arange(1, dtype=torch.int64).view(1, 1).repeat(batch_size, 1).numpy()
        inputs = dict(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            decoder_position_ids=decoder_position_ids,
        )

        # You can now execute the model
        model.generate(inputs, generation_len=150)
    """

    _hf_auto_class = AutoModelForSpeechSeq2Seq
    _pytorch_transforms = [CustomOpsTransform, AwqToMatmulNbitsTransform, GPTQToMatmulNbitsTransform, KVCacheTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(self, model: nn.Module, **kwargs):
        model_class_name = model.__class__.__name__
        if not (model_class_name.endswith("ForConditionalGeneration")):
            raise TypeError(f"Required pytorch module with ForConditionalGeneration, got {model_class_name}")

        super().__init__(model)
        self.model.config.use_cache = True
        self.num_layers = model.config.num_hidden_layers

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

    @property
    def get_model_config(self) -> dict:
        return self.model.config.__dict__

    def export(self, export_dir: Optional[str] = None) -> str:
        """
        Exports the model to ``ONNX`` format using ``torch.onnx.export``.

        ``Optional`` Args:
        :export_dir (str, optional): The directory path to store ONNX-graph.

        Returns:
            :str: Path of the generated ``ONNX`` graph.
        """
        inputs = self.model.get_dummy_inputs()
        dynamic_axes = self.model.get_onnx_dynamic_axes()
        output_names = self.model.get_output_names()
        return self._export(inputs, output_names, dynamic_axes, export_dir=export_dir)

    def compile(
        self,
        onnx_path: Optional[str] = None,
        compile_dir: Optional[str] = None,
        *,
        prefill_seq_len: Optional[int] = 1,
        encoder_ctx_len: Optional[int] = None,
        ctx_len: int = 150,
        full_batch_size: Optional[int] = None,
        kv_cache_batch_size: Optional[int] = None,
        batch_size: int = 1,
        num_devices: int = 1,
        num_cores: int = 16,  # FIXME: Make this mandatory arg
        mxfp6_matmul: bool = False,
        mxint8_kv_cache: bool = False,
        num_speculative_tokens: Optional[int] = None,
        **compiler_options,
    ) -> str:
        """
        This method compiles the exported ``ONNX`` model using the Cloud AI 100 Platform SDK compiler binary found at ``/opt/qti-aic/exec/qaic-exec`` and generates a ``qpc`` package.
        If the model has not been exported yet, this method will handle the export process.
        You can pass any other arguments that the `qaic-exec` takes as extra kwargs.

        ``Optional`` Args:
            :onnx_path (str, optional): Path to pre-exported onnx model.
            :compile_dir (str, optional): Path for saving the qpc generated.
            :encoder_ctx_len (int, optional): The maximum length of context for encoder, based on the AutoProcessor output. ``Defaults to checking config, if None in config then 1500``
            :ctx_len (int, optional): The maximum length of context to keep for decoding. ``Defaults to 150``.
            :batch_size (int, optional): Batch size. ``Defaults to 1``.
            :num_devices (int): Number of devices the model needs to be compiled for. Defaults to 1.
            :num_cores (int): Number of cores used to compile the model.
            :mxfp6_matmul (bool, optional): Whether to use ``mxfp6`` compression for weights. ``Defaults to False``.
            :aic_enable_depth_first (bool, optional): Enables DFS with default memory size. ``Defaults to False``.

            Other args are not yet implemented for AutoModelForSpeechSeq2Seq
        Returns:
            :str: Path of the compiled ``qpc`` package.
        """
        specializations, compiler_options = self.model.get_specializations(
            batch_size,
            encoder_ctx_len,
            ctx_len,
            **compiler_options,
        )

        if full_batch_size:
            logger.warning("Continuous batching is not yet enabled for AutoModelForSpeechSeq2Seq")

        if kv_cache_batch_size:
            logger.warning("Prefix caching is not yet enabled for AutoModelForSpeechSeq2Seq")

        if mxint8_kv_cache:
            logger.warning("mxint8 cache is not yet enabled for AutoModelForSpeechSeq2Seq")

        if num_speculative_tokens:
            logger.warning("Speculative decoding is not yet enabled for AutoModelForSpeechSeq2Seq")

        output_names = self.model.get_output_names()

        kv_cache_dtype = "float16"
        custom_io = {}

        custom_io["input_features"] = kv_cache_dtype

        # Slice output_names to get input names
        for output_name in output_names:
            if output_name.endswith("_RetainedState"):
                custom_io[output_name[: -len("_RetainedState")]] = kv_cache_dtype

        # Get output names
        for output_name in output_names:
            if output_name.endswith("_RetainedState"):
                custom_io[output_name] = kv_cache_dtype

        return self._compile(
            onnx_path,
            compile_dir,
            compile_only=True,
            retained_state=True,
            specializations=specializations,
            convert_to_fp16=True,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=num_devices,
            aic_num_cores=num_cores,
            custom_io=custom_io,
            **compiler_options,
        )

    def generate(
        self,
        inputs: torch.Tensor,
        generation_len: int,
        streamer: Optional[TextStreamer] = None,
        device_ids: List[int] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        This method generates output until ``endoftranscript`` or ``generation_len`` by executing the compiled ``qpc`` on ``Cloud AI 100`` Hardware cards.
        This is a sequential execution based on the ``batch_size`` of the compiled model and the number of audio tensor passed.

        ``Mandatory`` Args:
            :processor: autoprocessor to process inputs and decode logits
            :inputs (torch.Tensor): inputs to run the execution.
            :generation_len (int): length upto which to generate
            :device_id (List[int]): Ids of devices for running the qpc pass as [0] in case of normal model / [0, 1, 2, 3] in case of tensor slicing model
        Returns:
            :dict: Output from the ``AI_100`` or ``PyTorch`` runtime.
        """
        if not isinstance(self.qpc_path, Path):
            raise TypeError("Please run compile API first!")

        inputs = self.auto_correct_inputs(inputs)

        if self.qpc_session is None:
            self.qpc_session = QAICInferenceSession(str(self.qpc_path), device_ids)
            self.batch_size = self.qpc_session.bindings[0].dims[0]

        inputs["input_features"] = inputs["input_features"].numpy().astype(np.float16)

        # add start token id and initial position ids to inputs
        seq_len = 1
        inputs["input_ids"] = (
            torch.ones((self.batch_size, seq_len), dtype=torch.int64) * self.model.config.decoder_start_token_id
        ).numpy()
        inputs["position_ids"] = (
            torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(self.batch_size, 1).numpy()
        )

        self.qpc_session.skip_buffers(
            [x for x in self.qpc_session.input_names + self.qpc_session.output_names if x.startswith("past_")]
        )

        outputs = {
            "logits": np.random.randn(self.batch_size, 1, self.model.config.vocab_size).astype(np.float32),
        }
        self.qpc_session.set_buffers(outputs)

        # encoder run
        start = perf_counter()
        outputs = self.qpc_session.run(inputs)

        # array to hold generated tokens
        generated_ids = np.full((self.batch_size, generation_len + 1), self.model.config.eos_token_id)
        generated_ids[:, 0] = [self.model.config.decoder_start_token_id]
        logits = outputs["logits"]
        next_token = logits.argmax(-1)
        generated_ids[:, 1] = next_token.squeeze(1)

        if streamer:
            streamer.put(next_token)

        inputs["input_features"] = np.zeros((self.batch_size, self.model.config.num_mel_bins, 1)).astype(np.float16)

        loop_start = perf_counter()
        for num_tokens in range(generation_len):
            outputs = self.qpc_session.run(inputs)
            logits = outputs["logits"]
            next_token = logits.argmax(-1)
            generated_ids[:, num_tokens + 1] = next_token.squeeze(1)

            if next_token[0][0] == self.model.config.eos_token_id:
                break

            inputs["input_ids"] = next_token
            inputs["position_ids"] += 1

            if streamer:
                streamer.put(next_token)
        end = perf_counter()

        prefill_time, decode_perf, total_perf, total_time = calculate_latency(num_tokens, loop_start, start, end)

        return CloudAI100ExecInfoNew(
            batch_size=self.batch_size,
            generated_ids=generated_ids,
            perf_metrics=PerfMetrics(prefill_time, decode_perf, total_perf, total_time),
        )
