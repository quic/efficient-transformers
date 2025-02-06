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
    AutoProcessor,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    TextStreamer,
)

import QEfficient
from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.base.onnx_transforms import FP16ClipTransform, SplitTensorsTransform
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.generation.text_generation_inference import CloudAI100ExecInfoNew, PerfMetrics, get_compilation_dims
from QEfficient.transformers.models.pytorch_transforms import (
    CustomOpsTransform,
    KVCacheModuleMethodMapperTransform,
    KVCacheTransform,
    SpDTransform,
    VlmKVOffloadTransform,
    VlmNoKVOffloadTransform,
)
from QEfficient.transformers.quantizers.auto import QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING, with_replaced_quantizers
from QEfficient.transformers.quantizers.quant_transforms import AwqToMatmulNbitsTransform, GPTQToMatmulNbitsTransform
from QEfficient.utils import constants, get_padding_shape_from_config
from QEfficient.utils.cache import to_hashable
from QEfficient.utils.logging_utils import logger
from QEfficient.generation.text_generation_inference import calculate_latency, PerfMetrics, CloudAI100ExecInfo
from time import perf_counter

MODELS_WITH_ACCURACY_ISSUE_FOR_MXFP6 = ["MllamaForConditionalGeneration"]


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
    _pytorch_transforms = [AwqToMatmulNbitsTransform, GPTQToMatmulNbitsTransform, CustomOpsTransform, KVCacheTransform]
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
        # self.model.config.text_config.use_cache=True

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


class _QEffAutoModelForImageTextToTextDualQPC:
    _hf_auto_class = AutoModelForImageTextToText
    UNSUPPORTED_MODELS = ["LlavaForConditionalGeneration", "InternVLChatModel"]

    def __init__(
        self,
        model: nn.Module,
        **kwargs,
    ):
        if kwargs.pop("full_batch_size", None):
            raise NotImplementedError("Continuous batching is not supported for image-text-to-text models yet.")
        self.model = model
        self.config = model.config
        if self.model_name in self.UNSUPPORTED_MODELS:
            raise NotImplementedError(f"kv_offload is not yet supported for {self.model.__class__.__name__}")
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
        return [self.vision_model.qpc_path, self.lang_model.qpc_path]

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

    def compile(
        self,
        img_size: int,
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
        enable_qnn: bool = False,
        qnn_config: Optional[str] = None,
        **compiler_options,
    ) -> str:
        if (
            any(
                param is not None
                for param in [full_batch_size, kv_cache_batch_size, num_speculative_tokens, qnn_config]
            )
            or enable_qnn
        ):
            raise ValueError(
                f"Expected 'full_batch_size', 'kv_cache_batch_size', 'num_speculative_tokens', and 'qnn_config' to be None, and 'enable_qnn' to be False but got: "
                f"full_batch_size={full_batch_size}, kv_cache_batch_size={kv_cache_batch_size}, num_speculative_tokens={num_speculative_tokens}, "
                f"enable_qnn={enable_qnn}, qnn_config={qnn_config}"
            )

        output_names = self.model.get_output_names(kv_offload=True)

        specializations = self.model.get_specializations(
            batch_size=batch_size,
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            img_size=img_size,
            kv_offload=True,
            kv_offlaod=True,
            **compiler_options,
        )

        custom_io_vision = {}
        kv_cache_dtype = "mxint8" if mxint8_kv_cache else "float16"
        custom_io_vision["pixel_values"] = kv_cache_dtype
        for output_name in output_names["vision"]:
            custom_io_vision[output_name] = kv_cache_dtype

        if vision_onnx_path:
            self.vision_model.onnx_path = vision_onnx_path
        if lang_onnx_path:
            self.lang_model.onnx_path = lang_onnx_path

        if (self.vision_model.onnx_path is None and vision_onnx_path is None) or (
            self.lang_model.onnx_path is None and lang_onnx_path is None
        ):
            self.export()

        if mxfp6_matmul and self.model_name in MODELS_WITH_ACCURACY_ISSUE_FOR_MXFP6:
            logger.warning(
                "Due to accuracy issues of vision model fixing it's precision to fp16, while language model will be compiled for mxfp6"
            )

        self.vision_model._compile(
            compile_dir,
            compile_only=True,
            specializations=specializations["vision"],
            convert_to_fp16=True,
            mxfp6_matmul=False,
            mdp_ts_num_devices=num_devices,
            aic_num_cores=num_cores,
            custom_io=custom_io_vision,
            **compiler_options,
        )

        custom_io_lang = {}
        # Inputs
        for output_name in output_names["lang"]:
            if output_name.startswith("past_"):
                custom_io_lang[output_name[: -len("_RetainedState")]] = kv_cache_dtype

        # outputs
        for output_name in output_names["lang"]:
            if output_name.startswith("past_"):
                custom_io_lang[output_name] = kv_cache_dtype

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
            **compiler_options,
        )

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

        vision_inputs = {
            k: v for k, v in inputs.items() if k in {"pixel_values", "aspect_ratio_ids", "aspect_ratio_mask"}
        }

        vision_inputs["pixel_values"] = vision_inputs["pixel_values"].astype("float16")
        vision_outputs = vision_session.run(vision_inputs)

        lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}
        lang_inputs["position_ids"] = np.where(
            lang_inputs.pop("attention_mask"), np.arange(padded_len), -1
        )  # Need to use -1 as position_ids for invalid tokens

        vision_session.deactivate()
        lang_session.activate()

        lang_session.set_buffers(vision_outputs)

        # Run prefill
        for i in range(num_chunks):
            chunk_inputs = lang_inputs.copy()
            chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
            chunk_inputs["position_ids"] = lang_inputs["position_ids"][
                :, i * prefill_seq_len : (i + 1) * prefill_seq_len
            ]
            outputs = lang_session.run(chunk_inputs)

        prefill_time = perf_counter() - prefill_start
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
        total_time = decode_end - prefill_start
        total_perf = num_token / total_time

        return CloudAI100ExecInfoNew(
            batch_size=batch_size,
            generated_ids=generated_ids,
            perf_metrics=PerfMetrics(
                prefill_time=prefill_time, decode_perf=decode_perf, total_perf=total_perf, total_time=total_time
            ),
        )


class _QEFFAutoModelForImageTextToTextSingleQPC(QEFFTransformersBase):
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
        self._export(inputs, output_names, dynamic_axes, export_dir=export_dir)

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
        enable_qnn: bool = False,
        qnn_config: Optional[str] = None,
        **compiler_options,
    ) -> str:
        if (
            any(
                param is not None
                for param in [full_batch_size, kv_cache_batch_size, num_speculative_tokens, qnn_config]
            )
            or enable_qnn
        ):
            raise ValueError(
                f"Expected 'full_batch_size', 'kv_cache_batch_size', 'num_speculative_tokens', and 'qnn_config' to be None, and 'enable_qnn' to be False but got: "
                f"full_batch_size={full_batch_size}, kv_cache_batch_size={kv_cache_batch_size}, num_speculative_tokens={num_speculative_tokens}, "
                f"enable_qnn={enable_qnn}, qnn_config={qnn_config}"
            )

        output_names = self.model.get_output_names()

        # Get specializations from modelling file
        # TODO: expose this via the auto class as well
        specializations = self.model.get_specializations(
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
                custom_io[input_name[: -len("_RetainedState")]] = kv_cache_dtype

        # outputs
        for output_name in output_names:
            if output_name.endswith("_RetainedState"):
                custom_io[output_name] = kv_cache_dtype

        if self.model_name in MODELS_WITH_ACCURACY_ISSUE_FOR_MXFP6 and mxfp6_matmul:
            logger.warning(
                f"It is advised to use fp16 precision during compilation for {self.model.__class__.__name__} to avoid accuracy issues, got mxfp6_matmul=True"
            )

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
                + "got"
                + f"{[(k, v.shape, v.dtype) for k, v in inputs.items()]}"
            )

            raise RuntimeError(err_str)

        return {k: v for k, v in inputs.items() if k in [iinfo.name for iinfo in inputs_info]}

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


class QEFFAutoModelForImageTextToText:
    """
    A factory class for creating QEFFAutoModelForImageTextToText instances with for single and Dual QPC approach
    Attributes:
        _hf_auto_class (class): The Hugging Face AutoModel class for ImageTextToText models.
    """

    _hf_auto_class = AutoModelForImageTextToText

    def __new__(self, model: nn.Module, kv_offload: Optional[bool] = None, **kwargs):
        if model.config.architectures[0] in MODELS_WITH_ACCURACY_ISSUE_FOR_MXFP6 and not kv_offload:
            # For models with mxfp6 accuracy issue, we will use kv_offload=True by default
            if kv_offload is None:
                kv_offload = True
            else:
                logger.warning(f"Advised to use kv_offload=True for {model.__class__.__name__}")
        elif kv_offload is None:
            kv_offload = False

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
    _pytorch_transforms = [AwqToMatmulNbitsTransform, GPTQToMatmulNbitsTransform, CustomOpsTransform, KVCacheTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(
        self,
        model: nn.Module,
        continuous_batching: bool = False,
        is_tlm: bool = False,
        **kwargs,
    ):
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

        if is_tlm:
            # TODO: It is possible to always apply this transform and make value of indices as last indices by default in PyTorch
            self.model, transformed = SpDTransform.apply(self.model)
        self.is_tlm = is_tlm

    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname

    def __repr__(self) -> str:
        return self.__class__.__name__ + "\n" + self.model.__repr__

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, continuous_batching: bool = False, is_tlm: bool = False, *args, **kwargs
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

        kwargs.update({"attn_implementation": "eager", "low_cpu_mem_usage": False})
        model = cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        # This is support models that should be classified to in a different auto class but transformers load them via this class
        kv_offload = kwargs.pop("kv_offload", None)
        if model.__class__.__name__ in MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP:
            return MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP[model.__class__.__name__](
                model, kv_offload=kv_offload
            )

        return cls(model, is_tlm=is_tlm, continuous_batching=continuous_batching)

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
        enable_qnn: bool = False,
        qnn_config: Optional[str] = None,
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
            :enable_qnn (bool): Enables QNN Compilation. ``Defaults to False.``
            :qnn_config (str): Path of QNN Config parameters file. ``Defaults to None.``

        Returns:
            :str: Path of the compiled ``qpc`` package.
        """
        if self.is_tlm:
            # assert num_speculative_tokens cfg is acceptable if defined
            if num_speculative_tokens is None:
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

        if self.continuous_batching and full_batch_size is None:
            raise TypeError("missing required argument: 'full_batch_size'")

        if kv_cache_batch_size and not full_batch_size:
            raise ValueError(
                "Prefix caching is enabled only for continuous batching as of now. Please pass `full_batch_size` argument and make sure you pass `continuous_batching=True` in the `from_pretrained` call"
            )

        kv_cache_batch_size = (
            kv_cache_batch_size if kv_cache_batch_size else (full_batch_size if full_batch_size else batch_size)
        )
        # Define prefill specialization
        prefill_specialization = {
            # Prefill is always run with single BS for continuous batching.
            "batch_size": 1 if self.continuous_batching else batch_size,
            "seq_len": prefill_seq_len,
            "ctx_len": ctx_len,
            # TODO: should be renamed to kv_cache_batch_size in specialization too
        }
        prefill_specialization.update({"num_logits_to_keep": 1}) if self.is_tlm else ...
        if self.continuous_batching:
            prefill_specialization.update({"full_batch_size": kv_cache_batch_size})
        else:
            prefill_specialization.update({"batch_size": kv_cache_batch_size})
        prefill_specialization.update({"full_batch_exec_size": full_batch_size}) if full_batch_size else ...
        specializations = [
            prefill_specialization,
        ]

        # Skip decode specialization if we are not in continuous batching and prefill_seq_len=1 as this repeats prefill specialization
        if prefill_seq_len != 1 or self.continuous_batching:
            decode_specialization = {
                "batch_size": full_batch_size if self.continuous_batching else batch_size,
                "seq_len": num_speculative_tokens + 1 if self.is_tlm else 1,
                "ctx_len": ctx_len,
            }
            if self.continuous_batching:
                decode_specialization.update({"full_batch_size": kv_cache_batch_size})
            else:
                decode_specialization.update({"batch_size": kv_cache_batch_size})
            decode_specialization.update({"num_logits_to_keep": num_speculative_tokens + 1}) if self.is_tlm else ...
            specializations.append(decode_specialization)

        if enable_qnn:
            if compiler_options:
                logger.warning("Extra arguments to QNN compilation are supported via qnn_config.json only")

            qpc_path = self._qnn_compile(
                onnx_path,
                compile_dir,
                specializations=specializations,
                prefill_seq_len=prefill_seq_len,
                ctx_len=ctx_len,
                batch_size=batch_size,
                full_batch_size=full_batch_size,
                mdp_ts_num_devices=num_devices,
                num_cores=num_cores,
                mxfp6_matmul=mxfp6_matmul,
                mxint8_kv_cache=mxint8_kv_cache,
                qnn_config=qnn_config,
            )
        else:
            # Custom IO
            custom_io = {}
            kv_cache_dtype = "mxint8" if mxint8_kv_cache else "float16"
            for suffix in ["", "_RetainedState"]:
                for i in range(self.num_layers):
                    for kv in ["key", "value"]:
                        custom_io[f"past_{kv}.{i}{suffix}"] = kv_cache_dtype

            qpc_path = self._compile(
                onnx_path,
                compile_dir,
                compile_only=True,
                retained_state=True,
                specializations=specializations,
                convert_to_fp16=True,
                mxfp6_matmul=mxfp6_matmul,
                custom_io=custom_io,
                mdp_ts_num_devices=num_devices,
                num_speculative_tokens=num_speculative_tokens,
                aic_num_cores=num_cores,
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

class QEFFAutoModelForSpeechSeq2Seq(QEFFTransformersBase):
    """
    The QEFFAutoModelForSpeechSeq2Seq class is designed for transformers models with a sequence-to-sequence speech-to-text modeing head, including Whisper and other Encoder-Decoder speech models.
    Although it is possible to initialize the class directly, we highly recommend using the ``from_pretrained`` method for initialization.

    ``Mandatory`` Args:
        :model (nn.Module): PyTorch model

    .. code-block:: python # TODO - update this for speech

        from QEfficient import QEFFAutoModel
        from transformers import AutoTokenizer

        # Initialize the model using from_pretrained similar to transformers.AutoModel.
        model = QEFFAutoModel.from_pretrained("model_name")

        # Now you can directly compile the model for Cloud AI 100
        model.compile(num_cores=16, device_group=[0])  # Considering you have a Cloud AI 100 SKU

        #prepare input
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        inputs = tokenizer("My name is", return_tensors="pt")

        # You can now execute the model
        model.generate(inputs)
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
        encoder_seq_len = self.model.config.max_source_positions
        encoder_feature_count = self.model.config.num_mel_bins

        kv_cache_shape = get_padding_shape_from_config(self.model.config, bs, seq_len)
        kv_cross_cache_shape = get_padding_shape_from_config(self.model.config, bs, encoder_seq_len)
        example_inputs = {
            "input_features": torch.zeros((bs, encoder_feature_count, 1), dtype=torch.float32),
            "decoder_input_ids": torch.zeros((bs, seq_len), dtype=torch.int64),
            "decoder_position_ids": torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(bs, 1),
            "past_key_values": [[] for _ in range(self.num_layers)],
        }
        dynamic_axes = {
            "input_features": {0: "batch_size", 2: "feature_len"},
            "decoder_input_ids": {0: "batch_size", 1: "seq_len"},
            "decoder_position_ids": {0: "batch_size", 1: "seq_len"},
        }
        pkv_self_dynamic_axes = {
            0: "batch_size",
            2: "decoder_ctx_len",
        }
        pkv_cross_dynamic_axes = {
            0: "batch_size",
            2: "encoder_ctx_len",
        }
        output_names = ["logits"]

        for i in range(self.num_layers):
            for self_cross in ["self", "cross"]:
                for kv in ["key", "value"]:
                    example_inputs["past_key_values"][i].append(
                        torch.zeros(
                            kv_cache_shape if self_cross == "self" else kv_cross_cache_shape, dtype=torch.float32
                        )
                    )
                    dynamic_axes[f"past_{kv}_{self_cross}.{i}"] = (
                        pkv_self_dynamic_axes if self_cross == "self" else pkv_cross_dynamic_axes
                    )
                    output_names.append(f"past_{kv}_{self_cross}.{i}_RetainedState")

        self.onnx_path = self._export(
            example_inputs,
            output_names,
            dynamic_axes,
            export_dir=export_dir,
            encoder_decoder=True,
        )

        return self.onnx_path

    def compile(
        self,
        onnx_path: Optional[str] = None,
        compile_dir: Optional[str] = None,
        *,
        encoder_ctx_len: int = 1500,
        decoder_ctx_len: int = 150,
        feature_len: int = 3000,
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
        encoder_specializations = {
            "batch_size": batch_size,
            "seq_len": 1,
            "encoder_ctx_len": encoder_ctx_len,
            "decoder_ctx_len": decoder_ctx_len,
            "feature_len": feature_len,
        }

        decoder_specializations = {
            "batch_size": batch_size,
            "seq_len": 1,
            "encoder_ctx_len": encoder_ctx_len,
            "decoder_ctx_len": decoder_ctx_len,
            "feature_len": 1,  # important dummy feature so that torch.where knows whether to run cross attention or not
        }

        specializations = [encoder_specializations, decoder_specializations]

        self.qpc_path = self._compile(
            onnx_path,
            compile_dir,
            compile_only=True,
            retained_state=True,
            specializations=specializations,
            convert_to_fp16=True,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=num_devices,
            aic_num_cores=num_cores,
            **compiler_options,
        )

        return self.qpc_path

    def generate(
        self,
        processor: AutoProcessor,
        inputs: torch.Tensor,
        generation_len: int,
        sample_rate: int = 16000,
        device_ids: List[int] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        This method generates output until ``endoftranscript`` or ``generation_len`` by executing the compiled ``qpc`` on ``Cloud AI 100`` Hardware cards.
        This is a sequential execution based on the ``batch_size`` of the compiled model and the number of audio tensor passed.

        ``Mandatory`` Args:
            :processor: autoprocessor to process inputs and decode logits
            :inputs (np.ndarray): inputs to run the execution.
            :generation_len (int): length upto which to generate
            :sample_rate (int): sampling rate at which input audio is stored in inputs (needed for processor)
            :device_id (List[int]): Ids of devices for running the qpc pass as [0] in case of normal model / [0, 1, 2, 3] in case of tensor slicing model
        ``optional`` Args:
            :runtime_ai100 (bool, optional): ``AI_100`` and ``PyTorch`` runtime is supported as of now. Defaults to ``True`` for ``AI_100`` runtime.
        Returns:
            :dict: Output from the ``AI_100`` or ``PyTorch`` runtime.
        """
        if not isinstance(self.qpc_path, Path):
            raise TypeError("Please run compile API first!")

        if self.qpc_session is None:
            self.qpc_session = QAICInferenceSession(str(self.qpc_path), device_ids)
            self.batch_size = self.qpc_session.bindings[0].dims[0]

        seq_len = 1

        # prepare inputs
        input_features = (
            processor(inputs, sampling_rate=sample_rate, return_tensors="pt").input_features.numpy().astype(np.float32)
        )
        decoder_input_ids = (
            torch.ones((self.batch_size, seq_len), dtype=torch.int64) * self.model.config.decoder_start_token_id
        ).numpy()
        decoder_position_ids = (
            torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(self.batch_size, 1).numpy()
        )

        model_inputs = dict(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            decoder_position_ids=decoder_position_ids,
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
        outputs = self.qpc_session.run(model_inputs)

        # array to hold generated tokens
        generated_ids = np.full((self.batch_size, generation_len + 1), processor.tokenizer.pad_token_id)
        generated_ids[:, 0] = [self.model.config.decoder_start_token_id]
        logits = outputs["logits"]
        next_token = logits.argmax(-1)
        generated_ids[:, 1] = next_token.squeeze(1)

        model_inputs["input_features"] = np.random.randn(self.batch_size, self.model.config.num_mel_bins, 1).astype(
            np.float32
        )

        loop_start = perf_counter()
        for num_tokens in range(generation_len):
            outputs = self.qpc_session.run(model_inputs)
            logits = outputs["logits"]
            next_token = logits.argmax(-1)
            generated_ids[:, num_tokens + 1] = next_token.squeeze(1)

            if next_token[0][0] == processor.tokenizer.eos_token_id:
                break

            model_inputs["decoder_input_ids"] = next_token
            model_inputs["decoder_position_ids"] += 1
        end = perf_counter()

        prefill_time, decode_perf, total_perf, total_time = calculate_latency(num_tokens, loop_start, start, end)

        exec_info = CloudAI100ExecInfo(
            batch_size=self.batch_size,
            generated_texts=processor.batch_decode(generated_ids),
            generated_ids=generated_ids,
            perf_metrics=PerfMetrics(prefill_time, decode_perf, total_perf, total_time),
        )

        return exec_info
