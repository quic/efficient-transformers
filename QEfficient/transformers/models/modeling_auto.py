# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import warnings
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForCTC,
    AutoModelForImageTextToText,
    AutoModelForSpeechSeq2Seq,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    TextStreamer,
)

import QEfficient
from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.base.onnx_transforms import (
    FP16ClipTransform,
    SplitTensorsTransform,
)
from QEfficient.base.pytorch_transforms import SplitGateUpWeightsTransform
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.generation.text_generation_inference import (
    CloudAI100ExecInfoNew,
    PerfMetrics,
    calculate_latency,
    get_compilation_dims,
)
from QEfficient.generation.vlm_generation import VisionLanguageGeneration
from QEfficient.transformers.modeling_utils import DYNAMIC_SEQ_LEN_SUPPORTED_MODEL_ARCH
from QEfficient.transformers.models.pytorch_transforms import (
    BlockedKVAttentionTransform,
    CustomOpsTransform,
    KVCacheExternalModuleMapperTransform,
    KVCacheTransform,
    PoolingTransform,
    SamplerTransform,
    SpDTransform,
    VlmKVOffloadTransform,
    VlmNoKVOffloadTransform,
)
from QEfficient.transformers.quantizers.auto import QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING, with_replaced_quantizers
from QEfficient.transformers.quantizers.quant_transforms import (
    AwqToMatmulNbitsTransform,
    FP8DeQuantLinearToLinearTransform,
    GPTQToMatmulNbitsTransform,
    Mxfp4GptOssExpertDequantizeTransform,
)
from QEfficient.utils import (
    constants,
    get_padding_shape_from_config,
)
from QEfficient.utils.check_ccl_specializations import process_ccl_specializations
from QEfficient.utils.logging_utils import logger


class QEFFTransformersBase(QEFFBaseModel):
    """
    Base class for QEfficient wrappers around HuggingFace transformer models.

    This class provides common functionality for loading, representing, and managing
    HuggingFace models within the QEfficient framework. It serves as a parent
    for specific model types like `AutoModel`, `AutoModelForCausalLM`, etc.
    """

    _hf_auto_class: type

    def __init__(self, model: nn.Module, **kwargs) -> None:
        if (
            hasattr(model, "config")
            and hasattr(model.config, "quantization_config")
            and not isinstance(model.config.quantization_config, tuple(QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING.values()))
        ):
            raise AssertionError("Please use `from_pretrained` method to load quantized models")

        super().__init__(model, **kwargs)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "\n" + self.model.__repr__()

    @classmethod
    @with_replaced_quantizers
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        """
        Load a QEfficient transformer model from a pretrained HuggingFace model or local path.

        This is the recommended way to initialize any QEfficient transformer model.
        The interface is similar to ``transformers.AutoModel.from_pretrained``.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            Model card name from HuggingFace or local path to model directory.
        *args :
            Positional arguments passed directly to `cls._hf_auto_class.from_pretrained`.
        **kwargs :
            Keyword arguments passed directly to `cls._hf_auto_class.from_pretrained`.

            **Note:** `attn_implementation` and `low_cpu_mem_usage` are automatically set to "eager" and False respectively to ensure compatibility.

        Returns
        -------
        QEFFTransformersBase
            An instance of the specific QEFFAutoModel subclass, initialized with the pretrained weights.
        """
        if kwargs.get("attn_implementation", None) not in {None, "eager"}:
            logger.warning('Updating attn_implementation="eager"')

        if kwargs.get("low_cpu_mem_usage", None):
            logger.warning("Updating low_cpu_mem_usage=False")

        kwargs.update({"attn_implementation": "eager", "low_cpu_mem_usage": False})

        model = cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return cls(model, pretrained_model_name_or_path=pretrained_model_name_or_path)

    @property
    def model_name(self) -> str:
        """
        Get the name of the underlying HuggingFace model.

        Returns
        -------
        str
            The model's class name, with "QEff" or "QEFF" prefix removed if present.
        """
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname


class MultimodalUtilityMixin:
    """
    Mixin for multimodal models providing utilities like input auto-correction.

    This mixin ensures that inputs to multimodal models conform to the expected
    names, shapes, and dtypes defined by the model's `get_inputs_info` method.
    """

    def __new__(cls, *args, **kwargs):
        if cls is MultimodalUtilityMixin:
            raise TypeError(f"only children of '{cls.__name__}' may be instantiated")
        return object.__new__(cls)

    def auto_correct_inputs(self, inputs):
        """
        Validates and corrects model inputs to match expected specifications.

        Checks if the provided inputs dictionary contains all required keys and
        if the data types of the tensors match the model's specifications.
        It then filters the input dictionary to only include expected inputs.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            A dictionary of input tensors, where keys are input names and values are `torch.Tensor` objects.

        Returns
        -------
        Dict[str, torch.Tensor]
            A filtered dictionary of input tensors that match the model's expected inputs.

        Raises
        ------
        RuntimeError
            If any expected input is missing or has a mismatched data type.
        """
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
    QEfficient class for general transformer models from the HuggingFace hub (e.g., BERT, Sentence Transformers).

    This class provides a unified interface for loading, exporting, compiling, and running
    various encoder-only transformer models on Cloud AI 100 hardware. It supports pooling
    for embedding extraction.

    Example
    -------
    .. code-block:: python

        from QEfficient import QEFFAutoModel
        from transformers import AutoTokenizer

        model = QEFFAutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", pooling="mean")
        model.compile(num_cores=16)
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        inputs = tokenizer("My name is", return_tensors="pt")
        output = model.generate(inputs)
        print(output) # Output will be a dictionary containing extracted features.
    """

    _hf_auto_class = AutoModel
    _pytorch_transforms = [CustomOpsTransform, AwqToMatmulNbitsTransform, GPTQToMatmulNbitsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(self, model: nn.Module, pooling=None, **kwargs):
        """
        Initializes a QEFFAutoModel instance.

        Parameters
        ----------
        model : nn.Module
            The underlying HuggingFace PyTorch model.
        pooling : str or Callable, optional
            The pooling method to use for feature extraction.
            Options include: "mean", "max", "cls", "avg", or a custom Callable.
            Default is None (no pooling applied).
        **kwargs :
            Additional keyword arguments passed to the base class constructor.
        """
        super().__init__(model, **kwargs)

        # Make Embedding specific transforms like appending pooling
        if pooling:
            self.model, _ = PoolingTransform.apply(self.model, pooling)

        self.model.base_model.config.use_cache = True

        self.hash_params["qeff_auto_class"] = self.__class__.__name__

    @classmethod
    @with_replaced_quantizers
    def from_pretrained(cls, pretrained_model_name_or_path, pooling=None, *args, **kwargs):
        """
        Load a QEfficient transformer model from a pretrained HuggingFace model or local path.

        This is the recommended way to initialize a QEfficient transformer model. The interface is similar to
        ``transformers.AutoModel.from_pretrained``. Once initialized, you can use methods such as ``export``, ``compile``, and ``generate``.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            Model card name from HuggingFace or local path to model directory.
        pooling : str or Callable, optional
            The pooling method to use. Options include:
            - "mean": Mean pooling
            - "max": Max pooling
            - "cls": CLS token pooling
            - "avg": Average pooling
            - Callable: A custom pooling function
            - None: No pooling applied. Default is None.
        *args :
            Positional arguments passed directly to `cls._hf_auto_class.from_pretrained`.
        **kwargs :
            Additional keyword arguments passed directly to `cls._hf_auto_class.from_pretrained`.

            **Note:** `attn_implementation` and `low_cpu_mem_usage` are automatically
            set to "eager" and False respectively to ensure compatibility.

        Returns
        -------
        QEFFAutoModel
            An instance initialized with the pretrained weights.
        """
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
                model, kv_offload=kv_offload, **kwargs
            )

        return cls(model, pretrained_model_name_or_path=pretrained_model_name_or_path, pooling=pooling, **kwargs)

    @property
    def get_model_config(self) -> dict:
        """
        Get the model configuration as a dictionary.

        Returns
        -------
        dict
            The configuration dictionary of the underlying HuggingFace model.
        """
        return self.model.config.__dict__

    def export(self, export_dir: Optional[str] = None, use_onnx_subfunctions: bool = False) -> str:
        """
        Export the model to ONNX format using ``torch.onnx.export``.

        This method prepares example inputs and dynamic axes based on the model configuration,
        then exports the model to an ONNX graph suitable for compilation and deployment on Cloud AI 100 hardware.

        Parameters
        ----------
        export_dir : str, optional
            Directory path where the exported ONNX graph will be saved. If not provided,
            the default export directory is used.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False

        Returns
        -------
        str
            Path to the generated ONNX graph file.
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
            use_onnx_subfunctions=use_onnx_subfunctions,
        )

    def compile(
        self,
        onnx_path: Optional[str] = None,
        compile_dir: Optional[str] = None,
        *,
        seq_len: Union[int, List[int]] = 32,
        batch_size: int = 1,
        num_devices: int = 1,
        num_cores: int = 16,  # FIXME: Make this mandatory arg
        mxfp6_matmul: bool = False,
        use_onnx_subfunctions: bool = False,
        **compiler_options,
    ) -> str:
        """
        Compile the exported ONNX model using the Cloud AI 100 Platform SDK compiler.

        This method generates a ``qpc`` package. If the model has not been exported yet,
        this method will handle the export process. Additional arguments for the `qaic-exec`
        compiler can be passed as keyword arguments.

        Parameters
        ----------
        onnx_path : str, optional
            Path to a pre-exported ONNX model. If not provided, the model will be exported first.
        compile_dir : str, optional
            Directory to save the generated QPC package. If not provided, a default directory is used.
        seq_len : int or list of int, optional
            The length(s) of the prompt(s) to compile for. Can be a single integer or a list of integers
            to create multiple specializations. Default is 32.
        batch_size : int, optional
            Batch size. Default is 1.
        num_devices : int, optional
            Number of devices to compile for. Default is 1.
        num_cores : int, optional
            Number of cores to use for compilation.
        mxfp6_matmul : bool, optional
            Use MXFP6 compression for weights. Default is False.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False
        **compiler_options : dict
            Additional compiler options for QAIC or QNN compilers. These are passed directly
            to the underlying compilation command.

            **For QAIC Compiler:** Extra arguments for qaic-exec can be passed. Some common options include:

            - mos (int, optional): Effort level to reduce on-chip memory. Defaults to -1, meaning no effort. Defaults to -1.
            - aic_enable_depth_first (bool, optional): Enables DFS with default memory size. Defaults to False.
            - allow_mxint8_mdp_io (bool, optional): Allows MXINT8 compression of MDP IO traffic. Defaults to False.

            Params are converted to flags as below:

            - ``aic_num_cores=16`` -> ``-aic-num-cores=16``
            - ``convert_to_fp16=True`` -> ``-convert-to-fp16``

            **For QNN Compiler:** Following arguments can be passed as:

            - enable_qnn (bool): Enables QNN Compilation.
            - qnn_config (str): Path of QNN Config parameters file. Any extra parameters for QNN compilation can be passed via this file.

        Returns
        -------
        str
            Path to the compiled QPC package.

        """

        if isinstance(seq_len, list) and len(seq_len) >= 15:
            warnings.warn("Recommended: `seq_len` should contain fewer than 15 items.")

        specializations = [
            {"batch_size": batch_size, "seq_len": sl} for sl in (seq_len if isinstance(seq_len, list) else [seq_len])
        ]

        return self._compile(
            onnx_path=onnx_path,
            compile_dir=compile_dir,
            compile_only=True,
            specializations=specializations,
            convert_to_fp16=True,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=num_devices,
            aic_num_cores=num_cores,
            use_onnx_subfunctions=use_onnx_subfunctions,
            **compiler_options,
        )

    def generate(
        self,
        inputs: torch.Tensor,
        device_ids: List[int] = None,
        runtime_ai100: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Generate output by executing the compiled QPC on Cloud AI 100 hardware or using PyTorch runtime.

        This method runs sequential execution based on the compiled model's batch size and the number of prompts.
        If the number of prompts is not divisible by the batch size, the last batch will be dropped.

        Parameters
        ----------
        inputs : torch.Tensor or np.ndarray
            Input data for the model. For AI 100 runtime, this typically includes
            `input_ids` and `attention_mask`.
        device_ids : list of int, optional
            Device IDs for running the QPC. Defaults to `[0]` if not specified and `runtime_ai100` is True.
        runtime_ai100 : bool, optional
            Whether to use the AI 100 runtime for inference. If False, the PyTorch
            runtime will be used. Default is True.

        Returns
        -------
        torch.Tensor or np.ndarray
            Output from the AI 100 or PyTorch runtime. The type depends on the runtime and model.
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
        Generate features for a batch of inputs using the Cloud AI 100 hardware runtime.

        This method runs inference on the compiled QPC using the Cloud AI 100 accelerator.
        It automatically pads input tensors to match the compiled sequence length and handles session setup.

        Parameters
        ----------
        inputs : torch.Tensor or np.ndarray
            Input tensors for feature extraction. Must be a dictionary-like object
            including `input_ids` and `attention_mask`.
        device_ids : List[int], optional
            List of device IDs to use for inference. Defaults to [0].

        Returns
        -------
        np.ndarray
            Array containing the generated output features for each input in the batch.
        """

        if self.qpc_session is None:
            self.qpc_session = QAICInferenceSession(str(self.qpc_path), device_ids)
            self.batch_size = self.qpc_session.bindings[0].dims[0]

        # Dynamic switching to closest seq_Len based on input_ids_len
        input_ids_len = inputs["input_ids"].shape[1]

        for allowed_shape in self.qpc_session.allowed_shapes:
            seq_len_allowed = allowed_shape[1][1][1]

            if seq_len_allowed >= input_ids_len:
                self.seq_len = seq_len_allowed
                break

        # To handle single seq_len as we can't fetch allowed shapes for single seq_len
        self.seq_len = self.qpc_session.bindings[0].dims[1] if not hasattr(self, "seq_len") else self.seq_len

        input_ids = np.array(
            torch.nn.functional.pad(inputs["input_ids"], (0, self.seq_len - input_ids_len), "constant", 0)
        )
        attention_mask = np.array(
            torch.nn.functional.pad(
                inputs["attention_mask"], (0, self.seq_len - inputs["attention_mask"].size(1)), "constant", 0
            )
        )

        inputs = dict(input_ids=input_ids, attention_mask=attention_mask)

        # TODO: Remove try and catch after compiler fix
        try:
            outputs = {
                "output": np.random.randn(*list(self.qpc_session.bindings[2].dims)).astype(np.float32),
            }
            self.qpc_session.set_buffers(outputs)
            outputs = self.qpc_session.run(inputs)
        except Exception:
            outputs = {
                "output": np.random.randn(self.batch_size, self.seq_len, self.qpc_session.bindings[2].dims[1]).astype(
                    np.float32
                ),
            }
            self.qpc_session.set_buffers(outputs)
            outputs = self.qpc_session.run(inputs)
        return outputs

    def pytorch_feature_generate(self, model, inputs: Union[torch.Tensor, np.ndarray]) -> List[torch.Tensor]:
        """
        Generate features from a batch of inputs using the PyTorch model.

        This method runs the model in PyTorch (CPU/GPU) mode for feature extraction.

        Parameters
        ----------
        model : nn.Module
            The PyTorch model to use for inference.
        inputs : torch.Tensor or np.ndarray
            Input tensors for feature extraction. Expected to be a dictionary-like object.

        Returns
        -------
        List[torch.Tensor]
            List of output features generated by the model for each input.
        """
        return model(**inputs)


class QEffVisionEncoderForTextImageToTextModel(QEFFBaseModel):
    """
    QEfficient wrapper for the Vision Encoder component of a Text-to-Image-to-Text model.

    This class handles the export and compilation of the vision encoder part
    of multimodal models for optimal performance on Cloud AI 100 hardware.
    """

    _pytorch_transforms = [
        AwqToMatmulNbitsTransform,
        GPTQToMatmulNbitsTransform,
        CustomOpsTransform,
        KVCacheTransform,
        KVCacheExternalModuleMapperTransform,
    ]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(self, model: nn.modules, **kwargs):
        """
        Initializes the vision encoder component for multimodal models.

        Parameters
        ----------
        model : nn.Module
            The full HuggingFace multimodal model from which the vision encoder is extracted.
        **kwargs :
            Additional keyword arguments passed to the base class constructor.
        """
        super().__init__(model, **kwargs)
        self.model = model.get_qeff_vision_encoder()
        self.hash_params["qeff_auto_class"] = self.__class__.__name__

    def export(
        self,
        inputs,
        output_names,
        dynamic_axes,
        export_dir=None,
        offload_pt_weights=True,
        use_onnx_subfunctions: bool = False,
    ):
        """
        Exports the vision encoder component to ONNX format.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Example inputs for the ONNX export.
        output_names : List[str]
            List of output names for the ONNX graph.
        dynamic_axes : Dict[str, Dict[int, str]]
            Dynamic axes configuration for the ONNX graph.
        export_dir : str, optional
            Directory path where the exported ONNX graph will be saved. Default is None.
        offload_pt_weights : bool, optional
            If True, PyTorch weights will be offloaded after export. Default is True.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False

        Returns
        -------
        str
            Path to the generated ONNX graph file for the vision encoder.
        """
        return self._export(
            inputs,
            output_names,
            dynamic_axes,
            export_dir=export_dir,
            offload_pt_weights=offload_pt_weights,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )

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
        use_onnx_subfunctions: bool = False,
        **compiler_options,
    ) -> str:
        """
        Compiles the vision encoder component to a QPC package.

        Parameters
        ----------
        compile_dir : str
            Directory to save the generated QPC package.
        compile_only : bool
            If True, only compilation occurs without running inference.
        specializations : List[Dict[str, Union[int, str]]]
            List of dictionaries, each specifying a compilation specialization.
        convert_to_fp16 : bool
            If True, converts model to FP16 precision during compilation.
        mxfp6_matmul : bool
            If True, uses MXFP6 compression for MatMul weights.
        mdp_ts_num_devices : int
            Number of devices for multi-device (tensor slicing) compilation.
        aic_num_cores : int
            Number of cores to use for compilation.
        custom_io : Dict[str, str]
            Custom I/O configurations for the compiler.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False
        **compiler_options :
            Additional compiler options passed to the underlying compilation command.

        Returns
        -------
        str
            Path to the compiled QPC package for the vision encoder.
        """
        return self._compile(
            compile_dir=compile_dir,
            compile_only=compile_only,
            specializations=specializations,
            convert_to_fp16=convert_to_fp16,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=mdp_ts_num_devices,
            aic_num_cores=aic_num_cores,
            custom_io=custom_io,
            use_onnx_subfunctions=use_onnx_subfunctions,
            **compiler_options,
        )

    @property
    def model_name(self) -> str:
        """
        Get the name of the underlying vision encoder model.

        Returns
        -------
        str
            The model's class name, with "QEff" or "QEFF" prefix removed if present.
        """
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname

    @property
    def get_model_config(self) -> dict:
        """
        Get the configuration dictionary of the underlying HuggingFace vision model.

        Returns
        -------
        dict
            The configuration dictionary.
        """
        if hasattr(self.model.model, "vision_model"):
            return self.model.model.vision_model.config.__dict__
        return self.model.model.config.__dict__


class QEffCausalLMForTextImageToTextModel(QEFFBaseModel):
    """
    QEfficient wrapper for the Causal Language Model (decoder) component of a Text-to-Image-to-Text model.

    This class handles the export and compilation of the language decoder part
    of multimodal models for optimal performance on Cloud AI 100 hardware.
    """

    _pytorch_transforms = [
        AwqToMatmulNbitsTransform,
        GPTQToMatmulNbitsTransform,
        CustomOpsTransform,
        KVCacheTransform,
        VlmKVOffloadTransform,
        SplitGateUpWeightsTransform,
    ]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(self, model, **kwargs):
        """
        Initializes the language decoder component for multimodal models.

        Parameters
        ----------
        model : nn.Module
            The full HuggingFace multimodal model from which the language decoder is extracted.
        **kwargs :
            Additional keyword arguments passed to the base class constructor.
        """
        super().__init__(model, **kwargs)
        self.model = model.get_qeff_language_decoder()
        self.hash_params["qeff_auto_class"] = self.__class__.__name__

    def export(
        self,
        inputs,
        output_names,
        dynamic_axes,
        export_dir=None,
        offload_pt_weights=True,
        use_onnx_subfunctions: bool = False,
    ):
        """
        Exports the language decoder component to ONNX format.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Example inputs for the ONNX export.
        output_names : List[str]
            List of output names for the ONNX graph.
        dynamic_axes : Dict[str, Dict[int, str]]
            Dynamic axes configuration for the ONNX graph.
        export_dir : str, optional
            Directory path where the exported ONNX graph will be saved. Default is None.
        offload_pt_weights : bool, optional
            If True, PyTorch weights will be offloaded after export. Default is True.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False

        Returns
        -------
        str
            Path to the generated ONNX graph file for the language decoder.
        """
        return self._export(
            inputs,
            output_names,
            dynamic_axes,
            export_dir=export_dir,
            offload_pt_weights=offload_pt_weights,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )

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
        use_onnx_subfunctions: bool = False,
        **compiler_options,
    ) -> str:
        """
        Compiles the language decoder component to a QPC package.

        Parameters
        ----------
        compile_dir : str
            Directory to save the generated QPC package.
        compile_only : bool
            If True, only compilation occurs without running inference.
        specializations : List[Dict[str, Union[int, str]]]
            List of dictionaries, each specifying a compilation specialization.
        convert_to_fp16 : bool
            If True, converts model to FP16 precision during compilation.
        mxfp6_matmul : bool
            If True, uses MXFP6 compression for MatMul weights.
        mdp_ts_num_devices : int
            Number of devices for multi-device (tensor slicing) compilation.
        aic_num_cores : int
            Number of cores to use for compilation.
        custom_io : Dict[str, str]
            Custom I/O configurations for the compiler.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False
        **compiler_options :
            Additional compiler options passed to the underlying compilation command.

        Returns
        -------
        str
            Path to the compiled QPC package for the language decoder.
        """
        return self._compile(
            compile_dir=compile_dir,
            compile_only=compile_only,
            specializations=specializations,
            convert_to_fp16=convert_to_fp16,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=mdp_ts_num_devices,
            aic_num_cores=aic_num_cores,
            custom_io=custom_io,
            use_onnx_subfunctions=use_onnx_subfunctions,
            **compiler_options,
        )

    @property
    def model_name(self) -> str:
        """
        Get the name of the underlying language decoder model.

        Returns
        -------
        str
            The model's class name, with "QEff" or "QEFF" prefix removed if present.
        """
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname

    @property
    def get_model_config(self) -> dict:
        """
        Get the configuration dictionary of the underlying HuggingFace language model.

        Returns
        -------
        dict
            The configuration dictionary.
        """
        if hasattr(self.model, "language_model"):
            return self.model.language_model.config.__dict__
        return self.model.config.__dict__


class _QEffAutoModelForImageTextToTextDualQPC:
    """
    Internal class handling multimodal image-text-to-text models using a dual QPC approach.

    In this approach, the vision encoder and language model decoder are compiled
    into separate QPC packages. The vision encoder's KV cache might be offloaded
    to CPU or managed differently from the language model's KV cache.
    """

    _hf_auto_class = AutoModelForImageTextToText

    def __init__(
        self,
        model: nn.Module,
        continuous_batching: bool = False,
        qaic_config: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initializes the dual QPC multimodal model wrapper.

        Parameters
        ----------
        model : nn.Module
            The full HuggingFace multimodal model.
        **kwargs :
            Additional keyword arguments. `full_batch_size` is not supported here.

        Raises
        ------
        NotImplementedError
            If `full_batch_size` is provided.
        """
        if kwargs.pop("full_batch_size", None):
            raise NotImplementedError("Continuous batching is not supported for image-text-to-text models yet.")
        self.model = model
        self.config = model.config

        self.comp_ctx_lengths_prefill, self.comp_ctx_lengths_decode = process_ccl_specializations(qaic_config)

        self.vision_model = QEffVisionEncoderForTextImageToTextModel(model, **kwargs)
        self.lang_model = QEffCausalLMForTextImageToTextModel(model, **kwargs)
        self.continuous_batching = continuous_batching
        self.input_shapes, self.output_names = None, None

    @property
    def model_name(self) -> str:
        """
        Get the name of the underlying multimodal model.

        Returns
        -------
        str
            The model's class name, with "QEff" or "QEFF" prefix removed if present.
        """
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, qaic_config: Optional[dict] = None, **kwargs):
        """
        Load a QEfficient multimodal model for dual QPC from a pretrained HuggingFace model or local path.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            Model card name from HuggingFace or local path to model directory.
        **kwargs :
            Additional keyword arguments passed directly to `cls._hf_auto_class.from_pretrained`.
            Note: `attn_implementation` and `low_cpu_mem_usage` are automatically
            set to "eager" and False respectively to ensure compatibility.

        Returns
        -------
        _QEffAutoModelForImageTextToTextDualQPC
            An instance initialized with the pretrained weights.
        """
        if kwargs.get("attn_implementation", None) not in {None, "eager"}:
            logger.warning('Updating attn_implementation="eager"')

        if kwargs.get("low_cpu_mem_usage", None):
            logger.warning("Updating low_cpu_mem_usage=False")

        kwargs.update({"attn_implementation": "eager", "low_cpu_mem_usage": False})
        model = cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(
            model,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            qaic_config=qaic_config,
            **kwargs,
        )

    @property
    def onnx_path(self):
        """
        Get the ONNX paths for the vision and language model components.

        Returns
        -------
        List[str]
            A list containing the ONNX paths of the vision model and the language model.
        """
        return [self.vision_model.onnx_path, self.lang_model.onnx_path]

    @property
    def qpc_path(self):
        """
        Get the QPC paths for the vision and language model components.

        Returns
        -------
        Union[List[str], str, None]
            A list containing both QPC paths if both are compiled, or just one if only one is,
            or None if neither is compiled.
        """
        if self.vision_model.qpc_path and self.lang_model.qpc_path:
            return [self.vision_model.qpc_path, self.lang_model.qpc_path]
        elif self.vision_model.qpc_path:
            return self.vision_model.qpc_path
        else:
            return self.lang_model.qpc_path

    def export(
        self,
        export_dir: Optional[str] = None,
        use_onnx_subfunctions: bool = False,
        **kwargs,
    ) -> str:
        """
        Exports both the vision encoder and language decoder components to ONNX format.

        This method exports the vision component (optionally without offloading PyTorch weights)
        and the language component (with offloading PyTorch weights).

        Parameters
        ----------
        export_dir : str, optional
            Directory path where the exported ONNX graphs will be saved. Default is None.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False
        **kwargs :
            Additional keyword arguments.

        Returns
        -------
        List[str]
            A list containing the paths to the generated ONNX graph files for both components.
        """
        # TODO This is a temporary change as continous batching is enabled only for few models. Once support is added for all the models this exception handing can be removed.
        try:
            inputs = self.model.get_dummy_inputs(
                kv_offload=True,
                continuous_batching=self.continuous_batching,
                comp_ctx_lengths=self.comp_ctx_lengths_decode,
            )
            dynamic_axes = self.model.get_onnx_dynamic_axes(
                kv_offload=True,
                continuous_batching=self.continuous_batching,
                comp_ctx_lengths=self.comp_ctx_lengths_decode,
            )
        except TypeError:
            inputs = self.model.get_dummy_inputs(kv_offload=True, comp_ctx_lengths=self.comp_ctx_lengths_decode)
            dynamic_axes = self.model.get_onnx_dynamic_axes(
                kv_offload=True, comp_ctx_lengths=self.comp_ctx_lengths_decode
            )
        output_names = self.model.get_output_names(kv_offload=True)

        self.vision_model.export(
            inputs["vision"],
            output_names["vision"],
            dynamic_axes["vision"],
            export_dir=export_dir,
            offload_pt_weights=False,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )
        self.lang_model.export(
            inputs["lang"],
            output_names["lang"],
            dynamic_axes["lang"],
            export_dir=export_dir,
            offload_pt_weights=True,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )

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
        skip_vision: Optional[bool] = False,
        skip_lang: Optional[bool] = False,
        use_onnx_subfunctions: bool = False,
        **compiler_options,
    ) -> str:
        """
        Compiles both the vision encoder and language decoder components into QPC packages.

        Parameters
        ----------
        img_size : int, optional
            The image size to compile the vision model for. Default is None.
        vision_onnx_path : str, optional
            Path to a pre-exported ONNX file for the vision encoder. If None, it will be exported.
        lang_onnx_path : str, optional
            Path to a pre-exported ONNX file for the language decoder. If None, it will be exported.
        compile_dir : str, optional
            Directory to save the generated QPC packages.
        prefill_seq_len : int, optional
            Length of the prefill prompt for the language model. Default is None.
        ctx_len : int, optional
            Maximum context length for the language model. Default is None.
        batch_size : int, optional
            Batch size. Default is 1.
        full_batch_size : int, optional
            Not supported for this model; must be None.
        kv_cache_batch_size : int, optional
            Not supported for this model; must be None.
        num_devices : int, optional
            Number of devices to compile for. Default is 1.
        num_cores : int, optional
            Number of cores to use for compilation.
        mxfp6_matmul : bool, optional
            Use MXFP6 compression for weights in the language model. Default is False.
        mxint8_kv_cache : bool, optional
            Use MXINT8 compression for KV cache. Default is False.
        num_speculative_tokens : int, optional
            Not supported for this model; must be None.
        skip_vision : bool, optional
            If True, skips compilation of the vision encoder. Default is False.
        skip_lang : bool, optional
            If True, skips compilation of the language decoder. Default is False.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False
        **compiler_options : dict
            Additional compiler options for QAIC or QNN compilers.

        Returns
        -------
        Union[List[str], str, None]
            A list of paths to the compiled QPC packages, or a single path if only
            one component is compiled, or None if neither is compiled.

        Raises
        ------
        ValueError
            If `full_batch_size`, `kv_cache_batch_size`, or `num_speculative_tokens` are not None.
            If both `skip_lang` and `skip_vision` are True.
        """
        if skip_lang and skip_vision:
            raise ValueError("Expected at least one of 'skip_lang' or 'skip_vision' to be False")

        if self.continuous_batching and full_batch_size is None:
            raise TypeError("`full_batch_size` is required when `continuous_batching=True`.")

        if kv_cache_batch_size and not full_batch_size:
            raise ValueError(
                "KV caching requires continuous batching. Please set `full_batch_size` and "
                "enable `continuous_batching=True` in `from_pretrained`."
            )

        # Infer kv_cache_batch_size if not provided
        kv_cache_batch_size = kv_cache_batch_size or full_batch_size or batch_size

        output_names = self.model.get_output_names(kv_offload=True)

        # For supporting VLLM and Disaggregated with CCL
        if "comp_ctx_lengths_prefill" in compiler_options:
            self.comp_ctx_lengths_prefill = compiler_options.pop("comp_ctx_lengths_prefill")
            self.comp_ctx_lengths_decode = compiler_options.pop("comp_ctx_lengths_decode")

        specializations, compiler_options = self.model.get_specializations(
            batch_size=batch_size,
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            comp_ctx_lengths_prefill=self.comp_ctx_lengths_prefill,
            comp_ctx_lengths_decode=self.comp_ctx_lengths_decode,
            img_size=img_size,
            kv_offload=True,
            continuous_batching=self.continuous_batching,
            kv_cache_batch_size=kv_cache_batch_size,
            full_batch_size=full_batch_size,
            **compiler_options,
        )

        custom_io_vision = {}
        kv_cache_dtype = "mxint8" if mxint8_kv_cache else "float16"
        molmo = hasattr(self.model.config, "model_type") and self.model.config.model_type == "molmo"
        if molmo:
            custom_io_vision["image_masks"] = "float16"
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
            self.export(
                use_onnx_subfunctions=use_onnx_subfunctions,
            )

        # TODO this hould be removed once the continous batching is supported for all the models.
        compiler_options.pop("continuous_batching", None)
        compiler_options.pop("kv_cache_batch_size", None)
        compiler_options.pop("full_batch_size", None)

        if not skip_vision:
            self.vision_model._compile(
                compile_dir=compile_dir,
                compile_only=True,
                specializations=specializations["vision"],
                convert_to_fp16=True,
                mxfp6_matmul=constants.VISION_MXFP6_MATMUL,
                mdp_ts_num_devices=num_devices,
                aic_num_cores=num_cores,
                custom_io=custom_io_vision,
                mxint8_kv_cache=mxint8_kv_cache,
                use_onnx_subfunctions=use_onnx_subfunctions,
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
                compile_dir=compile_dir,
                compile_only=True,
                retained_state=True,
                specializations=specializations["lang"],
                convert_to_fp16=True,
                mxfp6_matmul=mxfp6_matmul,
                mdp_ts_num_devices=num_devices,
                aic_num_cores=num_cores,
                custom_io=custom_io_lang,
                mxint8_kv_cache=mxint8_kv_cache,
                use_onnx_subfunctions=use_onnx_subfunctions,
                **compiler_options,
            )
        return self.qpc_path

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer] = None,
        processor: Optional[AutoImageProcessor] = None,
        images: List[str] = None,
        prompts: List[str] = None,
        streamer: Optional[TextStreamer] = None,
        device_ids: List[int] = None,
        runtime_ai100: bool = True,
        generation_len: Optional[int] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Generates output by executing the compiled QPC(s) on Cloud AI 100 Hardware cards.

        This method coordinates inference between the vision encoder and language model decoder.

        Parameters
        ----------
        inputs : Dict[str, Union[torch.Tensor, np.ndarray]]
            Inputs to run the execution, typically includes `pixel_values`, `input_ids`,
            `attention_mask`, etc.
        tokenizer : PreTrainedTokenizer or PreTrainedTokenizerFast, optional
            Tokenizer for the model. Used when images and prompts are provided.
        processor : AutoImageProcessor, optional
            Processor for the model. Used when images and prompts are provided.
        images : List[str], optional
            List of image paths or PIL images to process.
        prompts : List[str], optional
            List of text prompts corresponding to the images.
        streamer : TextStreamer, optional
            A streamer object to display generated tokens in real-time. Default is None.
        device_ids : List[int], optional
            IDs of devices for running the QPC. E.g., `[0]` for a single device or
            `[0, 1, 2, 3]` for tensor slicing. Defaults to `[0]` if not specified.
        runtime_ai100 : bool, optional
            If True, uses the AI 100 runtime. PyTorch runtime is not supported for this model.
            Default is True.
        generation_len : int, optional
            The maximum number of tokens to generate. If None, it's inferred from `ctx_len`.

        Returns
        -------
        CloudAI100ExecInfoNew or np.ndarray
            Output from the AI 100 runtime, including generated IDs and performance metrics.

        Raises
        ------
        NotImplementedError
            If `runtime_ai100` is False.
        """
        if not runtime_ai100:
            raise NotImplementedError("PyTorch execution is not supported yet for this model!")

        # Use VisionLanguageGeneration for image-prompt pairs
        if (processor and images) or (tokenizer and prompts):
            # Create VisionLanguageGeneration instance
            batch_size_comp, ctx_len_comp, fbs = get_compilation_dims(self.lang_model.qpc_path)
            vlm_gen = VisionLanguageGeneration(
                qeff_model=self,
                lang_qpc_path=self.lang_model.qpc_path,
                vision_qpc_path=self.vision_model.qpc_path,
                tokenizer=tokenizer,
                processor=processor,
                device_id=device_ids,  # if device_ids is not None else [0],
                ctx_len=ctx_len_comp,
                full_batch_size=fbs,
                comp_ctx_lengths_prefill=self.comp_ctx_lengths_prefill,
                comp_ctx_lengths_decode=self.comp_ctx_lengths_decode,
            )

            # Call generate method
            return vlm_gen.generate(
                images=images,
                prompts=prompts,
                generation_len=generation_len,
                stream=streamer is not None,
            )

        # Fallback to kv_offload_generate for direct inputs (backward compatibility)
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
        """
        Performs generation for multimodal models with KV offloading to CPU.

        This method orchestrates the inference by running the vision encoder (if compiled)
        and then iteratively running the language decoder, managing KV cache states.

        Parameters
        ----------
        inputs : Dict[str, Union[torch.Tensor, np.ndarray]]
            Input tensors for the multimodal model.
        streamer : TextStreamer, optional
            A streamer object to display generated tokens in real-time. Default is None.
        device_ids : List[int], optional
            IDs of devices for running the QPC. Defaults to `[0]` if not specified.
        generation_len : int, optional
            The maximum number of tokens to generate. If None, it's inferred from `ctx_len`.

        Returns
        -------
        CloudAI100ExecInfoNew
            Execution information including generated IDs and performance metrics.

        Raises
        ------
        TypeError
            If the language model QPC is not compiled.
        AssertionError
            If `generation_len` is not greater than zero.
        """
        if not self.lang_model.qpc_path:
            raise TypeError("Please run compile API for language model first!")

        lang_session = QAICInferenceSession(self.lang_model.qpc_path, device_ids, activate=False)

        if self.vision_model.qpc_path:
            vision_session = QAICInferenceSession(self.vision_model.qpc_path, device_ids)

        batch_size, ctx_len, fbs = get_compilation_dims(self.lang_model.qpc_path)

        pad_token_id = 1

        # Skip inputs/outputs
        lang_session.skip_buffers(
            [
                x
                for x in lang_session.input_names + lang_session.output_names
                if x.startswith("past_") or x.endswith("_RetainedState")
            ]
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

        inputs["input_ids"] = torch.nn.functional.pad(
            inputs["input_ids"],
            (0, padded_len - input_ids_length),
            "constant",
            pad_token_id,
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
            k: v
            for k, v in inputs.items()
            if k
            in {"pixel_values", "image_masks", "image_input_idx", "valid_idx", "aspect_ratio_ids", "aspect_ratio_mask"}
        }

        vision_inputs_fp16 = {"pixel_values", "image_masks"}
        vision_inputs.update({k: vision_inputs[k].astype("float16") for k in vision_inputs_fp16 if k in vision_inputs})

        vision_start = perf_counter()

        vision_outputs = {}
        if vision_inputs:
            vision_outputs = vision_session.run(vision_inputs)
        vision_end = perf_counter()

        lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}

        if "position_ids" in inputs:
            lang_inputs["position_ids"] = inputs["position_ids"]
            lang_inputs.pop("attention_mask")
        else:
            lang_inputs["position_ids"] = np.where(
                lang_inputs.pop("attention_mask"), np.arange(padded_len), -1
            )  # Need to use -1 as position_ids for invalid tokens

        not_mllama = hasattr(self.model.config, "model_type") and self.model.config.model_type != "mllama"
        if not_mllama:
            lang_inputs["image_idx"] = np.array([[0]])

        if self.vision_model.qpc_path:
            vision_session.deactivate()
        lang_session.activate()

        lang_session.set_buffers(vision_outputs)

        if self.comp_ctx_lengths_prefill is not None:
            list_of_comp_ctx_lengths_prefill = [np.zeros(length) for length in self.comp_ctx_lengths_prefill]
            prefill_ccl_id = 0
            lang_inputs["comp_ctx_lengths"] = list_of_comp_ctx_lengths_prefill[prefill_ccl_id]

        lang_start = perf_counter()

        # Run prefill
        chunk_inputs = lang_inputs.copy()
        for i in range(num_chunks):
            if (
                self.comp_ctx_lengths_prefill is not None
                and (i + 1) * prefill_seq_len > self.comp_ctx_lengths_prefill[prefill_ccl_id]
            ):
                prefill_ccl_id = min(prefill_ccl_id + 1, len(self.comp_ctx_lengths_prefill) - 1)
                chunk_inputs["comp_ctx_lengths"] = list_of_comp_ctx_lengths_prefill[prefill_ccl_id]

            chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
            chunk_inputs["position_ids"] = lang_inputs["position_ids"][
                ..., i * prefill_seq_len : (i + 1) * prefill_seq_len
            ]
            outputs = lang_session.run(chunk_inputs)
            chunk_inputs["image_idx"] = outputs["image_idx_output"]

        prefill_time = perf_counter() - lang_start + vision_end - vision_start
        # Skip inputs/outputs again
        lang_session.skip_buffers(
            [
                x
                for x in lang_session.input_names + lang_session.output_names
                if x.startswith("past_") or x.endswith("_RetainedState")
            ]
        )
        if not_mllama:
            lang_session.skip_buffers(vision_outputs.keys())

        # Get first token
        lang_inputs["input_ids"] = outputs["logits"].argmax(2)
        lang_inputs["position_ids"] = np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1
        if "cross_attention_mask" in lang_inputs:
            bs, _, num_images, img_tiles = lang_inputs["cross_attention_mask"].shape
            lang_inputs["cross_attention_mask"] = torch.ones((bs, 1, num_images, img_tiles), dtype=torch.int64).numpy()
        generated_ids[:, 0] = lang_inputs["input_ids"].squeeze(1)

        if streamer:
            streamer.put(lang_inputs["input_ids"][0])

        # Decode loop
        if self.comp_ctx_lengths_decode is not None:
            max_ccl_id = len(self.comp_ctx_lengths_decode) - 1
            list_of_comp_ctx_lengths_decode = [np.zeros(length) for length in self.comp_ctx_lengths_decode]
            max_position_id = np.max(lang_inputs["position_ids"])
            ccl_id_initial = 0
            ccl_id = ccl_id_initial
            for i in range(ccl_id_initial, len(self.comp_ctx_lengths_decode)):
                if max_position_id < self.comp_ctx_lengths_decode[i]:
                    ccl_id = i
                    break
            lang_inputs["comp_ctx_lengths"] = list_of_comp_ctx_lengths_decode[ccl_id]

        decode_start = perf_counter()
        for num_token in range(1, generation_len):
            if self.comp_ctx_lengths_decode is not None:
                if max_position_id >= self.comp_ctx_lengths_decode[ccl_id] - 1:
                    ccl_id = min(ccl_id + 1, max_ccl_id)
                    lang_inputs["comp_ctx_lengths"] = list_of_comp_ctx_lengths_decode[ccl_id]

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
    """
    Internal class handling multimodal image-text-to-text models using a single QPC approach.

    In this approach, the entire multimodal model (vision encoder + language model decoder)
    is compiled into a single QPC package.
    """

    _hf_auto_class = AutoModelForImageTextToText
    _pytorch_transforms = [
        AwqToMatmulNbitsTransform,
        GPTQToMatmulNbitsTransform,
        CustomOpsTransform,
        KVCacheTransform,
        KVCacheExternalModuleMapperTransform,
        VlmNoKVOffloadTransform,
        SplitGateUpWeightsTransform,
    ]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(
        self,
        model: nn.Module,
        qaic_config: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initializes the single QPC multimodal model wrapper.

        Parameters
        ----------
        model : nn.Module
            The full HuggingFace multimodal model.
        **kwargs :
            Additional keyword arguments. `full_batch_size` is not supported here.

        Raises
        ------
        NotImplementedError
            If `full_batch_size` is provided.
        """
        if kwargs.pop("full_batch_size", None):
            raise NotImplementedError("Continuous batching is not supported for image-text-to-text models yet.")
        super().__init__(model, **kwargs)

        self.comp_ctx_lengths_prefill, self.comp_ctx_lengths_decode = process_ccl_specializations(qaic_config)

        # to handle internvl models
        if hasattr(self.model.config, "llm_config") and hasattr(self.model.config, "vision_config"):
            self.model.config.llm_config.use_cache = True
            self.model.config.llm_config._attn_implementation = "eager"
            self.model.config.vision_config.use_flash_attn = "false"
        else:
            if hasattr(self.model.config, "text_config"):
                self.model.config.text_config.use_cache = True
            else:
                self.model.config.use_cache = True
        self.hash_params["qeff_auto_class"] = self.__class__.__name__

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        qaic_config: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        """
        Load a QEfficient multimodal model for single QPC from a pretrained HuggingFace model or local path.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            Model card name from HuggingFace or local path to model directory.
        *args :
            Positional arguments passed directly to `cls._hf_auto_class.from_pretrained`.
        **kwargs :
            Additional keyword arguments passed directly to `cls._hf_auto_class.from_pretrained`.
            Note: `attn_implementation` and `low_cpu_mem_usage` are automatically
            set to "eager" and False respectively to ensure compatibility.
            Also, `_attn_implementation` and `use_flash_attn` are configured for VLM models.

        Returns
        -------
        _QEFFAutoModelForImageTextToTextSingleQPC
            An instance initialized with the pretrained weights.
        """
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

        return cls(
            model,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            qaic_config=qaic_config,
            **kwargs,
        )

    def export(
        self,
        export_dir: Optional[str] = None,
        use_onnx_subfunctions: bool = False,
        **kwargs,
    ) -> str:
        """
        Exports the entire multimodal model to ONNX format.

        Parameters
        ----------
        export_dir : str, optional
            Directory path where the exported ONNX graph will be saved. Default is None.
        **kwargs :
            Additional keyword arguments.

        Returns
        -------
        str
            Path to the generated ONNX graph file.
        """
        inputs = self.model.get_dummy_inputs(comp_ctx_lengths=self.comp_ctx_lengths_decode)
        dynamic_axes = self.model.get_onnx_dynamic_axes(comp_ctx_lengths=self.comp_ctx_lengths_decode)
        output_names = self.model.get_output_names()
        return self._export(
            inputs,
            output_names,
            dynamic_axes,
            export_dir=export_dir,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )

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
        use_onnx_subfunctions: bool = False,
        **compiler_options,
    ) -> str:
        """
        Compiles the exported ONNX model (single QPC) using the Cloud AI 100 Platform SDK compiler.

        This method generates a single ``qpc`` package for the entire multimodal model.

        Parameters
        ----------
        onnx_path : str, optional
            Path to a pre-exported ONNX model. If not provided, the model will be exported first.
        img_size : int, optional
            The image size to compile the vision part of the model for. Default is None.
        compile_dir : str, optional
            Directory to save the generated QPC package.
        prefill_seq_len : int, optional
            Length of the prefill prompt. Default is None.
        ctx_len : int, optional
            Maximum context length the compiled model can remember. Default is None.
        batch_size : int, optional
            Batch size. Default is 1.
        full_batch_size : int, optional
            Not supported for this model; must be None.
        kv_cache_batch_size : int, optional
            Not supported for this model; must be None.
        num_devices : int, optional
            Number of devices to compile for. Default is 1.
        num_cores : int, optional
            Number of cores to use for compilation.
        mxfp6_matmul : bool, optional
            Use MXFP6 compression for weights. Default is False.
        mxint8_kv_cache : bool, optional
            Use MXINT8 compression for KV cache. Default is False.
        num_speculative_tokens : int, optional
            Not supported for this model; must be None.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False
        **compiler_options : dict
            Additional compiler options for QAIC or QNN compilers.

        Returns
        -------
        str
            Path to the compiled QPC package.

        Raises
        ------
        ValueError
            If `full_batch_size`, `kv_cache_batch_size`, or `num_speculative_tokens` are not None.
        """
        if any(param is not None for param in [full_batch_size, kv_cache_batch_size, num_speculative_tokens]):
            raise ValueError(
                f"Expected 'full_batch_size', 'kv_cache_batch_size', 'num_speculative_tokens' to be None but got: "
                f"full_batch_size={full_batch_size}, kv_cache_batch_size={kv_cache_batch_size}, num_speculative_tokens={num_speculative_tokens}, "
            )

        # Infer kv_cache_batch_size if not provided
        kv_cache_batch_size = kv_cache_batch_size or full_batch_size or batch_size
        output_names = self.model.get_output_names()

        # For supporting VLLM and Disaggregated with CCL
        if "comp_ctx_lengths_prefill" in compiler_options:
            self.comp_ctx_lengths_prefill = compiler_options.pop("comp_ctx_lengths_prefill")
            self.comp_ctx_lengths_decode = compiler_options.pop("comp_ctx_lengths_decode")

        # Get specializations from modelling file
        # TODO: expose this via the auto class as well
        specializations, compiler_options = self.model.get_specializations(
            batch_size=batch_size,
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            comp_ctx_lengths_prefill=self.comp_ctx_lengths_prefill,
            comp_ctx_lengths_decode=self.comp_ctx_lengths_decode,
            kv_cache_batch_size=kv_cache_batch_size,
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

        # TODO this hould be removed once the continous batching is supported for all the models.
        compiler_options.pop("continuous_batching", None)
        compiler_options.pop("kv_cache_batch_size", None)
        compiler_options.pop("full_batch_size", None)

        self._compile(
            onnx_path=onnx_path,
            compile_dir=compile_dir,
            compile_only=True,
            retained_state=True,
            specializations=specializations,
            convert_to_fp16=True,
            mxfp6_matmul=mxfp6_matmul,
            custom_io=custom_io,
            mdp_ts_num_devices=num_devices,
            aic_num_cores=num_cores,
            mxint8_kv_cache=mxint8_kv_cache,
            use_onnx_subfunctions=use_onnx_subfunctions,
            **compiler_options,
        )
        return self.qpc_path

    def get_onnx_dynamic_axes(self):
        """
        Retrieves the dynamic axes configuration for ONNX export for this model.

        Returns
        -------
        Dict[str, Dict[int, str]]
            A dictionary specifying the dynamic axes for inputs.
        """
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
        Generates output by executing the compiled single QPC on Cloud AI 100 Hardware cards.

        Parameters
        ----------
        inputs : Dict[str, Union[torch.Tensor, np.ndarray]]
            Inputs to run the execution, typically includes `pixel_values`, `input_ids`,
            `attention_mask`, etc.
        streamer : TextStreamer, optional
            A streamer object to display generated tokens in real-time. Default is None.
        device_ids : List[int], optional
            IDs of devices for running the QPC. E.g., `[0]` for a single device or
            `[0, 1, 2, 3]` for tensor slicing. Defaults to `[0]` if not specified.
        runtime_ai100 : bool, optional
            If True, uses the AI 100 runtime. PyTorch runtime is not supported for this model.
            Default is True.
        generation_len : int, optional
            The maximum number of tokens to generate. If None, it's inferred from `ctx_len`.

        Returns
        -------
        CloudAI100ExecInfoNew or np.ndarray
            Output from the AI 100 runtime, including generated IDs and performance metrics.

        Raises
        ------
        NotImplementedError
            If `runtime_ai100` is False.
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
        """
        Performs generation for multimodal models using a single QPC on Cloud AI 100 hardware.

        Parameters
        ----------
        inputs : Dict[str, Union[torch.Tensor, np.ndarray]]
            Input tensors for the multimodal model.
        device_ids : List[int]
            IDs of devices for running the QPC.
        enable_debug_logs : bool, optional
            If True, enables debug logging for the QAIC inference session. Default is False.
        generation_len : int, optional
            The maximum number of tokens to generate. If None, it's inferred from `ctx_len`.
        streamer : TextStreamer, optional
            A streamer object to display generated tokens in real-time. Default is None.

        Returns
        -------
        CloudAI100ExecInfoNew
            Execution information including generated IDs and performance metrics.

        Raises
        ------
        AssertionError
            If `generation_len` is not greater than zero.
        """
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
        inputs["input_ids"] = torch.nn.functional.pad(
            inputs["input_ids"],
            (0, padded_len - input_ids_length),
            "constant",
            pad_token_id,
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
        inputs["image_idx"] = np.array([[0]])

        if self.comp_ctx_lengths_prefill is not None:
            list_of_comp_ctx_lengths_prefill = [np.zeros(length) for length in self.comp_ctx_lengths_prefill]
            prefill_ccl_id = 0
            inputs["comp_ctx_lengths"] = list_of_comp_ctx_lengths_prefill[prefill_ccl_id]

        qpc_session.activate()
        chunk_inputs = inputs.copy()
        prefill_start = perf_counter()

        # Run prefill
        for i in range(num_chunks):
            if (
                self.comp_ctx_lengths_prefill is not None
                and (i + 1) * prefill_seq_len > self.comp_ctx_lengths_prefill[prefill_ccl_id]
            ):
                prefill_ccl_id = min(prefill_ccl_id + 1, len(self.comp_ctx_lengths_prefill) - 1)
                chunk_inputs["comp_ctx_lengths"] = list_of_comp_ctx_lengths_prefill[prefill_ccl_id]

            chunk_inputs["input_ids"] = inputs["input_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
            chunk_inputs["position_ids"] = inputs["position_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
            outputs = qpc_session.run(chunk_inputs)
            chunk_inputs["image_idx"] = outputs["image_idx_output"]

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
        if self.comp_ctx_lengths_decode is not None:
            list_of_comp_ctx_lengths_decode = [np.zeros(length) for length in self.comp_ctx_lengths_decode]
            max_ccl_id = len(self.comp_ctx_lengths_decode) - 1
            max_position_id = np.max(inputs["position_ids"])
            ccl_id_initial = 0
            ccl_id = ccl_id_initial
            for i in range(ccl_id_initial, len(self.comp_ctx_lengths_decode)):
                if max_position_id < self.comp_ctx_lengths_decode[i]:
                    ccl_id = i
                    break
            inputs["comp_ctx_lengths"] = list_of_comp_ctx_lengths_decode[ccl_id]

        decode_start = perf_counter()
        for num_token in range(1, generation_len):
            if self.comp_ctx_lengths_decode is not None:
                if max_position_id >= self.comp_ctx_lengths_decode[ccl_id] - 1:
                    ccl_id = min(ccl_id + 1, max_ccl_id)
                    inputs["comp_ctx_lengths"] = list_of_comp_ctx_lengths_decode[ccl_id]

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
    def model_name(self) -> str:
        """
        Get the name of the underlying multimodal model.

        Returns
        -------
        str
            The model's class name, with "QEff" or "QEFF" prefix removed if present.
        """
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname

    @property
    def get_model_config(self) -> dict:
        """
        Get the configuration dictionary of the underlying HuggingFace model.

        Returns
        -------
        dict
            The configuration dictionary.
        """
        return self.model.config.__dict__


class QEFFAutoModelForImageTextToText:
    """
    QEfficient class for multimodal (image-text-to-text) models from the HuggingFace hub.

    This class supports both single and dual QPC (Quantized Package Compilation) approaches for efficient deployment on Cloud AI 100 hardware.
    It is recommended to use the ``from_pretrained`` method for initialization.

    Example
    -------
    .. code-block:: python

        import requests
        from PIL import Image
        from transformers import AutoProcessor, TextStreamer
        from QEfficient import QEFFAutoModelForImageTextToText

        HF_TOKEN = "" # Your HuggingFace token if needed
        model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        query = "Describe this image."
        image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"

        # STEP 1: Load processor and model
        processor = AutoProcessor.from_pretrained(model_name, token=HF_TOKEN)
        model = QEFFAutoModelForImageTextToText.from_pretrained(
            model_name, token=HF_TOKEN, attn_implementation="eager", kv_offload=False # kv_offload=False for single QPC
        )

        # STEP 2: Export & Compile
        model.compile(
            prefill_seq_len=32,
            ctx_len=512,
            img_size=560,
            num_cores=16,
            num_devices=1,
            mxfp6_matmul=False,
        )

        # STEP 3: Prepare inputs
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
            padding="max_length", # Consider padding strategy if max_length is crucial
            max_length=32,
        )

        # STEP 4: Run inference
        streamer = TextStreamer(processor.tokenizer)
        model.generate(inputs=inputs, streamer=streamer, generation_len=512)
    """

    _hf_auto_class = AutoModelForImageTextToText

    def __new__(
        self,
        model: nn.Module,
        kv_offload: Optional[bool] = True,
        continuous_batching: bool = False,
        qaic_config: Optional[dict] = None,
        **kwargs,
    ):
        """
        Instantiate the appropriate internal class for single or dual QPC mode.

        Parameters
        ----------
        model : nn.Module
            The loaded HuggingFace multimodal model.
        kv_offload : bool, optional
            If True, uses the dual QPC approach (vision encoder KV offloaded).
            If False, uses the single QPC approach (entire model in one QPC).
            Default is True.
        **kwargs :
            Additional keyword arguments passed to the constructor of the selected internal class.

        Returns
        -------
        Union[_QEffAutoModelForImageTextToTextDualQPC, _QEFFAutoModelForImageTextToTextSingleQPC]
            The wrapped model instance, configured for either dual or single QPC.
        """
        if kv_offload:
            return _QEffAutoModelForImageTextToTextDualQPC(
                model, continuous_batching, qaic_config=qaic_config, **kwargs
            )
        else:
            return _QEFFAutoModelForImageTextToTextSingleQPC(model, qaic_config=qaic_config, **kwargs)

    @classmethod
    @with_replaced_quantizers
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        kv_offload: Optional[bool] = None,
        continuous_batching: bool = False,
        qaic_config: Optional[dict] = None,
        **kwargs,
    ):
        """
        Load a QEfficient image-text-to-text model from a pretrained HuggingFace model or local path.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            Model card name from HuggingFace or local path to model directory.
        kv_offload : bool, optional
            If True, uses the dual QPC approach (vision encoder KV offloaded).
            If False, uses the single QPC approach (entire model in one QPC).
            If None, the default behavior of the internal classes is used (typically dual QPC).
        **kwargs :
            Additional arguments passed to HuggingFace's ``from_pretrained``.

            **Note:** `attn_implementation` and `low_cpu_mem_usage` are automatically set to "eager" and False respectively to ensure compatibility.
            `continuous_batching` is not supported for image-text-to-text models.

        Returns
        -------
        QEFFAutoModelForImageTextToText
            An instance initialized with the pretrained weights, wrapped for QEfficient.

        Raises
        ------
        NotImplementedError
            If `continuous_batching` is provided as True.
        """
        # TODO: add a check to see if kv_offload is allowed for given model by loading the config and checking architecture or type of config here.
        if continuous_batching and not kv_offload:
            NotImplementedError("Continuous batching is not supported for kv_offload = False")

        if kwargs.get("attn_implementation", None) not in {None, "eager"}:
            logger.warning('Updating attn_implementation="eager"')

        if kwargs.get("low_cpu_mem_usage", None):
            logger.warning("Updating low_cpu_mem_usage=False")

        kwargs.update({"attn_implementation": "eager", "low_cpu_mem_usage": False})

        model = cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(
            model,
            kv_offload=kv_offload,
            continuous_batching=continuous_batching,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            qaic_config=qaic_config,
            **kwargs,
        )


MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP = {
    "InternVLChatModel": QEFFAutoModelForImageTextToText,
    "MolmoForCausalLM": QEFFAutoModelForImageTextToText,
}


class QEFFAutoModelForCausalLM(QEFFBaseModel):
    """
    QEfficient class for Causal Language Models from the HuggingFace hub (e.g., GPT-2, Llama).

    This class provides a unified interface for loading, exporting, compiling, and generating
    text with causal language models on Cloud AI 100 hardware. It supports features like
    continuous batching, speculative decoding (TLM), and on-device sampling.

    Example
    -------
    .. code-block:: python

        from QEfficient import QEFFAutoModelForCausalLM
        from transformers import AutoTokenizer

        model = QEFFAutoModelForCausalLM.from_pretrained("gpt2")
        model.compile(num_cores=16)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model.generate(prompts=["Hi there!!"], tokenizer=tokenizer)
    """

    _hf_auto_class = AutoModelForCausalLM
    _pytorch_transforms = [
        AwqToMatmulNbitsTransform,
        GPTQToMatmulNbitsTransform,
        FP8DeQuantLinearToLinearTransform,
        Mxfp4GptOssExpertDequantizeTransform,
        CustomOpsTransform,
        KVCacheTransform,
        SplitGateUpWeightsTransform,
        KVCacheExternalModuleMapperTransform,
    ]
    _onnx_transforms = [
        FP16ClipTransform,
        SplitTensorsTransform,
    ]

    def __init__(
        self,
        model: nn.Module,
        continuous_batching: bool = False,
        qaic_config: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initializes a QEFFAutoModelForCausalLM instance.

        Parameters
        ----------
        model : nn.Module
            The underlying HuggingFace PyTorch Causal Language Model.
        continuous_batching : bool, optional
            If True, enables continuous batching mode for future compilation and execution.
            This setting must be consistent across `from_pretrained` and `compile` calls. Default is False.
        qaic_config : dict, optional
            A dictionary for QAIC-specific configurations. Supported keys include:
            - **speculative_model_type** (str): Specifies the type of Speculative Decoding model (e.g., "target").
            - **include_sampler** (bool): If True, enables on-device sampling of next tokens.
            - **return_pdfs** (bool): If True, returns probability distributions along with sampled tokens.
              For Speculative Decoding Target Language Models, this is always True.
            - **max_top_k_ids** (int): Maximum number of top K tokens (<= vocab size) to consider during sampling.
            - **num_kv_blocks** (int): Number of K/V blocks for BlockedKV attention implementation.
        **kwargs :
            Additional keyword arguments passed to the base class constructor.

        Raises
        ------
        TypeError
            If the provided `model` is not a CausalLM or LMHeadModel type.
        """
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
        # Set use_cache=True to get KV values as output during ONNX export
        model.config.use_cache = True

        self.comp_ctx_lengths_prefill, self.comp_ctx_lengths_decode = process_ccl_specializations(qaic_config)

        super().__init__(model, qaic_config=qaic_config, **kwargs)
        self.num_layers = model.config.num_hidden_layers
        self.continuous_batching = continuous_batching
        self.model.qaic_config = qaic_config
        self.model, transformed = SpDTransform.apply(self.model, qaic_config, **kwargs)
        self.is_tlm = transformed

        self.hash_params["qeff_auto_class"] = self.__class__.__name__

        # ---Sampling---
        # Note: SamplerTransform should be applied after all other transforms
        # are done. The role of the sampler is to just add nodes at the output of the
        # previous transform function.
        self.model, transformed = SamplerTransform.apply(self.model, qaic_config, **kwargs)
        # TODO : Update in qaic_config isn't updated in the hash due to SpDTransforms. Need to move
        # SpDTransforms to PytorchTransforms.
        if self.is_tlm:
            self.model.qaic_config["return_pdfs"] = True

        if self.model.qaic_config is not None and self.model.qaic_config.get("num_kv_blocks", None) is not None:
            BlockedKVAttentionTransform.apply(model, num_kv_blocks=self.model.qaic_config.get("num_kv_blocks"))

    @property
    def model_name(self) -> str:
        """
        Get the name of the underlying Causal Language Model.

        Returns
        -------
        str
            The model's class name, with "QEff" or "QEFF" prefix removed if present.
        """
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
        Load a QEfficient Causal Language Model from a pretrained HuggingFace model or local path.

        This is the recommended way to initialize a QEfficient Causal Language Model.
        The interface is similar to ``transformers.AutoModelForCausalLM.from_pretrained``.
        Once initialized, you can use methods such as ``export``, ``compile``, and ``generate``.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            Model card name from HuggingFace or local path to model directory.
        continuous_batching : bool, optional
            Whether this model will be used for continuous batching in the future.
            If not set to True here, the model cannot be exported/compiled for
            continuous batching later. Default is False.
        qaic_config : dict, optional
            QAIC config dictionary. Supported keys include:

            - **speculative_model_type** (str): Specify Speculative Decoding Target Language Models.
            - **include_sampler** (bool): Enable/Disable sampling of next tokens.
            - **return_pdfs** (bool): Return probability distributions along with sampled next tokens.
              For Speculative Decoding Target Language Model, ``return_pdfs=True`` always.
              Otherwise, ``return_pdfs=True`` for Speculative Decoding Draft Language Model
              and ``return_pdfs=False`` for regular model.
            - **max_top_k_ids** (int): Maximum number of top K tokens (<= vocab size) to consider during sampling.
              The values provided in ``top_ks`` tensor must be less than this maximum limit.

        *args :
            Positional arguments passed directly to `cls._hf_auto_class.from_pretrained`.
        **kwargs :
            Additional keyword arguments passed directly to `cls._hf_auto_class.from_pretrained`.

            **Note:** `attn_implementation` and `low_cpu_mem_usage` are automatically
            set to "eager" and False respectively to ensure compatibility.

        Returns
        -------
        QEFFAutoModelForCausalLM
            An instance initialized with the pretrained weights.
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
                model,
                kv_offload=kv_offload,
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                qaic_config=qaic_config,
                **kwargs,
            )
        return cls(
            model,
            continuous_batching=continuous_batching,
            qaic_config=qaic_config,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **kwargs,
        )

    @property
    def get_model_config(self) -> dict:
        """
        Get the model configuration as a dictionary.

        Returns
        -------
        dict
            The configuration dictionary of the underlying HuggingFace model.
        """
        return self.model.config.__dict__

    def export(self, export_dir: Optional[str] = None, use_onnx_subfunctions: bool = False, **kwargs) -> str:
        """
        Export the model to ONNX format using ``torch.onnx.export``.

        This method prepares example inputs and dynamic axes based on the model configuration,
        then exports the model to an ONNX graph suitable for compilation and deployment
        on Cloud AI 100 hardware. It handles KV cache inputs/outputs and sampler-related inputs.

        Parameters
        ----------
        export_dir : str, optional
            Directory path where the exported ONNX graph will be saved.
            If not provided, the default export directory is used.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False
        Returns
        -------
        str
            Path to the generated ONNX graph file.
        """
        bs: int = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        seq_len: int = constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
        fbs: int = constants.ONNX_EXPORT_EXAMPLE_FBS
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
        if self.comp_ctx_lengths_prefill is not None:
            example_inputs["comp_ctx_lengths"] = torch.randint(0, 512, (512,), dtype=torch.long)
            dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}

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
        output_names = []
        if self.model.qaic_config is not None and self.model.qaic_config.get("include_sampler", False):
            if self.model.qaic_config.get("return_pdfs", False):
                output_names.append("probs")
            output_names.append("next_tokens")
        else:
            output_names.append("logits")

        # TODO Update the get_padding_shape_from_config method to handle the case when the model config has attention_chunk_size or sliding_window and it should return a list of shapes for each layer
        if (
            hasattr(self.model.config, "model_type")
            and self.model.config.model_type in DYNAMIC_SEQ_LEN_SUPPORTED_MODEL_ARCH
        ):
            pkv_cache = self.model.get_dummy_pkv_cache(
                self.model.config, fbs if self.continuous_batching else bs, seq_len
            )
            for i in range(self.num_layers):
                for kv in ["key", "value"]:
                    example_inputs["past_key_values"][i].append(torch.zeros(pkv_cache[0][0].shape, dtype=torch.float32))
                    dynamic_axes[f"past_{kv}.{i}"] = pkv_dynamic_axes
                    output_names.append(f"past_{kv}.{i}_RetainedState")

        else:
            # HACK: create common function for this including above if condition code
            pkv_dynamic_axes = (
                self.model.get_pkv_dynamic_axes() if hasattr(self.model, "get_pkv_dynamic_axes") else pkv_dynamic_axes
            )
            pkv_dynamic_axes = (
                [pkv_dynamic_axes] * self.model.config.num_hidden_layers
                if isinstance(pkv_dynamic_axes, dict)
                else pkv_dynamic_axes
            )

            for i in range(self.num_layers):
                pkv_dynamic_axes[i][0] = "full_batch_size" if self.continuous_batching else "batch_size"
                for kv in ["key", "value"]:
                    example_inputs["past_key_values"][i].append(torch.zeros(kv_cache_shape, dtype=torch.float32))
                    dynamic_axes[f"past_{kv}.{i}"] = pkv_dynamic_axes[i]
                    output_names.append(f"past_{kv}.{i}_RetainedState")

        if self.continuous_batching:
            example_inputs["batch_index"] = torch.arange(bs).view(bs, 1)
            dynamic_axes["batch_index"] = {0: "batch_size"}

        if self.is_tlm:
            nlk = constants.ONNX_EXPORT_EXAMPLE_NLK  # Number of Logits to Keep
            example_inputs["num_logits_to_keep"] = torch.arange(nlk).view(nlk, 1)
            dynamic_axes["num_logits_to_keep"] = {0: "num_logits_to_keep"}

        if self.model.qaic_config is not None and self.model.qaic_config.get("include_sampler", False):
            example_inputs, output_names, dynamic_axes = self.get_sampling_inputs_and_outputs(
                example_inputs=example_inputs,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

        return self._export(
            example_inputs,
            output_names,
            dynamic_axes,
            export_dir=export_dir,
            use_onnx_subfunctions=use_onnx_subfunctions,
            offload_pt_weights=kwargs.get("offload_pt_weights", True),
        )

    def get_sampling_inputs_and_outputs(
        self,
        example_inputs: Dict[str, torch.Tensor],
        output_names: List[str],
        dynamic_axes: Dict[str, Dict[int, str]],
    ):
        """
        Updates the example inputs, output names, and dynamic axes to include
        parameters relevant for on-device sampling during ONNX export.

        Parameters
        ----------
        example_inputs : Dict[str, torch.Tensor]
            Current dictionary of example inputs.
        output_names : List[str]
            Current list of output names.
        dynamic_axes : Dict[str, Dict[int, str]]
            Current dictionary of dynamic axes configurations.

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], List[str], Dict[str, Dict[int, str]]]
            Updated example inputs, output names, and dynamic axes including
            sampling-related parameters.
        """
        bs: int = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        fbs: int = constants.ONNX_EXPORT_EXAMPLE_FBS

        example_inputs["last_accepted_output_tokens"] = torch.zeros(
            (bs, constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN), dtype=torch.int64
        )
        dynamic_axes["last_accepted_output_tokens"] = {0: "batch_size", 1: "seq_len"}

        example_inputs["past_repetition_penalty_buffer"] = torch.zeros(
            (fbs if self.continuous_batching else bs, self.model.config.vocab_size), dtype=torch.bool
        )
        dynamic_axes["past_repetition_penalty_buffer"] = {
            0: "full_batch_size" if self.continuous_batching else "batch_size",
        }
        output_names.append("past_repetition_penalty_buffer_RetainedState")

        example_inputs["repetition_penalties"] = (
            torch.ones((bs, 1), dtype=torch.float) * constants.ONNX_EXPORT_EXAMPLE_REPETITION_PENALTIES
        )
        dynamic_axes["repetition_penalties"] = {0: "batch_size"}

        example_inputs["past_presence_penalty_buffer"] = torch.zeros(
            (fbs if self.continuous_batching else bs, self.model.config.vocab_size), dtype=torch.bool
        )
        dynamic_axes["past_presence_penalty_buffer"] = {
            0: "full_batch_size" if self.continuous_batching else "batch_size",
        }
        output_names.append("past_presence_penalty_buffer_RetainedState")

        example_inputs["presence_penalties"] = (
            torch.zeros((bs, 1), dtype=torch.float) + constants.ONNX_EXPORT_EXAMPLE_PRESENCE_PENALTIES
        )
        dynamic_axes["presence_penalties"] = {0: "batch_size"}

        example_inputs["temperatures"] = (
            torch.ones((bs, 1), dtype=torch.float) * constants.ONNX_EXPORT_EXAMPLE_TEMPERATURES
        )
        dynamic_axes["temperatures"] = {0: "batch_size"}

        max_top_k_ids = self.model.qaic_config.get("max_top_k_ids", constants.ONNX_EXPORT_EXAMPLE_MAX_TOP_K_IDS)
        example_inputs["top_ks"] = torch.randint(1, max_top_k_ids, size=(bs, 1)).to(torch.int32)
        dynamic_axes["top_ks"] = {0: "batch_size"}

        example_inputs["top_ps"] = torch.ones((bs, 1), dtype=torch.float) * constants.ONNX_EXPORT_EXAMPLE_TOP_PS
        dynamic_axes["top_ps"] = {0: "batch_size"}

        example_inputs["min_ps"] = torch.ones((bs, 1), dtype=torch.float) * constants.ONNX_EXPORT_EXAMPLE_MIN_PS
        dynamic_axes["min_ps"] = {0: "batch_size"}

        example_inputs["random_numbers"] = torch.rand((bs, 1), dtype=torch.float)
        dynamic_axes["random_numbers"] = {0: "batch_size"}

        return example_inputs, output_names, dynamic_axes

    def build_prefill_specialization(
        self,
        prefill_seq_len: int = 32,
        ctx_len: int = 128,
        comp_ctx_lengths: Optional[int] = None,
        batch_size: int = 1,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
    ):
        """
        Builds a dictionary representing a compilation specialization for the prefill phase.

        Parameters
        ----------
        prefill_seq_len : int, optional
            Length of the prefill prompt. Default is 32.
        ctx_len : int, optional
            Maximum context length the compiled model can remember. Default is 128.
        batch_size : int, optional
            Batch size for the prefill. Default is 1.
        kv_cache_batch_size : int, optional
            Batch size for KV cache. If not provided, it defaults based on `full_batch_size` or `batch_size`.
        full_batch_size : int, optional
            Continuous batching batch size. Used if `continuous_batching` is enabled. Default is None.

        Returns
        -------
        Dict[str, Union[int, str]]
            A dictionary defining the prefill specialization.
        """
        if hasattr(self.model, "get_specializations"):
            spec = self.model.get_specializations(
                batch_size=1 if self.continuous_batching else batch_size,
                prefill_seq_len=prefill_seq_len,
                ctx_len=ctx_len,
            )[0]
        else:
            spec = {
                "batch_size": 1 if self.continuous_batching else batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
            }
        if comp_ctx_lengths is not None:
            spec["comp_ctx_lengths"] = comp_ctx_lengths
        spec["num_logits_to_keep"] = 1 if self.is_tlm else None
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
        comp_ctx_lengths: Optional[int] = None,
        batch_size: int = 1,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
        num_speculative_tokens: Optional[int] = None,
    ):
        """
        Builds a dictionary representing a compilation specialization for the decode phase.

        Parameters
        ----------
        prefill_seq_len : int, optional
            Length of the prefill prompt. Used to avoid duplicate specializations. Default is 32.
        ctx_len : int, optional
            Maximum context length the compiled model can remember. Default is 128.
        batch_size : int, optional
            Batch size for the decode phase. Default is 1.
        kv_cache_batch_size : int, optional
            Batch size for KV cache. If not provided, it defaults based on `full_batch_size` or `batch_size`.
        full_batch_size : int, optional
            Continuous batching batch size. Used if `continuous_batching` is enabled. Default is None.
        num_speculative_tokens : int, optional
            Number of speculative tokens for Speculative Decoding Target Language Model. Default is None.

        Returns
        -------
        Optional[Dict[str, Union[int, str]]]
            A dictionary defining the decode specialization, or None if it would be a duplicate
            of the prefill specialization (e.g., if prefill_seq_len is 1 and not continuous batching).
        """
        if prefill_seq_len == 1 and not self.continuous_batching:
            return None  # Avoid duplication with prefill

        if hasattr(self.model, "get_specializations"):
            spec = self.model.get_specializations(
                batch_size=full_batch_size if self.continuous_batching else batch_size,
                prefill_seq_len=(num_speculative_tokens + 1) if self.is_tlm else 1,
                ctx_len=ctx_len,
            )[1]
        else:
            spec = {
                "batch_size": full_batch_size if self.continuous_batching else batch_size,
                "seq_len": (num_speculative_tokens + 1) if self.is_tlm else 1,
                "ctx_len": ctx_len,
            }
        if comp_ctx_lengths is not None:
            spec["comp_ctx_lengths"] = comp_ctx_lengths

        spec["num_logits_to_keep"] = (num_speculative_tokens + 1) if self.is_tlm else None

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
        use_onnx_subfunctions: bool = False,
        **compiler_options,
    ) -> str:
        """
        Compile the exported ONNX model using the Cloud AI 100 Platform SDK compiler.

        This method generates a ``qpc`` package. If the model has not been exported yet,
        this method will handle the export process. Additional arguments for the `qaic-exec`
        compiler can be passed as keyword arguments.

        Parameters
        ----------
        onnx_path : str, optional
            Path to a pre-exported ONNX model. If not provided, the model will be exported first.
        compile_dir : str, optional
            Directory to save the generated QPC package. If not provided, a default directory is used.
        prefill_seq_len : int, optional
            Length of the prefill prompt. Default is 32.
        ctx_len : int, optional
            Maximum context length the compiled model can remember. Default is 128.
        batch_size : int, optional
            Batch size. Default is 1.
        full_batch_size : int, optional
            Continuous batching batch size. Required if `continuous_batching=True` was
            set during `from_pretrained`.
        kv_cache_batch_size : int, optional
            Batch size for KV cache. If not provided, it defaults to `full_batch_size` (if
            continuous batching) or `batch_size`.
        num_devices : int, optional
            Number of devices to compile for. Default is 1.
        num_cores : int, optional
            Number of cores to use for compilation.
        mxfp6_matmul : bool, optional
            Use MXFP6 compression for weights. Default is False.
        mxint8_kv_cache : bool, optional
            Use MXINT8 compression for KV cache. Default is False.
        num_speculative_tokens : int, optional
            Number of speculative tokens for Speculative Decoding Target Language Model.
            Required if the model is configured as a Target Language Model (`is_tlm=True`).
        prefill_only : bool, optional
            If True, compiles only for the prefill stage. If False, compiles only for
            the decode stage. If None, compiles for both stages. Default is None.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False
        **compiler_options : dict
            Additional compiler options for QAIC or QNN compilers.

            **For QAIC Compiler:** Extra arguments for qaic-exec can be passed. Some common options include:

            - mos (int, optional): Effort level to reduce on-chip memory. Defaults to -1, meaning no effort. Defaults to -1.
            - aic_enable_depth_first (bool, optional): Enables DFS with default memory size. Defaults to False.
            - allow_mxint8_mdp_io (bool, optional): Allows MXINT8 compression of MDP IO traffic. Defaults to False.

            Params are converted to flags as below:

            - ``aic_num_cores=16`` -> ``-aic-num-cores=16``
            - ``convert_to_fp16=True`` -> ``-convert-to-fp16``

            **For QNN Compiler:** Following arguments can be passed as:

            - enable_qnn (bool): Enables QNN Compilation.
            - qnn_config (str): Path of QNN Config parameters file. Any extra parameters for QNN compilation can be passed via this file.

        Returns
        -------
        str
            Path to the compiled QPC package.

        Raises
        ------
        TypeError
            If `prefill_only` is not a boolean.
            If `full_batch_size` is None when `continuous_batching` is True.
            If `num_speculative_tokens` is None when the model is a TLM.
        ValueError
            If KV caching is requested without continuous batching (`full_batch_size`).
            If `include_sampler` is True and `num_speculative_tokens` is greater than 0.
            If `num_speculative_tokens` is not an integer greater than 1.
            If `prefill_seq_len` is less than `num_speculative_tokens + 1` for TLM models.

        """

        # For supporting VLLM and Disaggregated with CCL
        if "comp_ctx_lengths_prefill" in compiler_options and "comp_ctx_lengths_decode" in compiler_options:
            comp_ctx_lengths_prefill = compiler_options.pop("comp_ctx_lengths_prefill")
            comp_ctx_lengths_decode = compiler_options.pop("comp_ctx_lengths_decode")
            if isinstance(comp_ctx_lengths_prefill, str):
                import ast

                try:
                    # Safely evaluate the string to a Python list for disaggregated input
                    self.comp_ctx_lengths_prefill = ast.literal_eval(comp_ctx_lengths_prefill)
                    self.comp_ctx_lengths_decode = ast.literal_eval(comp_ctx_lengths_decode)

                except (ValueError, SyntaxError):
                    raise ValueError("Invalid format for comp_ctx_lengths. Expected a list-like string.")
            else:
                self.comp_ctx_lengths_prefill = comp_ctx_lengths_prefill
                self.comp_ctx_lengths_decode = comp_ctx_lengths_decode

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

        if (
            self.model.qaic_config is not None
            and self.model.qaic_config.get("include_sampler", False)
            and num_speculative_tokens is not None
            and num_speculative_tokens > 0
        ):
            raise ValueError("Currently, sampler does not support `num_speculative_tokens` > 0.")

        # Infer kv_cache_batch_size if not provided
        kv_cache_batch_size = kv_cache_batch_size or full_batch_size or batch_size

        # --- Specializations ---
        specializations = []
        if prefill_only is None or prefill_only or prefill_seq_len == 1:
            if self.comp_ctx_lengths_prefill is not None:
                # Adding elements from self.comp_ctx_lengths_prefill to prefill_specialization
                for i in range(0, len(self.comp_ctx_lengths_prefill)):
                    specializations.append(
                        self.build_prefill_specialization(
                            prefill_seq_len=prefill_seq_len,
                            ctx_len=ctx_len,
                            comp_ctx_lengths=self.comp_ctx_lengths_prefill[i],
                            batch_size=batch_size,
                            kv_cache_batch_size=kv_cache_batch_size,
                            full_batch_size=full_batch_size,
                        )
                    )

            else:
                specializations.append(
                    self.build_prefill_specialization(
                        prefill_seq_len=prefill_seq_len,
                        ctx_len=ctx_len,
                        batch_size=batch_size,
                        kv_cache_batch_size=kv_cache_batch_size,
                        full_batch_size=full_batch_size,
                    )
                )

        if prefill_only is None or not prefill_only:
            if self.comp_ctx_lengths_decode is not None:
                # Adding elements from self.comp_ctx_lengths_decode to decode_specialization
                for i in range(0, len(self.comp_ctx_lengths_decode)):
                    decode_spec = self.build_decode_specialization(
                        prefill_seq_len=prefill_seq_len,
                        ctx_len=ctx_len,
                        comp_ctx_lengths=self.comp_ctx_lengths_decode[i],
                        batch_size=batch_size,
                        kv_cache_batch_size=kv_cache_batch_size,
                        full_batch_size=full_batch_size,
                        num_speculative_tokens=num_speculative_tokens,
                    )
                    if decode_spec:
                        specializations.append(decode_spec)

            else:
                decode_spec = self.build_decode_specialization(
                    prefill_seq_len=prefill_seq_len,
                    ctx_len=ctx_len,
                    batch_size=batch_size,
                    kv_cache_batch_size=kv_cache_batch_size,
                    full_batch_size=full_batch_size,
                    num_speculative_tokens=num_speculative_tokens,
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
            use_onnx_subfunctions=use_onnx_subfunctions,
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
        Generate output by executing the compiled QPC on Cloud AI 100 hardware.

        This method runs sequential execution based on the compiled model's batch size and the number of prompts.
        If the number of prompts is not divisible by the batch size, the last batch will be dropped.

        Parameters
        ----------
        tokenizer : PreTrainedTokenizer or PreTrainedTokenizerFast
            Tokenizer for the model.
        prompts : list of str
            List of prompts to generate output for.
        device_id : list of int, optional
            Device IDs for running the QPC. Defaults to `[0]` if not specified.
        runtime_ai100 : bool, optional
            Whether to use AI 100 runtime. Default is True.
        **kwargs :
            Additional keyword arguments. Currently supports:
            - `generation_len (int, optional)`: The maximum number of tokens to generate.

        Returns
        -------
        CloudAI100ExecInfoNew
            Output from the AI 100 runtime, containing generated IDs and performance metrics.

        Raises
        ------
        TypeError
            If the QPC path is not set (i.e., `compile` was not run).
        NotImplementedError
            If `runtime_ai100` is False.
        """
        if runtime_ai100:
            if not isinstance(self.qpc_path, Path):
                raise TypeError("Please run compile API first!")
            generation_len = kwargs.pop("generation_len", None)
            return QEfficient.cloud_ai_100_exec_kv(
                tokenizer=tokenizer,
                qpc_path=self.qpc_path,
                prompt=prompts,
                comp_ctx_lengths_prefill=self.comp_ctx_lengths_prefill,
                comp_ctx_lengths_decode=self.comp_ctx_lengths_decode,
                device_id=device_id,
                generation_len=generation_len,
                automation=kwargs.pop("automation", False),
                iteration=kwargs.pop("iteration", 1),
                is_tlm=self.is_tlm,
                **kwargs,
            )
        else:
            raise NotImplementedError("Only AI_100 runtime is supported right now via generate API")

    def check_and_get_num_speculative_tokens(self, num_speculative_tokens: Optional[int], prefill_seq_len: int):
        """
        Validates and retrieves the number of speculative tokens for TLM models.

        Parameters
        ----------
        num_speculative_tokens : int, optional
            The number of speculative tokens provided by the user.
        prefill_seq_len : int
            The prefill sequence length.

        Returns
        -------
        int
            The determined number of speculative tokens.

        Raises
        ------
        TypeError
            If `num_speculative_tokens` is None when `is_tlm` is True.
        ValueError
            If `num_speculative_tokens` is not an integer greater than 1.
            If `prefill_seq_len` is less than `num_speculative_tokens + 1`.
        """
        if hasattr(self.model.config, "speculative_config"):
            num_speculative_tokens_ = self.model.config.speculative_config["num_speculative_tokens"]
            if num_speculative_tokens is not None:
                logger.warning(
                    f"arg `num_speculative_tokens` is a fixed value of {num_speculative_tokens_} for this model."
                    f" Passed value of {num_speculative_tokens} will be ignored."
                )
            num_speculative_tokens = num_speculative_tokens_
        elif num_speculative_tokens is None:
            raise TypeError("missing required argument `num_speculative_tokens` as `is_tlm` instance variable is True.")

        if not isinstance(num_speculative_tokens, int) and num_speculative_tokens:
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
    QEfficient class for sequence-to-sequence speech-to-text models (e.g., Whisper, Encoder-Decoder speech models).

    This class enables efficient export, compilation, and inference of speech models on Cloud AI 100 hardware.
    It is recommended to use the ``from_pretrained`` method for initialization.

    Example
    -------
    .. code-block:: python

        from datasets import load_dataset
        from transformers import AutoProcessor
        from QEfficient import QEFFAutoModelForSpeechSeq2Seq

        base_model_name = "openai/whisper-tiny"
        ## STEP 1 -- load audio sample, using a standard english dataset, can load specific files if longer audio needs to be tested; also load initial processor
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        data = ds[0]["audio"]["array"]
        # reshape to so shape corresponds to data with batch size 1
        data = data.reshape(-1)
        sample_rate = ds[0]["audio"]["sampling_rate"]
        processor = AutoProcessor.from_pretrained(base_model_name)

        ## STEP 2 -- init base model
        qeff_model = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(base_model_name)

        ## STEP 3 -- export and compile model
        qeff_model.compile()

        ## STEP 4 -- generate output for loaded input and processor
        exec_info = qeff_model.generate(inputs=processor(data, sampling_rate=sample_rate, return_tensors="pt"), generation_len=25)

        ## STEP 5 (optional) -- use processor to decode output
        print(processor.batch_decode(exec_info.generated_ids)[0])
    """

    _hf_auto_class = AutoModelForSpeechSeq2Seq
    _pytorch_transforms = [CustomOpsTransform, AwqToMatmulNbitsTransform, GPTQToMatmulNbitsTransform, KVCacheTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(self, model: nn.Module, **kwargs):
        """
        Initialize a QEFFAutoModelForSpeechSeq2Seq instance.

        Parameters
        ----------
        model : nn.Module
            A PyTorch model with a sequence-to-sequence speech-to-text head (e.g., Whisper).
        **kwargs :
            Additional keyword arguments passed to the base class constructor.

        Raises
        ------
        TypeError
            If the model is not a supported speech-to-text model (i.e., not a `ForConditionalGeneration` model).
        """
        model_class_name = model.__class__.__name__
        if not (model_class_name.endswith("ForConditionalGeneration")):
            raise TypeError(f"Required pytorch module with ForConditionalGeneration, got {model_class_name}")

        model.config.use_cache = True
        super().__init__(model, **kwargs)
        self.num_layers = model.config.num_hidden_layers
        self.hash_params["qeff_auto_class"] = self.__class__.__name__

    @property
    def get_model_config(self) -> dict:
        """
        Get the configuration dictionary of the underlying HuggingFace model.

        Returns
        -------
        dict
            The configuration dictionary.
        """
        return self.model.config.__dict__

    def export(self, export_dir: Optional[str] = None, use_onnx_subfunctions: bool = False) -> str:
        """
        Export the model to ONNX format using ``torch.onnx.export``.

        This method prepares example inputs and dynamic axes based on the model configuration,
        then exports the model to an ONNX graph suitable for compilation and deployment on Cloud AI 100 hardware.

        Parameters
        ----------
        export_dir : str, optional
            Directory path where the exported ONNX graph will be saved.
            If not provided, the default export directory is used.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False

        Returns
        -------
        str
            Path to the generated ONNX graph file.
        """
        inputs = self.model.get_dummy_inputs()
        dynamic_axes = self.model.get_onnx_dynamic_axes()
        output_names = self.model.get_output_names()
        return self._export(
            inputs,
            output_names,
            dynamic_axes,
            export_dir=export_dir,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )

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
        use_onnx_subfunctions: bool = False,
        **compiler_options,
    ) -> str:
        """
        Compile the exported ONNX model using the Cloud AI 100 Platform SDK compiler.

        This method generates a ``qpc`` package. If the model has not been exported yet,
        this method will handle the export process. Additional arguments for the `qaic-exec`
        compiler can be passed as keyword arguments.

        Parameters
        ----------
        onnx_path : str, optional
            Path to a pre-exported ONNX model. If not provided, the model will be exported first.
        compile_dir : str, optional
            Directory to save the generated QPC package.
        prefill_seq_len : int, optional
            Prefill sequence length. This parameter is typically not critically used for
            SpeechSeq2Seq models' decoder compilation as the first decoder input is `seq_len=1`.
            Default is 1.
        encoder_ctx_len : int, optional
            Maximum context length for the encoder part of the model. If None, it's inferred
            from the model configuration or defaults (e.g., 1500 for Whisper).
        ctx_len : int, optional
            Maximum decoder context length. This defines the maximum output sequence length
            the compiled model can handle. Default is 150.
        batch_size : int, optional
            Batch size. Default is 1.
        num_devices : int, optional
            Number of devices to compile for. Default is 1.
        num_cores : int, optional
            Number of cores to use for compilation.
        mxfp6_matmul : bool, optional
            Use MXFP6 compression for weights. Default is False.
        mxint8_kv_cache : bool, optional
            Use MXINT8 compression for KV cache. Default is False.
        full_batch_size : int, optional
            Not yet supported for this model.
        kv_cache_batch_size : int, optional
            Not yet supported for this model.
        num_speculative_tokens : int, optional
            Not yet supported for this model.
        use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False
        **compiler_options : dict
            Additional compiler options for QAIC.

            **For QAIC Compiler:** Extra arguments for qaic-exec can be passed. Some common options include:

            - mos (int, optional): Effort level to reduce on-chip memory. Defaults to -1, meaning no effort. Defaults to -1.
            - aic_enable_depth_first (bool, optional): Enables DFS with default memory size. Defaults to False.
            - allow_mxint8_mdp_io (bool, optional): Allows MXINT8 compression of MDP IO traffic. Defaults to False.

            Params are converted to flags as below:

            - ``aic_num_cores=16`` -> ``-aic-num-cores=16``
            - ``convert_to_fp16=True`` -> ``-convert-to-fp16``

        Returns
        -------
        str
            Path to the compiled QPC package.

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
            onnx_path=onnx_path,
            compile_dir=compile_dir,
            compile_only=True,
            retained_state=True,
            specializations=specializations,
            convert_to_fp16=True,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=num_devices,
            aic_num_cores=num_cores,
            custom_io=custom_io,
            use_onnx_subfunctions=use_onnx_subfunctions,
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
        Generate output until ``<|endoftext|>`` token or `generation_len` is reached,
        by executing the compiled QPC on Cloud AI 100 hardware.

        This method performs sequential execution based on the compiled model's batch size
        and the provided audio tensors. It manages the iterative decoding process and KV cache.

        Parameters
        ----------
        inputs : Dict[str, np.ndarray]
            Model inputs for inference, typically a dictionary containing:
            - `input_features` (np.ndarray): Preprocessed audio features.
            - `decoder_input_ids` (np.ndarray): Initial decoder input IDs (e.g., start token).
            - `decoder_position_ids` (np.ndarray): Initial decoder position IDs.
            These should be prepared to match the compiled model's expectations.
        generation_len : int
            Maximum number of tokens to generate. The generation stops if this limit is reached
            or the model generates an end-of-sequence token.
        streamer : TextStreamer, optional
            Streamer to receive generated tokens in real-time. Default is None.
        device_ids : List[int], optional
            Device IDs for running the QPC. Defaults to `[0]` if not specified.

        Returns
        -------
        CloudAI100ExecInfoNew
            Output from the AI 100 runtime, including generated IDs and performance metrics.

        Raises
        ------
        TypeError
            If the QPC path is not set (i.e., `compile` was not run).
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


class QEFFAutoModelForCTC(QEFFTransformersBase):
    """
    The QEFFAutoModelForCTC class is designed for transformer models with a Connectionist Temporal Classification (CTC) speech-to-text head,
    including Wav2Vec2 and other encoder-only speech models optimized for alignment-free transcription.
    Although it is possible to initialize the class directly, we highly recommend using the ``from_pretrained`` method for initialization.

    ``Mandatory`` Args:
        :model (nn.Module): PyTorch model

    .. code-block:: python
        import torchaudio
        from QEfficient import QEFFAutoModelForCTC
        from transformers import AutoProcessor

        # Initialize the model using from_pretrained similar to transformers.AutoModelForCTC.
        model=QEFFAutoModelForCTC.from_pretrained(model_name)

        # Now you can directly compile the model for Cloud AI 100
        model.compile(num_cores=16)  # Considering you have a Cloud AI 100 SKU

        #prepare input
        processor = AutoProcessor.from_pretrained(model_name)
        input_audio, sample_rate = [...] # audio data loaded in via some external audio package, such as librosa or soundfile

        # Resample the input_audio if necessary
        if input_audio.shape[0] > 1:
            input_audio = input_audio.mean(dim=0)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            input_audio = resampler(input_audio)

        # You can now execute the model
        out = model.generate(processor,inputs=input_audio)
    """

    _hf_auto_class = AutoModelForCTC
    _pytorch_transforms = [CustomOpsTransform, AwqToMatmulNbitsTransform, GPTQToMatmulNbitsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, **kwargs)
        self.model.base_model.config.use_cache = True

        self.hash_params["qeff_auto_class"] = self.__class__.__name__

    @classmethod
    @with_replaced_quantizers
    def from_pretrained(cls, pretrained_model_name_or_path, pooling=None, *args, **kwargs):
        """
        This method serves as the easiest entry point into using QEfficient. The interface is designed to be similar to transformers.AutoModelForCTC.
        Once the model is initialized, you can use other methods such as export, compile, and generate on the same object.

        Args:
            pretrained_model_name_or_path (str): The name or path of the pre-trained model.

        .. code-block:: python

        import torchaudio
        from QEfficient import QEFFAutoModelForCTC
        from transformers import AutoProcessor

        # Initialize the model using from_pretrained similar to transformers.AutoModelForCTC.
        model=QEFFAutoModelForCTC.from_pretrained(model_name)

        # Now you can directly compile the model for Cloud AI 100
        model.compile(num_cores=16)  # Considering you have a Cloud AI 100 SKU

        #prepare input
        processor = AutoProcessor.from_pretrained(model_name)
        input_audio, sample_rate = [...] # audio data loaded in via some external audio package, such as librosa or soundfile

        # Resample the input_audio if necessary
        if input_audio.shape[0] > 1:
            input_audio = input_audio.mean(dim=0)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            input_audio = resampler(input_audio)

        # You can now execute the model
        out = model.generate(processor,inputs=input_audio)
        """
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
                model, kv_offload=kv_offload, **kwargs
            )

        return cls(model, pretrained_model_name_or_path=pretrained_model_name_or_path, pooling=pooling, **kwargs)

    @property
    def get_model_config(self) -> dict:
        return self.model.config.__dict__

    def export(self, export_dir: Optional[str] = None, use_onnx_subfunctions: bool = False) -> str:
        """
        Exports the model to ``ONNX`` format using ``torch.onnx.export``.

        ``Optional`` Args:
           :export_dir (str, optional): The directory path to store ONNX-graph.
           :use_onnx_subfunctions: bool, optional
            whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False

        Returns:
            :str: Path of the generated ``ONNX`` graph.
        """
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        seq_len = constants.WAV2VEC2_MAX_SEQ_LEN

        example_inputs = {
            "input_values": torch.zeros((bs, seq_len), dtype=torch.float32),
        }

        dynamic_axes = {"input_values": {0: "batch_size", 1: "seq_len"}}

        output_names = ["logits"]

        return self._export(
            example_inputs,
            output_names,
            dynamic_axes,
            export_dir=export_dir,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )

    def compile(
        self,
        onnx_path: Optional[str] = None,
        compile_dir: Optional[str] = None,
        *,
        seq_len: Union[int, List[int]] = 480000,
        batch_size: int = 1,
        num_devices: int = 1,
        num_cores: int = 16,  # FIXME: Make this mandatory arg
        mxfp6_matmul: bool = False,
        use_onnx_subfunctions: bool = False,
        **compiler_options,
    ) -> str:
        """
        This method compiles the exported ``ONNX`` model using the Cloud AI 100 Platform SDK compiler binary found at ``/opt/qti-aic/exec/qaic-exec`` and generates a ``qpc`` package.
        If the model has not been exported yet, this method will handle the export process.
        You can pass any other arguments that the `qaic-exec` takes as extra kwargs.

        ``Optional`` Args:
            :onnx_path (str, optional): Path to pre-exported onnx model.
            :compile_dir (str, optional): Path for saving the qpc generated.
            :seq_len (Union[int, List[int]]): The length of the prompt should be less that ``seq_len``. ``Defaults to 32``.
            :batch_size (int, optional): Batch size. ``Defaults to 1``.
            :num_devices (int): Number of devices the model needs to be compiled for. Defaults to 1.
            :num_cores (int): Number of cores used to compile the model.
            :mxfp6_matmul (bool, optional): Whether to use ``mxfp6`` compression for weights. ``Defaults to False``.
            :use_onnx_subfunctions: bool, optional: whether to enable ONNX subfunctions during export. Exporting PyTorch model to ONNX with modules as subfunctions helps to reduce export/compile time. Defaults to False
            :compiler_options (dict, optional): Additional compiler options.

                For QAIC Compiler: Extra arguments for qaic-exec can be passed.
                    :aic_enable_depth_first (bool, optional): Enables DFS with default memory size. ``Defaults to False``.
                    :allow_mxint8_mdp_io (bool, optional): Allows MXINT8 compression of MDP IO traffic. ``Defaults to False.``

                    Params are converted to flags as below:

                    - aic_hw_version=ai100 -> -aic-hw-version=ai100
                    - aic_hw_version=ai200 -> -aic-hw-version=ai200

                For QNN Compiler: Following arguments can be passed.
                    :enable_qnn (bool): Enables QNN Compilation.
                    :qnn_config (str): Path of QNN Config parameters file. Any extra parameters for QNN compilation can be passed via this file.

        Returns:
            :str: Path of the compiled ``qpc`` package.
        """

        specializations = [
            {"batch_size": batch_size, "seq_len": sl} for sl in (seq_len if isinstance(seq_len, list) else [seq_len])
        ]

        return self._compile(
            onnx_path=onnx_path,
            compile_dir=compile_dir,
            compile_only=True,
            specializations=specializations,
            convert_to_fp16=True,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=num_devices,
            aic_num_cores=num_cores,
            use_onnx_subfunctions=use_onnx_subfunctions,
            **compiler_options,
        )

    def generate(
        self,
        processor,
        inputs: torch.Tensor,
        device_ids: List[int] = None,
        runtime_ai100: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        This method generates output by executing PyTorch runtime or the compiled ``qpc`` on ``Cloud AI 100`` Hardware cards.
        ``Mandatory`` Args:
            :inputs (Union[torch.Tensor, np.ndarray]): inputs to run the execution.
            :processor (AutoProcessor): The Processor to use for encoding the waveform.
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

            return self.cloud_ai_100_feature_generate(processor, inputs=inputs, device_ids=device_ids)
        # PyTorch runtime
        else:
            return self.pytorch_feature_generate(processor, model=self.model, inputs=inputs)

    def cloud_ai_100_feature_generate(
        self,
        processor,
        inputs: torch.Tensor,
        device_ids: List[int] = [0],
    ) -> np.ndarray:
        """
        Generates features with list of prompts using AI 100 runtime.

        ``Mandatory`` Args:
            :inputs (Union[torch.Tensor, np.ndarray]): inputs to run the execution.
            :processor (AutoProcessor): The Processor to use for encoding the waveform.
        ``Optional`` Args:
            device_ids (List[int], optional): A list of device IDs to use for the session. Defaults to [0].

        """

        if self.qpc_session is None:
            self.qpc_session = QAICInferenceSession(str(self.qpc_path), device_ids)
            self.batch_size = self.qpc_session.bindings[0].dims[0]

        # Dynamic switching to closest seq_Len based on input_ids_len
        inputs = processor(inputs, return_tensors="pt")
        input_ids_len = inputs["input_values"].shape[-1]

        for allowed_shape in self.qpc_session.allowed_shapes:
            seq_len_allowed = allowed_shape[1][1][1]

            if seq_len_allowed >= input_ids_len:
                self.seq_len = seq_len_allowed
                break

        # To handle single seq_len as we can't fetch allowed shapes for single seq_len
        self.seq_len = self.qpc_session.bindings[0].dims[1] if not hasattr(self, "seq_len") else self.seq_len
        input_values = np.array(
            torch.nn.functional.pad(inputs["input_values"], (0, self.seq_len - input_ids_len), "constant", 0)
        )
        inputs = dict(input_values=input_values)
        outputs = self.qpc_session.run(inputs)
        logits = outputs["logits"]
        predicted_ids = np.argmax(logits, axis=-1)
        transcriptions = processor.batch_decode(torch.tensor(predicted_ids))
        return transcriptions

    def pytorch_feature_generate(self, processor, model, inputs: Union[torch.Tensor, np.ndarray]) -> List[torch.Tensor]:
        """
        Generates features from a list of text prompts using a PyTorch model.

        ``Mandatory`` Args:
            :model: The transformed PyTorch model used for generating features.
            :inputs (Union[torch.Tensor, np.ndarray]): inputs to run the execution.
            :processor (AutoProcessor): The Processor to use for encoding the waveform.

        """
        input_values = processor(
            inputs[0], return_tensors="pt", max_length=self.seq_len, truncation=True, padding="max_length"
        ).input_values
        logits = model(input_values[0]).logits
        logits = logits.detach().numpy()
        predicted_ids = np.argmax(logits, axis=-1)
        transcriptions = processor.batch_decode(predicted_ids)
        return transcriptions
