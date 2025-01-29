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
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

import QEfficient
from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.base.onnx_transforms import FP16ClipTransform, SplitTensorsTransform
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.models.pytorch_transforms import CustomOpsTransform, KVCacheTransform, SpDTransform
from QEfficient.transformers.quantizers.auto import QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING, with_replaced_quantizers
from QEfficient.transformers.quantizers.quant_transforms import AwqToMatmulNbitsTransform, GPTQToMatmulNbitsTransform
from QEfficient.utils import constants, get_padding_shape_from_config
from QEfficient.utils.cache import to_hashable
from QEfficient.generation.text_generation_inference import calculate_latency, PerfMetrics, CloudAI100ExecInfo
from time import perf_counter

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


class QEFFAutoModelForCausalLM(QEFFTransformersBase):
    """
    The QEFF class is designed for manipulating any causal language model from the HuggingFace hub.
    Although it is possible to initialize the class directly, we highly recommend using the ``from_pretrained`` method for initialization.

    ``Mandatory`` Args:
        :model (nn.Module):  PyTorch model
        :continuous_batching (bool): Weather this model will be used for continuous batching in future. If this is not set True here, the model can not be exported/compiled for continuous batching later.
        :is_tlm (bool): Whether this is a Speculative Decoding Target Language Model. If set to True, `num_logits_to_keep` input array will have to be fed to control the number of returned logits during prefill/decode.


    .. code-block:: python

        from QEfficient import QEFFAutoModelForCausalLM

        model = QEFFAutoModelForCausalLM.from_pretrained(model_name, num_hidden_layers=2)
        model.compile(prefill_seq_len=32, ctx_len=1024)

        model.generate(prompts=["Hi there!!"])
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
        model_class_name = model.__class__.__name__
        if not (model_class_name.endswith("ForCausalLM") or model_class_name.endswith("LMHeadModel")):
            raise TypeError(f"Required pytorch module for CausalLM or LMHeadModel, got {model_class_name}")

        # TODO: remove from version 1.20
        if kwargs.pop("full_batch_size", None):
            continuous_batching = True
            warnings.warn(
                "full_batch_size argument is deprecated. Use continuous_batching=True instead.", DeprecationWarning, 2
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

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, continuous_batching: bool = False, is_tlm: bool = False, *args, **kwargs
    ):
        """
        This method serves as the easiest entry point into using QEfficient. The interface is designed to be similar to transformers.AutoModelForCausalLM.
        Once the model is initialized, you can use other methods such as export, compile, and generate on the same object.

        Args:
            :pretrained_name_or_path (str): Model card name from HuggingFace or local path to model directory.
            :continuous_batching (bool): Whether this model will be used for continuous batching in future. If this is not set True here, the model can not be exported/compiled for continuous batching later.
            :is_tlm (bool): Whether this is a Speculative Decoding Target Language Model. If set to True, `num_logits_to_keep` input array will have to be fed to control the number of returned logits during prefill/decode.
            :args, kwargs: Additional arguments to pass to transformers.AutoModelForCausalLM.

        .. code-block:: python

            from QEfficient import QEFFAutoModelForCausalLM

            # Initialize the model using from_pretrained similar to transformers.AutoModelForCausalLM
            model = QEFFAutoModelForCausalLM.from_pretrained("gpt2")

            # Now you can directly compile the model for Cloud AI 100
            model.compile(num_cores=6, device_group=[0])  # Considering you have a Cloud AI 100 Standard SKU

            # You can now execute the model
            model.generate(prompts=["Hi there!!"])
        """

        if kwargs.pop("full_batch_size", None):
            continuous_batching = True
            warnings.warn(
                "full_batch_size argument is deprecated. Use continuous_batching=True instead.", DeprecationWarning, 2
            )

        self = super().from_pretrained(pretrained_model_name_or_path, is_tlm=is_tlm, *args, **kwargs)
        self.continuous_batching = continuous_batching
        return self

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
            # TODO: should be renamed to kv_cache_batch_size in specialzation too
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
            :prompts (List[str]): List of prompts to run the execution.
            :device_id (List[int]): Ids of devices for running the qpc pass as [0] in case of normal model / [0, 1, 2, 3] in case of tensor slicing model
        ``optional`` Args:
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
        model.compile(num_cores=16, device_group=[0])  # Considering you have a Cloud AI 100 SKU

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
            model.compile(num_cores=16, device_group=[0])  # Considering you have a Cloud AI 100 SKU

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
            :device_id (List[int]): Ids of devices for running the qpc pass as [0] in case of normal model / [0, 1, 2, 3] in case of tensor slicing model
        ``optional`` Args:
            :runtime_ai100 (bool, optional): ``AI_100`` and ``PyTorch`` runtime is supported as of now. Defaults to ``True`` for ``AI_100`` runtime.
            :eq_len (int, optional): Sequence length for the inputs. Defaults to constants.Constants.CTX_LEN.
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
            model: The transformed PyTorch model used for generating features.
            :inputs (Union[torch.Tensor, np.ndarray]): inputs to run the execution.

        Returns:
            torch.Tensor: A list of output features generated by the model for each prompt.
        """
        return model(**inputs)


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
        super().__init__(model)
        self.model.config.use_cache = True
        self.num_layers = model.config.num_hidden_layers

        k_proj_weights, v_proj_weights = [], []
        for i in range(self.model.config.decoder_layers):
            k_proj_weights.append(self.model.model.decoder.layers[i].encoder_attn.k_proj)
            v_proj_weights.append(self.model.model.decoder.layers[i].encoder_attn.v_proj)
        self.model.model.encoder.add_cross_attention_module(k_proj_weights, v_proj_weights)

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
        encoder_seq_len = constants.ONNX_EXPORT_EXAMPLE_ENCODER_SEQ_LEN
        encoder_feature_count = self.model.config.num_mel_bins

        kv_cache_shape = get_padding_shape_from_config(self.model.config, bs, seq_len)
        kv_cross_cache_shape = get_padding_shape_from_config(self.model.config, bs, encoder_seq_len)
        example_inputs = {
            "input_features": torch.zeros((bs, encoder_feature_count, 1), dtype=torch.float32),
            "decoder_input_ids": torch.zeros((bs, seq_len), dtype=torch.int64),
            "decoder_position_ids": torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(bs, 1),
            "past_key_values": [[] for _ in range(self.num_layers)],
            "cross_key_values": [],
        }
        dynamic_axes = {
            "input_features": {0: "batch_size", 2: "feature_len"},
            "decoder_input_ids": {0: "batch_size", 1: "seq_len"},
            "decoder_position_ids": {0: "batch_size", 1: "seq_len"},
            "cross_key_values": {0: "batch_size", 2: "enc_ctx_len"},
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

        # need a separate loop to ensure order of output_names
        for i in range(self.num_layers):
            for kv in ["key", "value"]:
                example_inputs["cross_key_values"].append(torch.zeros(kv_cross_cache_shape, dtype=torch.float32))
                dynamic_axes[f"cross_{kv}_.{i}"] = pkv_cross_dynamic_axes
                output_names.append(f"cross_{kv}.{i}_RetainedState")

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
            "feature_len": 1,  # important dummy feature so that torch.where knows whether to run encoder or not
        }

        specializations = [encoder_specializations, decoder_specializations]

        self.qpc_path = self._compile(
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

        return self.qpc_path

    def generate(
        self,
        processor: AutoProcessor,
        inputs: torch.Tensor,
        generation_len: int,
        sample_rate: int = 16000,
        device_ids: List[int] = None,
        runtime_ai100: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        This method generates output until ``endoftranscript`` or ``generation_len`` by executing the compiled ``qpc`` on ``Cloud AI 100`` Hardware cards.
        This is a sequential execution based on the ``batch_size`` of the compiled model and the number of audio tensor passed.

        ``Mandatory`` Args:
            :processor (AutoProcessor): AutoProcessor, usually loaded by AutoProcessor.from_pretrained to extract feature data from input array
            :inputs (Union[torch.Tensor, np.ndarray]): inputs to run the execution.
            :device_id (List[int]): Ids of devices for running the qpc pass as [0] in case of normal model / [0, 1, 2, 3] in case of tensor slicing model
        ``optional`` Args:
            :runtime_ai100 (bool, optional): ``AI_100`` and ``PyTorch`` runtime is supported as of now. Defaults to ``True`` for ``AI_100`` runtime.
        Returns:
            :dict: Output from the ``AI_100`` or ``PyTorch`` runtime.
        """
        # AI_100 runtime
        if runtime_ai100:
            if not isinstance(self.qpc_path, Path):
                raise TypeError("Please run compile API first!")

            return self.cloud_ai_100_feature_generate(
                processor=processor,
                inputs=inputs,
                device_ids=device_ids,
                generation_len=generation_len,
                sample_rate=sample_rate,
            )
        # PyTorch runtime
        else:
            return self.pytorch_feature_generate(model=self.model, inputs=inputs)

    def cloud_ai_100_feature_generate(
        self,
        processor: AutoProcessor,
        inputs: torch.Tensor,
        sample_rate: int,
        generation_len: int,
        device_ids: List[int] = [0],
    ) -> np.ndarray:
        """
        Generates features with list of prompts using AI 100 runtime.

        ``Mandatory`` Args:
            :inputs (Union[torch.Tensor, np.ndarray]): inputs to run the execution.
        ``Optional`` Args:
            device_ids (List[int], optional): A list of device IDs to use for the session. Defaults to [0].

        Returns:
           CloudAI100ExecInfo: AI100 inference information
        """

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
        decoder_position_ids = torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(self.batch_size, 1)

        model_inputs = dict(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            decoder_position_ids=decoder_position_ids,
        )

        self.qpc_session.skip_buffers(
            [
                x
                for x in self.qpc_session.input_names + self.qpc_session.output_names
                if x.startswith("past_") or x.startswith("cross_")
            ]
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

            model_inputs["input_ids"] = next_token
            model_inputs["position_ids"] += 1
        end = perf_counter()

        prefill_time, decode_perf, total_perf, total_time = calculate_latency(num_tokens, loop_start, start, end)

        exec_info = CloudAI100ExecInfo(
            batch_size=self.batch_size,
            generated_texts=processor.batch_decode(generated_ids),
            generated_ids=generated_ids,
            perf_metrics=PerfMetrics(prefill_time, decode_perf, total_perf, total_time),
        )

        return exec_info

    def pytorch_feature_generate(self, model, inputs: Union[torch.Tensor, np.ndarray]) -> List[torch.Tensor]:
        """
        Generates features from a list of text prompts using a PyTorch model.

        ``Mandatory`` Args:
            model: The transformed PyTorch model used for generating features.
            :inputs (Union[torch.Tensor, np.ndarray]): inputs to run the execution.

        Returns:
            torch.Tensor: A list of output features generated by the model for each prompt.
        """
        # TODO: implement decode loop for pytorch
        return model(**inputs)
