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
from QEfficient.transformers.models.pytorch_transforms import CustomOpsTransform, KVCacheTransform, SamplerTransform, SpDTransform
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
    def from_pretrained(cls, pretrained_model_name_or_path: str, is_tlm: bool = False, include_sampler: bool = False, return_pdfs: bool = False, *args, **kwargs):
        if kwargs.get("attn_implementation", None) not in {None, "eager"}:
            logger.warning('Updating attn_implementation="eager"')

        if kwargs.get("low_cpu_mem_usage", None):
            logger.warning("Updating low_cpu_mem_usage=False")

        kwargs.update({"attn_implementation": "eager", "low_cpu_mem_usage": False})

        model = cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return cls(model, is_tlm=is_tlm, include_sampler=include_sampler, return_pdfs=return_pdfs)

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
        :include_sampler (bool): Enable/Disable sampling of next tokens during decode.
        :return_pdfs (bool): Return probability distributions (logits/probs) or sampled next tokens. If `is_tlm`=True, then `return_pdfs`=True always. If `is_tlm`=False, then `return_pdfs`=True for Speculative Decoding Draft Language Model and `return_pdfs`=False for regular model. 


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
        include_sampler: bool = False,
        return_pdfs: bool = False,
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
            self.model.return_pdfs = True
        self.is_tlm = is_tlm

        if include_sampler:  # Sampling 
            self.model, transformed = SamplerTransform.apply(self.model)
            self.model.return_pdfs = return_pdfs
        self.include_sampler = include_sampler

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, continuous_batching: bool = False, is_tlm: bool = False, include_sampler: bool = False, return_pdfs: bool = False, *args, **kwargs
    ):
        """
        This method serves as the easiest entry point into using QEfficient. The interface is designed to be similar to transformers.AutoModelForCausalLM.
        Once the model is initialized, you can use other methods such as export, compile, and generate on the same object.

        Args:
            :pretrained_name_or_path (str): Model card name from HuggingFace or local path to model directory.
            :continuous_batching (bool): Whether this model will be used for continuous batching in future. If this is not set True here, the model can not be exported/compiled for continuous batching later.
            :is_tlm (bool): Whether this is a Speculative Decoding Target Language Model. If set to True, `num_logits_to_keep` input array will have to be fed to control the number of returned logits during prefill/decode.
            :include_sampler (bool): Enable/Disable sampling of next tokens during decode.
            :return_pdfs (bool): Return probability distributions (logits/probs) or sampled next tokens. If `is_tlm`=True, then `return_pdfs`=True always. If `is_tlm`=False, then `return_pdfs`=True for Speculative Decoding Draft Language Model and `return_pdfs`=False for regular model. 
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

        self = super().from_pretrained(pretrained_model_name_or_path, is_tlm=is_tlm, include_sampler=include_sampler, return_pdfs=return_pdfs, *args, **kwargs)
        self.continuous_batching = continuous_batching
        return self

    @property
    def model_hash(self) -> str:
        # Compute the hash with: model_config, continuous_batching, transforms
        mhash = hashlib.sha256()
        mhash.update(to_hashable(self.model.config.to_diff_dict()))
        mhash.update(to_hashable({"continuous_batching": self.continuous_batching}))
        mhash.update(to_hashable({"is_tlm": self.is_tlm}))
        mhash.update(to_hashable({"include_sampler": self.include_sampler}))
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

        if self.include_sampler:
            nlk = constants.ONNX_EXPORT_EXAMPLE_NLK  # Number of Logits to Keep
            max_top_k_ids = constants.ONNX_EXPORT_EXAMPLE_MAX_TOP_K_IDS
            
            example_inputs["last_accepted_output_tokens"] = torch.randint(low=0, high=self.model.config.vocab_size, size=(bs, nlk))
            dynamic_axes["last_accepted_output_tokens"] = {0: "batch_size", 1: "num_logits_to_keep"}

            example_inputs["repetition_penalty_retain_state"] = torch.zeros(bs, self.model.config.vocab_size, dtype=torch.int32)
            dynamic_axes["repetition_penalty_retain_state"] = {0: "batch_size", 1: "vocab_size"}
            output_names.append("repetition_penalty_retain_state_RetainedState")

            example_inputs["repetition_penalties"] = torch.ones(bs, dtype=torch.float) * 0.5
            dynamic_axes["repetition_penalties"] = {0: "batch_size"}

            example_inputs["presence_penalty_retain_state"] = torch.zeros(bs, self.model.config.vocab_size, dtype=torch.int32)
            dynamic_axes["presence_penalty_retain_state"] = {0: "batch_size", 1: "vocab_size"}
            output_names.append("presence_penalty_retain_state_RetainedState")

            example_inputs["presence_penalties"] = torch.zeros(bs, dtype=torch.float) + 0.5
            dynamic_axes["presence_penalties"] = {0: "batch_size"}

            example_inputs["temperatures"] = torch.ones(bs, dtype=torch.float)
            dynamic_axes["temperatures"] = {0: "batch_size"}

            example_inputs["top_ks"] = torch.randint(1, max_top_k_ids, size=(bs,)).to(torch.int32)
            dynamic_axes["top_ks"] = {0: "batch_size"}

            example_inputs["top_ps"] = torch.ones(bs, dtype=torch.float) * 0.80
            dynamic_axes["top_ps"] = {0: "batch_size"}

            example_inputs["min_ps"] = torch.ones(bs, dtype=torch.float) * 0.99
            dynamic_axes["min_ps"] = {0: "batch_size"}

            example_inputs["random_numbers"] = torch.rand(bs, dtype=torch.float)
            dynamic_axes["random_numbers"] = {0: "batch_size"}

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
        if self.include_sampler:
             prefill_specialization.update({
                 "vocab_size": self.model.config.vocab_size,
                 "max_top_k_ids": constants.Constants.MAX_TOP_K_IDS,
             })
        prefill_specialization.update({"num_logits_to_keep": 1})
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
            if self.include_sampler:
                decode_specialization.update({
                    "vocab_size": self.model.config.vocab_size,
                    "max_top_k_ids": constants.Constants.MAX_TOP_K_IDS,
                })
            if self.continuous_batching:
                decode_specialization.update({"full_batch_size": kv_cache_batch_size})
            else:
                decode_specialization.update({"batch_size": kv_cache_batch_size})
            decode_specialization.update({"num_logits_to_keep": num_speculative_tokens + 1})
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
