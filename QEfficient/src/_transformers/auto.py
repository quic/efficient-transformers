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
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

import QEfficient
from QEfficient.src._transformers.runtime_args import (
    QEFFAutoModelForCausalLMAI100RuntimeArgs,
    QEFFAutoModelForCausalLMCPUORTRuntimeArgs,
)
from QEfficient.src.base import QEFFBaseModel, Runtime
from QEfficient.transformers.modeling_utils import TransformersToQEffModulesDict
from QEfficient.utils import get_qpc_dir_name_infer, load_hf_tokenizer, qpc_exists

# Dictionary that defines the interface from transformers to be used underneath the QEFF interface
QEFFAutoModelToTransformersAutoModelMap = {
    "QEFFAutoModelForCausalLM": AutoModelForCausalLM,
    "QEFFAutoModel": AutoModel,
}


class QEFFTransformersBase(QEFFBaseModel):
    """
    Parent class for models QEFF provides from transformers i.e. (AutoModel, AutoModelForCausalLM, AutoModelForAudioClassification etc.) from src/transformers/models/auto/modeling_auto.py file.
    """

    def __init__(self, model: nn.Module, pretrained_model_name_or_path: str, **kwargs) -> None:
        super().__init__()
        assert (
            model.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING.values()
            or
            # FIXME: Use model architectures here instead of complete dictionary TransformersToQEffModulesDict
            model.__class__ in TransformersToQEffModulesDict.values()
        ), f"Given model{model.__class__.__name__} could not be found in transformers library i.e. {MODEL_FOR_CAUSAL_LM_MAPPING.values()}"  # type: ignore
        self.model: nn.Module = model
        self.model_card_name = kwargs.pop("model_card_name", None)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.kwargs = kwargs
        self._tokenizer = None
        if kwargs.get("transform", True):
            self.transform()

    def __repr__(self) -> str:
        return self.model.__repr__()

    @property
    def is_transformed(self) -> bool:
        return getattr(self.model, "qeff_transformed", False)
    
    @property
    def tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        if self._tokenizer is None:
            self._tokenizer = self.get_tokenizer()
        return self._tokenizer

    def get_model_card_name(self) -> str:
        # FIXME: use getter
        if self.model_card_name is None:
            # Handle when pretrained_model_name_or_path is a path and we don't know the model_card_name
            assert not os.path.isdir(self.pretrained_model_name_or_path), f"Please provide `model_card_name` argument as valid string, got {self.model_card_name}"
            self.model_card_name = self.pretrained_model_name_or_path

        return self.model_card_name

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        """
        This method accepts All the parameters that are acceptable by transformers.AutoModelForCausalLM.
        There are few additional parameters that this method can take.
        ---------
        :param transform: bool. Whether to optimize model for KV retention; default is True. Pass False to get BertStyle model.
        :param model_card_name: str. HuggingFace model card name or name of the model if custom, used for deciding folder name while saving ONNX/qpc files.
        """
        kwargs.update(
            {"use_cache": True}
        )  # Always pass use_cache = True, to get KV values as output during ONNX export
        kwargs.update({"attn_implementation": "eager"})  # Always use eager mode for attention implementation
        model_card_name = kwargs.pop("model_card_name", None)

        model = QEFFAutoModelToTransformersAutoModelMap[cls.__name__].from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        
        kwargs.update({"model_card_name": model_card_name})
        return cls(model, pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs)
        
    def get_tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=self.pretrained_model_name_or_path, **self.kwargs)
        return tokenizer

    def transform_export(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Reached too far!!")

    def transform_export_compile(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Reached too far!!")

    def transform(self):
        # FIXME: break down transform into optmization passes i.e. HW specific optimization(RMSNorm), KV retention pass etc.
        QEfficient.transform(self)
        return self


class QEFFAutoModelForCausalLM(QEFFTransformersBase):
    """
    QEFF class for manipulating any causal language model from HuggingFace hub.
    """

    def execute(self, *args, **kwargs):  # type: ignore
        raise NotImplementedError("Reached too far!!")
    
    def set_runtime(self, runtime: Runtime, runtime_args: Union[QEFFAutoModelForCausalLMAI100RuntimeArgs, QEFFAutoModelForCausalLMCPUORTRuntimeArgs]) -> None:
        assert runtime != Runtime.CPU_PT, "Please use self.runtime=Runtime.CPU_PT for updating runtime to CPU_PT"

        if runtime == Runtime.CPU_ORT:
            assert isinstance(runtime_args, QEFFAutoModelForCausalLMCPUORTRuntimeArgs), f"Expected runtime_args of type {QEFFAutoModelForCausalLMCPUORTRuntimeArgs.__class__}, got {type(runtime_args)}"
            self.ort_runtime_args = runtime_args
        elif runtime == Runtime.AI_100:
            assert isinstance(runtime_args, QEFFAutoModelForCausalLMAI100RuntimeArgs),  f"Expected runtime_args of type {QEFFAutoModelForCausalLMAI100RuntimeArgs.__class__}, got {type(runtime_args)}"
            self.cloud_ai_100_runtime_args = runtime_args
        # FIXME: Add logger info
        self.runtime = runtime

    def export_and_compile(self, num_cores: int, device_group: List[int], batch_size: int = 1, prompt_len: int = 32, ctx_len: int = 128,
                mxfp6: bool = True, mxint8: bool = False, mos: int = -1, aic_enable_depth_first: bool = False, qpc_dir_suffix: Optional[str] = None,
                full_batch_size: int = 1) -> str:
        """
        Exports the Pytorch model to ONNX and saves it locally.
        -------------
        hf_model_card: str. Model card name on HuggingFace, used for deciding where to save the onnx, qpc files i.e. qeff_models/HF_model_card/(onnx or qpc).
        onnx_model_path: str. Path for saving onnx model.
        """
        self.export()
        self.compile(num_cores=num_cores, device_group=device_group, batch_size=batch_size, prompt_len=prompt_len, ctx_len=ctx_len,
                     mxfp6=mxfp6, mxint8=mxint8, mos=mos, aic_enable_depth_first=aic_enable_depth_first, qpc_dir_suffix=qpc_dir_suffix,
                     full_batch_size=full_batch_size)
        return self.cloud_ai_100_runtime_args.qpc_dir_path


    def export(self, **kwargs) -> str:
        assert self.is_transformed, "Please first run transform on the QEFFAutoModelForCausalLM object"
        model_card_name = self.get_model_card_name()
        base_path, onnx_path = QEfficient.export(model_name=model_card_name, model_kv=self, tokenizer=self.tokenizer)
        assert self.runtime == Runtime.CPU_ORT, "Something went wrong while exporting model to ONNX"
        return onnx_path

    def compile(self, num_cores: int, device_group: List[int], batch_size: int = 1, prompt_len: int = 32, ctx_len: int = 128,
                mxfp6: bool = True, mxint8: bool = False, mos: int = -1, aic_enable_depth_first: bool = False, qpc_dir_suffix: Optional[str] = None,
                full_batch_size: int = 1) -> str:
        # Prepare qpc dir path
        qpc_base_dir_name = get_qpc_dir_name_infer(num_cores=num_cores, mos=mos, batch_size=batch_size, prompt_len=prompt_len, ctx_len=ctx_len, mxfp6=mxfp6, mxint8=mxint8, device_group=device_group)
        qpc_base_dir_name = qpc_base_dir_name + "_" + qpc_dir_suffix if qpc_dir_suffix else qpc_base_dir_name
        _, qpc_dir_path = qpc_exists(model_name = self.get_model_card_name(), qpc_base_dir_name=qpc_base_dir_name)

        # Compile
        QEfficient.compile(onnx_path=self.ort_runtime_args.onnx_model_path, qpc_path=os.path.dirname(qpc_dir_path),
                           num_cores=num_cores, device_group=device_group, aic_enable_depth_first=aic_enable_depth_first,
                           mos=mos, batch_size=batch_size, prompt_len=prompt_len, ctx_len=ctx_len, mxfp6=mxfp6,
                           mxint8=mxint8, full_batch_size=full_batch_size)
        cloud_ai_100_runtime_args = QEFFAutoModelForCausalLMAI100RuntimeArgs(qpc_dir_path=qpc_dir_path, device_group=device_group)
        self.set_runtime(runtime=Runtime.AI_100, runtime_args=cloud_ai_100_runtime_args)

        return qpc_dir_path


class QEffAutoModel(QEFFTransformersBase):
    def execute(self, *args, **kwargs):  # type: ignore
        raise NotImplementedError("Reached too far!!")

    def export(self):
        raise NotImplementedError("Reached too far!!")

    def compile(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Reached too far!!")
