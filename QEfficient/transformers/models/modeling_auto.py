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
from QEfficient.base.modeling_qeff import QEFFBaseModel, Runtime
from QEfficient.base.pytorch_transforms import CustomOpsTransform, KVCacheTransform
from QEfficient.transformers.modeling_utils import TransformersToQEffModulesDict
from QEfficient.utils import get_qpc_dir_path, load_hf_tokenizer
from QEfficient.utils.logging_utils import logger

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
        super().__init__(model)
        assert (
            model.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING.values()
            or
            # FIXME: Use model architectures here instead of complete dictionary TransformersToQEffModulesDict
            model.__class__ in TransformersToQEffModulesDict.values()
        ), f"Given model{model.__class__.__name__} could not be found in transformers library i.e. {MODEL_FOR_CAUSAL_LM_MAPPING.values()}"  # type: ignore
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
            self.transform()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}\n" + self.model.__repr__()

    @property
    def tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        if self._tokenizer is None:
            self._tokenizer = self.get_tokenizer()
        return self._tokenizer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        """
        This method accepts All the parameters that are acceptable by transformers.AutoModelForCausalLM.
        There are few additional parameters that this method can take.
        ---------
        :transform: bool. Whether to optimize model for KV retention; default is True. Pass False to get BertStyle model.
        :model_card_name: str. HuggingFace model card name or name of the model if custom, used for deciding folder name while saving ONNX/qpc files.
        """
        model_card_name = kwargs.pop(
            "model_card_name", None
        )  # Remove model_card_name from kwargs for transformers APIs

        model = QEFFAutoModelToTransformersAutoModelMap[cls.__name__].from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        return cls(
            model,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            model_card_name=model_card_name,
            **kwargs,
        )

    def get_tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=self.pretrained_model_name_or_path, **self.kwargs)
        return tokenizer


class QEFFAutoModelForCausalLM(QEFFTransformersBase):
    """
    QEFF class for manipulating any causal language model from HuggingFace hub.
    """

    _pytorch_transforms = [CustomOpsTransform, KVCacheTransform]

    def transform(self):
        for transform in self._pytorch_transforms:
            transform.apply(self.model)
        self.is_transformed = True

    def execute(self, *args, **kwargs):  # type: ignore
        raise NotImplementedError("Reached too far!!")

    def export(self, model_card_name: Optional[str] = None) -> str:
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
