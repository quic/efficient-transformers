# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from typing import Any

import torch.nn as nn
from transformers.models.auto import AutoModel, AutoModelForCausalLM
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

import QEfficient
from QEfficient.src.base import QEFFBaseModel
from QEfficient.transformers.modeling_utils import TransformersToQEffModulesDict

# Dictionary that defines the interface from transformers to be used underneath the QEFF interface
QEFFAutoModelToTransformersAutoModelMap = {
    "QEFFAutoModelForCausalLM": AutoModelForCausalLM,
    "QEFFAutoModel": AutoModel,
}


class QEFFTransformersBase(QEFFBaseModel):
    """
    Parent class for models QEFF provides from transformers i.e. (AutoModel, AutoModelForCausalLM, AutoModelForAudioClassification etc.) from src/transformers/models/auto/modeling_auto.py file.
    """

    def __init__(self, model: nn.Module, transform: bool = True) -> None:
        assert (
            model.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING.values()
            or
            # FIXME: Use model architectures here instead of complete dictionary TransformersToQEffModulesDict
            model.__class__ in TransformersToQEffModulesDict.values()
        ), f"Given model{model.__class__.__name__} could not be found in transformers library i.e. {MODEL_FOR_CAUSAL_LM_MAPPING.values()}"  # type: ignore
        self.model: nn.Module = model
        if transform:
            self.transform()

    def __repr__(self) -> str:
        return self.model.__repr__()

    @property
    def is_transformed(self) -> bool:
        return getattr(self.model, "qeff_transformed", False)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        """
        This method accepts All the parameters that are acceptable by transformers.AutoModelForCausalLM.
        There are few additional parameters that this method can take:
        :param transform:bool. Whether to optimize model for KV retention; default is True. Pass False to get BertStyle model.
        """
        transform: bool = kwargs.get("transform", True)
        kwargs.update(
            {"use_cache": True}
        )  # Always pass use_cache = True, to get KV values as output during ONNX export
        kwargs.update({"attn_implementation": "eager"})  # Always use eager mode for attention implementation

        model = QEFFAutoModelToTransformersAutoModelMap[cls.__name__].from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        return cls(model, transform=transform)

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

    def export(self):
        raise NotImplementedError("Reached too far!!")

    def compile(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Reached too far!!")


class QEffAutoModel(QEFFTransformersBase):
    def execute(self, *args, **kwargs):  # type: ignore
        raise NotImplementedError("Reached too far!!")

    def export(self):
        raise NotImplementedError("Reached too far!!")

    def compile(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Reached too far!!")
