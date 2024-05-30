# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

import QEfficient
from QEfficient.transformers.modeling_utils import TransformersToQEffModulesDict


class QEFF_MODEL_TYPE(Enum):
    CAUSALLM = "LLM"
    DIFFUSION = "STABLE_DIFFUSION"
    AWQ = "AWQ"


class QEFFBaseModel(ABC):
    """
    This class acts as parent class for all the varieties of model class (i.e. LLMs, SD, quantized etc.).
    Enforces certain methods to be implemented by child classes.

    All the child classes must provide way to load, transform(optimize), exoprt to ONNX etc. capabilities.
    """
    def __init__(self) -> None:
        super().__init__()
        # Users can call generate or execute
        self.generate = self.execute

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        raise NotImplementedError("Must implement for child classes")

    @property
    def is_transformed(self) -> bool:
        raise NotImplementedError("Must implement for child classes")

    @abstractmethod
    def transform_export(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def transform_export_compile(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def transform(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def export(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def compile(self, *args, **kwargs) -> Any:
        pass


class QEFFAutoModelForCausalLM(QEFFBaseModel):
    """
    QEFF class for manipulating any causal language model from HuggingFace hub.
    """
    def __init__(self, model: nn.Module, pretrained_model_name_or_path: str) -> None:
        assert (model.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING.values() or
                model.__class__ in TransformersToQEffModulesDict.values()), f"Given model{model.__class__.__name__} could not be found in transformers library i.e. {MODEL_FOR_CAUSAL_LM_MAPPING.values()}" # type: ignore
        self.model: nn.Module = model
        self.model_files_path = pretrained_model_name_or_path

    def __repr__(self) -> str:
        return self.model.__repr__()
    
    @property
    def is_transformed(self) -> bool:
        return getattr(self.model, "qeff_transformed", False)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return cls(model=model, pretrained_model_name_or_path=pretrained_model_name_or_path)
    
    def transform_export(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Reached too far!!")
    
    def transform_export_compile(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Reached too far!!")
    
    def execute(self, *args, **kwargs): # type: ignore
        raise NotImplementedError("Reached too far!!")
    
    def transform(self):
        QEfficient.transform(self)
        return self

    def export(self):
        raise NotImplementedError("Reached too far!!")
    
    def compile(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Reached too far!!")
