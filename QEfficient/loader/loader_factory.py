# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

import QEfficient


class QEFFBaseAutoModelFactory(ABC):
    
    def __init__(self) -> None:
        super().__init__()
        # Users can call generate or execute
        self.generate = self.execute
    
    @abstractmethod
    def from_pretrained(self, pretrained_model_name_or_path: str, *args, **kwargs):
        pass
    
    @property
    def is_transformed(self) -> bool:
        raise NotImplementedError("Must implement for child classes")

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        pass
    
    @abstractmethod
    def transform(self, *args, **kwargs) -> Any:
        pass
    
    @abstractmethod
    def export(self, *args, **kwargs) -> Any:
        pass
    

class QEFFAutoModelForCausalLM(QEFFBaseAutoModelFactory):
    def __init__(self, model: nn.Module, pretrained_model_name_or_path: str) -> None:
        assert model.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING.values(), f"Given model{model.__class__.__name__} could not be found in transformers library i.e. {MODEL_FOR_CAUSAL_LM_MAPPING.values()}" # type: ignore
        self.model: nn.Module = model
        self.model_files_path = pretrained_model_name_or_path

    @property
    def is_transformed(self) -> bool:
        return getattr(self.model, "qeff_transformed", False)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return cls(model=model, pretrained_model_name_or_path=pretrained_model_name_or_path)
    
    def execute(self, *args, **kwargs): # type: ignore
        raise NotImplementedError("Reached too far!!")
    
    def transform(self):
        QEfficient.transform(self.model)
        return self

    def export(self):
        raise NotImplementedError("Reached too far!!")
    
    def __repr__(self) -> None:
        print(self.model)


class QEFF_MODEL_TYPE(Enum):
    LLM = "LLM"
    STABLE_DIFFUSION = "STABLE_DIFFUSION"
    AWQ = "AWQ"


MODEL_TYPE_TO_QEFF_AUTO_MODEL_MAP = {
    QEFF_MODEL_TYPE.LLM: QEFFAutoModelForCausalLM
}

AUTO_MODEL_MAP_TO_MODEL_TYPE_MAP = {v:k for k,v in MODEL_TYPE_TO_QEFF_AUTO_MODEL_MAP.items()}


def get_hf_model_type(hf_model_path: str):
    assert os.path.isdir(hf_model_path), "Pleae pass local dir path where the model is downloaded use `QEfficient.utils.login_and_download_hf_lm` for downloading hf model"
    config, kwargs = AutoConfig.from_pretrained(
                hf_model_path,
                return_unused_kwargs=True,
            )
    if config.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING:
        # FIXME: Add logic to handle if quantization config is stored in separate quant_config.json outside of config, also create a separate function for this and below lines
        quant_config = getattr(config, "quantization_config", getattr(config, "quant_config", None))
        if quant_config is not None:
            if quant_config.get("quant_method", None) == "awq":
                return QEFF_MODEL_TYPE.AWQ
            else:
                raise NotImplementedError(f"current model type is not yet supported {type(config)}")
        else:
            return QEFF_MODEL_TYPE.LLM
    else:
        raise NotImplementedError(f"model type {type(config)} is not yet supported")
