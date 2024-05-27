# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import os
from typing import Any
from abc import ABC, abstractmethod
from enum import Enum
from typing import Union

from qtpy import API
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

from QEfficient.utils.run_utils import ApiRunner, run_hf_lm_model_with_pt
import QEfficient

class QEFFBaseAutoModelFactory(ABC):
    
    def __init__(self) -> None:
        super().__init__()
        # Users can call generate or execute
        self.generate = self.execute
    
    @abstractmethod
    def from_pretrained(self, pretrained_model_name_or_path: str, *args, **kwargs):
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        pass
    
    @abstractmethod
    def transform(self, *args, **kwargs) -> Any:
        pass
    
    @abstractmethod
    def export(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Reached too far!!")
    

class QEFFAutoModelForCausalLM(QEFFBaseAutoModelFactory):
    def __init__(self, model: nn.Module, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], pretrained_model_name_or_path: str) -> None:
        assert model.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING.values(), f"Given model{model.__class__.__name__} could not be found in transformers library i.e. {MODEL_FOR_CAUSAL_LM_MAPPING.values()}" # type: ignore
        self.model = model
        self.tokenizer = tokenizer
        self.model_files_path = pretrained_model_name_or_path
        self._model_executor = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return cls(model=model, tokenizer=tokenizer, pretrained_model_name_or_path=pretrained_model_name_or_path)
    
    def _run_kv_lm_model_with_pt(self, prompt, prompt_len, ctx_len):
        api_runner = ApiRunner(self.tokenizer, prompt=prompt, prompt_len=prompt_len, ctx_len=ctx_len)
        return api_runner.run_kv_model_on_pytorch(self.model, )

    def execute(self, prompt: str, prompt_len: int = None, ctx_len: int = None, max_gen_length: int = 128): # type: ignore
        if getattr(self.model, "qeff_transformed", False):
            output_ids = run_hf_lm_model_with_pt(self.model, self.tokenizer, prompt, max_gen_length)
        else:
            output_ids = self._run_kv_lm_model_with_pt(prompt, prompt_len, ctx_len)
        return output_ids
    
    def transform(self):
        QEfficient.transform(self.model)
        return self

    def export(self):
        pass


class QEFF_MODEL_TYPE(Enum):
    LLM = "LLM"
    STABLE_DIFFUSION = "STABLE_DIFFUSION"
    AWQ = "AWQ"


MODEL_TYPE_TO_QEFF_AUTO_MODEL_MAP= {
    QEFF_MODEL_TYPE.LLM: QEFFAutoModelForCausalLM
}


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
