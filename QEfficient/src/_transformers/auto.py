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
<<<<<<< HEAD
    def __init__(self, model: nn.Module, transform:bool = True) -> None:
        assert (model.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING.values() or
                # FIXME: Use model architectures here instead of complete dictionary TransformersToQEffModulesDict
                model.__class__ in TransformersToQEffModulesDict.values()), f"Given model{model.__class__.__name__} could not be found in transformers library i.e. {MODEL_FOR_CAUSAL_LM_MAPPING.values()}" # type: ignore
        self.model: nn.Module = model
        if transform:
=======

    def __init__(self, model: nn.Module, pretrained_model_name_or_path: str, **kwargs) -> None:
        super().__init__()
        assert (
            model.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING.values()
            or
            # FIXME: Use model architectures here instead of complete dictionary TransformersToQEffModulesDict
            model.__class__ in TransformersToQEffModulesDict.values()
        ), f"Given model{model.__class__.__name__} could not be found in transformers library i.e. {MODEL_FOR_CAUSAL_LM_MAPPING.values()}"  # type: ignore
        self.model_card_name = kwargs.pop("model_card_name", None)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.model: nn.Module = model
        try:
            self.model.config.use_cache = True
        except Exception:
            logger.info("Could not set config.use_cache=True, might result into errors while executing the model")
        self.kwargs = kwargs
        self._tokenizer = None
        if kwargs.get("transform", True):
>>>>>>> 09cb3eb (Updated the assert condition for bs > 1 and full batch size >1)
            self.transform()

    def __repr__(self) -> str:
        return self.model.__repr__()
    
    @property
    def is_transformed(self) -> bool:
        return getattr(self.model, "qeff_transformed", False)

<<<<<<< HEAD
=======
    @property
    def tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        if self._tokenizer is None:
            self._tokenizer = self.get_tokenizer()
        return self._tokenizer

    def get_model_card_name(self) -> str:
        # FIXME: use getter
        if self.model_card_name is None:
            # Handle when pretrained_model_name_or_path is a path and we don't know the model_card_name
            assert not os.path.isdir(
                self.pretrained_model_name_or_path
            ), f"Please provide `model_card_name` argument as valid string, got {self.model_card_name}"
            self.model_card_name = self.pretrained_model_name_or_path

        return self.model_card_name

>>>>>>> 09cb3eb (Updated the assert condition for bs > 1 and full batch size >1)
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        """
        This method accepts All the parameters that are acceptable by transformers.AutoModelForCausalLM.
        There are few additional parameters that this method can take:
        :param transform:bool. Whether to optimize model for KV retention; default is True. Pass False to get BertStyle model.
        """
<<<<<<< HEAD
        transform: bool = kwargs.get("transform", True)
        kwargs.update({"use_cache": True})  # Always pass use_cache = True, to get KV values as output during ONNX export 
        kwargs.update({"attn_implementation" : "eager"}) # Always use eager mode for attention implementation
        
        model = QEFFAutoModelToTransformersAutoModelMap[cls.__name__].from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return cls(model, transform=transform)
        

    def transform_export(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Reached too far!!")
    
    def transform_export_compile(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Reached too far!!")
        
=======
        kwargs.update(
            {"use_cache": True}
        )  # Always pass use_cache = True, to get KV values as output during ONNX export
        model_card_name = kwargs.pop("model_card_name", None)
        model = QEFFAutoModelToTransformersAutoModelMap[cls.__name__].from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        kwargs.update({"model_card_name": model_card_name})
        return cls(model, pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs)

    def get_tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=self.pretrained_model_name_or_path, **self.kwargs)
        return tokenizer

>>>>>>> 09cb3eb (Updated the assert condition for bs > 1 and full batch size >1)
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
