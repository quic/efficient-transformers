# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

import torch.nn as nn
import transformers
from transformers import AutoTokenizer

from QEfficient.finetune.experimental.core.component_registry import registry
from QEfficient.finetune.experimental.core.logger import Logger
from QEfficient.finetune.experimental.core.utils.dataset_utils import insert_pad_token

logger = Logger(__name__)


class BaseModel(nn.Module, ABC):
    """Shared skeleton for every finetunable model in the system."""

    def __init__(self, model_name: str, **model_kwargs: Any) -> None:
        super().__init__()
        self.model_name = model_name
        self.model_kwargs: Dict[str, Any] = model_kwargs
        self._model: Optional[nn.Module] = None
        self._tokenizer: Any = None  # HF tokenizers are not nn.Modules.

    # Factory constructor: load model after __init__ finishes
    @classmethod
    def create(cls, model_name: str, **model_kwargs: Any) -> "BaseModel":
        obj = cls(model_name, **model_kwargs)
        # load model after __init__ finishes
        module = obj.load_model()
        if not isinstance(module, nn.Module):
            raise TypeError(f"load_model() must return nn.Module, got {type(module)}")
        obj._model = module
        return obj

    @abstractmethod
    def load_model(self) -> nn.Module:
        """Load and return the underlying torch.nn.Module."""
        pass

    def load_tokenizer(self) -> Any:
        """Override if the model exposes a tokenizer."""
        warnings.warn(f"{type(self).__name__} does not provide a tokenizer.", category=UserWarning)
        return None

    # Lazy accessors
    @property
    def model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("Model not loaded; use .create(...) to load.")
        return self._model

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            self._tokenizer = self.load_tokenizer()
        return self._tokenizer

    # nn.Module API surface
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return super().train(mode)

    def eval(self):
        return self.train(False)


@registry.model("hf")
class HFModel(BaseModel):
    """HuggingFace-backed model with optional quantization."""

    def __init__(
        self,
        model_name: str,
        auto_class_name: str = "AutoModelForCausalLM",
        *,
        tokenizer_name: Optional[str] = None,
        **model_kwargs: Any,
    ) -> None:
        super().__init__(model_name, **model_kwargs)
        self.tokenizer_name = tokenizer_name or model_name
        self.auto_class: Type = self._resolve_auto_class(auto_class_name)

    @staticmethod
    def _resolve_auto_class(auto_class_name: str) -> Type:
        if not hasattr(transformers, auto_class_name):
            candidates = sorted(name for name in dir(transformers) if name.startswith("AutoModel"))
            raise ValueError(
                f"Unsupported Auto class '{auto_class_name}'. Available candidates: {', '.join(candidates)}"
            )
        return getattr(transformers, auto_class_name)

    # def _build_quant_config(self) -> Optional[BitsAndBytesConfig]:
    #     if not self.model_kwargs.get("load_in_4bit"):
    #         return None
    #     return BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type=self.model_kwargs.get("bnb_4bit_quant_type", "nf4"),
    #         bnb_4bit_compute_dtype=self.model_kwargs.get("bnb_4bit_compute_dtype", torch.float16),
    #         bnb_4bit_use_double_quant=self.model_kwargs.get("bnb_4bit_use_double_quant", True),
    #     )

    def configure_model_kwargs(self) -> Dict[str, Any]:
        """Hook for subclasses to tweak HF `.from_pretrained` kwargs."""

        extra = dict(self.model_kwargs)
        # extra["quantization_config"] = self._build_quant_config()
        return extra

    def load_model(self) -> nn.Module:
        logger.log_rank_zero(f"Loading HuggingFace model '{self.model_name}' via {self.auto_class.__name__}")

        return self.auto_class.from_pretrained(
            self.model_name,
            **self.configure_model_kwargs(),
        )

    def load_tokenizer(self) -> AutoTokenizer:
        """Load Hugging Face tokenizer."""
        logger.log_rank_zero(f"Loading tokenizer '{self.tokenizer_name}'")
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        insert_pad_token(tokenizer)
        return tokenizer
