# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
MODEL_TYPE_TO_QEFF_AUTO_MODEL_MAP dictionary defines the mapping between names of the varities of Transformer model defined in
QEFF_MODEL_TYPE and the classes that implement the methods i.e.(compile, export etc.) for those types.

QEFFAutoModel provides a common interface for loading the HuggingFace models using either the HF card name of local path of downloaded model.
"""

import os
from enum import Enum
from typing import Any, Dict, Type

from transformers import AutoConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils._utils import login_and_download_hf_lm


class QEFF_MODEL_TYPE(Enum):
    """
    Defines Names of the different varities of transformer models.
    """

    CAUSALLM = "LLM"
    DIFFUSION = "DIFFUSION"
    AWQ = "AWQ"


MODEL_TYPE_TO_QEFF_AUTO_MODEL_MAP: Dict[QEFF_MODEL_TYPE, Type[QEFFBaseModel]] = {
    QEFF_MODEL_TYPE.CAUSALLM: QEFFAutoModelForCausalLM
}

AUTO_MODEL_MAP_TO_MODEL_TYPE_MAP: Dict[Type[QEFFBaseModel], QEFF_MODEL_TYPE] = {
    v: k for k, v in MODEL_TYPE_TO_QEFF_AUTO_MODEL_MAP.items()
}


def get_hf_model_type(hf_model_path: str) -> QEFF_MODEL_TYPE:
    """
    Loads model config file and returns the type of the model (i.e. LLMs, SD, quantized etc.) as supported by the library.
    """
    assert os.path.isdir(
        hf_model_path
    ), "Pleae pass local dir path where the model is downloaded; use `QEfficient.utils.login_and_download_hf_lm` for downloading hf model"
    config, kwargs = AutoConfig.from_pretrained(
        hf_model_path,
        return_unused_kwargs=True,
    )

    if config.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING:
        return QEFF_MODEL_TYPE.CAUSALLM
    else:
        raise NotImplementedError(f"model type {type(config)} is not yet supported")


class QEFFCommonLoader:
    """
    Provides HuggingFace model loading interface same as transformers APIs.
    Supports loading any model on HuggingFace.
    """

    def __init__(self, *args: Any, **kwds: Any) -> None:
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)`"
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs) -> QEFFBaseModel:
        """
        Downloads HuggingFace model if already doesn't exist locally, returns QEffAutoModel object based on type of model.
        """
        if not os.path.isdir(pretrained_model_name_or_path):
            # Save model_card_name if passed
            model_card_name = kwargs.pop("model_card_name", pretrained_model_name_or_path)
            kwargs.update({"model_card_name": model_card_name})
            pretrained_model_name_or_path = login_and_download_hf_lm(pretrained_model_name_or_path, *args, **kwargs)
        model_type = get_hf_model_type(hf_model_path=pretrained_model_name_or_path)
        qeff_auto_model_class = MODEL_TYPE_TO_QEFF_AUTO_MODEL_MAP[model_type]
        assert issubclass(
            qeff_auto_model_class, QEFFBaseModel
        ), f"Expected class that inherits {QEFFBaseModel}, got {type(qeff_auto_model_class)}"

        return qeff_auto_model_class.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs
        )
