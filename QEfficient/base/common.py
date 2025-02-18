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

from typing import Any

from transformers import AutoConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM


class QEFFCommonLoader:
    """
    Provides HuggingFace model loading interface same as transformers APIs.
    Supports loading any model on HuggingFace.
    Wrapper on top of Auto Classes
    """

    def __init__(self, *args: Any, **kwds: Any) -> None:
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)`"
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs) -> QEFFBaseModel:
        """
        Downloads HuggingFace model if already doesn't exist locally, returns QEFFAutoModel object based on type of model.
        """
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        architecture = config.architectures[0] if config.architectures else None

        if architecture in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            model_class = QEFFAutoModelForCausalLM
        else:
            raise NotImplementedError(
                f"Unknown architecture={architecture}, either use specific auto model class for loading the model or raise an issue for support!"
            )

        local_model_dir = kwargs.pop("local_model_dir", None)
        hf_token = kwargs.pop("hf_token", None)
        continuous_batching = True if kwargs.pop("full_batch_size", None) else False

        qeff_model = model_class.from_pretrained(
            pretrained_model_name_or_path=(local_model_dir if local_model_dir else pretrained_model_name_or_path),
            token=hf_token,
            continuous_batching=continuous_batching,
            **kwargs,
        )
        return qeff_model
