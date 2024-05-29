# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from typing import Any, Type

from QEfficient.loader.loader_factory import (
    MODEL_TYPE_TO_QEFF_AUTO_MODEL_MAP,
    QEFFBaseModel,
    get_hf_model_type,
)
from QEfficient.utils._utils import login_and_download_hf_lm


class QEFFAutoModel:
    """
    Provides HuggingFace model loading interface same as transformers APIs.
    Supports loading any model on HuggingFace.
    """
    def __init__(self, *args: Any, **kwds: Any) -> None:
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)`")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs) -> Type[QEFFBaseModel]:
        """
        Downloads HuggingFace model if already doesn't exist locally, returns QEffAutoModel object based on type of model.
        """
        pretrained_model_name_or_path = pretrained_model_name_or_path if os.path.isdir(pretrained_model_name_or_path) \
            else login_and_download_hf_lm(pretrained_model_name_or_path, *args, **kwargs)
        model_type = get_hf_model_type(hf_model_path=pretrained_model_name_or_path)
        qeff_auto_model_class = MODEL_TYPE_TO_QEFF_AUTO_MODEL_MAP[model_type]
        assert issubclass(qeff_auto_model_class, QEFFBaseModel), f"Expected class that inherits {QEFFBaseModel}, got {type(qeff_auto_model_class)}"

        return qeff_auto_model_class.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
