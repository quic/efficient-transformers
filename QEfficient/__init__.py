# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Any, Union

from QEfficient.loader import QEFFAutoModel  # noqa: F401
from QEfficient.loader.loader_factory import AUTO_MODEL_MAP_TO_MODEL_TYPE_MAP, QEFF_MODEL_TYPE, QEFFAutoModelForCausalLM
from QEfficient.transformers.modeling_utils import transform as transform_hf


def transform(model: Union[QEFFAutoModelForCausalLM, Any], form_factor="cloud"):
    """Low level apis in library
    model : instance of nn.Module
    type : Transformers | Diffusers, default : Transformers
    """
    assert form_factor == "cloud", "Only form_factor='cloud' is supported as of now!"
    if AUTO_MODEL_MAP_TO_MODEL_TYPE_MAP.get(model.__class__, None) == QEFF_MODEL_TYPE.LLM:
        transform_hf(model.model, form_factor)
        return model
    else:
        raise NotImplementedError(f"Recieved unsupported class of type {type(model)}")
