# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import hashlib

import torch.nn as nn
import transformers

from QEfficient.base.common import AUTO_MODEL_MAP_TO_MODEL_TYPE_MAP, QEFF_MODEL_TYPE
from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_utils import TransformersToQEffModulesDict
from QEfficient.utils.logging_utils import logger


def replace_module_with_qeff_layers(model: nn.Module) -> None:
    """
    Replaces the transformers nn.Module classes with optmized QEff classes in place.
    ----------
    :param model: `torch.nn.Module` - Base PyTorch model.
    """
    # Replace if module class is registed in TransformersToQEffModulesDict
    target_module = TransformersToQEffModulesDict.get(model.__class__)
    if target_module is not None:
        model.__class__ = target_module

    # Iterate over child modules
    for _, module in model.named_children():
        replace_module_with_qeff_layers(module)


def get_params_hash(model: nn.Module) -> str:
    """
    Creates a Hash of all the parameters values i.e. weights using SHA256 algo.
    --------
    :param model: `torch.nn.Module` - Base PyTorch model.
    :returns: `str` - Hash string
    """
    hasher = hashlib.sha256()
    for _, params in model.named_parameters():
        hasher.update(params.data.numpy().tobytes())

    return hasher.hexdigest()


def transform_lm(model: nn.Module) -> nn.Module:
    """
    Replaces some Transformers torch.nn.Module layers for equivalent optimized modules for Cloud AI 100.
    ---------

    :model: `torch.nn.Module` - PyTorch model.

    Returns:
        PyTorch Module with replaced QEff layers.
    """

    # Introducnig qeff_transformed attribue in model to check status of transform
    if getattr(model, "qeff_transformed", False):
        print("Model is already transformed")
        return model

    # Get Hash of all params for checking later
    prior_params_hash = get_params_hash(model)
    logger.warning(f"The model {model.__class__} layers has been updated to QEff layers in-place")
    # Replace with QEff layers
    replace_module_with_qeff_layers(model)

    # Check with new params hash
    later_params_hash = get_params_hash(model)
    assert (
        prior_params_hash == later_params_hash
    ), "Weights were changed in the transform process, please report an issue"

    # Replace the Dyanmic cache utils update api
    transformers.cache_utils.DynamicCache.update = QEffDynamicCache.update

    setattr(model, "qeff_transformed", True)
    return model.eval()


def transform(model: QEFFBaseModel, form_factor="cloud"):
    """
    This function serves for optimizing any kind of model (i.e. LLM, SD, AWQ etc.) for Cloud AI 100.
    Will replace the torch.nn.Module layers of passed QEffModel with optimized implementation of the same.

    model: `torch.nn.Module` - object of any instance of class that is child of `QEFFBaseAutoModelFactory`
    form_factor: `str` - form factor configuration for optmizing the model, available options=["cloud", "edge"].
    """
    assert form_factor == "cloud", "Only form_factor='cloud' is supported as of now!"
    # FIXME: move this to class and use model.transform()
    if AUTO_MODEL_MAP_TO_MODEL_TYPE_MAP.get(model.__class__, None) == QEFF_MODEL_TYPE.CAUSALLM:
        transform_lm(model.model)  # type: ignore
        return model
    else:
        raise NotImplementedError(f"Recieved unsupported class of type {type(model)}")
