# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from QEfficient.transformers.spd.turbo import build_and_attach_turbo
from QEfficient.utils.spd_utils import get_speculative_config, get_speculative_weights

model_type_registry = dict(turbo=build_and_attach_turbo)


def build_and_attach_mlp(model, pretrained_model_name_or_path, speculative_model_type: str, **kwargs):
    speculative_config: dict = get_speculative_config(pretrained_model_name_or_path, **kwargs)
    speculative_weights: str = get_speculative_weights(pretrained_model_name_or_path, **kwargs)

    if (model_type := speculative_config.get("model_type")) is None:
        speculative_config["model_type"] = speculative_model_type
    else:
        if model_type != speculative_model_type:
            raise ValueError(
                f"`model_type` key from speculator config ({model_type} does not match input model type ({speculative_model_type})."
            )
    func = model_type_registry[speculative_model_type]
    model = func(model, speculative_config, speculative_weights)
    model.config.speculative_config = speculative_config
    return model
