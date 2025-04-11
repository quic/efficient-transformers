# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from QEfficient.transformers.spd.turbo import build_and_attach_turbo

model_type_registry = dict(turbo=build_and_attach_turbo)


def build_and_attach_mlp(model, config: dict):
    model_type = config["model_type"]
    if model_type not in model_type_registry:
        raise NotImplementedError(f"model type {model_type} does not have a registered `build_and_attach` function.")
    func = model_type_registry[model_type]
    return func(model, config)
