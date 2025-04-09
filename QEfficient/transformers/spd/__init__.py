# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from QEfficient.transformers.spd.causal_lm_forward import tlm_forward
from QEfficient.transformers.spd.turbo import build_and_attach_turbo


def build_and_attach_mlp(model, config):
    model_type = config["model_type"]
    func_name = f"build_and_attach_{model_type}"
    func = globals().get(func_name)
    if func:
        return func(model, config)
    else:
        raise ValueError(f"function {func_name} not found.")
