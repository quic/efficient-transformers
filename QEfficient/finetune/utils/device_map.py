# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import math


def get_device_map(rank, num_pp_stages, num_layers):
    """Returns device map for model layers and given process rank based on number of pipeline stages.

    Args:
        rank (int): process rank
        num_pp_stages (int): number of stages in pipeline
        num_layers (int): total number of layers in the models

    Returns:
        Dict: A dictionary of layers and corresponding device id.

    Notes:
        - This device map structure is verified for llama models only.
    """
    device_map = {
        "model.embed_tokens": rank * num_pp_stages,
        "lm_head": rank * num_pp_stages,
        "model.norm": rank * num_pp_stages + (num_pp_stages - 1),
        "model.rotary_emb": rank * num_pp_stages + (num_pp_stages - 1),
    }
    n_layer_per_stage = math.ceil(num_layers / num_pp_stages)
    for j in range(num_pp_stages):
        for i in range(n_layer_per_stage * j, min(n_layer_per_stage * (j + 1), num_layers)):
            device_map[f"model.layers.{i}"] = rank * num_pp_stages + j
    return device_map
