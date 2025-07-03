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

    Example:
        for meta-llama/Llama-3.2-1B, 2x pp and 2x ddp,(total 4 devices)
        2x pp - each copy of model is split in 2 stages.
        2x ddp -  there will 2 copies of the model.

        Process rank 0 across device ids 0,1
        {'model.embed_tokens': 0, 'lm_head': 0, 'model.norm': 1, 'model.rotary_emb': 1, 'model.layers.0': 0, 'model.layers.1': 0, 'mo
        del.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'm
        odel.layers.8': 1, 'model.layers.9': 1, 'model.layers.10': 1, 'model.layers.11': 1, 'model.layers.12': 1, 'model.layers.13':
        1, 'model.layers.14': 1, 'model.layers.15': 1}

        Process rank 0 across device ids 2,3
        {'model.embed_tokens': 2, 'lm_head': 2, 'model.norm': 3, 'model.rotary_emb': 3, 'model.layers.0': 2, 'model.layers.1': 2, 'mo
        del.layers.2': 2, 'model.layers.3': 2, 'model.layers.4': 2, 'model.layers.5': 2, 'model.layers.6': 2, 'model.layers.7': 2, 'm
        odel.layers.8': 3, 'model.layers.9': 3, 'model.layers.10': 3, 'model.layers.11': 3, 'model.layers.12': 3, 'model.layers.13':
        3, 'model.layers.14': 3, 'model.layers.15': 3}
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
