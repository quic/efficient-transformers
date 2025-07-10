# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import math

from transformers import AutoConfig

from QEfficient.utils._utils import get_num_layers_from_config


def get_device_map(model_name, num_pp_stages, rank):
    """Returns device map for model layers based number of pipeline stages and given process rank.

    Args:
        model_name (str): model name to get the device map for.
        num_pp_stages (int): number of stages in pipeline
        rank (int): process rank

    Returns:
        Dict: A dictionary of layers and corresponding device id.

    Notes:
        - This device map structure is verified for llama models only.

    Example:
        Configuration for meta-llama/Llama-3.2-1B
        Total devices: 4 (2x PP x 2x DDP)

        PP (Pipeline Parallelism): Each copy of the model is split into 2 stages
        DDP (Distributed Data Parallel): 2 model copies run in parallel

        |-------------------------------------------------------------------------------
        | Process Rank |   Assigned Device IDs  | Model Component |
        |-------------------------------------------------------------------------------
        | Rank 0       | 0                 | model.embed_tokens |
        |              |                   | model.lm_head |
        |              |                   | model.layers.0 - model.layers.7 |
        |-------------------------------------------------------------------------------
        | Rank 0       | 1                 | model.norm |
        |              |                   | model.rotary_emb |
        |              |                   | model.layers.8 - model.layers.15 |
        |-------------------------------------------------------------------------------
        | Rank 1       | 2                 | model.embed_tokens |
        |              |                   | model.lm_head |
        |              |                   | model.layers.0 - model.layers.7 |
        |-------------------------------------------------------------------------------
        | Rank 1       | 3                 | model.norm |
        |              |                   | model.rotary_emb |
        |              |                   | model.layers.8 - model.layers.15 |
        |-------------------------------------------------------------------------------
    """

    config = AutoConfig.from_pretrained(model_name)
    num_layers = get_num_layers_from_config(config)

    first_device = rank * num_pp_stages
    last_device = rank * num_pp_stages + (num_pp_stages - 1)

    if config.tie_word_embeddings:
        lm_head_device = first_device
    else:
        lm_head_device = last_device

    device_map = {
        "model.embed_tokens": first_device,
        "lm_head": lm_head_device,
        "model.norm": last_device,
        "model.rotary_emb": last_device,
    }

    n_layer_per_stage = math.ceil(num_layers / num_pp_stages)

    for j in range(num_pp_stages):
        for i in range(n_layer_per_stage * j, min(n_layer_per_stage * (j + 1), num_layers)):
            device_map[f"model.layers.{i}"] = first_device + j

    return device_map
