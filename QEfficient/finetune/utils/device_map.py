# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import numpy as np
import torch
from transformers import AutoConfig

from QEfficient.finetune.utils.helper import get_local_rank
from QEfficient.utils._utils import get_num_layers_from_config


def get_device_map(train_config):
    """Returns device map for the given model.

    Args:
        train_config (TrainConfig): Training configuration object contaning model name and number of pipeline stages etc.

    Returns:
        Dict: A dictionary of layers and corresponding device id.
    """
    torch_device = torch.device(train_config.device)
    num_available_devices = getattr(torch, torch_device.type).device_count()
    if train_config.enable_pp:
        if train_config.enable_ddp:
            device_map = custom_device_map(train_config)
        elif train_config.num_pp_stages < num_available_devices:
            device_map = custom_device_map(train_config)
        elif train_config.num_pp_stages == num_available_devices:
            device_map = "auto"
    else:
        device_map = None

    return device_map


def custom_device_map(train_config):
    """Returns custom device map for model layers based number of pipeline stages and given process rank.

    Args:
        train_config (TrainConfig): Training configuration object contaning model name and number of pipeline stages etc.

    Returns:
        Dict: A dictionary of layers and corresponding device id.

    Notes:
        - This device map structure is verified for llama models only.

    Example:
        Configuration for meta-llama/Llama-3.2-1B
        Total devices: 4 (2x PP x 2x DDP)

        PP (Pipeline Parallelism): Each copy of the model is split into 2 stages
        DDP (Distributed Data Parallel): 2 model copies run in parallel

        |--------------------------------------------------------------------------|
        | Process Rank |   Assigned Device IDs  | Model Component                  |
        |--------------------------------------------------------------------------|
        | Rank 0       | 0                      | model.embed_tokens               |
        |              |                        | model.lm_head                    |
        |              |                        | model.layers.0 - model.layers.7  |
        |--------------------------------------------------------------------------|
        | Rank 0       | 1                      | model.norm                       |
        |              |                        | model.rotary_emb                 |
        |              |                        | model.layers.8 - model.layers.15 |
        |--------------------------------------------------------------------------|
        | Rank 1       | 2                      | model.embed_tokens               |
        |              |                        | model.lm_head                    |
        |              |                        | model.layers.0 - model.layers.7  |
        |--------------------------------------------------------------------------|
        | Rank 1       | 3                      | model.norm                       |
        |              |                        | model.rotary_emb                 |
        |              |                        | model.layers.8 - model.layers.15 |
        |--------------------------------------------------------------------------|
    """

    model_config = AutoConfig.from_pretrained(train_config.model_name)
    num_layers = get_num_layers_from_config(model_config)
    num_pp_stages = train_config.num_pp_stages
    local_rank = get_local_rank()
    first_device = local_rank * num_pp_stages
    last_device = local_rank * num_pp_stages + (num_pp_stages - 1)

    if model_config.tie_word_embeddings:
        lm_head_device = first_device
    else:
        lm_head_device = last_device

    device_map = {
        "model.embed_tokens": first_device,
        "lm_head": lm_head_device,
        "model.norm": last_device,
        "model.rotary_emb": last_device,
    }
    n_layer_per_stage = np.ceil(num_layers / num_pp_stages)

    pp_stage_ids = np.arange(num_pp_stages)
    pp_device_map = np.repeat(pp_stage_ids, n_layer_per_stage)

    for i in range(num_layers):
        device_map[f"model.layers.{i}"] = pp_device_map[i] + local_rank * num_pp_stages

    return device_map
