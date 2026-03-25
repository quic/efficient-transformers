# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Utility functions for creating device maps for pipeline parallelism.
"""

from typing import Dict, Optional

import numpy as np
import torch
from transformers import AutoConfig

from QEfficient.finetune.experimental.core.utils.dist_utils import get_local_rank
from QEfficient.utils._utils import get_num_layers_from_config


def get_device_map(
    model_name: str,
    device: str,
    pp_degree: int = 1,
) -> Optional[Dict[str, int]]:
    """
    Returns device map for the given model based on PP and DDP configuration.

    Args:
        model_name: Name of the model to load configuration from.
        device: Device type (e.g., 'cuda', 'qaic').
        pp_degree: Pipeline parallelism degree (number of pipeline stages). > 1 enables PP.
    Returns:
        Dict: A dictionary mapping layer names to device IDs, or None if no PP.
    """
    if pp_degree <= 1:
        return None

    torch_device = torch.device(device)
    num_available_devices = getattr(torch, torch_device.type).device_count()

    if pp_degree > num_available_devices:
        raise ValueError(
            f"pp_degree ({pp_degree}) cannot exceed the number of available {device} devices "
            f"({num_available_devices}). Reduce pp_degree or use a node with more devices."
        )
    elif pp_degree == num_available_devices:
        device_map = "auto"
    else:  # pp_degree < num_available_devices
        device_map = custom_device_map(model_name, device, pp_degree)

    return device_map


def custom_device_map(model_name: str, device: str, pp_degree: int) -> Dict[str, int]:
    """
    Returns custom device map for model layers based on number of pipeline stages and process rank.

    Args:
        model_name: Name of the model to load configuration from.
        device: Device type (e.g., 'cuda', 'qaic').
        pp_degree: Pipeline parallelism degree (number of pipeline stages).

    Returns:
        Dict: A dictionary mapping layer names to device IDs.

    Notes:
        - This device map structure is verified for llama models primarily.
        - For other architectures, you may need to adjust the layer naming conventions.
        - Layers are distributed as evenly as possible: the first (num_layers % pp_degree)
          stages receive one extra layer each.

    Example:
        Example config for PP + DDP is provided below as it works for only PP as well.
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

    model_config = AutoConfig.from_pretrained(model_name)
    num_layers = get_num_layers_from_config(model_config)
    local_rank = get_local_rank()

    if num_layers < pp_degree:
        raise ValueError(
            f"Number of model layers ({num_layers}) must be >= pp_degree ({pp_degree}). "
            f"Cannot split {num_layers} layers across {pp_degree} pipeline stages."
        )

    first_device = local_rank * pp_degree
    last_device = local_rank * pp_degree + (pp_degree - 1)

    # Handle tied embeddings
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

    # Distribute layers as evenly as possible across stages.
    # The first (num_layers % pp_degree) stages get one extra layer each.
    base_layers, remainder = divmod(num_layers, pp_degree)
    layers_per_stage = np.array([base_layers + (1 if i < remainder else 0) for i in range(pp_degree)])

    # Create device assignment per layer
    pp_device_map = np.repeat(np.arange(pp_degree), layers_per_stage)

    # Assign each layer to a device
    for i in range(num_layers):
        device_map[f"model.layers.{i}"] = int(pp_device_map[i] + local_rank * pp_degree)

    return device_map


def validate_pp_config(
    pp_degree: int,
    device: str,
    local_world_size: int = 1,
) -> None:
    """
    Validate pipeline parallelism configuration.

    Args:
        pp_degree: Pipeline parallelism degree (number of pipeline stages). Must be > 1 to enable PP.
        device: Device type (e.g., 'cuda', 'qaic').
        local_world_size: Number of processes per node for DDP.

    Raises:
        AssertionError: If configuration is invalid.
    """
    if pp_degree > 1:
        # Validate device availability
        torch_device = torch.device(device)
        num_available_devices = getattr(torch, torch_device.type).device_count()

        assert local_world_size * pp_degree <= num_available_devices, (
            f"Number of devices required per node (LOCAL_WORLD_SIZE * pp_degree = "
            f"{local_world_size} * {pp_degree} = {local_world_size * pp_degree}) "
            f"should be <= locally available devices ({num_available_devices})."
        )
