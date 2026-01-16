# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Utility functions for preparing training configurations.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import is_accelerate_available

from QEfficient.finetune.experimental.core.config_manager import ConfigManager


def prepare_training_config(
    config_manager: ConfigManager,
    include_num_input_tokens_seen: bool = False,
    use_cpu: bool = False,
) -> Dict[str, Any]:
    """
    Prepare and transform training configuration for trainer initialization.

    Args:
        config_manager: ConfigManager instance with loaded configuration
        include_num_input_tokens_seen: Override for include_num_input_tokens_seen (default: False)
        use_cpu: Override for use_cpu (default: False)

    Returns:
        Dictionary of training arguments ready for trainer initialization
    """
    training_config = dict(config_manager.get_training_config())

    # Handle scheduler configuration
    scheduler_config = config_manager.get_scheduler_config()
    training_config.setdefault("lr_scheduler_type", scheduler_config.get("scheduler_name"))
    training_config.setdefault("warmup_ratio", scheduler_config.get("warmup_ratio"))
    training_config.setdefault("warmup_steps", scheduler_config.get("warmup_steps"))

    # Handle dataset configuration for dataloader settings
    dataset_config = config_manager.get_dataset_config()
    training_config.setdefault("dataloader_pin_memory", dataset_config.get("dataloader_pin_memory"))
    training_config.setdefault("dataloader_persistent_workers", dataset_config.get("dataloader_persistent_workers"))
    training_config.setdefault("dataloader_prefetch_factor", dataset_config.get("dataloader_prefetch_factor"))
    training_config.setdefault("dataloader_drop_last", dataset_config.get("dataloader_drop_last"))
    training_config.setdefault("dataloader_num_workers", dataset_config.get("dataloader_num_workers"))
    training_config.setdefault("group_by_length", dataset_config.get("group_by_length"))

    # Handle DDP configuration
    if training_config.get("ddp_config") is not None:
        ddp_config = training_config.pop("ddp_config")
        training_config = {**training_config, **ddp_config}

    # Handle accelerator configuration
    accelerate_config_path = training_config.pop("accelerator_config", None)
    if accelerate_config_path:
        try:
            with open(accelerate_config_path, "r", encoding="utf-8") as file:
                accelerate_config = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to load accelerate config from {accelerate_config_path}: {e}")

        parallelism_dict = accelerate_config.get("parallelism_config", None)
        if parallelism_dict is not None:
            if is_accelerate_available("1.10.1"):
                from accelerate.parallelism_config import ParallelismConfig
            else:
                raise RuntimeError("Accelerate package with version 1.10.1 or higher is required.")
            parallelism_config = ParallelismConfig(
                dp_replicate_size=parallelism_dict.get("dp_replicate_size", 1),
                dp_shard_size=parallelism_dict.get("dp_shard_size", 1),
                tp_size=parallelism_dict.get("tp_size", 1),
                cp_size=parallelism_dict.get("cp_size", 1),
            )
            training_config["parallelism_config"] = parallelism_config

        fsdp_config = accelerate_config.get("fsdp_config", None)
        if fsdp_config is not None:
            training_config["fsdp_config"] = fsdp_config

    return training_config
