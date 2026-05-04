# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Utility functions for preparing training configurations.
"""

from typing import Any, Dict

from accelerate.utils import ParallelismConfig

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

    Returns:
        Dictionary of training arguments ready for trainer initialization
    """
    # Get training config as dict and create mutable copy to avoid mutating original
    training_config = dict(config_manager.get_training_config())

    # torch_dtype is a model loading concern and is handled in model config.
    parallelism_config = {}
    tp_degree = training_config.get("tp_degree", 1)
    pp_degree = training_config.get("pp_degree", 1)
    ddp_degree = training_config.get("ddp_degree", 1)

    if tp_degree > 1:
        parallelism_config["tp_size"] = tp_degree

    # ddp_degree is a TP+DDP shaping hint and should not force data-parallel sizing
    # for pure DDP (single-node or multi-node), where Accelerate derives world size
    # from launcher environment (RANK/WORLD_SIZE/LOCAL_WORLD_SIZE).
    if ddp_degree > 1 and (tp_degree > 1 or pp_degree > 1):
        parallelism_config["dp_replicate_size"] = ddp_degree

    if parallelism_config:  # Only inject if at least one parallelism dimension is active
        pc = ParallelismConfig(**parallelism_config)
        training_config["parallelism_config"] = pc

    # Keep AMP/autocast flags decoupled from model `torch_dtype`.
    # `fp16`/`bf16` here control mixed precision execution
    # (and GradScaler path for fp16).
    training_config["fp16"] = bool(training_config.get("fp16", False))
    training_config["bf16"] = bool(training_config.get("bf16", False))

    training_config["data_seed"] = training_config.get("seed")

    # Handle scheduler configuration
    scheduler_config = config_manager.get_scheduler_config()
    training_config.setdefault("lr_scheduler_type", scheduler_config.get("scheduler_name"))

    # Set warmup_ratio and warmup_steps from scheduler_config if they exist and are not None
    warmup_ratio = scheduler_config.get("warmup_ratio")
    if warmup_ratio is not None:
        training_config["warmup_ratio"] = warmup_ratio
    warmup_steps = scheduler_config.get("warmup_steps")
    if warmup_steps is not None:
        training_config["warmup_steps"] = warmup_steps

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
        if not isinstance(ddp_config, dict):
            from dataclasses import asdict, is_dataclass

            if is_dataclass(ddp_config):
                ddp_config = asdict(ddp_config)
            else:
                raise TypeError(
                    f"ddp_config must be a dict or DdpConfig dataclass instance, "
                    f"got {type(ddp_config).__name__}: {ddp_config}"
                )

        # Merge ddp_config into training_config
        training_config = {**training_config, **ddp_config}

    return training_config
