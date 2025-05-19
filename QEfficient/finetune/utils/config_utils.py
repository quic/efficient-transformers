# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import inspect
import json
import os
from dataclasses import asdict
from typing import Any, Dict

import torch.distributed as dist
import torch.utils.data as data_utils
import yaml
from peft import (
    AdaptionPromptConfig,
    PrefixTuningConfig,
)
from peft import LoraConfig as PeftLoraConfig
from transformers.data import DataCollatorForSeq2Seq

import QEfficient.finetune.configs.dataset_config as datasets
from QEfficient.finetune.configs.peft_config import LoraConfig, PrefixConfig
from QEfficient.finetune.configs.training import TrainConfig
from QEfficient.finetune.data.sampler import DistributedLengthBasedBatchSampler
from QEfficient.finetune.dataset.dataset_config import DATASET_PREPROC


def update_config(config, **kwargs):
    """Update the attributes of a config object based on provided keyword arguments.

    Args:
        config: The configuration object (e.g., TrainConfig, LoraConfig) or a list/tuple of such objects.
        **kwargs: Keyword arguments representing attributes to update.

    Raises:
        ValueError: If an unknown parameter is provided and the config type doesn't support nested updates.
    """
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                config_name, param_name = k.split(".", 1)
                if type(config).__name__.lower() == config_name.lower():
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        raise ValueError(f"Config '{config_name}' does not have parameter: '{param_name}'")
            else:
                config_type = type(config).__name__
                # FIXME (Meet): Once logger is available put this in debug level.
                print(f"[WARNING]: Unknown parameter '{k}' for config type '{config_type}'")


def generate_peft_config(train_config: TrainConfig, peft_config_file: str = None, **kwargs) -> Any:
    """Generate a PEFT-compatible configuration from a custom config based on peft_method.

    Args:
        train_config (TrainConfig): Training configuration with peft_method.
        custom_config: Custom configuration object (e.g., LoraConfig).

    Returns:
        Any: A PEFT-specific configuration object (e.g., PeftLoraConfig).

    Raises:
        RuntimeError: If the peft_method is not supported.
    """
    if peft_config_file:
        peft_config_data = load_config_file(peft_config_file)
        validate_config(peft_config_data, config_type="lora")
        peft_config = PeftLoraConfig(**peft_config_data)
    else:
        config_map = {
            "lora": (LoraConfig, PeftLoraConfig),
            "prefix": (PrefixConfig, PrefixTuningConfig),
            "adaption_prompt": (None, AdaptionPromptConfig),
        }

        if train_config.peft_method not in config_map:
            raise RuntimeError(f"Peft config not found: {train_config.peft_method}")

        config_cls, peft_config_cls = config_map[train_config.peft_method]
        if config_cls is None:
            params = kwargs
        else:
            config = config_cls()
            update_config(config, **kwargs)
            params = asdict(config)

        peft_config = peft_config_cls(**params)
    return peft_config


def generate_dataset_config(dataset_name: str) -> Any:
    """Generate a dataset configuration based on the specified dataset.

    Args:
        dataset_name (str): Name of the dataset to be used for finetuning.

    Returns:
        Any: A dataset configuration object.

    Raises:
        AssertionError: If the dataset name is not recognized.
    """
    supported_datasets = DATASET_PREPROC.keys()
    assert dataset_name in supported_datasets, f"Given dataset '{dataset_name}' is not supported."
    # FIXME (Meet): Replace below logic by creating using auto registry of datasets.
    dataset_config = {k: v for k, v in inspect.getmembers(datasets)}[dataset_name]()
    return dataset_config


def get_dataloader_kwargs(train_config, dataset, dataset_processer, mode):
    kwargs = {}
    batch_size = train_config.batch_size_training if mode == "train" else train_config.val_batch_size
    if train_config.enable_ddp:
        if train_config.enable_sorting_for_ddp:
            if train_config.context_length:
                raise ValueError(
                    "Sorting cannot be done with padding, Please disable sorting or pass context_length as None to disable padding"
                )
            else:
                kwargs["batch_sampler"] = DistributedLengthBasedBatchSampler(
                    dataset,
                    batch_size=batch_size,
                    rank=dist.get_rank(),
                    num_replicas=dist.get_world_size(),
                    shuffle=False,
                )
        else:
            kwargs["sampler"] = data_utils.DistributedSampler(
                dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
            )
            kwargs["batch_size"] = batch_size
            kwargs["drop_last"] = True
    else:
        kwargs["batch_size"] = batch_size
        kwargs["drop_last"] = True
    kwargs["collate_fn"] = DataCollatorForSeq2Seq(dataset_processer)
    return kwargs


def validate_config(config_data: Dict[str, Any], config_type: str = "lora") -> None:
    """Validate the provided YAML/JSON configuration for required fields and types.

    Args:
        config_data (Dict[str, Any]): The configuration dictionary loaded from YAML/JSON.
        config_type (str): Type of config to validate ("lora" for LoraConfig, default: "lora").

    Raises:
        ValueError: If required fields are missing or have incorrect types.
        FileNotFoundError: If the config file path is invalid (handled upstream).

    Notes:
        - Validates required fields for LoraConfig: r, lora_alpha, target_modules.
        - Ensures types match expected values (int, float, list, etc.).
    """
    if config_type.lower() != "lora":
        raise ValueError(f"Unsupported config_type: {config_type}. Only 'lora' is supported.")

    required_fields = {
        "r": int,
        "lora_alpha": int,
        "target_modules": list,
    }
    optional_fields = {
        "bias": str,
        "task_type": str,
        "lora_dropout": float,
        "inference_mode": bool,
    }

    # Check for missing required fields
    missing_fields = [field for field in required_fields if field not in config_data]
    if missing_fields:
        raise ValueError(f"Missing required fields in {config_type} config: {missing_fields}")

    # Validate types of required fields
    for field, expected_type in required_fields.items():
        if not isinstance(config_data[field], expected_type):
            raise ValueError(
                f"Field '{field}' in {config_type} config must be of type {expected_type.__name__}, "
                f"got {type(config_data[field]).__name__}"
            )

    # Validate target_modules contains strings
    if not all(isinstance(mod, str) for mod in config_data["target_modules"]):
        raise ValueError("All elements in 'target_modules' must be strings")

    # Validate types of optional fields if present
    for field, expected_type in optional_fields.items():
        if field in config_data and not isinstance(config_data[field], expected_type):
            raise ValueError(
                f"Field '{field}' in {config_type} config must be of type {expected_type.__name__}, "
                f"got {type(config_data[field]).__name__}"
            )


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load a configuration from a YAML or JSON file.

    Args:
        config_path (str): Path to the YAML or JSON file.

    Returns:
        Dict[str, Any]: The loaded configuration as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            return yaml.safe_load(f)
        elif config_path.endswith(".json"):
            return json.load(f)
        else:
            raise ValueError("Unsupported config file format. Use .yaml, .yml, or .json")
