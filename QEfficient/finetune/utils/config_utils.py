# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import inspect
import json
import os
from dataclasses import asdict
from typing import Any, Dict, Optional
from collections import namedtuple

import yaml
from peft import LoraConfig as PeftLoraConfig

import QEfficient.finetune.configs.dataset_config as qeff_datasets
from QEfficient.finetune.configs.peft_config import LoraConfig
from QEfficient.finetune.configs.training import TrainConfig
from QEfficient.finetune.dataset.dataset_config import DATASET_PREPROC
from QEfficient.finetune.utils.helper import Peft_Method
from QEfficient.finetune.utils.logging_utils import logger


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
                        logger.raise_error(
                            f"Config '{config_name}' does not have parameter: '{param_name}'", ValueError
                        )
            else:
                config_type = type(config).__name__
                logger.debug(f"Unknown parameter '{k}' for config type '{config_type}'")


def generate_peft_config(train_config: TrainConfig, **kwargs) -> Any:
    """Generate a PEFT-compatible configuration from a custom config based on peft_method.

    Args:
        train_config (TrainConfig): Training configuration with peft_method.

    Returns:
        Any: A PEFT-specific configuration object (e.g., PeftLoraConfig).

    Raises:
        RuntimeError: If the peft_method is not supported.
    """
    if train_config.peft_config_file:
        peft_config_data = load_config_file(train_config.peft_config_file)
        validate_config(peft_config_data, config_type=Peft_Method.LORA)
        peft_config = PeftLoraConfig(**peft_config_data)
    else:
        config_map = {Peft_Method.LORA: (LoraConfig, PeftLoraConfig)}
        if train_config.peft_method not in config_map:
            logger.raise_error(f"Peft config not found: {train_config.peft_method}", RuntimeError)

        config_cls, peft_config_cls = config_map[train_config.peft_method]
        if config_cls is None:
            params = kwargs
        else:
            config = config_cls()
            update_config(config, **kwargs)
            params = asdict(config)

        peft_config = peft_config_cls(**params)
    return peft_config


def generate_dataset_config(dataset_name: str, custom_dataset_config: Optional[str] = None) -> Any:
    """Generate a dataset configuration based on the specified dataset.

    Args:
        dataset_name (str): Name of the dataset to be used for finetuning.
        custom_dataset_config (str): Dataset config json file for custom datset.
            This file contains dataset specific arguments to be used in dataset
            preprocessing step.

    Returns:
        Any: A dataset configuration object.

    Raises:
        AssertionError: If the dataset name is not recognized.
    """
    supported_datasets = DATASET_PREPROC.keys()
    assert dataset_name in supported_datasets, f"Given dataset '{dataset_name}' is not supported."
    # FIXME (Meet): Replace below logic by creating using auto registry of datasets.
    dataset_config = {k: v for k, v in inspect.getmembers(qeff_datasets)}[dataset_name]()
    if dataset_name == "custom_dataset":
        custom_dataset_dict = asdict(dataset_config)
        custom_dataset_dict_override = load_config_file(custom_dataset_config)
        # Override existing and add new params to dataset_config.
        custom_dataset_dict.update(custom_dataset_dict_override)

        custom_dataset_class = namedtuple("custom_dataset", custom_dataset_dict.keys())
        dataset_config = custom_dataset_class(**custom_dataset_dict)
    return dataset_config


def validate_config(config_data: Dict[str, Any], config_type: str = Peft_Method.LORA) -> None:
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
    if config_type.lower() != Peft_Method.LORA:
        logger.raise_error(f"Unsupported config_type: {config_type}. Only 'lora' is supported.", ValueError)

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
        logger.raise_error(f"Missing required fields in {config_type} config: {missing_fields}", ValueError)

    # Validate types of required fields
    for field, expected_type in required_fields.items():
        if not isinstance(config_data[field], expected_type):
            logger.raise_error(
                f"Field '{field}' in {config_type} config must be of type {expected_type.__name__}, "
                f"got {type(config_data[field]).__name__}",
                ValueError,
            )

    # Validate target_modules contains strings
    if not all(isinstance(mod, str) for mod in config_data["target_modules"]):
        logger.raise_error("All elements in 'target_modules' must be strings", ValueError)

    # Validate types of optional fields if present
    for field, expected_type in optional_fields.items():
        if field in config_data and not isinstance(config_data[field], expected_type):
            logger.raise_error(
                f"Field '{field}' in {config_type} config must be of type {expected_type.__name__}, "
                f"got {type(config_data[field]).__name__}",
                ValueError,
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
        logger.raise_error(f"Config file not found: {config_path}", FileNotFoundError)

    with open(config_path, "r") as f:
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            return yaml.safe_load(f)
        elif config_path.endswith(".json"):
            return json.load(f)
        else:
            logger.raise_error("Unsupported config file format. Use .yaml, .yml, or .json", ValueError)
