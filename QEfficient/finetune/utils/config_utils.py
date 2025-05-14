# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import inspect
from dataclasses import asdict

import yaml
from peft import LoraConfig as PeftLoraConfig

import QEfficient.finetune.configs.dataset_config as datasets
from QEfficient.finetune.configs.peft_config import LoraConfig
from QEfficient.finetune.configs.training import TrainConfig
from QEfficient.finetune.dataset.dataset_config import DATASET_PREPROC


def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                # allow --some_config.some_param=True
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        # In case of specialized config we can warn user
                        assert False, f"Warning: {config_name} does not accept parameter: {k}"
            elif isinstance(config, train_config):
                assert False, f"Warning: unknown parameter {k}"


def generate_peft_config(train_config, kwargs):
    configs = (lora_config, prefix_config)
    peft_configs = (LoraConfig, AdaptionPromptConfig, PrefixTuningConfig)
    names = tuple(c.__name__.rstrip("_config") for c in configs)

    if train_config.peft_method not in names:
        raise RuntimeError(f"Peft config not found: {train_config.peft_method}")

    config = configs[names.index(train_config.peft_method)]()

    Raises:
        RuntimeError: If the peft_method is not supported.
    """
    if peft_config_file:
        peft_config_data = load_config_file(peft_config_file)
        validate_config(peft_config_data, config_type="lora")
        peft_config = PeftLoraConfig(**peft_config_data)
    else:
        config_map = {"lora": (LoraConfig, PeftLoraConfig)}
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


def generate_dataset_config(train_config, kwargs):
    names = tuple(DATASET_PREPROC.keys())
    assert train_config.dataset in names, f"Unknown dataset: {train_config.dataset}"
    dataset_config = {k: v for k, v in inspect.getmembers(datasets)}[train_config.dataset]()
    update_config(dataset_config, **kwargs)
    return dataset_config


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
