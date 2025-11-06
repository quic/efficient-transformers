# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import os
from typing import Optional

from QEfficient.utils._utils import load_json


def config_manager(cls, config_source: Optional[str] = None):
    """
    JSON-based compilation configuration manager for diffusion pipelines.

    Supports loading configuration from JSON files only. Automatically detects
    model type and handles model-specific requirements.
    Initialize the configuration manager.

    Args:
        config_source: Path to JSON configuration file. If None, uses default config.
    """
    if config_source is None:
        config_source = cls.get_default_config_path()

    if not isinstance(config_source, str):
        raise ValueError("config_source must be a path to JSON configuration file")

    # Direct use of load_json utility - no wrapper needed
    if not os.path.exists(config_source):
        raise FileNotFoundError(f"Configuration file not found: {config_source}")

    cls.custom_config = load_json(config_source)


def set_module_device_ids(cls):
    """
    Set device IDs for each module based on the custom configuration.

    Iterates through all modules in the pipeline and assigns device IDs
    from the configuration file to each module's device_ids attribute.
    """
    config_modules = cls.custom_config["modules"]
    for module_name, module_obj in cls.has_module:
        module_obj.device_ids = config_modules[module_name]["execute"]["device_ids"]
