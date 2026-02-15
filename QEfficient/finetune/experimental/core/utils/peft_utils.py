# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Utility functions for PEFT (Parameter-Efficient Fine-Tuning) configuration.
"""

from dataclasses import asdict
from typing import Any, Optional

from peft import LoraConfig


def convert_peft_config_to_lora_config(peft_config: Any) -> Optional[LoraConfig]:
    """
    Convert PeftConfig (dataclass or dict) to LoraConfig from peft library.

    Args:
        peft_config: PeftConfig dataclass instance or dict

    Returns:
        LoraConfig instance or None if PEFT is not enabled
    """
    if peft_config is None:
        return None

    # Convert dataclass to dictionary if needed
    if hasattr(peft_config, "__dict__") and not isinstance(peft_config, dict):
        peft_dict = asdict(peft_config)
    else:
        peft_dict = peft_config

    # Map PeftConfig fields to LoraConfig fields
    lora_config_dict = {
        "r": peft_dict.get("lora_r"),
        "lora_alpha": peft_dict.get("lora_alpha"),
        "lora_dropout": peft_dict.get("lora_dropout"),
        "target_modules": peft_dict.get("target_modules"),
        "bias": peft_dict.get("bias"),
        "task_type": peft_dict.get("task_type"),
    }

    return LoraConfig(**lora_config_dict)
