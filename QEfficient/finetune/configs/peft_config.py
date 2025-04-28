# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import List


@dataclass
class LoraConfig:
    """LoRA-specific configuration for parameter-efficient fine-tuning.

    Attributes:
        r (int): LoRA rank (default: 8).
        lora_alpha (int): LoRA scaling factor (default: 32).
        target_modules (List[str]): Modules to apply LoRA to (default: ["q_proj", "v_proj"]).
        bias (str): Bias handling in LoRA (default: "none").
        task_type (str): Task type for LoRA (default: "CAUSAL_LM").
        lora_dropout (float): Dropout rate for LoRA (default: 0.0).
        inference_mode (bool): Whether model is in inference mode (default: False).
    """

    r: int = 8
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05
    inference_mode: bool = False  # should be False for finetuning


# CAUTION prefix tuning is currently not supported
@dataclass
class PrefixConfig:
    num_virtual_tokens: int = 30
    task_type: str = "CAUSAL_LM"
