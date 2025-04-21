# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import List


# Currently, the support is for Lora Configs only
# In future, we can expand to llama_adapters and prefix tuning
# TODO: vbaddi: Check back once FSDP is enabled
@dataclass
class lora_config:
    r: int = 8
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05
    inference_mode: bool = False  # should be False for finetuning


# CAUTION prefix tuning is currently not supported
@dataclass
class prefix_config:
    num_virtual_tokens: int = 30
    task_type: str = "CAUSAL_LM"
