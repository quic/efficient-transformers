# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
from trl import SFTConfig, SFTTrainer

from QEfficient.finetune.experimental.core.component_registry import registry
from QEfficient.finetune.experimental.core.config_manager import PeftConfig

# def get_default_peft_config():
#     """Get default peft config from config_manager and convert to peft library config."""
#     peft_config_params = PeftConfig()
#     return peft_config_params.to_peft_config()


@registry.trainer_module(name="sft", args_cls=SFTConfig, required_kwargs={"peft_config": PeftConfig})
class SFTTrainerModule(SFTTrainer):
    pass  # Just using the standard SFTTrainer
