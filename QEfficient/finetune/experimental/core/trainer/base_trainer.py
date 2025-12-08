# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
from transformers import Trainer, TrainingArguments

from QEfficient.finetune.experimental.core.component_registry import registry


@registry.trainer_module(name="base", args_cls=TrainingArguments, required_kwargs={})
class BaseTrainer(Trainer):
    pass  # Just using the standard Trainer
