# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Optimizer components for the training system.
"""

from typing import Type

import torch.optim as optim
from torch.optim import Optimizer

from QEfficient.finetune.experimental.core.component_registry import registry


def register_optimizer(optimizer_name: str, cls: Type[Optimizer]) -> None:
    """Register a new optimizer class."""
    registry.optimizer(optimizer_name)(cls)


register_optimizer("adam", optim.Adam)
register_optimizer("adamw", optim.AdamW)
register_optimizer("sgd", optim.SGD)


def get_optimizer_cls(optimizer_name: str) -> Type[Optimizer]:
    optimizer_cls = registry.get_optimizer(optimizer_name)
    if optimizer_cls is None:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    return optimizer_cls
