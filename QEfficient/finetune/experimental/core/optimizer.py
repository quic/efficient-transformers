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

registry.optimizer("Adam")(optim.Adam)
registry.optimizer("AdamW")(optim.AdamW)
registry.optimizer("SGD")(optim.SGD)


def get_optimizer_cls(optimizer_name: str) -> Type[Optimizer]:
    """
    Get optimizer class from registry.
    Args: optimizer_name: Name of the optimizer to retrieve.
    Returns: Optimizer class.
    Raises: ValueError: If optimizer name is not found in registry.
    """
    optimizer_cls = registry.get_optimizer(optimizer_name)
    if optimizer_cls is None:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    return optimizer_cls


def get_optimizer(opt_config):
    """
    Create optimizer from config.
    Args: opt_config: Dictionary containing optimizer configuration.
    Returns: Tuple of optimizer class and its arguments.
    """
    opt_name = opt_config.pop("optimizer_name")
    opt_cls = get_optimizer_cls(opt_name)
    opt_config["lr"] = float(opt_config["lr"])
    optimizer_cls_and_kwargs = (opt_cls, opt_config)
    return optimizer_cls_and_kwargs
