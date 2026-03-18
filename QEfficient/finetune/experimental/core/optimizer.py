# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Optimizer components for the training system.
"""

import torch.optim as optim

from QEfficient.finetune.experimental.core.component_registry import registry

registry.optimizer("Adam")(optim.Adam)
registry.optimizer("AdamW")(optim.AdamW)
registry.optimizer("SGD")(optim.SGD)


def prepare_optimizer(opt_config):
    """
    Create optimizer from config.
    Args: opt_config: Dictionary containing optimizer configuration.
    Returns: Tuple of optimizer class and its arguments.
    """
    opt_name = opt_config.pop("optimizer_name")
    opt_cls = registry.get_optimizer(opt_name)
    opt_config["lr"] = float(opt_config["lr"])
    optimizer_cls_and_kwargs = (opt_cls, opt_config)
    return optimizer_cls_and_kwargs
