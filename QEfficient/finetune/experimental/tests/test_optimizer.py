# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import inspect
import sys
from pathlib import Path

import pytest
import torch.nn as nn
import torch.optim as optim

from QEfficient.finetune.experimental.core.optimizer import get_optimizer_cls, register_optimizer

sys.path.insert(0, str(Path(__file__).parent.parent))
OPTIMIZER_CONFIGS = {
    "adam": {
        "optimizer_name": "adam",
        "opt_cls": optim.Adam,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False,
    },
    "adamw": {
        "optimizer_name": "adamw",
        "opt_cls": optim.AdamW,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False,
    },
    "sgd": {
        "optimizer_name": "sgd",
        "opt_cls": optim.SGD,
        "lr": 1e-4,
        "momentum": 0.9,
        "weight_decay": 0.01,
        "dampening": 0.0,
        "nesterov": False,
    },
}


@pytest.fixture
def dummy_model():
    return nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
    )


@pytest.mark.parametrize("opt_name", OPTIMIZER_CONFIGS.keys())
def test_optimizers(opt_name, dummy_model):
    """Test that all optimizers can be created with their configs."""
    # Register optimizer class
    config = OPTIMIZER_CONFIGS[opt_name]
    register_optimizer(config["optimizer_name"], config["opt_cls"])
    optimizer_class = get_optimizer_cls(config["optimizer_name"])
    assert optimizer_class is not None
    assert optimizer_class == config["opt_cls"]
    valid_params = inspect.signature(optimizer_class).parameters
    filtered_config = {k: v for k, v in config.items() if k in valid_params}
    opt_inst = optimizer_class(dummy_model.parameters(), **filtered_config)
    assert isinstance(opt_inst, optim.Optimizer)
    assert len(list(opt_inst.param_groups)) == 1
    assert opt_inst.param_groups[0]["lr"] == config["lr"]
    if "weight_decay" in config:
        assert opt_inst.param_groups[0]["weight_decay"] == config["weight_decay"]
    if "betas" in config:
        assert opt_inst.param_groups[0]["betas"] == config["betas"]
    if "eps" in config:
        assert opt_inst.param_groups[0]["eps"] == config["eps"]
    if "momentum" in config:
        assert opt_inst.param_groups[0]["momentum"] == config["momentum"]
    if "dampening" in config:
        assert opt_inst.param_groups[0]["dampening"] == config["dampening"]
    if "nesterov" in config:
        assert opt_inst.param_groups[0]["nesterov"] == config["nesterov"]
    if "amsgrad" in config:
        assert opt_inst.param_groups[0]["amsgrad"] == config["amsgrad"]
