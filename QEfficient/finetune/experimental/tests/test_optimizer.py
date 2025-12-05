# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy

import pytest
import torch.nn as nn
import torch.optim as optim

from QEfficient.finetune.experimental.core.component_registry import registry
from QEfficient.finetune.experimental.core.optimizer import prepare_optimizer

OPTIMIZER_CONFIGS = {
    "Adam": {
        "optimizer_name": "Adam",
        "opt_cls": optim.Adam,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False,
    },
    "AdamW": {
        "optimizer_name": "AdamW",
        "opt_cls": optim.AdamW,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False,
    },
    "SGD": {
        "optimizer_name": "SGD",
        "opt_cls": optim.SGD,
        "lr": 1e-4,
        "momentum": 0.9,
        "weight_decay": 0.01,
        "dampening": 0.0,
        "nesterov": False,
    },
    "RMSprop": {
        "optimizer_name": "RMSprop",
        "opt_cls": optim.RMSprop,
    },
}

REGISTRY_CONFIG = {
    "RMSprop": {
        "optimizer_name": "RMSprop",
        "opt_cls": optim.RMSprop,
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
    """Test that all registered optimizers can be created with their configs."""
    config = copy.deepcopy(OPTIMIZER_CONFIGS[opt_name])

    config.pop("opt_cls")
    try:
        optimizer_class_and_kwargs = prepare_optimizer(config)
        assert optimizer_class_and_kwargs is not None
    except ValueError as e:
        assert "Unknown optimizer" in str(e)
        return
    optimizer_class = optimizer_class_and_kwargs[0]
    opt_inst = optimizer_class(dummy_model.parameters(), **optimizer_class_and_kwargs[1])
    assert isinstance(opt_inst, optim.Optimizer)
    assert len(list(opt_inst.param_groups)) == 1

    for key in ["lr", "weight_decay", "betas", "eps", "momentum", "dampening", "nesterov", "amsgrad"]:
        if key in config:
            assert opt_inst.param_groups[0][key] == config[key], f"{key} mismatch"


@pytest.mark.parametrize("opt_name, opt_cls", REGISTRY_CONFIG.items())
def test_registered_optimizer(opt_name, opt_cls):
    """Test that the optimizer registerd correctly."""
    registry.optimizer(opt_name)(opt_cls)
    optimizer_class = registry.get_optimizer(opt_name)
    assert optimizer_class is not None
    assert optimizer_class == opt_cls
