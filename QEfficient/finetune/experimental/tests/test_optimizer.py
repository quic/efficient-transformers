# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import sys
from pathlib import Path

import pytest
import torch.optim as optim

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.finetune.experimental.core.component_registry import ComponentFactory

sys.path.insert(0, str(Path(__file__).parent.parent))
OPTIMIZER_CONFIGS = {
    "adam": {
        "name": "adam",
        "opt_cls": optim.Adam,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False,
    },
    "adamw": {
        "name": "adamw",
        "opt_cls": optim.AdamW,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False,
    },
    "sgd": {
        "name": "sgd",
        "opt_cls": optim.SGD,
        "lr": 1e-4,
        "momentum": 0.9,
        "weight_decay": 0.01,
        "dampening": 0.0,
        "nesterov": False,
    },
}


@pytest.fixture
def ref_model():
    return QEFFAutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")


@pytest.mark.parametrize("opt_name", OPTIMIZER_CONFIGS.keys())
def test_optimizers(opt_name, ref_model):
    """Test that all optimizers can be created with their configs."""
    # Create optimizer using the factory
    config = OPTIMIZER_CONFIGS[opt_name]
    opt_inst = ComponentFactory.create_optimizer(**config, model_params=ref_model.model.parameters())
    assert opt_inst is not None
    assert isinstance(opt_inst, optim.Optimizer)
    assert len(list(opt_inst.param_groups)) == 1
