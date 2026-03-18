# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import pytest
from transformers import TrainerCallback

from QEfficient.finetune.experimental.core.callbacks import create_callbacks
from QEfficient.finetune.experimental.core.component_registry import registry


class ModelSummaryCallback(TrainerCallback):
    def __init__(self):
        pass


# Setup test data
CALLBACK_CONFIGS = {
    "early_stopping": {
        "name": "early_stopping",
        "early_stopping_patience": 3,
        "early_stopping_threshold": 0.001,
    },
    "tensorboard": {"name": "tensorboard", "tb_writer": "SummaryWriter"},
    "model_summary": {
        "name": "model_summary",
        "max_depth": 1,
    },
}

REGISTRY_CALLBACK_CONFIGS = {
    "model_summary": {
        "name": "model_summary",
        "max_depth": 1,
        "callback_class": ModelSummaryCallback,
    },
}


@pytest.mark.parametrize("callback_name", CALLBACK_CONFIGS.keys())
def test_callbacks(callback_name):
    """Test that registered callbacks that can be created with their configs."""
    # Create callbacks using the factory
    config = CALLBACK_CONFIGS[callback_name]
    try:
        callback_inst = create_callbacks(**config)
    except ValueError as e:
        assert "Unknown callback" in str(e)
        return
    assert callback_inst is not None
    assert isinstance(callback_inst, TrainerCallback)


@pytest.mark.parametrize("callback_name,callback_class", REGISTRY_CALLBACK_CONFIGS.items())
def test_callbacks_registery(callback_name, callback_class):
    """Test that a callback registered correctly."""
    registry.callback(callback_name)(callback_class)
    callback = registry.get_callback(callback_name)
    assert callback is not None
    assert callback == callback_class
