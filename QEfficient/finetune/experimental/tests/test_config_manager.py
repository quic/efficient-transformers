# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
from pathlib import Path

import pytest

from QEfficient.finetune.experimental.core.config_manager import ConfigManager


@pytest.fixture
def config_path() -> Path:
    here = Path(__file__).resolve().parent
    return (here / "test_config.yaml").resolve()


def test_default_config():
    config_manager = ConfigManager()
    assert config_manager is not None
    assert config_manager.config is not None


def test_config(config_path):
    config_manager = ConfigManager(config_path=config_path)
    assert isinstance(config_manager, ConfigManager)

    # Test that all required fields are present
    missing = [
        a
        for a in ("model", "dataset", "optimizers", "scheduler", "callbacks", "training")
        if not hasattr(config_manager, a)
    ]
    assert not missing, f"Missing attributes: {missing}"
    trainer_config = config_manager.get_training_config()
    assert trainer_config is not None
    assert isinstance(trainer_config, dict)
    assert (hasattr(trainer_config, attr) for attr in ("output_dir", "train_batch_size", "num_epochs", "ddp_config"))
    dataset_config = config_manager.get_dataset_config()
    assert dataset_config is not None
    assert isinstance(dataset_config, dict)
    assert (hasattr(dataset_config, attr) for attr in ("dataset_type", "dataset_name", "tokenizer_name"))
    model_config = config_manager.get_model_config()
    assert model_config is not None
    assert isinstance(model_config, dict)
    assert (hasattr(model_config, attr) for attr in ("model_type", "model_name", "use_peft", "peft_config"))
    scheduler_config = config_manager.get_scheduler_config()
    assert scheduler_config is not None
    assert isinstance(scheduler_config, dict)
    assert (hasattr(scheduler_config, attr) for attr in ("scheduler_name"))
    callback_config = config_manager.get_callback_config()
    assert callback_config is not None
    assert isinstance(callback_config, dict)
    assert (hasattr(callback_config, attr) for attr in ("earlystopping"))
    optimizer_config = config_manager.get_optimizer_config()
    assert optimizer_config is not None
    assert isinstance(optimizer_config, dict)
    assert (hasattr(optimizer_config, attr) for attr in ("optimizer_name", "lr"))
