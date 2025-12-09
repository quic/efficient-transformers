# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


from pathlib import Path

import pytest

from QEfficient.finetune.experimental.core.config_manager import ConfigManager, parse_arguments


@pytest.fixture
def config_path() -> Path:
    here = Path(__file__).resolve().parent
    return (here / "test_config.yaml").resolve()


def test_config(config_path):
    # parse the yaml file
    master_config = parse_arguments(config_path)
    config_manager = ConfigManager(master_config)
    # Test that the config manager is initialized correctly
    assert isinstance(config_manager, ConfigManager)

    # Test that all required fields are present
    missing = [
        a
        for a in ("model", "dataset", "optimizers", "scheduler", "callbacks", "training")
        if not hasattr(config_manager, a)
    ]
    assert not missing, f"Missing attributes: {missing}"
    trainer_config = config_manager.get_training_config()
    assert (hasattr(trainer_config, attr) for attr in ("output_dir", "train_batch_size", "num_epochs"))
    dataset_config = config_manager.get_dataset_config()
    assert (hasattr(dataset_config, attr) for attr in ("dataset_type", "dataset_name", "tokenizer_name"))
    model_config = config_manager.get_model_config()
    assert (hasattr(model_config, attr) for attr in ("model_type", "model_name", "use_peft"))
    scheduler_config = config_manager.get_scheduler_config()
    assert (hasattr(scheduler_config, attr) for attr in ("scheduler_name"))
    callback_config = config_manager.get_callback_config()
    assert (hasattr(callback_config, attr) for attr in ("earlystopping"))
    optimizer_config = config_manager.get_optimizer_config()
    assert (hasattr(optimizer_config, attr) for attr in ("optimizer_name", "lr"))
