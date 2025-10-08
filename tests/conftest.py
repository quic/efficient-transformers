# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
import shutil

import pytest
from transformers import AutoConfig

from QEfficient.utils.constants import QEFF_MODELS_DIR
from QEfficient.utils.logging_utils import logger
from QEfficient.utils.test_utils import ModelConfig


def get_custom_model_config_dict(configs):
    """
    Converts a list of custom model configuration dictionaries into a dictionary
    mapping model names to their corresponding AutoConfig objects.

    Args:
        configs (List[Dict]): A list of dictionaries, each containing model configuration parameters.

    Returns:
        Dict[str, AutoConfig]: A dictionary where keys are model names and values are AutoConfig objects.
    """
    config_dict = {}
    for config in configs:
        model_name = config["model_name"]
        config_dict[model_name] = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=config["model_name"] in ModelConfig.EXTERNAL_MODELS,
            **config.get("additional_params", {}),
        )
    return config_dict


# Pytest fixture to load custom model configs from a JSON file
@pytest.fixture(scope="session")
def custom_causal_model_config_dict():
    with open("tests/transformers/models/custom_tiny_model_configs.json", "r") as f:
        custom_model_configs_data = json.load(f)
    return get_custom_model_config_dict(custom_model_configs_data)


@pytest.fixture(scope="session")
def worker_device_id(request):
    """
    Assigns a unique device ID to each pytest-xdist worker.
    Supports devices 0-5, cycles through them if more workers exist.
    
    Worker 'master' or no worker gets device 0
    Worker 'gw0' gets device 0
    Worker 'gw1' gets device 1
    ...
    Worker 'gw6' gets device 0 (cycles back if needed)
    """
    worker_id = getattr(request.config, 'workerinput', {}).get('workerid', 'master')
    
    # Total available devices
    NUM_DEVICES = 6
    
    if worker_id == 'master':
        device_id = 0
    else:
        # Extract number from 'gw0', 'gw1', etc. and use modulo for cycling
        worker_num = int(worker_id.replace('gw', ''))
        device_id = worker_num % NUM_DEVICES
    
    logger.info(f"Worker {worker_id} assigned to device {device_id}")
    return [device_id]


def qeff_models_clean_up():
    if os.path.exists(QEFF_MODELS_DIR):
        shutil.rmtree(QEFF_MODELS_DIR)
        logger.info(f"\n.............Cleaned up {QEFF_MODELS_DIR}")


def pytest_sessionstart(session):
    logger.info("PYTEST Session Starting ...")
    qeff_models_clean_up()


def pytest_sessionfinish(session, exitstatus):
    inside_worker = getattr(session.config, "workerinput", None)
    if inside_worker is None:
        qeff_models_clean_up()
        logger.info("...PYTEST Session Ended.")
