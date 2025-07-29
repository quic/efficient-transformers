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

external_models = {"hpcai-tech/grok-1"}


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
        config_dict[config["model_name"]] = AutoConfig.from_pretrained(
            config["model_name"],
            trust_remote_code=config["model_name"] in external_models,
            **config.get("additional_params", {}),
        )
    return config_dict


@pytest.fixture(scope="session")
def custom_causal_model_config_dict():
    with open("tests/transformers/models/custom_tiny_model_configs.json", "r") as f:
        custom_model_configs_data = json.load(f)
    return get_custom_model_config_dict(custom_model_configs_data)


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
