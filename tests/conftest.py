# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import shutil

from transformers import logging

from QEfficient.utils.cache import QEFF_HOME


def qeff_models_clean_up():
    qeff_dir = QEFF_HOME
    if os.path.exists(qeff_dir):
        shutil.rmtree(qeff_dir)
        print(f"\n.............Cleaned up {qeff_dir}")


def pytest_sessionstart(session):
    print("\n############################### Pytest Session Starting ###############################\n")

    # Suppress transformers warnings about unused weights when loading models with fewer layers
    logging.set_verbosity_error()

    qeff_models_clean_up()


def pytest_configure(config):
    """Register custom markers for test categorization."""
    config.addinivalue_line("markers", "llm_model: mark test as a pure LLM model inference test")
    config.addinivalue_line(
        "markers", "feature: mark test as a feature-specific test (SPD, sampler, prefix caching, LoRA, etc.)"
    )


def pytest_runtest_teardown(item, nextitem):
    """Clean up after each test case."""
    qeff_models_clean_up()


def pytest_sessionfinish(session, exitstatus):
    inside_worker = getattr(session.config, "workerinput", None)
    if inside_worker is None:
        qeff_models_clean_up()
        print("\n############################### Pytest Session Ended ###############################\n")
