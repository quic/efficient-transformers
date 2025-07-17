# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import shutil

from QEfficient.utils.logging_utils import logger


def qeff_models_clean_up():
    qeff_cache_home = os.environ.get("QEFF_HOME")
    if os.path.exists(qeff_cache_home):
        shutil.rmtree(qeff_cache_home)
        logger.info(f"\n.............Cleaned up {qeff_cache_home}")


def pytest_sessionstart(session):
    logger.info("PYTEST Session Starting ...")
    qeff_models_clean_up()


def pytest_sessionfinish(session, exitstatus):
    inside_worker = getattr(session.config, "workerinput", None)
    if inside_worker is None:
        qeff_models_clean_up()
        logger.info("...PYTEST Session Ended.")
