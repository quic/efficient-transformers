# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
QEFF_DIR = os.path.dirname(UTILS_DIR)
ROOT_DIR = os.path.dirname(QEFF_DIR)


# Store the qeff_models inside the ~/.cache directory or over-ride with an env variable.
def get_models_dir():
    """
    Determine the directory for storing QEFF models.
    Priority:
    1. Use $XDG_CACHE_HOME/qeff_models if XDG_CACHE_HOME is set.
    2. Use QEFF_HOME if set in environment.
    3. Default to ~/.cache/qeff_models.
    Sets QEFF_MODELS_DIR environment variable if not already set.
    Returns:
        str: Path to the QEFF models directory.
    """
    qeff_cache_home = os.environ.get("QEFF_HOME")
    # Check if XDG_CACHE_HOME is set
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if qeff_cache_home:
        qeff_models_dir = os.path.join(qeff_cache_home, "qeff_models")
    # Check if QEFF_MODELS_DIR is set
    elif xdg_cache_home:
        qeff_models_dir = os.path.join(xdg_cache_home, "qeff_models")
    else:
        # Use ~/.cache/qeff_models as the default
        qeff_models_dir = os.path.join(os.path.expanduser("~"), ".cache", "qeff_models")

    # Set QEFF_MODELS_DIR environment variable
    return qeff_models_dir


QEFF_MODELS_DIR = get_models_dir()


class Constants:
    # Export Constants.
    SEQ_LEN = 32
    CTX_LEN = 32
    PROMPT_LEN = 8
    INPUT_STR = ["My name is"]

    CACHE_DIR = os.path.join(ROOT_DIR, "cache_dir")

    GB = 2**30
    MAX_QPC_LIMIT = 30
