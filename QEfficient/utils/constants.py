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
BASE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "qeff_models")
QEFF_MODELS_DIR = os.path.join(os.environ.get("QEFF_MODELS_DIR", BASE_DIR), "qeff_models")


class Constants:
    # Export Constants.
    seq_length = 32
    input_str = "My Name is"

    CTX_LEN = 32
    PROMPT_LEN = 8
    INPUT_STRING = ["My name is"]

    CACHE_DIR = os.path.join(ROOT_DIR, "cache_dir")

    GB = 2**30
    MAX_QPC_LIMIT = 30
