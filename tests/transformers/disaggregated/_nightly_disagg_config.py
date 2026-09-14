# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""Config loader for nightly disaggregated HF/ORT/QAIC parity tests."""

import json
from pathlib import Path

import pytest

_CONFIG_PATH = Path(__file__).with_name("nightly_disagg_configs.json")


def nightly_disagg_configs(model_key: str) -> list:
    with _CONFIG_PATH.open(encoding="utf-8") as handle:
        configs = json.load(handle)

    model_config = configs[model_key]
    model_id = model_config["model_id"]
    return [
        pytest.param({"model_id": model_id, **test_config}, id=test_config["id"])
        for test_config in model_config["test_configs"]
    ]
