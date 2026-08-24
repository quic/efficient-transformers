# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""Config loader for the on_qaic disagg_dma parity tests."""

import json
from pathlib import Path

_CONFIG_PATH = Path(__file__).with_name("test_config.json")


def disagg_dma_config(model_key: str, config_id: str | None = None) -> dict:
    with _CONFIG_PATH.open(encoding="utf-8") as handle:
        configs = json.load(handle)

    model_config = configs[model_key]
    model_id = model_config["model_id"]
    test_configs = model_config["test_configs"]

    if config_id is None:
        test_config = test_configs[0]
    else:
        matches = [tc for tc in test_configs if tc["id"] == config_id]
        if not matches:
            raise KeyError(f"No test_config with id '{config_id}' for model_key '{model_key}'")
        test_config = matches[0]

    return {"model_id": model_id, **test_config}
