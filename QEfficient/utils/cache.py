# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import json
import os
from pathlib import Path

QEFF_HOME: Path = None
if "QEFF_HOME" in os.environ:
    QEFF_HOME = Path(os.environ["QEFF_HOME"])
elif "XDG_CACHE_HOME" in os.environ:
    QEFF_HOME = Path(os.environ["XDG_CACHE_HOME"]) / "qeff_models"
else:
    QEFF_HOME = Path("~/.cache/qeff_models").expanduser()


def json_serializable(obj):
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def to_hashable(obj) -> bytes:
    """
    Converts obj to bytes such that same object will result in same hash
    """
    return json.dumps(
        obj,
        skipkeys=False,
        ensure_ascii=True,
        check_circular=True,
        allow_nan=False,
        indent=None,
        separators=(",", ":"),
        default=json_serializable,
        sort_keys=True,
    ).encode()
