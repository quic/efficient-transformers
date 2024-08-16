# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import json
import os

QEFF_HOME = None
if "QEFF_HOME" in os.environ:
    QEFF_HOME = os.environ["QEFF_HOME"]
elif "XDG_CACHE_HOME" in os.environ:
    QEFF_HOME = os.path.join(os.environ["XDG_CACHE_HOME"], "qeff_models")
else:
    QEFF_HOME = os.path.expanduser("~/.cache/qeff_models")


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
