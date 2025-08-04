# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import hashlib
import json
from typing import Dict

from QEfficient.utils.constants import HASH_HEXDIGEST_STR_LEN


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


def hash_dict_params(dict_items: Dict, hash_string_size: int = HASH_HEXDIGEST_STR_LEN):
    """
    Takes a dictionary of items and returns a SHA256 hash object
    """
    mhash = hashlib.sha256(to_hashable(dict_items))
    return mhash.hexdigest()[:hash_string_size]
