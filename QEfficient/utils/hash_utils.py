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
        # Convert set to a sorted list of strings for consistent hashing
        return sorted([cls.__name__ if isinstance(cls, type) else str(cls) for cls in obj])
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def to_hashable(obj) -> bytes:
    """
    Converts obj to bytes such that same object will result in same hash
    """
    try:
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
    except TypeError as e:
        raise TypeError(f"Unable to convert object: {obj} to bytes") from e


def hash_dict_params(dict_items: Dict, hash_string_size: int = HASH_HEXDIGEST_STR_LEN):
    """
    Takes a dictionary of items and returns a SHA256 hash object
    """
    mhash = hashlib.sha256(to_hashable(dict_items))
    return mhash.hexdigest()[:hash_string_size]


def create_export_hash(**kwargs):
    """
    This Method prepares all the model params required to create the hash for export directory.
    """
    export_hash_params = kwargs.get("model_params")

    export_params = {}
    export_params["output_names"] = kwargs.get("output_names")
    export_params["dynamic_axes"] = kwargs.get("dynamic_axes")
    if kwargs.get("use_onnx_subfunctions"):
        export_params["use_onnx_subfunctions"] = True
    export_hash_params["export_params"] = export_params

    export_kwargs = kwargs.get("export_kwargs")
    if export_kwargs:
        export_hash_params.update(export_kwargs)

    onnx_transform_kwargs = kwargs.get("onnx_transform_kwargs")
    if onnx_transform_kwargs:
        export_hash_params.update(onnx_transform_kwargs)
    if export_hash_params.get("peft_config") is not None and not isinstance(export_hash_params["peft_config"], dict):
        export_hash_params["peft_config"] = export_hash_params["peft_config"].to_dict()

    return hash_dict_params(export_hash_params), export_hash_params
