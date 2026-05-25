# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Iterable, Optional

from QEfficient.utils.constants import ATTENTION_HEAD_CONFIG_KEYS, HIDDEN_SIZE_CONFIG_KEYS, KV_HEAD_CONFIG_KEYS


def get_first_config_value(config, names: Iterable[str], default=None, cast_int: bool = False):
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value) if cast_int else value
    return default


def resolve_attention_heads(config) -> Optional[int]:
    return get_first_config_value(config, ATTENTION_HEAD_CONFIG_KEYS, cast_int=True)


def resolve_kv_heads(config) -> Optional[int]:
    value = get_first_config_value(config, KV_HEAD_CONFIG_KEYS, cast_int=True)
    if value is None:
        value = resolve_attention_heads(config)
    return value


def resolve_hidden_size(config) -> Optional[int]:
    return get_first_config_value(config, HIDDEN_SIZE_CONFIG_KEYS, cast_int=True)


def set_kv_head_aliases(config, value: int):
    setattr(config, "num_key_value_heads", value)
    for key in KV_HEAD_CONFIG_KEYS:
        if hasattr(config, key):
            setattr(config, key, value)
