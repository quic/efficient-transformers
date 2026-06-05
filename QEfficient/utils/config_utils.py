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


def calculate_num_replicate_kv_heads(num_devices: int, text_model_config) -> int:
    """
    Choose a KV-repeat value from model config and device count.

    Primary criteria:
    1. num_kv_heads * repeat is divisible by num_devices
    2. num_attention_heads is divisible by (num_kv_heads * repeat)

    Fallback:
    repeat = num_attention_heads / num_kv_heads (integer-truncated if needed).
    """
    num_attention_heads = resolve_attention_heads(text_model_config)
    num_kv_heads = resolve_kv_heads(text_model_config)

    if num_attention_heads is None or num_kv_heads is None or num_attention_heads < 1 or num_kv_heads < 1:
        return 1

    num_devices = max(1, int(num_devices))
    max_repeat = max(1, int(num_attention_heads / num_kv_heads))

    for repeat in range(max_repeat, 0, -1):
        repeated_kv_heads = num_kv_heads * repeat
        if (repeated_kv_heads % num_devices == 0) and (num_attention_heads % repeated_kv_heads == 0):
            return repeat

    return 1
