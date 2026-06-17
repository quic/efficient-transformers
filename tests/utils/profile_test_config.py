# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Centralised test-config loader.

Tests call::

    from tests.utils.profile_test_config import load_test_config
    config_data = load_test_config("causal_model_configs")

``load_test_config`` loads the unified config file from ``tests/configs/``.

Two execution paths are supported via the ``QEFF_TEST_PROFILE`` environment
variable:

* **original models** (default, unset or ``full_layers_model``):
  loads ``tests/configs/test_models_config.json``
* **tiny models** (``QEFF_TEST_PROFILE=tiny_model``):
  loads ``tests/configs/test_models_config_tiny.json``

The ``name`` argument is accepted for backward compatibility but is no longer
used to select a file — all test domains live in the single unified file and
callers access them via their existing top-level key
(e.g. ``config_data["causal_lm_models"]``).
"""

from __future__ import annotations

import copy
import functools
import json
import os
from pathlib import Path

_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
_UNIFIED_CONFIG = "test_models_config.json"
_UNIFIED_CONFIG_TINY = "test_models_config_tiny.json"
_PROFILE_ENV = "QEFF_TEST_PROFILE"
_TINY_PROFILES = {"tiny_model"}

_SKIP_NO_TINY = frozenset(
    {
        "Snowflake/Llama-3.1-SwiftKV-8B-Instruct",
        "allenai/Molmo-7B-D-0924",
        "Qwen/Qwen3-VL-Reranker-2B",
        "Qwen/Qwen3-VL-Reranker-8B",
        "TheBloke/Llama-2-7B-GPTQ",
        "google/gemma-3-4b-it",
        "llava-hf/llava-1.5-7b-hf",
        "Qwen/Qwen3-VL-Embedding-8B",
    }
)

_EXTRA_TINY_OVERRIDES = {
    "meta-llama/Meta-Llama-3-8B": "llamafactory/tiny-random-Llama-3",
    "hallisky/lora-type-narrative-llama-3-8b": "llamafactory/tiny-random-Llama-3-lora",
    "hallisky/lora-grade-elementary-llama-3-8b": "llamafactory/tiny-random-Llama-3-lora",
    "distilbert/distilgpt2": "hf-internal-testing/tiny-random-GPT2LMHeadModel",
}

_STRING_LIST_KEYS = {
    "seq_classification_models",
    "speech_seq2seq_models",
    "audio_embedding_models",
}

_DISAGG_DUMMY_PARAMS_BY_TYPE = {
    "gpt_oss": {
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "intermediate_size": 256,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_local_experts": 4,
        "head_dim": 32,
        "max_position_embeddings": 512,
        "vocab_size": 201088,
        "sliding_window": 128,
    },
    "qwen3_moe": {
        "hidden_size": 256,
        "intermediate_size": 256,
        "max_position_embeddings": 512,
        "max_window_layers": 48,
        "moe_intermediate_size": 768,
        "num_attention_heads": 2,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "vocab_size": 151936,
    },
}


def _entry_with_model_name(entry):
    if isinstance(entry, str):
        return {"model_name": entry}
    return copy.deepcopy(entry)


def _normalize_domain(key: str, value):
    if isinstance(value, dict) and "models" in value:
        shared_params = {k: copy.deepcopy(v) for k, v in value.items() if k != "models"}
        models = value["models"]
    else:
        shared_params = {}
        models = value

    if key in _STRING_LIST_KEYS:
        return [entry if isinstance(entry, str) else entry["model_name"] for entry in models]

    normalized = []
    for entry in models:
        model_entry = _entry_with_model_name(entry)
        if shared_params:
            model_entry = {**copy.deepcopy(shared_params), **model_entry}
        if key == "disaggregated_dummy_models" and "additional_params" not in model_entry:
            model_entry["additional_params"] = copy.deepcopy(
                _DISAGG_DUMMY_PARAMS_BY_TYPE.get(model_entry.get("model_type"), {})
            )
        normalized.append(model_entry)
    return normalized


def _normalize_config(data: dict) -> dict:
    return {key: _normalize_domain(key, value) for key, value in data.items() if not key.startswith("_")}


def _model_names(value):
    if isinstance(value, dict) and "models" in value:
        models = value["models"]
    else:
        models = value

    names = []
    for entry in models:
        if isinstance(entry, str):
            names.append(entry)
        elif isinstance(entry, dict) and "model_name" in entry:
            names.append(entry["model_name"])
    return names


@functools.lru_cache(maxsize=1)
def _load_overrides() -> dict:
    original_path = _CONFIGS_DIR / _UNIFIED_CONFIG
    tiny_path = _CONFIGS_DIR / _UNIFIED_CONFIG_TINY
    try:
        original_data = json.loads(original_path.read_text())
        tiny_data = json.loads(tiny_path.read_text())
    except (OSError, json.JSONDecodeError):
        return dict(_EXTRA_TINY_OVERRIDES)

    overrides = dict(_EXTRA_TINY_OVERRIDES)
    for key, original_value in original_data.items():
        if key.startswith("_") or key not in tiny_data:
            continue

        original_names = [name for name in _model_names(original_value) if name not in _SKIP_NO_TINY]
        tiny_names = _model_names(tiny_data[key])
        for original_name, tiny_name in zip(original_names, tiny_names):
            if original_name != tiny_name:
                overrides.setdefault(original_name, tiny_name)
    return overrides


@functools.lru_cache(maxsize=1)
def _load_skip_set() -> frozenset:
    return _SKIP_NO_TINY


def _tiny_lane_active() -> bool:
    return os.environ.get(_PROFILE_ENV, "").strip() in _TINY_PROFILES


def is_skipped_model(model_id: str) -> bool:
    if not isinstance(model_id, str) or not _tiny_lane_active():
        return False
    return model_id in _load_skip_set()


def resolve_model_id(model_id: str) -> str:
    if not isinstance(model_id, str) or not _tiny_lane_active():
        return model_id
    return _load_overrides().get(model_id, model_id)


def load_test_config(name: str) -> dict:
    """Load and return the unified test config dict.

    Args:
        name: Accepted for backward compatibility; no longer used to select a
              file.  All callers access their domain via a top-level key on the
              returned dict (e.g. ``config_data["causal_lm_models"]``).

    Returns:
        Parsed unified config dict.  When ``QEFF_TEST_PROFILE=tiny_model`` the
        tiny-model variant is returned; otherwise the original-model config.

    Raises:
        FileNotFoundError: If the config file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    profile = os.environ.get(_PROFILE_ENV, "").strip()
    filename = _UNIFIED_CONFIG_TINY if profile == "tiny_model" else _UNIFIED_CONFIG
    config_path = _CONFIGS_DIR / filename
    if not config_path.is_file():
        raise FileNotFoundError(f"Test config not found: {config_path}")
    return _normalize_config(json.loads(config_path.read_text()))
