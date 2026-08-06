#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Helpers for classifying nightly models as older or newer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

MODEL_AGE_ENV_VAR = "NIGHTLY_MODEL_AGE"
NEWER_MODELS_ONLY_ENV_VAR = "NIGHTLY_NEWER_MODELS_ONLY"
MODEL_BATCH_INDEX_ENV_VAR = "NIGHTLY_MODEL_BATCH_INDEX"
MODEL_BATCH_SIZE_ENV_VAR = "NIGHTLY_MODEL_BATCH_SIZE"
MODEL_AGE_OLDER = "older"
MODEL_AGE_NEWER = "newer"
MODEL_AGE_UNKNOWN = "unknown"

PIPELINE_ROOT = Path(__file__).resolve().parent
VALIDATED_MODELS_PATH = PIPELINE_ROOT / "configs" / "validated_models.json"

MODEL_CLASS_TO_MODEL_KEY = {
    "audio_embedding_model_configs": "audio_embedding_models",
    "audio_model_configs": "audio_models",
    "causal_pipeline_configs": "causal_lm_models",
    "embedding_model_configs": "embedding_models",
    "image_text_to_text_model_configs": "image_text_to_text_models",
    "sequence_model_configs": "sequence_models",
}

CSV_CLASS_TO_MODEL_KEY = {
    "audio_embedding_model": "audio_embedding_models",
    "audio_model": "audio_models",
    "causal_model": "causal_lm_models",
    "embedding_model": "embedding_models",
    "image_text_to_text_model": "image_text_to_text_models",
    "sequence_model": "sequence_models",
}


def load_validated_models_config() -> dict[str, Any]:
    with VALIDATED_MODELS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_model_key(model_class: str) -> str:
    return MODEL_CLASS_TO_MODEL_KEY.get(model_class, CSV_CLASS_TO_MODEL_KEY.get(model_class, model_class))


def get_newer_models(config: dict[str, Any], model_key: str) -> set[str]:
    model_age_groups = config.get("model_age_groups", {})
    newer_models = model_age_groups.get("newer_models", {})
    return set(newer_models.get(model_key, []))


def get_model_age(model_name: str, model_class: str, config: dict[str, Any] | None = None) -> str:
    config = config or load_validated_models_config()
    model_key = get_model_key(model_class)
    if model_name in get_newer_models(config, model_key):
        return MODEL_AGE_NEWER
    if model_name in set(config.get(model_key, [])):
        return MODEL_AGE_OLDER
    return MODEL_AGE_UNKNOWN


def is_newer_model(model_name: str, model_class: str, config: dict[str, Any] | None = None) -> bool:
    return get_model_age(model_name, model_class, config) == MODEL_AGE_NEWER


def filter_models_by_age(models: list[str], model_key: str, requested_age: str | None = None) -> list[str]:
    requested_age = (requested_age or os.environ.get(MODEL_AGE_ENV_VAR, "")).strip().lower()
    if _newer_models_only_enabled():
        if requested_age == MODEL_AGE_OLDER:
            return []
        if requested_age not in {MODEL_AGE_OLDER, MODEL_AGE_NEWER}:
            requested_age = MODEL_AGE_NEWER

    if requested_age not in {MODEL_AGE_OLDER, MODEL_AGE_NEWER}:
        return models

    config = load_validated_models_config()
    newer_models = get_newer_models(config, model_key)
    if requested_age == MODEL_AGE_NEWER:
        return [model_name for model_name in models if model_name in newer_models]
    return [model_name for model_name in models if model_name not in newer_models]


def filter_models_for_nightly(
    models: list[str],
    model_key: str,
    requested_age: str | None = None,
    batch_index: str | int | None = None,
    batch_size: str | int | None = None,
) -> list[str]:
    filtered_models = filter_models_by_age(models, model_key, requested_age)
    parsed_batch_index = _parse_non_negative_int(
        batch_index if batch_index is not None else os.environ.get(MODEL_BATCH_INDEX_ENV_VAR)
    )
    parsed_batch_size = _parse_positive_int(
        batch_size if batch_size is not None else os.environ.get(MODEL_BATCH_SIZE_ENV_VAR)
    )
    if parsed_batch_index is None or parsed_batch_size is None:
        return filtered_models

    start = parsed_batch_index * parsed_batch_size
    end = start + parsed_batch_size
    return filtered_models[start:end]


def count_models_by_age(models: list[str], model_key: str, requested_age: str | None = None) -> int:
    return len(filter_models_by_age(models, model_key, requested_age))


def _newer_models_only_enabled() -> bool:
    return os.environ.get(NEWER_MODELS_ONLY_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_non_negative_int(value: str | int | None) -> int | None:
    parsed_value = _parse_int(value)
    if parsed_value is None or parsed_value < 0:
        return None
    return parsed_value


def _parse_positive_int(value: str | int | None) -> int | None:
    parsed_value = _parse_int(value)
    if parsed_value is None or parsed_value <= 0:
        return None
    return parsed_value


def _parse_int(value: str | int | None) -> int | None:
    if isinstance(value, int):
        return value
    if value is None:
        return None
    value = value.strip()
    if not value.isdigit():
        return None
    return int(value)
