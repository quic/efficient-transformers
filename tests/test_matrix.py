# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
from pathlib import Path
from typing import Any, Iterable


TEST_SCOPE_ENV = "QEFF_TEST_SCOPE"
PR_SCOPE = "pr"
EXHAUSTIVE_SCOPE = "exhaustive"
VALID_SCOPES = {PR_SCOPE, EXHAUSTIVE_SCOPE}


def get_test_scope() -> str:
    scope = os.environ.get(TEST_SCOPE_ENV, EXHAUSTIVE_SCOPE).strip().lower()
    if scope not in VALID_SCOPES:
        raise ValueError(f"Unsupported {TEST_SCOPE_ENV}={scope!r}. Expected one of {sorted(VALID_SCOPES)}.")
    return scope


def load_test_config(config_path: str | os.PathLike) -> dict[str, Any]:
    with open(Path(config_path), "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def select_test_entries(config_path: str | os.PathLike, suite_name: str, scope: str | None = None) -> list[Any]:
    entries = load_test_config(config_path)[suite_name]
    scope = get_test_scope() if scope is None else scope
    if scope == EXHAUSTIVE_SCOPE:
        return entries

    selected_entries = []
    seen_coverage_keys = set()
    for entry in entries:
        if not _runs_in_scope(entry, scope):
            continue
        coverage_key = get_coverage_key(entry)
        if coverage_key in seen_coverage_keys:
            continue
        seen_coverage_keys.add(coverage_key)
        selected_entries.append(entry)
    return selected_entries


def model_names(entries: Iterable[Any]) -> list[str]:
    return [_entry_id(entry) for entry in entries]


def entries_by_model_name(entries: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {entry["model_name"]: entry for entry in entries}


def entries_by_id(entries: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {entry["id"]: entry for entry in entries}


def get_coverage_key(entry: Any) -> str:
    if not isinstance(entry, dict):
        return str(entry)

    if entry.get("coverage_key"):
        return str(entry["coverage_key"])

    model_name = str(entry.get("model_name") or entry.get("target_model_name") or entry.get("id") or entry)
    model_type = entry.get("model_type")
    if model_type:
        return _typed_coverage_key(model_name, str(model_type))
    return model_name


def assert_pr_coverage(config_path: str | os.PathLike, suite_name: str, required_keys: set[str]) -> None:
    selected_keys = {get_coverage_key(entry) for entry in select_test_entries(config_path, suite_name, PR_SCOPE)}
    missing_keys = required_keys - selected_keys
    assert not missing_keys, f"Missing PR coverage keys for {suite_name}: {sorted(missing_keys)}"


def _runs_in_scope(entry: Any, scope: str) -> bool:
    if not isinstance(entry, dict):
        return True
    tiers = entry.get("tiers")
    if tiers is None:
        return True
    return scope in tiers


def _entry_id(entry: Any) -> str:
    if isinstance(entry, dict):
        if "model_name" in entry:
            return entry["model_name"]
        if "id" in entry:
            return entry["id"]
    return str(entry)


def _typed_coverage_key(model_name: str, model_type: str) -> str:
    lower_name = model_name.lower()
    if "awq" in lower_name:
        return f"{model_type}:awq"
    if "gptq" in lower_name:
        return f"{model_type}:gptq"
    if "fp8" in lower_name:
        return f"{model_type}:fp8"
    if "swiftkv" in lower_name:
        return f"{model_type}:swiftkv"
    if "reranker" in lower_name:
        return f"{model_type}:reranker"
    if "embedding" in lower_name:
        return f"{model_type}:embedding"
    return model_type
