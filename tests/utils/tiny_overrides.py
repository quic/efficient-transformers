# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Profile-gated tiny-random model overrides for the per-PR CI profiles.

The Jenkins pipeline (scripts/Jenkinsfile / Jenkinsfile.fast) selects one of
three test profiles via the `TEST_PROFILE` parameter, surfaced to pytest
through the `QEFF_TEST_PROFILE` env var:

  - dummy_layers_model  → use tiny replacements (per-PR fast lane)
  - few_layers_model    → use tiny replacements (per-PR fast lane)
  - full_layers_model   → use original IDs verbatim (nightly)

Tests call `resolve_model_id("Qwen/Qwen2-0.5B")` and get either the original
ID (full_layers, or no override registered) or a tiny-random sibling. The
mapping lives in tests/configs/tiny_overrides.json so adding a swap is a
one-line JSON change with no code edit.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

_PROFILE_ENV = "QEFF_TEST_PROFILE"
_TINY_PROFILES = {"dummy_layers_model", "few_layers_model"}
_OVERRIDE_FILE = Path(__file__).resolve().parents[1] / "configs" / "tiny_overrides.json"


@functools.lru_cache(maxsize=1)
def _load_overrides() -> dict:
    if not _OVERRIDE_FILE.is_file():
        return {}
    try:
        data = json.loads(_OVERRIDE_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data.get("overrides", {}) or {}


@functools.lru_cache(maxsize=1)
def _load_skip_set() -> frozenset:
    if not _OVERRIDE_FILE.is_file():
        return frozenset()
    try:
        data = json.loads(_OVERRIDE_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return frozenset()
    return frozenset(data.get("skip_no_tiny", []) or [])


def is_skipped_model(model_id: str) -> bool:
    """True when this model is in the skip_no_tiny list AND the tiny lane
    is active. Tests should call this in their parametrize body and
    `pytest.skip(...)` when it returns True."""
    if not isinstance(model_id, str) or not _tiny_lane_active():
        return False
    return model_id in _load_skip_set()


def _tiny_lane_active() -> bool:
    """Return True when the per-PR fast lane is active.

    Default behavior: when QEFF_TEST_PROFILE is unset, treat the run as a
    developer/local invocation and DO NOT remap (preserve current behavior).
    Only the tiny profiles opt in. To force the tiny lane locally, export
    QEFF_TEST_PROFILE=dummy_layers_model.
    """
    profile = os.environ.get(_PROFILE_ENV, "").strip()
    return profile in _TINY_PROFILES


def resolve_model_id(model_id: str) -> str:
    """Map a real HF model ID to its tiny-random sibling under the per-PR profile.

    Returns the original ID unchanged when:
      - QEFF_TEST_PROFILE is unset or set to full_layers_model (nightly), or
      - no override is registered for this model.
    """
    if not isinstance(model_id, str) or not _tiny_lane_active():
        return model_id
    return _load_overrides().get(model_id, model_id)
