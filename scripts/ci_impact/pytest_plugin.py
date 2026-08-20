# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Pytest collection filter for a generated CI impact plan."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from .core import SCHEMA_VERSION, SELECTIVE_OMITTED_MARKERS, STAGES, _stages_for


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("ci-impact")
    group.addoption("--impact-plan", type=Path, help="path to .ci-impact-plan.json")
    group.addoption("--impact-stage", choices=STAGES, help="Jenkins stage key represented by this pytest run")
    group.addoption(
        "--impact-catalog",
        type=Path,
        default=Path(".ci-impact-catalog.json"),
        help="path to write the collect-only pytest catalog",
    )
    group.addoption("--impact-head", help="git HEAD SHA represented by the collect-only pytest catalog")
    group.addoption(
        "--impact-profile-omissions",
        choices=("dummy_layers_model", "few_layers_model", "full_layers_model"),
        help="run only changed tests excluded by this layer profile",
    )


def _load(config: pytest.Config) -> tuple[dict[str, object], str] | None:
    path = config.getoption("--impact-plan")
    stage = config.getoption("--impact-stage")
    if path is None and stage is None:
        return None
    if path is None or stage is None:
        raise pytest.UsageError("--impact-plan and --impact-stage must be provided together")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise pytest.UsageError(f"cannot load impact plan {path}: {error}") from error
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise pytest.UsageError(f"unsupported impact plan schema: {payload.get('schema_version')!r}")
    if payload.get("mode") not in {"full", "selective", "install_only", "no_tests"}:
        raise pytest.UsageError(f"invalid impact mode: {payload.get('mode')!r}")
    stages = payload.get("stages")
    if not isinstance(stages, dict) or stage not in stages:
        raise pytest.UsageError(f"impact plan has no stage {stage!r}")
    return payload, stage


def _matches(nodeid: str, nodeids: list[str]) -> bool:
    return nodeid in set(nodeids)


def _omitted_by_profile(item: pytest.Item, profile: str) -> bool:
    markers = {marker.name for marker in item.iter_markers()}
    excluded = {
        "dummy_layers_model": {"full_layers", "few_layers"},
        "few_layers_model": {"full_layers", "dummy_layers"},
        "full_layers_model": {"dummy_layers", "few_layers"},
    }
    return bool(markers & excluded[profile])


def _omitted_from_selective_ci(item: pytest.Item) -> bool:
    return bool({marker.name for marker in item.iter_markers()} & SELECTIVE_OMITTED_MARKERS)


def _head(config: pytest.Config) -> str:
    configured = config.getoption("--impact-head")
    if configured:
        return configured
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()


def _write_catalog(config: pytest.Config, items: list[pytest.Item]) -> None:
    tests = []
    for item in items:
        path = item.nodeid.split("::", 1)[0]
        markers = {marker.name for marker in item.iter_markers()}
        if markers & SELECTIVE_OMITTED_MARKERS:
            continue
        stages = sorted(_stages_for(path, markers))
        if stages:
            tests.append({"nodeid": item.nodeid, "stages": stages})
    payload = {
        "schema_version": SCHEMA_VERSION,
        "head": _head(config),
        "tests": sorted(tests, key=lambda test: test["nodeid"]),
    }
    path = config.getoption("--impact-catalog")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.option.collectonly is True:
        _write_catalog(config, items)

    loaded = _load(config)
    if loaded is None:
        return
    payload, stage = loaded
    if payload["mode"] == "full":
        return
    stage_plan = payload["stages"][stage]
    field = "profile_override_nodeids" if config.getoption("--impact-profile-omissions") else "nodeids"
    prefixes = stage_plan[field]
    selected = [item for item in items if _matches(item.nodeid, prefixes)]
    profile = config.getoption("--impact-profile-omissions")
    if profile:
        selected = [item for item in selected if _omitted_by_profile(item, profile)]
    if payload["mode"] == "selective":
        selected = [item for item in selected if not _omitted_from_selective_ci(item)]
    selected_ids = {id(item) for item in selected}
    deselected = [item for item in items if id(item) not in selected_ids]
    items[:] = selected
    if deselected:
        config.hook.pytest_deselected(items=deselected)
