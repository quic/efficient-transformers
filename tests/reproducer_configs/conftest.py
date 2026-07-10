# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Pytest hooks for reported reproducer config tests."""

from __future__ import annotations

import pytest

SCENARIO_CLI_OPTION = "--qeff-reproducer-scenario"


# Parse comma-separated or repeated scenario names from pytest CLI values.
def _split_scenario_names(values: list[str]) -> tuple[str, ...]:
    """Return unique scenario names from repeated or comma-separated options.

    The helper exists so developers can run one or more just-added reproducers
    without collecting unrelated scenarios from the full reported-config matrix.
    """
    names = []
    for value in values:
        for name in value.split(","):
            stripped_name = name.strip()
            if stripped_name and stripped_name not in names:
                names.append(stripped_name)
    return tuple(names)


# Register a targeted reproducer selector for local developer loops.
def pytest_addoption(parser: pytest.Parser) -> None:
    """Add the scenario-selection option used by reproducer config tests."""
    group = parser.getgroup("qeff reproducer configs")
    group.addoption(
        SCENARIO_CLI_OPTION,
        action="append",
        default=[],
        metavar="NAME[,NAME...]",
        help="Run only the named reproducer scenario(s). Can be repeated or comma-separated.",
    )


# Deselect non-targeted reproducer items after parametrization.
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep only requested reproducer scenarios when the CLI selector is set.

    The hook exists so ``--qeff-reproducer-scenario`` runs exactly the requested
    scenario tests and drops catalog guards or empty-stage sentinel parameters.
    """
    selected_names = set(_split_scenario_names(config.getoption(SCENARIO_CLI_OPTION) or []))
    if not selected_names:
        return

    available_names = set()
    for item in items:
        scenario = getattr(getattr(item, "callspec", None), "params", {}).get("scenario")
        scenario_name = getattr(scenario, "name", None)
        if scenario_name:
            available_names.add(scenario_name)

    missing_names = selected_names - available_names
    if missing_names:
        names = ", ".join(sorted(missing_names))
        raise pytest.UsageError(f"unknown reproducer scenario(s): {names}")

    kept_items = []
    deselected_items = []
    for item in items:
        scenario = getattr(getattr(item, "callspec", None), "params", {}).get("scenario")
        scenario_name = getattr(scenario, "name", None)
        if scenario_name in selected_names:
            kept_items.append(item)
        else:
            deselected_items.append(item)

    if deselected_items:
        config.hook.pytest_deselected(items=deselected_items)
    items[:] = kept_items
