# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Root conftest for the dynamo test suite.

Responsibilities:
* Register dynamo pytest markers.
* Enforce the minimum PyTorch version for dynamo export.
* Expose shared fixtures for the collision-safe workdir and QAIC device pool.
* Capture per-test outcomes + durations, aggregate them across xdist workers,
  and emit HTML + JSON reports at session end.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import pytest

from .utils.hardware_manager import DevicePool
from .utils.hash_manager import work_root, workdir_for
from .utils.report_generator import aggregate_results, write_html_report, write_json_report

MIN_TORCH_VERSION = (2, 13)

_DYNAMO_RESULTS: List[Dict] = []
_XDIST_DYNAMO_RESULTS: List[Dict] = []
_REPORT_PATHS: Dict[str, Path] = {}


# --- Markers -----------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line("markers", "dynamo: mark tests that validate the dynamo exporter path")
    config.addinivalue_line("markers", "dynamo_runtime: mark tests that execute compiled dynamo artifacts on QAIC")
    config.addinivalue_line("markers", "dynamo_compile: mark tests that compile dynamo exports")
    config.addinivalue_line("markers", "dynamo_export: mark tests that only exercise the dynamo export path")
    config.addinivalue_line("markers", "multi_device: mark dynamo tests that require more than one QAIC device")
    config.addinivalue_line("markers", "precision(name): tag a test with the numeric precision it validates")


# --- Version / environment guard --------------------------------------------
def _torch_version_tuple():
    try:
        import torch
    except ImportError:
        return None
    parts = torch.__version__.split("+")[0].split(".")
    numeric = []
    for part in parts[:2]:
        try:
            numeric.append(int(part))
        except ValueError:
            numeric.append(0)
    while len(numeric) < 2:
        numeric.append(0)
    return tuple(numeric)


def pytest_collection_modifyitems(config, items):
    torch_version = _torch_version_tuple()
    if torch_version is None or torch_version >= MIN_TORCH_VERSION:
        return
    reason = f"dynamo tests require torch>={MIN_TORCH_VERSION[0]}.{MIN_TORCH_VERSION[1]}, found {torch_version}"
    skip_marker = pytest.mark.skip(reason=reason)
    for item in items:
        if "dynamo" in item.keywords:
            item.add_marker(skip_marker)


# --- Shared fixtures --------------------------------------------------------
@pytest.fixture(scope="session")
def device_pool() -> DevicePool:
    """Cross-process device pool (4 QAIC devices by default)."""
    return DevicePool()


@pytest.fixture
def dynamo_workdir(request):
    """Collision-safe compile/export workdir builder.

    Usage inside a test::

        workdir = dynamo_workdir(architecture="llama", feature="compile_fp16")
    """

    def _builder(*, architecture: str, feature: str, precision: str = "fp32", extras=None) -> Path:
        return workdir_for(
            architecture=architecture,
            feature=feature,
            precision=precision,
            extras=extras,
            root=work_root(),
        )

    return _builder


# --- Result capture ---------------------------------------------------------
def _report_reason(report) -> str:
    wasxfail = getattr(report, "wasxfail", None)
    if wasxfail:
        return str(wasxfail)
    if report.outcome == "skipped":
        longrepr = getattr(report, "longrepr", None)
        if isinstance(longrepr, tuple) and len(longrepr) == 3:
            return str(longrepr[2])
        return str(longrepr or "")
    if report.outcome == "failed":
        longrepr = getattr(report, "longreprtext", None)
        return str(longrepr or "")
    if report.outcome == "passed":
        return ""
    return report.outcome.upper()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    # Capture the call phase, or a setup phase that was skipped up-front.
    if report.when != "call" and not (report.when == "setup" and report.outcome == "skipped"):
        return

    meta = getattr(item, "_dynamo_case", None)
    if meta is None:
        return

    if report.outcome == "skipped" and getattr(report, "wasxfail", None):
        outcome_label = "XFAIL"
    else:
        outcome_label = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP"}.get(
            report.outcome, report.outcome.upper()
        )

    _DYNAMO_RESULTS.append(
        {
            **meta,
            "nodeid": item.nodeid,
            "outcome": outcome_label,
            "reason": _report_reason(report),
            "duration": float(getattr(report, "duration", 0.0) or 0.0),
        }
    )


# --- xdist worker aggregation -----------------------------------------------
def pytest_testnodedown(node, error):
    worker_results = node.workeroutput.get("dynamo_results", [])
    if worker_results:
        _XDIST_DYNAMO_RESULTS.extend(worker_results)


def pytest_sessionfinish(session, exitstatus):
    if getattr(session.config, "workerinput", None) is not None:
        session.config.workeroutput["dynamo_results"] = list(_DYNAMO_RESULTS)
        return

    all_results = list(_DYNAMO_RESULTS) + list(_XDIST_DYNAMO_RESULTS)
    if not all_results:
        return

    report_root = Path(
        os.environ.get(
            "QEFF_DYNAMO_REPORT_DIR",
            Path.cwd() / "test-results" / "dynamo",
        )
    )
    json_path = report_root / "dynamo-report.json"
    html_path = report_root / "dynamo-report.html"
    rows = aggregate_results(all_results)
    write_json_report(json_path, rows)
    write_html_report(html_path, rows)
    _REPORT_PATHS["json"] = json_path
    _REPORT_PATHS["html"] = html_path


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _REPORT_PATHS:
        return
    terminalreporter.write_sep("=", "Dynamo Coverage Report")
    terminalreporter.write_line(f"JSON: {_REPORT_PATHS['json']}")
    terminalreporter.write_line(f"HTML: {_REPORT_PATHS['html']}")
