# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pytest

from tests.ci_perf.perf_schema import ModelPerfRecord
from tests.ci_perf.storage import compute_dir_size, write_stage_report

_RECORDER_KEY = pytest.StashKey()


def _build_key(model_name: str, config: dict) -> str:
    """Build a stable composite DB key from model name + sorted config params."""
    parts = "::".join(f"{k}={v}" for k, v in sorted(config.items()))
    return f"{model_name}::{parts}"


class PerfRecorder:
    """Session-level accumulator of per-model performance records."""

    def __init__(self) -> None:
        self._records: dict[str, dict] = {}

    def record(
        self,
        model_name: str,
        exec_info,
        config: dict,
        onnx_path=None,
        qpc_path=None,
        request=None,
    ) -> None:
        """
        Extract perf_metrics from *exec_info* and store under a composite key.

        Args:
            model_name: HuggingFace model id.
            exec_info:  CloudAI100ExecInfo (or compatible) with .perf_metrics attribute.
            config:     Dict of inference params, e.g. {"batch_size": 1, "seq_len": 128}.
            onnx_path:  Optional path to ONNX export directory — sizes computed if provided.
            qpc_path:   Optional path to QPC compiled directory — sizes computed if provided.
            request:    pytest FixtureRequest for node ID tagging (optional).
        """
        if exec_info is None:
            return
        pm = getattr(exec_info, "perf_metrics", None)
        if pm is None:
            return

        key = _build_key(model_name, config)
        nodeid = request.node.nodeid if request is not None else ""

        record = ModelPerfRecord(
            prefill_time=getattr(pm, "prefill_time", None),
            decode_perf=getattr(pm, "decode_perf", None),
            total_perf=getattr(pm, "total_perf", None),
            total_time=getattr(pm, "total_time", None),
            onnx_size_bytes=compute_dir_size(onnx_path),
            qpc_size_bytes=compute_dir_size(qpc_path),
            batch_size=getattr(exec_info, "batch_size", 1),
            test_nodeid=nodeid,
        )
        self._records[key] = record.to_dict()


# ---------------------------------------------------------------------------
# pytest plugin hooks
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Create a PerfRecorder on the controller process only."""
    if not hasattr(config, "workerinput"):
        config.stash[_RECORDER_KEY] = PerfRecorder()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """
    Worker path: push collected records back to the controller via workeroutput.
    Controller path (non-xdist or after all workers done): merge + write report.
    """
    if hasattr(session.config, "workerinput"):
        # xdist worker — push records to controller
        recorder: Optional[PerfRecorder] = session.config.stash.get(_RECORDER_KEY, None)
        if recorder is not None:
            session.config.workeroutput["ci_perf_records"] = recorder._records
        return

    # Controller path
    merged: dict[str, dict] = {}

    # Collect records forwarded by xdist workers
    for worker_records in getattr(session.config, "_ci_perf_worker_outputs", []):
        merged.update(worker_records)

    # Also collect any records from the controller itself (non-xdist case)
    controller_recorder: Optional[PerfRecorder] = session.config.stash.get(_RECORDER_KEY, None)
    if controller_recorder is not None:
        merged.update(controller_recorder._records)

    if merged and os.environ.get("CI_PERF_REPORT_DIR"):
        write_stage_report(merged)


def pytest_testnodedown(node, error) -> None:
    """
    Called on the controller when an xdist worker finishes.
    Collects the worker's ci_perf_records from workeroutput.
    """
    records = node.workeroutput.get("ci_perf_records", {})
    if records:
        if not hasattr(node.config, "_ci_perf_worker_outputs"):
            node.config._ci_perf_worker_outputs = []
        node.config._ci_perf_worker_outputs.append(records)


@pytest.fixture(scope="session")
def perf_recorder(request: pytest.FixtureRequest) -> PerfRecorder:
    """Session-scoped fixture that accumulates per-model perf records."""
    recorder = request.config.stash.get(_RECORDER_KEY, None)
    if recorder is None:
        # Fallback for xdist workers that receive a fresh stash
        recorder = PerfRecorder()
        request.config.stash[_RECORDER_KEY] = recorder
    return recorder
