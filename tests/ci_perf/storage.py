# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tests.ci_perf.perf_schema import SCHEMA_VERSION, StagePerfReport


def compute_dir_size(path) -> Optional[int]:
    """Return total bytes of all files under *path*, or None if path is None/missing."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def write_stage_report(records: dict[str, dict]) -> None:
    """
    Build a StagePerfReport from *records* and env vars, then write atomically
    to $CI_PERF_REPORT_DIR/{CI_STAGE_NAME}/perf_report.json.

    No-op if CI_PERF_REPORT_DIR is not set (local dev runs).
    """
    report_dir = os.environ.get("CI_PERF_REPORT_DIR")
    if not report_dir:
        return

    stage_name = os.environ.get("CI_STAGE_NAME", "UNKNOWN")
    target_dir = Path(report_dir) / stage_name
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "perf_report.json"

    report = StagePerfReport(
        schema_version=SCHEMA_VERSION,
        stage_name=stage_name,
        ci_build_tag=os.environ.get("CI_BUILD_TAG", ""),
        git_sha=os.environ.get("GIT_SHA", ""),
        hardware={
            "node_label": os.environ.get("NODE_LABEL", ""),
            "nsp_count": os.environ.get("NSP_COUNT", ""),
        },
        timestamp=datetime.now(timezone.utc).isoformat(),
        models=records,
    )

    _atomic_write(target, report.to_dict())


def load_stage_report(report_dir: str | Path, stage_name: str) -> dict[str, Any]:
    """Load perf_report.json for a given stage. Returns empty dict on missing file."""
    path = Path(report_dir) / stage_name / "perf_report.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_db(db_path: str | Path) -> dict[str, Any]:
    """Load baseline_db.json, returning an empty skeleton if the file does not exist."""
    p = Path(db_path)
    if not p.exists():
        return {"schema_version": SCHEMA_VERSION, "hardware_profiles": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def update_baseline(
    db_path: str | Path,
    hw_key: str,
    stage_name: str,
    models_data: dict[str, dict],
    build_tag: str,
    git_sha: str,
) -> None:
    """
    Atomically update baseline_db.json with metrics from the current run.
    Creates the file if it does not exist.
    """
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    db = load_db(p)
    hw = db["hardware_profiles"].setdefault(hw_key, {"stages": {}})
    stage = hw["stages"].setdefault(stage_name, {"models": {}})

    now = datetime.now(timezone.utc).isoformat()
    for model_key, metrics in models_data.items():
        entry = {
            "prefill_time": metrics.get("prefill_time"),
            "decode_perf": metrics.get("decode_perf"),
            "total_perf": metrics.get("total_perf"),
            "total_time": metrics.get("total_time"),
            "onnx_size_bytes": metrics.get("onnx_size_bytes"),
            "qpc_size_bytes": metrics.get("qpc_size_bytes"),
            "baseline_sha": git_sha,
            "baseline_timestamp": now,
        }
        stage["models"][model_key] = entry

    db["last_updated"] = now
    db["last_updated_by"] = build_tag

    _atomic_write(p, db)


def _atomic_write(target: Path, data: dict) -> None:
    """Write *data* as JSON to *target* atomically via a temp file."""
    tmp = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
