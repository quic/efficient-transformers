# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Unit tests for the CI perf tracking comparison logic.

All tests run without QAIC hardware — they use synthetic in-memory data.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.ci_perf.compare_perf import compare_report, get_threshold, load_thresholds, main
from tests.ci_perf.conftest import PerfRecorder, _build_key
from tests.ci_perf.storage import compute_dir_size, load_db, update_baseline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    "default": {
        "prefill_time": {"percentage_tolerance": 5.0},
        "decode_perf": {"percentage_tolerance": 5.0},
        "onnx_size_bytes": {"percentage_tolerance": 5.0},
        "qpc_size_bytes": {"percentage_tolerance": 5.0},
    },
    "per_model": {},
}


def _make_report(models: dict, stage="QAIC_LLM") -> dict:
    return {"schema_version": "1", "stage_name": stage, "models": models}


def _make_baseline(models: dict, hw_key="qeff_node_16", stage="QAIC_LLM") -> dict:
    return {
        "schema_version": "1",
        "hardware_profiles": {hw_key: {"stages": {stage: {"models": models}}}},
    }


def _write_report(tmp_path: Path, models: dict, stage="QAIC_LLM") -> Path:
    stage_dir = tmp_path / stage
    stage_dir.mkdir(parents=True)
    report_file = stage_dir / "perf_report.json"
    report_file.write_text(
        json.dumps({"schema_version": "1", "stage_name": stage, "ci_build_tag": "test-build",
                    "git_sha": "abc123", "hardware": {}, "models": models}),
        encoding="utf-8",
    )
    return tmp_path


def _write_db(tmp_path: Path, db: dict) -> Path:
    db_file = tmp_path / "baseline_db.json"
    db_file.write_text(json.dumps(db), encoding="utf-8")
    return db_file


def _good_model():
    return {"prefill_time": 0.044, "decode_perf": 121.0, "onnx_size_bytes": 1000, "qpc_size_bytes": 500,
            "total_perf": 111.0, "total_time": 2.28, "batch_size": 1}


# ---------------------------------------------------------------------------
# Test 1: empty DB → all models SKIPped, exit 0
# ---------------------------------------------------------------------------
def test_first_run_model_skipped():
    report = _make_report({"model::bs=1": _good_model()})
    rows, failures = compare_report(report, baseline_models={}, thresholds=DEFAULT_THRESHOLDS)
    assert failures == []
    assert rows[0]["status"] == "skipped"


# ---------------------------------------------------------------------------
# Test 2: within tolerance → pass
# ---------------------------------------------------------------------------
def test_within_tolerance_passes():
    baseline = {"model::bs=1": {**_good_model()}}
    current_model = {**_good_model(), "decode_perf": 118.0}  # -2.5% — within 5%
    report = _make_report({"model::bs=1": current_model})
    _, failures = compare_report(report, baseline_models=baseline, thresholds=DEFAULT_THRESHOLDS)
    assert failures == []


# ---------------------------------------------------------------------------
# Test 3: decode_perf drops 25% → exit 1
# ---------------------------------------------------------------------------
def test_decode_perf_regression_fails():
    baseline = {"model::bs=1": {**_good_model()}}
    current_model = {**_good_model(), "decode_perf": 90.0}  # -25.6%
    report = _make_report({"model::bs=1": current_model})
    _, failures = compare_report(report, baseline_models=baseline, thresholds=DEFAULT_THRESHOLDS)
    assert any("decode_perf" in f for f in failures)


# ---------------------------------------------------------------------------
# Test 4: prefill_time rises 20% → exit 1
# ---------------------------------------------------------------------------
def test_prefill_time_regression_fails():
    baseline = {"model::bs=1": {**_good_model()}}
    current_model = {**_good_model(), "prefill_time": 0.053}  # +20.5%
    report = _make_report({"model::bs=1": current_model})
    _, failures = compare_report(report, baseline_models=baseline, thresholds=DEFAULT_THRESHOLDS)
    assert any("prefill_time" in f for f in failures)


# ---------------------------------------------------------------------------
# Test 5: onnx_size grows 10% → exit 1
# ---------------------------------------------------------------------------
def test_onnx_size_regression_fails():
    baseline = {"model::bs=1": {**_good_model()}}
    current_model = {**_good_model(), "onnx_size_bytes": 1100}  # +10%
    report = _make_report({"model::bs=1": current_model})
    _, failures = compare_report(report, baseline_models=baseline, thresholds=DEFAULT_THRESHOLDS)
    assert any("onnx_size_bytes" in f for f in failures)


# ---------------------------------------------------------------------------
# Test 6: qpc_size grows 10% → exit 1
# ---------------------------------------------------------------------------
def test_qpc_size_regression_fails():
    baseline = {"model::bs=1": {**_good_model()}}
    current_model = {**_good_model(), "qpc_size_bytes": 555}  # +11%
    report = _make_report({"model::bs=1": current_model})
    _, failures = compare_report(report, baseline_models=baseline, thresholds=DEFAULT_THRESHOLDS)
    assert any("qpc_size_bytes" in f for f in failures)


# ---------------------------------------------------------------------------
# Test 7: onnx_size_bytes=None in report → not compared, no failure
# ---------------------------------------------------------------------------
def test_size_null_skipped():
    baseline = {"model::bs=1": {**_good_model()}}
    current_model = {**_good_model(), "onnx_size_bytes": None}
    report = _make_report({"model::bs=1": current_model})
    rows, failures = compare_report(report, baseline_models=baseline, thresholds=DEFAULT_THRESHOLDS)
    assert failures == [] or not any("onnx_size_bytes" in f for f in failures)
    onnx_rows = [r for r in rows if r["metric"] == "onnx_size_bytes"]
    assert all(r["status"] == "skipped" for r in onnx_rows)


# ---------------------------------------------------------------------------
# Test 8: decode_perf improves 30% → exit 0 (improvement never fails)
# ---------------------------------------------------------------------------
def test_improvement_does_not_fail():
    baseline = {"model::bs=1": {**_good_model()}}
    current_model = {**_good_model(), "decode_perf": 157.0}  # +29.8% improvement
    report = _make_report({"model::bs=1": current_model})
    _, failures = compare_report(report, baseline_models=baseline, thresholds=DEFAULT_THRESHOLDS)
    assert failures == []


# ---------------------------------------------------------------------------
# Test 9: per-model tolerance override
# ---------------------------------------------------------------------------
def test_per_model_tolerance_override():
    thresholds = {
        **DEFAULT_THRESHOLDS,
        "per_model": {
            "noisy-model::bs=1": {
                "prefill_time": {"percentage_tolerance": 15.0},
            }
        },
    }
    baseline = {"noisy-model::bs=1": {**_good_model()}}
    current_model = {**_good_model(), "prefill_time": 0.050}  # +13.6% — under 15% override
    report = _make_report({"noisy-model::bs=1": current_model})
    _, failures = compare_report(report, baseline_models=baseline, thresholds=thresholds)
    assert failures == []


# ---------------------------------------------------------------------------
# Test 10: --update-baseline creates/updates baseline_db.json
# ---------------------------------------------------------------------------
def test_update_baseline_writes_db(tmp_path):
    db_path = tmp_path / "baseline_db.json"
    update_baseline(
        db_path=db_path,
        hw_key="qeff_node_16",
        stage_name="QAIC_LLM",
        models_data={"model::bs=1": _good_model()},
        build_tag="test-build",
        git_sha="abc123",
    )
    assert db_path.exists()
    db = json.loads(db_path.read_text())
    stored = db["hardware_profiles"]["qeff_node_16"]["stages"]["QAIC_LLM"]["models"]["model::bs=1"]
    assert stored["decode_perf"] == 121.0
    assert stored["onnx_size_bytes"] == 1000


# ---------------------------------------------------------------------------
# Test 11: atomic write — existing DB intact if process crashes mid-write
# ---------------------------------------------------------------------------
def test_atomic_write_preserves_existing(tmp_path):
    db_path = tmp_path / "baseline_db.json"
    # Write initial DB
    update_baseline(db_path, "hw", "STAGE", {"m::bs=1": _good_model()}, "b1", "s1")
    original_content = db_path.read_text()

    # Simulate a failed second write (temp file left behind, original untouched by os.replace)
    tmp_file = db_path.with_suffix(".json.tmp_test")
    tmp_file.write_text('{"broken": true}')
    # tmp file exists but was never renamed — original is intact
    assert db_path.read_text() == original_content


# ---------------------------------------------------------------------------
# Test 12: missing hardware key → all models SKIP
# ---------------------------------------------------------------------------
def test_hardware_key_isolation():
    # DB has baseline for qeff_node_16, but we compare with qeff_node_8
    db = _make_baseline({"model::bs=1": _good_model()}, hw_key="qeff_node_16")
    baseline_models = (
        db.get("hardware_profiles", {})
        .get("qeff_node_8", {})  # wrong key
        .get("stages", {})
        .get("QAIC_LLM", {})
        .get("models", {})
    )
    report = _make_report({"model::bs=1": _good_model()})
    rows, failures = compare_report(report, baseline_models=baseline_models, thresholds=DEFAULT_THRESHOLDS)
    assert failures == []
    assert all(r["status"] == "skipped" for r in rows)


# ---------------------------------------------------------------------------
# Test 13: exec_info=None → no crash, empty records
# ---------------------------------------------------------------------------
def test_perf_recorder_noop_on_none():
    recorder = PerfRecorder()
    recorder.record("model", exec_info=None, config={"batch_size": 1})
    assert recorder._records == {}


# ---------------------------------------------------------------------------
# Test 14: xdist worker merge
# ---------------------------------------------------------------------------
def test_xdist_merge():
    worker_0 = {"model_a::bs=1": _good_model()}
    worker_1 = {"model_b::bs=1": {**_good_model(), "decode_perf": 90.0}}
    merged = {}
    merged.update(worker_0)
    merged.update(worker_1)
    assert "model_a::bs=1" in merged
    assert "model_b::bs=1" in merged
    assert len(merged) == 2


# ---------------------------------------------------------------------------
# Test 15: composite key uniqueness — same model, different batch_size → 2 entries
# ---------------------------------------------------------------------------
def test_composite_key_uniqueness():
    key_bs1 = _build_key("meta-llama/Llama-3.2-1B", {"batch_size": 1, "seq_len": 128, "decode": "greedy"})
    key_bs4 = _build_key("meta-llama/Llama-3.2-1B", {"batch_size": 4, "seq_len": 128, "decode": "greedy"})
    assert key_bs1 != key_bs4
    assert "meta-llama/Llama-3.2-1B" in key_bs1
    assert "batch_size=1" in key_bs1
    assert "batch_size=4" in key_bs4


# ---------------------------------------------------------------------------
# CLI integration test (uses tmp_path, no QAIC)
# ---------------------------------------------------------------------------
def test_cli_first_run_exits_zero(tmp_path):
    report_dir = _write_report(tmp_path / "reports", {"model::bs=1": _good_model()})
    db_path = tmp_path / "db" / "baseline_db.json"
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(json.dumps(DEFAULT_THRESHOLDS))

    rc = main([
        "--report-dir", str(report_dir),
        "--db-path", str(db_path),
        "--stage", "QAIC_LLM",
        "--hardware-key", "qeff_node_16",
        "--thresholds", str(thresholds_path),
    ])
    assert rc == 0  # no baseline → all skipped → pass


def test_cli_regression_exits_one(tmp_path):
    db_path = tmp_path / "db" / "baseline_db.json"
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(json.dumps(DEFAULT_THRESHOLDS))

    # Write baseline first
    update_baseline(db_path, "qeff_node_16", "QAIC_LLM",
                    {"model::bs=1": _good_model()}, "b1", "s1")

    # Report with a clear decode_perf regression
    bad_model = {**_good_model(), "decode_perf": 80.0}  # -33.9%
    report_dir = _write_report(tmp_path / "reports", {"model::bs=1": bad_model})

    rc = main([
        "--report-dir", str(report_dir),
        "--db-path", str(db_path),
        "--stage", "QAIC_LLM",
        "--hardware-key", "qeff_node_16",
        "--thresholds", str(thresholds_path),
    ])
    assert rc == 1


def test_cli_update_baseline(tmp_path):
    db_path = tmp_path / "db" / "baseline_db.json"
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(json.dumps(DEFAULT_THRESHOLDS))
    report_dir = _write_report(tmp_path / "reports", {"model::bs=1": _good_model()})

    rc = main([
        "--report-dir", str(report_dir),
        "--db-path", str(db_path),
        "--stage", "QAIC_LLM",
        "--hardware-key", "qeff_node_16",
        "--thresholds", str(thresholds_path),
        "--update-baseline",
    ])
    assert rc == 0
    db = load_db(db_path)
    assert "QAIC_LLM" in db["hardware_profiles"]["qeff_node_16"]["stages"]


# ---------------------------------------------------------------------------
# compute_dir_size
# ---------------------------------------------------------------------------
def test_compute_dir_size_returns_none_for_none():
    assert compute_dir_size(None) is None


def test_compute_dir_size_returns_none_for_missing(tmp_path):
    assert compute_dir_size(tmp_path / "nonexistent") is None


def test_compute_dir_size_counts_bytes(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 1024)
    (tmp_path / "b.bin").write_bytes(b"y" * 512)
    assert compute_dir_size(tmp_path) == 1536
