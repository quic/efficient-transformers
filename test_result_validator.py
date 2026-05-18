# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import csv
import json

from .result_validator import (
    ValidationTolerances,
    load_validation_tolerances,
    validate_artifact_file,
    validate_artifacts,
)


def test_validate_artifacts_passes_within_regression_tolerance():
    previous = {"model-a": _artifact_payload()}
    current = {
        "model-a": _artifact_payload(
            export_time=104,
            compile_time=104,
            size="104.00 MB",
            prefill_time=1.04,
            decode_perf=96,
            total_perf=96,
            total_time=1.04,
            generated_ids=[1, 2, 3.01],
        )
    }

    rows = validate_artifacts(
        current, previous, ValidationTolerances(percentage_tolerance=5.0, token_mad_tolerance=1e-2)
    )

    assert rows[0]["status"] == "passed"
    assert rows[0]["export_time_pct_diff"] == 4.0
    assert rows[0]["decode_pct_diff"] == -4.0


def test_validate_artifacts_fails_regressions_above_tolerance():
    previous = {"model-a": _artifact_payload()}
    current = {
        "model-a": _artifact_payload(
            export_time=106,
            compile_time=106,
            size="106.00 MB",
            prefill_time=1.06,
            decode_perf=94,
            total_perf=94,
            total_time=1.06,
            generated_ids=[1, 2, 4],
        )
    }

    rows = validate_artifacts(
        current, previous, ValidationTolerances(percentage_tolerance=5.0, token_mad_tolerance=1e-2)
    )

    assert rows[0]["status"] == "failed"
    assert "export_time_pct_diff" in rows[0]["failure_reason"]
    assert "decode_pct_diff" in rows[0]["failure_reason"]
    assert "tokens_mad" in rows[0]["failure_reason"]


def test_validate_artifacts_reports_missing_optional_metrics_as_na():
    previous = {"model-a": {"export_time": 10}}
    current = {"model-a": {"export_time": 10}}

    rows = validate_artifacts(current, previous, ValidationTolerances())

    assert rows[0]["status"] == "passed"
    assert rows[0]["compile_time_pct_diff"] == "N/A"
    assert rows[0]["tokens_mad"] == "N/A"


def test_validate_artifacts_fails_missing_previous_model():
    rows = validate_artifacts({"model-a": _artifact_payload()}, {}, ValidationTolerances())

    assert rows[0]["status"] == "failed"
    assert rows[0]["failure_reason"] == "Model not found in previous nightly results."


def test_validate_artifacts_sums_multiple_size_fields():
    previous = {
        "model-a": {
            "batch_size": 1,
            "encoder_onnx_and_qpc_dir size": "1.00 GB",
            "decoder_onnx_and_qpc_dir size": "512.00 MB",
        }
    }
    current = {
        "model-a": {
            "batch_size": 8,
            "encoder_onnx_and_qpc_dir size": "1.00 GB",
            "decoder_onnx_and_qpc_dir size": "512.00 MB",
        }
    }

    rows = validate_artifacts(current, previous, ValidationTolerances())

    assert rows[0]["status"] == "passed"
    assert rows[0]["onnx_qpc_size_before"] == 1.5 * 1024**3
    assert rows[0]["onnx_qpc_size_pct_diff"] == 0.0


def test_validate_artifacts_uses_na_for_zero_baseline_percentage():
    previous = {"model-a": {"export_time": 0}}
    current = {"model-a": {"export_time": 10}}

    rows = validate_artifacts(current, previous, ValidationTolerances())

    assert rows[0]["status"] == "passed"
    assert rows[0]["export_time_pct_diff"] == "N/A"


def test_validate_artifacts_uses_common_prefix_for_token_mad():
    previous = {"model-a": {"generated_ids": [[1, 2, 3, 999]]}}
    current = {"model-a": {"generated_ids": [[2, 4, 6]]}}

    rows = validate_artifacts(current, previous, ValidationTolerances(token_mad_tolerance=10))

    assert rows[0]["tokens_mad"] == 2.0


def test_validate_artifact_file_writes_csv(tmp_path):
    previous_path = tmp_path / "previous.json"
    current_path = tmp_path / "current.json"
    csv_path = tmp_path / "validation.csv"
    previous_path.write_text(json.dumps({"model-a": _artifact_payload()}), encoding="utf-8")
    current_path.write_text(json.dumps({"model-a": _artifact_payload(export_time=101)}), encoding="utf-8")

    rows = validate_artifact_file(current_path, previous_path, csv_path, ValidationTolerances())

    assert rows[0]["status"] == "passed"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["model_name"] == "model-a"
    assert csv_rows[0]["export_time_pct_diff"] == "1.000000"


def test_load_validation_tolerances_uses_model_class_override():
    configs = {
        "validation_configs": {
            "default": {"percentage_tolerance": 5.0, "token_mad_tolerance": 0.01},
            "model_class_tolerances": {"causal_pipeline_configs": {"percentage_tolerance": 7.5}},
        }
    }

    tolerances = load_validation_tolerances(configs, "causal_pipeline_configs")

    assert tolerances.percentage_tolerance == 7.5
    assert tolerances.token_mad_tolerance == 0.01


def _artifact_payload(
    export_time=100,
    compile_time=100,
    size="100.00 MB",
    prefill_time=1,
    decode_perf=100,
    total_perf=100,
    total_time=1,
    generated_ids=None,
):
    if generated_ids is None:
        generated_ids = [1, 2, 3]
    return {
        "export_time": export_time,
        "compile_time": compile_time,
        "size": size,
        "perf_metrics": {
            "prefill_time": prefill_time,
            "decode_perf": decode_perf,
            "total_perf": total_perf,
            "total_time": total_time,
        },
        "generated_ids": generated_ids,
    }
