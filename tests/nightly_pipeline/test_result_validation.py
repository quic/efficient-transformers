# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from pathlib import Path

import pytest

from .nightly_email_report import build_summary, render_html
from .result_validator import (
    ValidationTolerances,
    all_rows_passed,
    load_recorded_test_failure_rows,
    load_validation_tolerances,
    validate_artifact_file,
    validate_artifacts,
    write_validation_csv,
)

MODEL_ARTIFACTS = [
    ("causal_pipeline_configs", "causal_model_artifacts.json", "causal_model_validation.csv"),
    ("embedding_model_configs", "embedding_model_artifacts.json", "embedding_model_validation.csv"),
    ("audio_model_configs", "audio_model_artifacts.json", "audio_model_validation.csv"),
    ("audio_embedding_model_configs", "audio_embedding_model_artifacts.json", "audio_embedding_model_validation.csv"),
    (
        "image_text_to_text_model_configs",
        "image_text_to_text_model_artifacts.json",
        "image_text_to_text_model_validation.csv",
    ),
    ("sequence_model_configs", "sequence_model_artifacts.json", "sequence_model_validation.csv"),
]


@pytest.mark.nightly
@pytest.mark.parametrize("model_class, artifact_filename, csv_filename", MODEL_ARTIFACTS)
def test_validate_nightly_results(model_class, artifact_filename, csv_filename, artifacts_dir, get_pipeline_config):
    previous_artifacts_dir = os.environ.get("NIGHTLY_PIPELINE_PREVIOUS_ARTIFACTS_DIR")
    current_artifact_file = artifacts_dir / artifact_filename
    previous_artifact_file = None
    if previous_artifacts_dir is not None:
        previous_artifacts_path = Path(previous_artifacts_dir).expanduser().resolve()
        if previous_artifacts_path.is_dir():
            previous_artifact_file = previous_artifacts_path / artifact_filename
    output_csv_file = artifacts_dir / csv_filename

    if not current_artifact_file.exists():
        rows = load_recorded_test_failure_rows(artifacts_dir, model_class)
        assert rows, f"Current nightly artifact file is missing: {current_artifact_file}"
        write_validation_csv(output_csv_file, model_class, rows)
        assert all_rows_passed(rows), _failure_summary(model_class, rows)
        return
    if previous_artifact_file is not None:
        assert previous_artifact_file.exists(), f"Previous nightly artifact file is missing: {previous_artifact_file}"

    tolerances = load_validation_tolerances(get_pipeline_config, model_class)
    assert isinstance(tolerances, ValidationTolerances)

    rows = validate_artifact_file(
        current_artifact_file, previous_artifact_file, output_csv_file, model_class, tolerances
    )

    assert output_csv_file.exists(), f"Validation CSV was not created: {output_csv_file}"
    assert all_rows_passed(rows), _failure_summary(model_class, rows)


def _failure_summary(model_class, rows):
    failures = [f"{row['model_name']}: {row['failure_reason']}" for row in rows if row.get("status") != "passed"]
    return f"Nightly validation failed for {model_class}: " + " | ".join(failures)


def _causal_payload(generated_ids):
    return {
        "export_time": 1.0,
        "compile_time": 1.0,
        "size": "1 MB",
        "generated_texts": "hello",
        "generated_ids": generated_ids,
        "perf_metrics": {
            "prefill_time": 1.0,
            "decode_perf": 1.0,
            "total_perf": 1.0,
            "total_time": 1.0,
        },
    }


def test_older_model_validation_failure_is_warning():
    rows = validate_artifacts(
        {"openai-community/gpt2": _causal_payload([100, 101])},
        {"openai-community/gpt2": _causal_payload([1, 2])},
        "causal_pipeline_configs",
        ValidationTolerances(token_mad_tolerance=0.0),
    )

    assert rows[0]["model_age"] == "older"
    assert rows[0]["status"] == "warning"
    assert all_rows_passed(rows)


def test_newer_model_validation_failure_is_failed():
    rows = validate_artifacts(
        {"Qwen/Qwen3-30B-A3B-Instruct-2507": _causal_payload([100, 101])},
        {"Qwen/Qwen3-30B-A3B-Instruct-2507": _causal_payload([1, 2])},
        "causal_pipeline_configs",
        ValidationTolerances(token_mad_tolerance=0.0),
    )

    assert rows[0]["model_age"] == "newer"
    assert rows[0]["status"] == "failed"
    assert not all_rows_passed(rows)


def test_report_layout_and_warning_summary(tmp_path):
    class_rows = {
        "causal_model": [
            {"model_name": "openai-community/gpt2", "model_age": "older", "status": "warning", "failure_reason": "old"},
            {
                "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "model_age": "newer",
                "status": "passed",
                "failure_reason": "",
            },
        ]
    }
    summary = build_summary(class_rows)
    metadata = {
        "status": "passed_with_warnings",
        "job_name": "nightly",
        "build_number": "1",
        "build_url": "http://jenkins/build/1",
        "node_name": "node",
        "branch": "main",
        "pr_number": "N/A",
        "commit_id": "abc123",
        "docker_image": "qeff:test",
        "artifacts_dir": str(tmp_path),
        "previous_artifacts_dir": "previous",
        "start_time": "start",
        "end_time": "end",
        "total_duration": "1m",
    }
    environment = {
        "qaic_apps_version": "apps",
        "qaic_platform_version": "platform",
        "qaic_factory_version": "factory",
        "qnn_sdk_root": "/qnn",
        "qefficient_version": "1.0",
        "torch_version": "2.0",
        "transformers_version": "4.0",
    }

    html = render_html(class_rows, summary, metadata, environment, tmp_path)

    assert "Failure Spotlight" not in html
    assert "Performance Regression Watch" not in html
    assert "Current-only Comparisons" not in html
    assert "Build and SDK Details" in html
    assert "Warnings" in html
    assert html.index("Model Class Details") < html.index("Validation Summary")
