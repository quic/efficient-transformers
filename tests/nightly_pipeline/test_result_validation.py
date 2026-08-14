# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
from pathlib import Path

import pytest

from .nightly_utils import get_execution_modes
from .result_validator import (
    ValidationTolerances,
    all_rows_passed,
    load_recorded_test_failure_rows,
    load_validation_tolerances,
    validate_artifact_file,
    write_validation_csv,
)

PIPELINE_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "pipeline_configs.json"
with open(PIPELINE_CONFIG_PATH, "r") as pipeline_config_file:
    PIPELINE_CONFIG = json.load(pipeline_config_file)

MODE_ARTIFACTS = {
    "causal_pipeline_configs": {
        "non_cb": ("causal_model_artifacts.json", "causal_model_validation.csv"),
        "cb": ("causal_model_cb_artifacts.json", "causal_model_cb_validation.csv"),
    },
    "image_text_to_text_model_configs": {
        "non_cb": ("image_text_to_text_model_artifacts.json", "image_text_to_text_model_validation.csv"),
        "cb": ("image_text_to_text_model_cb_artifacts.json", "image_text_to_text_model_cb_validation.csv"),
    },
}

ALWAYS_ON_MODEL_ARTIFACTS = [
    ("embedding_model_configs", "embedding_model_artifacts.json", "embedding_model_validation.csv"),
    ("audio_model_configs", "audio_model_artifacts.json", "audio_model_validation.csv"),
    ("audio_embedding_model_configs", "audio_embedding_model_artifacts.json", "audio_embedding_model_validation.csv"),
    ("sequence_model_configs", "sequence_model_artifacts.json", "sequence_model_validation.csv"),
]


def _resolve_mode_artifacts(model_class):
    mode_artifacts = MODE_ARTIFACTS[model_class]
    resolved = []
    for execution_mode in get_execution_modes(PIPELINE_CONFIG, model_class):
        artifact_entry = mode_artifacts.get(execution_mode)
        if artifact_entry is None:
            raise KeyError(f"Missing validation artifact mapping for mode '{execution_mode}' in '{model_class}'")
        artifact_filename, csv_filename = artifact_entry
        resolved.append((model_class, artifact_filename, csv_filename))
    return resolved


MODEL_ARTIFACTS = [
    *_resolve_mode_artifacts("causal_pipeline_configs"),
    *ALWAYS_ON_MODEL_ARTIFACTS,
    *_resolve_mode_artifacts("image_text_to_text_model_configs"),
]

STATUS_RANK = {"passed": 0, "warning": 1, "failed": 2}


def _merge_rows_with_recorded_failures(rows, failure_rows):
    merged_rows = {row.get("model_name"): dict(row) for row in rows if row.get("model_name")}
    ordered_model_names = [row.get("model_name") for row in rows if row.get("model_name")]

    for failure_row in failure_rows:
        model_name = failure_row.get("model_name")
        if not model_name:
            continue

        row = merged_rows.get(model_name)
        if row is None:
            row = {"model_name": model_name}
            merged_rows[model_name] = row
            ordered_model_names.append(model_name)

        current_status = str(row.get("status", "passed")).lower()
        failure_status = str(failure_row.get("status", "warning")).lower()
        if STATUS_RANK.get(failure_status, 0) >= STATUS_RANK.get(current_status, 0):
            row["status"] = failure_status

        row["model_age"] = failure_row.get("model_age", row.get("model_age", "unknown"))

        failure_reason = str(failure_row.get("failure_reason") or "pytest test failed")
        current_reason = str(row.get("failure_reason") or "")
        if current_reason and current_reason not in {"N/A", failure_reason}:
            row["failure_reason"] = f"{failure_reason}; {current_reason}"
        else:
            row["failure_reason"] = failure_reason

    return [merged_rows[model_name] for model_name in ordered_model_names]


@pytest.mark.nightly
@pytest.mark.parametrize("model_class, artifact_filename, csv_filename", MODEL_ARTIFACTS)
def test_validate_nightly_results(model_class, artifact_filename, csv_filename, artifacts_dir, get_pipeline_config):
    previous_artifacts_dir = os.environ.get("NIGHTLY_PIPELINE_PREVIOUS_ARTIFACTS_DIR", None)
    if previous_artifacts_dir is not None:
        previous_artifacts_dir = previous_artifacts_dir.strip()
    model_mode = "cb" if "_cb_" in artifact_filename else "non_cb"
    failure_rows = load_recorded_test_failure_rows(artifacts_dir, model_class, model_mode=model_mode)
    current_artifact_file = artifacts_dir / artifact_filename
    previous_artifact_file = None
    if previous_artifacts_dir:
        previous_artifacts_path = Path(previous_artifacts_dir).expanduser().resolve()
        assert previous_artifacts_path.is_dir(), (
            "NIGHTLY_PIPELINE_PREVIOUS_ARTIFACTS_DIR must point to an existing directory. "
            f"Received: {previous_artifacts_path}"
        )
        previous_artifact_file = previous_artifacts_path / artifact_filename
    output_csv_file = artifacts_dir / csv_filename

    if not current_artifact_file.exists():
        rows = _merge_rows_with_recorded_failures([], failure_rows)
        assert rows, f"Current nightly artifact file is missing: {current_artifact_file}"
        write_validation_csv(output_csv_file, model_class, rows)
        assert all_rows_passed(rows), _failure_summary(model_class, rows)
        return
    if previous_artifact_file is not None and not previous_artifact_file.exists():
        previous_artifact_file = None

    tolerances = load_validation_tolerances(get_pipeline_config, model_class)
    assert isinstance(tolerances, ValidationTolerances)

    rows = validate_artifact_file(
        current_artifact_file, previous_artifact_file, output_csv_file, model_class, tolerances
    )
    if failure_rows:
        rows = _merge_rows_with_recorded_failures(rows, failure_rows)
        write_validation_csv(output_csv_file, model_class, rows)

    assert output_csv_file.exists(), f"Validation CSV was not created: {output_csv_file}"
    assert all_rows_passed(rows), _failure_summary(model_class, rows)


def _failure_summary(model_class, rows):
    failures = [f"{row['model_name']}: {row['failure_reason']}" for row in rows if row.get("status") != "passed"]
    return f"Nightly validation failed for {model_class}: " + " | ".join(failures)
