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


@pytest.mark.nightly
@pytest.mark.parametrize("model_class, artifact_filename, csv_filename", MODEL_ARTIFACTS)
def test_validate_nightly_results(model_class, artifact_filename, csv_filename, artifacts_dir, get_pipeline_config):
    previous_artifacts_dir = os.environ.get("NIGHTLY_PIPELINE_PREVIOUS_ARTIFACTS_DIR", None)
    model_mode = "cb" if "_cb_" in artifact_filename else "non_cb"
    current_artifact_file = artifacts_dir / artifact_filename
    previous_artifact_file = None
    if previous_artifacts_dir is not None:
        previous_artifacts_path = Path(previous_artifacts_dir).expanduser().resolve()
        assert previous_artifacts_path.is_dir(), (
            "NIGHTLY_PIPELINE_PREVIOUS_ARTIFACTS_DIR must point to an existing directory. "
            f"Received: {previous_artifacts_path}"
        )
        previous_artifact_file = previous_artifacts_path / artifact_filename
    output_csv_file = artifacts_dir / csv_filename

    if not current_artifact_file.exists():
        rows = load_recorded_test_failure_rows(artifacts_dir, model_class, model_mode=model_mode)
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
