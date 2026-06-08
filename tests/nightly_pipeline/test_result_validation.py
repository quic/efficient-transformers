# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from pathlib import Path

import pytest

from .result_validator import ValidationTolerances, all_rows_passed, load_validation_tolerances, validate_artifact_file

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

    assert current_artifact_file.exists(), f"Current nightly artifact file is missing: {current_artifact_file}"
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
