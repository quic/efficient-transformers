# -----------------------------------------------------------------------------

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# -----------------------------------------------------------------------------


import json
import os
import time

import pytest

from QEfficient import QEFFAutoModelForSequenceClassification

model_config_path = os.path.join(os.path.dirname(__file__), "../configs/validated_models.json")
with open(model_config_path, "r") as f:
    config = json.load(f)

test_models = config["sequence_models"]


@pytest.mark.parametrize("model_name", test_models)
def test_export_compile_sequence_model(model_name, get_model_config, sequence_model_artifacts):
    """Test export and compile for sequnce models."""
    config, pipeline_configs = get_model_config
    export_params = pipeline_configs["sequence_model_configs"][0].get("export_params", {})
    compile_params = pipeline_configs["sequence_model_configs"][0].get("compile_params", {})

    # Initialize model entry
    if model_name not in sequence_model_artifacts:
        sequence_model_artifacts[model_name] = {}

    # Export loading time
    print(f"\nLoading model for export: {model_name}")
    export_load_start = time.time()
    qeff_model = QEFFAutoModelForSequenceClassification.from_pretrained(model_name)
    export_loading_time = time.time() - export_load_start

    # Export time
    print(f"\nExporting for model: {model_name}")
    export_start = time.time()
    onnx_path = qeff_model.export(**export_params)
    export_time = time.time() - export_start
    print(f"\nExport is done for model: {model_name} and onnx_path: {onnx_path}")

    # Compile
    print(f"\nCompiling for model: {model_name}")
    compile_start = time.time()
    qpc_path = qeff_model.compile(onnx_path=onnx_path, **compile_params)
    compile_time = time.time() - compile_start
    print(f"\nCompilation is done for model: {model_name} and qpc path: {qpc_path}")

    sequence_model_artifacts[model_name].update(
        {
            "export_loading_time": export_loading_time,
            "onnx_path": onnx_path,
            "export_time": export_time,
            "qpc_path": qpc_path,
            "compile_time": compile_time,
        }
    )
