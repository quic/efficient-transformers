# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
import time

import pytest

from QEfficient import QEFFAutoModelForSpeechSeq2Seq

from ..nightly_utils import pre_export_compile_utils

model_config_path = os.path.join(os.path.dirname(__file__), "../configs/validated_models.json")
with open(model_config_path, "r") as f:
    config = json.load(f)

test_models = config["audio_models"]


@pytest.mark.parametrize("model_name", test_models)
def test_export_compile_audio_model(model_name, get_pipeline_config, audio_model_artifacts):
    """Test export and compile of audio model."""

    export_params, compile_params = pre_export_compile_utils(model_name, "audio_model_configs", get_pipeline_config)

    # Initialize model entry
    if model_name not in audio_model_artifacts:
        audio_model_artifacts[model_name] = {}

    # Export loading time
    print(f"\nLoading model for export: {model_name}")
    export_load_start = time.time()
    qeff_model = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    export_loading_time = time.time() - export_load_start
    print(f"\nModel loading is done for model: {model_name} in {export_loading_time:.2f} seconds.")

    # Export time
    print(f"\nExporting for model: {model_name}")
    export_start = time.time()
    onnx_path = qeff_model.export(**export_params)
    export_time = time.time() - export_start
    print(f"\nExport is done for model: {model_name} and onnx_path: {onnx_path} in {export_time:.2f} seconds.")

    # Compile
    print(f"\nCompiling for model: {model_name}")
    compile_start = time.time()
    qpc_path = qeff_model.compile(onnx_path=onnx_path, **compile_params)
    compile_time = time.time() - compile_start
    print(f"\nCompilation is done for model: {model_name} and qpc path: {qpc_path} in {compile_time:.2f} seconds.")

    audio_model_artifacts[model_name].update(
        {
            "export_loading_time": export_loading_time,
            "onnx_path": onnx_path,
            "export_time": export_time,
            "qpc_path": qpc_path,
            "compile_time": compile_time,
        }
    )
