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
import torch

from QEfficient import QEFFAutoModelForCTC

from ..nightly_utils import NIGHTLY_SKIPPED_MODELS

model_config_path = os.path.join(os.path.dirname(__file__), "../configs/validated_models.json")
with open(model_config_path, "r") as f:
    config = json.load(f)

test_models = config["audio_embedding_models"]


@pytest.mark.parametrize("model_name", test_models)
def test_export_compile_audio_embedding_model(model_name, get_model_config, audio_embedding_model_artifacts):

    if model_name in NIGHTLY_SKIPPED_MODELS:
        pytest.skip(f"Skipping {model_name} as it is in nightly skipped models list.")

    config, pipeline_configs = get_model_config
    export_params = pipeline_configs["audio_embedding_model_configs"][0].get("export_params", {})
    compile_params = pipeline_configs["audio_embedding_model_configs"][0].get("compile_params", {})

    # Initialize model entry
    if model_name not in audio_embedding_model_artifacts:
        audio_embedding_model_artifacts[model_name] = {}

    # Export loading time
    print(f"\nLoading model for export: {model_name}")
    export_load_start = time.time()
    qeff_model = QEFFAutoModelForCTC.from_pretrained(model_name, torch_dtype=torch.float32)
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

    audio_embedding_model_artifacts[model_name].update(
        {
            "export_loading_time": export_loading_time,
            "onnx_path": onnx_path,
            "export_time": export_time,
            "qpc_path": qpc_path,
            "compile_time": compile_time,
        }
    )
