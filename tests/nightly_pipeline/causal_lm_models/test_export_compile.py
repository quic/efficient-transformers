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

from QEfficient import QEFFAutoModelForCausalLM

from ..model_age_utils import filter_models_for_nightly
from ..nightly_utils import get_execution_modes, is_continuous_batching_mode, pre_export_compile_utils_with_mode

model_config_path = os.path.join(os.path.dirname(__file__), "../configs/validated_models.json")
with open(model_config_path, "r") as f:
    config = json.load(f)

pipeline_config_path = os.path.join(os.path.dirname(__file__), "../configs/pipeline_configs.json")
with open(pipeline_config_path, "r") as f:
    pipeline_config = json.load(f)

test_models = filter_models_for_nightly(config["causal_lm_models"], "causal_lm_models")
execution_modes = get_execution_modes(pipeline_config, "causal_pipeline_configs")


def _get_artifacts_store(execution_mode, causal_model_artifacts, causal_model_cb_artifacts):
    if is_continuous_batching_mode(execution_mode):
        return causal_model_cb_artifacts
    return causal_model_artifacts


@pytest.mark.parametrize("model_name", test_models)
@pytest.mark.parametrize("execution_mode", execution_modes)
def test_export_compile_causal_lm(
    model_name,
    execution_mode,
    causal_model_artifacts,
    causal_model_cb_artifacts,
    get_pipeline_config,
):
    export_params, compile_params = pre_export_compile_utils_with_mode(
        model_name, "causal_pipeline_configs", get_pipeline_config, execution_mode
    )
    model_artifacts = _get_artifacts_store(execution_mode, causal_model_artifacts, causal_model_cb_artifacts)

    # Initialize model entry
    if model_name not in model_artifacts:
        model_artifacts[model_name] = {}

    # Export loading time
    print(f"\nLoading model for export: {model_name} ({execution_mode})")
    export_load_start = time.time()
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        model_name, continuous_batching=is_continuous_batching_mode(execution_mode)
    )
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

    # Store metrics
    model_artifacts[model_name].update(
        {
            "onnx_path": onnx_path,
            "export_loading_time": export_loading_time,
            "export_time": export_time,
            "qpc_path": qpc_path,
            "compile_time": compile_time,
        }
    )
