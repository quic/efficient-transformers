import json
import os
import time

import pytest

from QEfficient import QEFFAutoModelForCausalLM

model_config_path = os.path.join(os.path.dirname(__file__), "../configs/validated_models.json")
with open(model_config_path, "r") as f:
    config = json.load(f)

test_models = config["causal_lm_models"]

pipeline_config_path = os.path.join(os.path.dirname(__file__), "../configs/pipeline_configs.json")
with open(pipeline_config_path, "r") as f:
    pipeline_configs = json.load(f)

compile_params = pipeline_configs["causal_pipeline_configs"][0].get("compile_params", {})


@pytest.mark.parametrize("model_name", test_models)
def test_compile_causal_lm(model_name, model_artifacts, get_model_config):
    config, pipeline_configs = get_model_config
    compile_params = pipeline_configs["causal_pipeline_configs"][0].get("compile_params", {})

    # Retrieve onnx_path from previous stage
    if model_name not in model_artifacts or "onnx_path" not in model_artifacts[model_name]:
        pytest.skip(f"ONNX path not available for {model_name}. Run test_export.py first.")

    onnx_path = model_artifacts[model_name]["onnx_path"]

    # Compile loading time
    print(f"\nLoading model for compilation: {model_name}")
    compile_load_start = time.time()
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_name)
    compile_loading_time = time.time() - compile_load_start

    # Compile time
    print(f"\nCompiling for model: {model_name}")
    compile_start = time.time()
    qpc_path = qeff_model.compile(onnx_path=onnx_path, **compile_params)
    compile_time = time.time() - compile_start

    print(f"\nCompilation is done for model: {model_name} and qpc path: {qpc_path}")

    # Store metrics
    model_artifacts[model_name].update(
        {
            "qpc_path": qpc_path,
            "compile_loading_time": compile_loading_time,
            "compile_time": compile_time,
        }
    )
