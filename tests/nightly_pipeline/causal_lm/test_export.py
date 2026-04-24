import json
import os
import time

import pytest

from QEfficient import QEFFAutoModelForCausalLM

model_config_path = os.path.join(os.path.dirname(__file__), "../configs/validated_models.json")
with open(model_config_path, "r") as f:
    config = json.load(f)

test_models = config["causal_lm_models"]


@pytest.mark.parametrize("model_name", test_models)
def test_export_causal_lm(model_name, model_artifacts, get_model_config):
    config, pipeline_configs = get_model_config
    export_params = pipeline_configs["causal_pipeline_configs"][0].get("export_params", {})

    # Initialize model entry
    if model_name not in model_artifacts:
        model_artifacts[model_name] = {}

    # Export loading time
    print(f"\nLoading model for export: {model_name}")
    export_load_start = time.time()
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_name)
    export_loading_time = time.time() - export_load_start

    # Export time
    print(f"\nExporting for model: {model_name}")
    export_start = time.time()
    onnx_path = qeff_model.export(**export_params)
    export_time = time.time() - export_start

    print(f"\nExport is done for model: {model_name} and onnx_path: {onnx_path}")

    # Store metrics
    model_artifacts[model_name].update(
        {
            "onnx_path": onnx_path,
            "export_loading_time": export_loading_time,
            "export_time": export_time,
        }
    )
