import json
import os

import pytest
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

model_config_path = os.path.join(os.path.dirname(__file__), "../configs/validated_models.json")
with open(model_config_path, "r") as f:
    config = json.load(f)

test_models = config["causal_lm_models"]


@pytest.mark.parametrize("model_name", test_models)
def test_generate_causal_lm(model_name, model_artifacts, get_model_config):
    config, pipeline_configs = get_model_config
    compile_params = pipeline_configs["causal_pipeline_configs"][0].get("compile_params", {})

    # Retrieve onnx_path from previous stage
    if model_name not in model_artifacts or "onnx_path" not in model_artifacts[model_name]:
        pytest.skip(f"ONNX path not available for {model_name}. Run test_export.py first.")

    generation_params = pipeline_configs["causal_pipeline_configs"][0].get(
        "generation_params", {"prompt": "My name is"}
    )

    # Retrieve qpc_path from previous stage
    if model_name not in model_artifacts or "qpc_path" not in model_artifacts[model_name]:
        pytest.skip(f"QPC path not available for {model_name}. Run test_compile.py first.")

    _ = model_artifacts[model_name]["qpc_path"]
    onnx_path = model_artifacts[model_name].get("onnx_path")

    print(f"\nLoading model for generation: {model_name}")
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    onnx_path = qeff_model.compile(onnx_path=onnx_path, **compile_params)

    print(f"\nGenerating for model: {model_name}")
    exec_info = qeff_model.generate(tokenizer=tokenizer, **generation_params)

    print(f"\nGeneration complete for model: {model_name}")

    # Store all metrics and execution info
    model_artifacts[model_name].update(
        {
            "batch_size": exec_info.batch_size,
            "generated_texts": exec_info.generated_texts,
            "generated_ids": exec_info.generated_ids[:20],  # Converted to list by conftest serializer
            "perf_metrics": {
                "prefill_time": exec_info.perf_metrics.prefill_time,
                "decode_perf": exec_info.perf_metrics.decode_perf,
                "total_perf": exec_info.perf_metrics.total_perf,
                "total_time": exec_info.perf_metrics.total_time,
            },
        }
    )
