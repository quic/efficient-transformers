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

    _ = qeff_model.compile(onnx_path=onnx_path, **compile_params)

    print(f"\nGenerating for model: {model_name}")
    exec_info = qeff_model.generate(tokenizer=tokenizer, **generation_params)

    print(f"\nGeneration complete for model: {model_name}")

    def get_onnx_and_qpc_size(dir):
        total_size = 0
        for root, dirs, files in os.walk(dir):
            for name in files:
                file_path = os.path.join(root, name)
                if not os.path.islink(file_path):  # avoid counting symlinks
                    total_size += os.path.getsize(file_path)
        print(f"Total size of {dir}: {total_size} bytes")
        return total_size

    def human_readable(size):
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024

    onnx_and_qpc_dir = os.path.dirname(onnx_path)
    size = get_onnx_and_qpc_size(onnx_and_qpc_dir)
    size = human_readable(size)
    # Store all metrics and execution info
    model_artifacts[model_name].update(
        {
            "batch_size": exec_info.batch_size,
            "generated_texts": exec_info.generated_texts,
            "generated_ids": exec_info.generated_ids[0][0][:20],  # Converted to list by conftest serializer
            "onnx_and_qpc_dir": onnx_and_qpc_dir,
            "size": size,
            "perf_metrics": {
                "prefill_time": exec_info.perf_metrics.prefill_time,
                "decode_perf": exec_info.perf_metrics.decode_perf,
                "total_perf": exec_info.perf_metrics.total_perf,
                "total_time": exec_info.perf_metrics.total_time,
            },
        }
    )
