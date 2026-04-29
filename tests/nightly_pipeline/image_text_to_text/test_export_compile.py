import json
import os
import time

import pytest

from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from QEfficient.utils.test_utils import ModelConfig

model_config_path = os.path.join(os.path.dirname(__file__), "../configs/validated_models.json")
with open(model_config_path, "r") as f:
    config = json.load(f)

test_models = config["image_text_to_text_models"]


@pytest.mark.parametrize("model_name", test_models[:1])
@pytest.mark.parametrize("kv_offload", [True])
def test_export_compile_image_text_to_text_model(
    model_name, kv_offload, image_text_to_text_model_artifacts, get_model_config
):

    config, pipeline_configs = get_model_config
    export_params = pipeline_configs["image_text_to_text_model_configs"][0].get("export_params", {})
    compile_params = pipeline_configs["image_text_to_text_model_configs"][0].get("compile_params", {})

    # Initialize model entry
    if model_name not in image_text_to_text_model_artifacts:
        image_text_to_text_model_artifacts[model_name] = {}

    # Export loading time
    print(f"\nLoading model for export: {model_name}")
    export_load_start = time.time()
    if model_name in ModelConfig.INTERNVL_MODELS or model_name in ModelConfig.MOLMO_MODELS:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            model_name,
            kv_offload=kv_offload,
        )
    else:
        qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
            model_name,
            kv_offload=kv_offload,
        )
    export_loading_time = time.time() - export_load_start

    print(qeff_model.model.config)
    # Export time
    print(f"\nExporting for model: {model_name}")
    export_start = time.time()
    onnx_path = qeff_model.export(**export_params)
    export_time = time.time() - export_start
    print(f"\nExport is done for model: {model_name} and onnx_path: {onnx_path}")

    # Compile
    if model_name in ModelConfig.INTERNVL_MODELS:
        compile_params["num_patches"] = 1
    else:
        config = qeff_model.model.config
        img_size = 336
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "image_size"):
            img_size = config.vision_config.image_size
        compile_params["img_size"] = img_size

    print(f"\nCompiling for model: {model_name}")
    compile_start = time.time()
    if kv_offload:
        qpc_path = qeff_model.compile(vision_onnx_path=onnx_path[0], lang_onnx_path=onnx_path[1], **compile_params)
    else:
        qpc_path = qeff_model.compile(onnx_path=onnx_path, **compile_params)
    compile_time = time.time() - compile_start
    print(f"\nCompilation is done for model: {model_name} and qpc path: {qpc_path}")

    # Store metrics
    image_text_to_text_model_artifacts[model_name].update(
        {
            "onnx_path": onnx_path,
            "export_loading_time": export_loading_time,
            "export_time": export_time,
            "qpc_path": qpc_path,
            "compile_time": compile_time,
        }
    )
