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

from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from QEfficient.utils.test_utils import ModelConfig

from ..model_age_utils import filter_models_for_nightly
from ..nightly_utils import get_execution_modes, is_continuous_batching_mode, pre_export_compile_utils_with_mode

model_config_path = os.path.join(os.path.dirname(__file__), "../configs/validated_models.json")
with open(model_config_path, "r") as f:
    config = json.load(f)

pipeline_config_path = os.path.join(os.path.dirname(__file__), "../configs/pipeline_configs.json")
with open(pipeline_config_path, "r") as f:
    pipeline_config = json.load(f)

test_models = filter_models_for_nightly(config["image_text_to_text_models"], "image_text_to_text_models")
execution_modes = get_execution_modes(pipeline_config, "image_text_to_text_model_configs")
QWEN_CB_MODEL_TYPES = {"qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"}


def _get_artifacts_store(execution_mode, image_text_to_text_model_artifacts, image_text_to_text_model_cb_artifacts):
    if is_continuous_batching_mode(execution_mode):
        return image_text_to_text_model_cb_artifacts
    return image_text_to_text_model_artifacts


def _apply_cb_compile_overrides(qeff_model, compile_params, execution_mode):
    if not is_continuous_batching_mode(execution_mode):
        return

    model_type = getattr(qeff_model.model.config, "model_type", "")
    if model_type in QWEN_CB_MODEL_TYPES:
        compile_params["prefill_seq_len"] = 64
        compile_params["ctx_len"] = 2048


@pytest.mark.parametrize("model_name", test_models)
@pytest.mark.parametrize("kv_offload", [True])
@pytest.mark.parametrize("execution_mode", execution_modes)
def test_export_compile_image_text_to_text_model(
    model_name,
    kv_offload,
    execution_mode,
    image_text_to_text_model_artifacts,
    image_text_to_text_model_cb_artifacts,
    get_pipeline_config,
):
    export_params, compile_params = pre_export_compile_utils_with_mode(
        model_name, "image_text_to_text_model_configs", get_pipeline_config, execution_mode
    )
    model_artifacts = _get_artifacts_store(
        execution_mode, image_text_to_text_model_artifacts, image_text_to_text_model_cb_artifacts
    )

    # Initialize model entry
    if model_name not in model_artifacts:
        model_artifacts[model_name] = {}

    # Export loading time
    print(f"\nLoading model for export: {model_name} ({execution_mode})")
    export_load_start = time.time()
    if model_name in ModelConfig.INTERNVL_MODELS or model_name in ModelConfig.MOLMO_MODELS:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            model_name,
            kv_offload=kv_offload,
            continuous_batching=is_continuous_batching_mode(execution_mode),
            trust_remote_code=True,
        )
    else:
        qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
            model_name,
            kv_offload=kv_offload,
            continuous_batching=is_continuous_batching_mode(execution_mode),
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
    if model_name in ModelConfig.INTERNVL_MODELS:
        compile_params["num_patches"] = 1
    else:
        config = qeff_model.model.config
        img_size = 336
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "image_size"):
            img_size = config.vision_config.image_size
        compile_params["img_size"] = img_size

    _apply_cb_compile_overrides(qeff_model, compile_params, execution_mode)

    print(f"\nCompiling for model: {model_name}")
    compile_start = time.time()
    if kv_offload:
        qpc_path = qeff_model.compile(vision_onnx_path=onnx_path[0], lang_onnx_path=onnx_path[1], **compile_params)
    else:
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
