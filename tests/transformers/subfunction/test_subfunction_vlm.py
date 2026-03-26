# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import copy
import json
from typing import Optional

import onnx
import pytest
import requests
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
)

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText
from QEfficient.utils._utils import get_num_layers_vlm
from QEfficient.utils.test_utils import load_vlm_model, load_vlm_model_from_config

NEW_GENERATION_TOKENS = 10


CONFIG_PATH = "tests/configs/image_text_model_configs.json"

with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    multimodal_models = config_data["image_text_subfunction_models"]

test_mm_models = [model_config["model_name"] for model_config in multimodal_models]
model_config_dict = {model["model_name"]: model for model in multimodal_models}


def has_QwenLayer_function(onnx_path):
    """Check if ONNX model contains QEffqwenlayer function definition."""
    model = onnx.load(onnx_path, load_external_data=False)
    function_names = [f.name for f in model.functions]
    QwenLayer_functions = [name for name in function_names if "QEffQwen2_5_VLDecoderLayer" in name]
    return len(QwenLayer_functions) > 0, QwenLayer_functions


def check_image_text_to_text_subfunction_core(
    model_name: str,
    img_size: int,
    img_url: str,
    query: str,
    prompt_len: int,
    ctx_len: int,
    max_gen_len: int = 20,
    batch_size: int = 1,
    n_layer: int = 1,
    kv_offload: bool = False,
    num_devices: int = 1,
    enable_qnn: Optional[bool] = False,
    qnn_config: Optional[str] = None,
    config: Optional[AutoConfig] = None,
):
    if config is None:
        model_config = {"model_name": model_name}
        model_config["img_size"] = img_size
        config = AutoConfig.from_pretrained(model_config["model_name"], trust_remote_code=True, padding=True)
        config.text_config.num_hidden_layers = n_layer
        config.vision_config.num_hidden_layers = n_layer
        model_hf = load_vlm_model(config)
        qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
            model_name,
            kv_offload=kv_offload,
            config=config,
        )
    else:
        model_hf = load_vlm_model_from_config(config)
        qeff_model = QEFFAutoModelForImageTextToText(
            copy.deepcopy(model_hf),
            kv_offload=kv_offload,
            config=config,
        )

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
    n_layer = get_num_layers_vlm(config)
    image = Image.open(requests.get(img_url, stream=True).raw)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

    with_sub_func_onnx = qeff_model.export(use_onnx_subfunctions=True, offload_pt_weights=False)

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    if hasattr(qeff_model.model.config, "model_type") and qeff_model.model.config.model_type == "qwen2_5_vl":
        inputs = qeff_model.model.prepare_inputs_for_generation(
            inputs=inputs, prefill_seq_len=prompt_len, batch_size=batch_size
        )
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

    # Verify that the model with subfunctions has QEffQwen2_5_VLDecoderLayer function definition
    has_qwenlayer, qwenlayer_names = has_QwenLayer_function(with_sub_func_onnx[-1])
    assert has_qwenlayer, (
        "Model exported with use_onnx_subfunctions=True should contain QEffQwen2_5_VLDecoderLayer function definition"
    )
    print(f"\nQwenLayer functions found: {qwenlayer_names}")

    qeff_model.compile(
        img_size=img_size,
        num_devices=num_devices,
        prefill_seq_len=prompt_len,
        ctx_len=ctx_len,
        mxfp6=False,
        enable_qnn=enable_qnn,
        qnn_config=qnn_config,
    )


@pytest.mark.feature
@pytest.mark.regular
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True])
def test_custom_image_text_to_text_subfunction(model_name, kv_offload):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model,  without continuous batching with subfunction.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``Qwen/Qwen2.5-VL-3B-Instruct``
    """
    torch.manual_seed(42)
    img_size = model_config_dict[model_name].get("img_size")
    custom_config = model_config_dict[model_name].get("additional_params", {})
    model_type = model_config_dict[model_name].get("model_type", None)
    hf_config = AutoConfig.for_model(model_type, trust_remote_code=True, **custom_config)
    hf_config.name_or_path = model_name
    check_image_text_to_text_subfunction_core(
        model_name=model_name,
        prompt_len=model_config_dict[model_name]["prompt_len"],
        ctx_len=model_config_dict[model_name]["ctx_len"],
        max_gen_len=NEW_GENERATION_TOKENS,
        img_size=img_size,
        img_url=model_config_dict[model_name]["img_url"],
        query=model_config_dict[model_name]["text_prompt"],
        n_layer=model_config_dict[model_name]["num_layers"],
        batch_size=model_config_dict[model_name]["batch_size"],
        kv_offload=kv_offload,
        config=hf_config,
    )


@pytest.mark.feature
@pytest.mark.nightly
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True])
def test_image_text_to_text_subfunction(model_name, kv_offload):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model,  without continuous batching with subfunction.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``Qwen/Qwen2.5-VL-3B-Instruct``
    """
    torch.manual_seed(42)
    img_size = model_config_dict[model_name].get("img_size")
    check_image_text_to_text_subfunction_core(
        model_name=model_name,
        prompt_len=model_config_dict[model_name]["prompt_len"],
        ctx_len=model_config_dict[model_name]["ctx_len"],
        max_gen_len=NEW_GENERATION_TOKENS,
        img_size=img_size,
        img_url=model_config_dict[model_name]["img_url"],
        query=model_config_dict[model_name]["text_prompt"],
        n_layer=model_config_dict[model_name]["num_layers"],
        batch_size=model_config_dict[model_name]["batch_size"],
        kv_offload=kv_offload,
    )
