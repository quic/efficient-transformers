# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import json
import os
from io import BytesIO
from typing import List

import pytest
import requests
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    TextStreamer,
)

from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from QEfficient.utils.test_utils import InternProcessor, ModelConfig

from ..nightly_utils import get_onnx_and_qpc_size, pre_generate_utils

model_config_path = os.path.join(os.path.dirname(__file__), "../configs/validated_models.json")
with open(model_config_path, "r") as f:
    config = json.load(f)

test_models = config["image_text_to_text_models"]

# Model types that use the newer unified processor API (apply_chat_template returns pixel_values)
_UNIFIED_PROCESSOR_MODEL_TYPES = {
    "llama4",
    "gemma3",
    "gemma4",
}

# Model types that require the qwen_vl_utils.process_vision_info path
_QWEN_VL_MODEL_TYPES = {
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen3_5",
    "qwen3_5_moe",
}

# Model types that use the legacy processor(image, text) API
_LEGACY_PROCESSOR_MODEL_TYPES = {
    "mistral3",
    "llava",
    "llava_next",
}

# Image resize dimensions required by specific models (width, height)
_MODEL_IMAGE_SIZES = {
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": (1540, 1540),
    "ibm-granite/granite-vision-3.2-2b": (1610, 1109),
}


def _prepare_internvl_inputs(
    model_name: str,
    img_url: str,
    query: str,
):
    """Preprocessing for InternVL models (early-fusion, CausalLM-based)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    model_hf = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    processor = InternProcessor(model_hf, tokenizer)
    prompt = [query]
    img_url_list = [img_url]
    pixel_values = []
    num_patches_list = []
    questions = []
    for i in range(len(prompt)):
        img = requests.get(img_url_list[i], stream=True)
        image = Image.open(BytesIO(img.content)).convert("RGB")
        image = image.resize((448, 448))
        pixel_value = processor.load_image(image, max_num=12)
        num_patches_list.append(pixel_value.shape[0])
        pixel_values.append(pixel_value)
        question = "<image>\n" + prompt[i]
        questions.append(question)

    pixel_values = torch.cat(pixel_values, dim=0)
    messages: List[List[str]] = []
    roles = ("<|im_start|>user\n", "<|im_start|>assistant\n")
    prompt = processor(pixel_values, questions, messages, roles, num_patches_list=num_patches_list)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs["pixel_values"] = pixel_values.clone()
    return inputs, processor


def _prepare_molmo_inputs(
    model_name: str,
    img_url: str,
    query: str,
):
    """Preprocessing for Molmo models (custom processor with trust_remote_code)."""
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
    img = requests.get(img_url, stream=True)
    image = Image.open(BytesIO(img.content)).convert("RGB")
    image = image.resize((536, 354))

    inputs = processor.process(images=[image], text=query)
    inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
    inputs["attention_mask"] = torch.ones(inputs["input_ids"].shape, dtype=torch.int64)
    valid = inputs["image_input_idx"] > 0
    valid = valid.reshape(1, -1)
    inputs["valid_idx"] = torch.nonzero(valid)[:, 1].unsqueeze(0)
    inputs["pixel_values"] = inputs.pop("images")
    return inputs, processor


def _prepare_unified_processor_inputs(
    model_name: str,
    qeff_model,
    img_url: str,
    query: str,
):
    """
    Preprocessing for models whose processor.apply_chat_template handles image encoding
    (Llama-4, Gemma-3). Image URL is embedded directly in the message dict.
    """
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": img_url},
                {"type": "text", "text": query},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(qeff_model.model.config.torch_dtype)
    return inputs, processor


def _prepare_qwen_vl_inputs(
    model_name: str,
    qeff_model,
    img_url: str,
    query: str,
    prompt_len: int,
    batch_size: int,
):
    """
    Preprocessing for Qwen VL / Qwen3.5 models. Uses qwen_vl_utils.process_vision_info
    to extract visual tokens separately before calling the processor.
    """
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
    img = requests.get(img_url, stream=True)
    image = Image.open(BytesIO(img.content)).convert("RGB")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query},
            ],
        },
    ]
    texts = [processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)]
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs=inputs, prefill_seq_len=prompt_len, batch_size=batch_size
    )
    return inputs, processor


def _prepare_legacy_processor_inputs(
    model_name: str,
    qeff_model,
    img_url: str,
    query: str,
):
    """
    Preprocessing for models using the legacy processor(image, text) API
    (Mistral-3, LLaVA, Granite).
    """
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
    image = Image.open(requests.get(img_url, stream=True).raw)
    if model_name in _MODEL_IMAGE_SIZES:
        image = image.resize(_MODEL_IMAGE_SIZES[model_name])

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": query},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt", add_special_tokens=False)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(qeff_model.model.config.torch_dtype)
    return inputs, processor


@pytest.mark.parametrize("model_name", test_models)
@pytest.mark.parametrize("kv_offload", [True])
def test_generate_image_text_to_text_model(
    model_name, kv_offload, image_text_to_text_model_artifacts, get_pipeline_config
):
    compile_params, generate_params = pre_generate_utils(
        model_name, "image_text_to_text_model_configs", get_pipeline_config, image_text_to_text_model_artifacts
    )

    img_url = generate_params.pop("image_url", None)
    query = generate_params.pop("query", None)
    prompt_len = compile_params.get("prefill_seq_len", 1)
    batch_size = 1

    onnx_path = image_text_to_text_model_artifacts[model_name].get("onnx_path")

    if model_name in ModelConfig.INTERNVL_MODELS or model_name in ModelConfig.MOLMO_MODELS:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            model_name,
            kv_offload=kv_offload,
            trust_remote_code=True,
        )
    else:
        qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
            model_name,
            kv_offload=kv_offload,
        )

    if model_name in ModelConfig.INTERNVL_MODELS:
        compile_params["num_patches"] = 1
    else:
        model_cfg = qeff_model.model.config
        img_size = 336
        if hasattr(model_cfg, "vision_config") and hasattr(model_cfg.vision_config, "image_size"):
            img_size = model_cfg.vision_config.image_size
        compile_params["img_size"] = img_size

    if kv_offload:
        _ = qeff_model.compile(vision_onnx_path=onnx_path[0], lang_onnx_path=onnx_path[1], **compile_params)
    else:
        _ = qeff_model.compile(onnx_path=onnx_path, **compile_params)

    # --- Preprocessing: dispatch to the appropriate helper ---
    model_type = getattr(qeff_model.model.config, "model_type", "")

    if model_name in ModelConfig.INTERNVL_MODELS:
        inputs, processor = _prepare_internvl_inputs(
            model_name, img_url, query
        )
    elif model_name in ModelConfig.MOLMO_MODELS:
        inputs, processor = _prepare_molmo_inputs(
            model_name, img_url, query
        )
    elif model_type in _UNIFIED_PROCESSOR_MODEL_TYPES:
        inputs, processor = _prepare_unified_processor_inputs(
            model_name, qeff_model, img_url, query
        )
    elif model_type in _QWEN_VL_MODEL_TYPES:
        inputs, processor = _prepare_qwen_vl_inputs(
            model_name, qeff_model, img_url, query, prompt_len, batch_size
        )
    elif model_type in _LEGACY_PROCESSOR_MODEL_TYPES:
        inputs, processor = _prepare_legacy_processor_inputs(
            model_name, qeff_model, img_url, query
        )
    else:
        raise ValueError(
            f"No preprocessing path defined for model '{model_name}' with model_type='{model_type}'. "
            "Add an entry to one of the model-type sets or add a dedicated helper."
        )

    streamer = TextStreamer(processor.tokenizer)
    print("QPC Outputs (QAIC):")
    exec_info = qeff_model.generate(inputs=inputs, streamer=streamer, **generate_params)
    print(exec_info)
    generated_text = processor.tokenizer.batch_decode(exec_info.generated_ids, skip_special_tokens=True)
    cloud_ai_100_tokens = exec_info.generated_ids[:, :-1]

    encoder_onnx_and_qpc_dir = None
    encoder_onnx_and_qpc_dir_size = None
    decoder_onnx_and_qpc_dir = None
    decoder_onnx_and_qpc_dir_size = None

    if kv_offload:
        encoder_onnx_and_qpc_dir = os.path.dirname(onnx_path[0])
        encoder_onnx_and_qpc_dir_size = get_onnx_and_qpc_size(encoder_onnx_and_qpc_dir)
        decoder_onnx_and_qpc_dir = os.path.dirname(onnx_path[1])
        decoder_onnx_and_qpc_dir_size = get_onnx_and_qpc_size(decoder_onnx_and_qpc_dir)
    else:
        decoder_onnx_and_qpc_dir = os.path.dirname(onnx_path)
        decoder_onnx_and_qpc_dir_size = get_onnx_and_qpc_size(decoder_onnx_and_qpc_dir)

    # Store all metrics and execution info
    artifacts_update = {
        "batch_size": exec_info.batch_size,
        "generated_text": generated_text,
        "generated_ids": cloud_ai_100_tokens,
        "decoder_onnx_and_qpc_dir": decoder_onnx_and_qpc_dir,
        "decoder_onnx_and_qpc_dir size": decoder_onnx_and_qpc_dir_size,
        "perf_metrics": {
            "prefill_time": exec_info.perf_metrics.prefill_time,
            "decode_perf": exec_info.perf_metrics.decode_perf,
            "total_perf": exec_info.perf_metrics.total_perf,
            "total_time": exec_info.perf_metrics.total_time,
        },
    }

    if kv_offload:
        artifacts_update["encoder_onnx_and_qpc_dir"] = encoder_onnx_and_qpc_dir
        artifacts_update["encoder_onnx_and_qpc_dir size"] = encoder_onnx_and_qpc_dir_size

    image_text_to_text_model_artifacts[model_name].update(artifacts_update)
