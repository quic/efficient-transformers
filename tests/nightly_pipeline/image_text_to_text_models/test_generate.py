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
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    TextStreamer,
)

from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from QEfficient.utils.test_utils import InternProcessor, ModelConfig

from ..nightly_utils import NIGHTLY_SKIPPED_MODELS, get_onnx_and_qpc_size

model_config_path = os.path.join(os.path.dirname(__file__), "../configs/validated_models.json")
with open(model_config_path, "r") as f:
    config = json.load(f)

test_models = config["image_text_to_text_models"]


@pytest.mark.parametrize("model_name", test_models)
@pytest.mark.parametrize("kv_offload", [True])
def test_generate_image_text_to_text_model(
    model_name, kv_offload, image_text_to_text_model_artifacts, get_model_config
):

    if model_name in NIGHTLY_SKIPPED_MODELS:
        pytest.skip(f"Skipping {model_name} as it is in nightly skipped models list.")

    config, pipeline_configs = get_model_config
    compile_params = pipeline_configs["image_text_to_text_model_configs"][0].get("compile_params", {})
    generate_params = pipeline_configs["image_text_to_text_model_configs"][0].get("generate_params", {})
    img_url = generate_params.pop("image_url", None)
    query = generate_params.pop("query", None)
    generation_len = generate_params.get("generation_len", 25)

    # Retrieve onnx_path from previous stage
    if (
        model_name not in image_text_to_text_model_artifacts
        or "onnx_path" not in image_text_to_text_model_artifacts[model_name]
    ):
        pytest.skip(f"ONNX path not available for {model_name}. Run test_export.py first.")

    # Retrieve qpc_path from previous stage
    if (
        model_name not in image_text_to_text_model_artifacts
        or "qpc_path" not in image_text_to_text_model_artifacts[model_name]
    ):
        pytest.skip(f"QPC path not available for {model_name}. Run test_compile.py first.")

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
        config = qeff_model.model.config
        img_size = 336
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "image_size"):
            img_size = config.vision_config.image_size
        compile_params["img_size"] = img_size
    if kv_offload:
        _ = qeff_model.compile(vision_onnx_path=onnx_path[0], lang_onnx_path=onnx_path[1], **compile_params)
    else:
        _ = qeff_model.compile(onnx_path=onnx_path, **compile_params)

    if model_name in ModelConfig.INTERNVL_MODELS:
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
        batch_size, prompt_len = inputs["input_ids"].shape
        inputs["pixel_values"] = pixel_values.clone()
        generation_config = dict(max_new_tokens=generation_len, do_sample=False)
        generation_config["eos_token_id"] = tokenizer.convert_tokens_to_ids("<|im_end|>\n".strip())

    elif model_name in ModelConfig.MOLMO_MODELS:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
        img = requests.get(img_url, stream=True)
        image = Image.open(BytesIO(img.content)).convert("RGB")
        image = image.resize((536, 354))
        inputs = processor.process(images=[image], text=query)
        inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
        generation_config = GenerationConfig(max_new_tokens=generation_len, stop_strings="<|endoftext|>")
        batch_size, prompt_len = inputs["input_ids"].shape
        inputs["attention_mask"] = torch.ones((inputs["input_ids"].shape), dtype=torch.int64)
        valid = inputs["image_input_idx"] > 0
        valid = valid.reshape(1, -1)
        inputs["valid_idx"] = torch.nonzero(valid)[:, 1].unsqueeze(0)
        inputs["pixel_values"] = inputs.pop("images")

    else:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
        image = Image.open(requests.get(img_url, stream=True).raw)
        if model_name == "mistralai/Mistral-Small-3.1-24B-Instruct-2503":
            image = image.resize((1540, 1540))
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
            inputs["pixel_values"] = inputs["pixel_values"].to(qeff_model.model.config.torch_dtype)
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        if hasattr(qeff_model.model.config, "model_type") and qeff_model.model.config.model_type in [
            "qwen2_5_vl",
            "qwen3_vl",
            "qwen3_vl_moe",
        ]:
            inputs = qeff_model.model.prepare_inputs_for_generation(
                inputs=inputs, prefill_seq_len=prompt_len, batch_size=batch_size
            )
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(qeff_model.model.config.torch_dtype)

    streamer = TextStreamer(processor.tokenizer)
    print("QPC Outputs (QAIC):")
    exec_info = qeff_model.generate(inputs=inputs, streamer=streamer, **generate_params)
    print(exec_info)
    cloud_ai_100_tokens = exec_info.generated_ids[:, :-1]

    if kv_offload:
        encoder_onnx_and_qpc_dir = os.path.dirname(onnx_path[0])
        encoder_onnx_and_qpc_dir_size = get_onnx_and_qpc_size(encoder_onnx_and_qpc_dir)
        decoder_onnx_and_qpc_dir = os.path.dirname(onnx_path[1])
        decoder_onnx_and_qpc_dir_size = get_onnx_and_qpc_size(decoder_onnx_and_qpc_dir)
    else:
        decoder_onnx_and_qpc_dir = os.path.dirname(onnx_path)
        decoder_onnx_and_qpc_dir_size = get_onnx_and_qpc_size(decoder_onnx_and_qpc_dir)

    # Store all metrics and execution info
    image_text_to_text_model_artifacts[model_name].update(
        {
            "batch_size": exec_info.batch_size,
            "generated_ids": cloud_ai_100_tokens,
            "encoder_onnx_and_qpc_dir": encoder_onnx_and_qpc_dir,
            "encoder_onnx_and_qpc_dir size": encoder_onnx_and_qpc_dir_size,
            "decoder_onnx_and_qpc_dir": decoder_onnx_and_qpc_dir,
            "decoder_onnx_and_qpc_dir size": decoder_onnx_and_qpc_dir_size,
            "perf_metrics": {
                "prefill_time": exec_info.perf_metrics.prefill_time,
                "decode_perf": exec_info.perf_metrics.decode_perf,
                "total_perf": exec_info.perf_metrics.total_perf,
                "total_time": exec_info.perf_metrics.total_time,
            },
        }
    )
