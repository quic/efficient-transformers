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
    TextStreamer,
)

from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from QEfficient.utils.test_utils import InternProcessor, ModelConfig

from ..model_age_utils import filter_models_for_nightly
from ..nightly_utils import (
    get_execution_modes,
    get_onnx_and_qpc_size,
    is_continuous_batching_mode,
    pre_generate_utils,
)

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


def _expand_to_full_batch(values, full_batch_size):
    if len(values) < full_batch_size:
        return (values * (full_batch_size // len(values) + 1))[:full_batch_size]
    return values[:full_batch_size]


def _apply_cb_compile_overrides(qeff_model, compile_params, execution_mode):
    if not is_continuous_batching_mode(execution_mode):
        return

    model_type = getattr(qeff_model.model.config, "model_type", "")
    if model_type in QWEN_CB_MODEL_TYPES:
        compile_params["prefill_seq_len"] = 64
        compile_params["ctx_len"] = 2048


def _extract_generated_ids(exec_info, execution_mode):
    generated_ids = exec_info.generated_ids
    if is_continuous_batching_mode(execution_mode):
        if isinstance(generated_ids, list):
            generated_ids = generated_ids[0]
        return generated_ids[:, :20]

    first_batch = generated_ids[0]
    if hasattr(first_batch, "ndim") and first_batch.ndim > 1:
        return first_batch[0][:20]
    return first_batch[:20]


@pytest.mark.parametrize("model_name", test_models)
@pytest.mark.parametrize("kv_offload", [True])
@pytest.mark.parametrize("execution_mode", execution_modes)
def test_generate_image_text_to_text_model(
    model_name,
    kv_offload,
    execution_mode,
    image_text_to_text_model_artifacts,
    image_text_to_text_model_cb_artifacts,
    get_pipeline_config,
):
    model_artifacts = _get_artifacts_store(
        execution_mode, image_text_to_text_model_artifacts, image_text_to_text_model_cb_artifacts
    )
    compile_params, generate_params = pre_generate_utils(
        model_name,
        "image_text_to_text_model_configs",
        get_pipeline_config,
        model_artifacts,
        execution_mode=execution_mode,
    )

    img_url = generate_params.pop("image_url", None)
    query = generate_params.pop("query", None)
    prompt_len = compile_params.get("prefill_seq_len", 1)
    batch_size = 1

    onnx_path = model_artifacts[model_name].get("onnx_path")
    cb_mode = is_continuous_batching_mode(execution_mode)

    if model_name in ModelConfig.INTERNVL_MODELS or model_name in ModelConfig.MOLMO_MODELS:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            model_name,
            kv_offload=kv_offload,
            continuous_batching=cb_mode,
            trust_remote_code=True,
        )
    else:
        qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
            model_name,
            kv_offload=kv_offload,
            continuous_batching=cb_mode,
        )

    if model_name in ModelConfig.INTERNVL_MODELS:
        compile_params["num_patches"] = 1
    else:
        config = qeff_model.model.config
        img_size = 336
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "image_size"):
            img_size = config.vision_config.image_size
        compile_params["img_size"] = img_size

    _apply_cb_compile_overrides(qeff_model, compile_params, execution_mode)

    if kv_offload:
        _ = qeff_model.compile(vision_onnx_path=onnx_path[0], lang_onnx_path=onnx_path[1], **compile_params)
    else:
        _ = qeff_model.compile(onnx_path=onnx_path, **compile_params)

    if cb_mode:
        full_batch_size = compile_params.get("full_batch_size")
        if full_batch_size is None:
            raise ValueError("`full_batch_size` is required for continuous batching mode.")

        prompts = generate_params.pop("prompts", None)
        if isinstance(prompts, str):
            prompts = [prompts]
        if not prompts:
            prompts = [query] if query else ["Can you describe the image in detail?"]

        image_urls = generate_params.pop("image_urls", None)
        if isinstance(image_urls, str):
            image_urls = [image_urls]
        if not image_urls:
            image_urls = [img_url] if img_url else []
        if not image_urls:
            raise ValueError(f"No image URLs configured for continuous batching in model: {model_name}")

        prompts = _expand_to_full_batch(prompts, full_batch_size)
        image_urls = _expand_to_full_batch(image_urls, full_batch_size)

        image_height = None
        image_width = None
        if model_name in ModelConfig.INTERNVL_MODELS:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
            model_hf = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            processor = InternProcessor(model_hf, tokenizer)
            image_height = 448
            image_width = 448
        elif model_name in ModelConfig.MOLMO_MODELS:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            image_height = 354
            image_width = 536
        else:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
            use_fast = model_name != "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=use_fast)
            model_type = getattr(qeff_model.model.config, "model_type", "")
            if model_type in QWEN_CB_MODEL_TYPES:
                image_height = 354
                image_width = 536
            if model_name == "mistralai/Mistral-Small-3.1-24B-Instruct-2503":
                image_height = 1540
                image_width = 1540
            if model_name == "ibm-granite/granite-vision-3.2-2b":
                image_height = 1109
                image_width = 1610

        print("QPC Outputs (QAIC):")
        exec_info = qeff_model.generate(
            tokenizer=tokenizer,
            processor=processor,
            prompts=prompts,
            images=image_urls,
            image_height=image_height,
            image_width=image_width,
            **generate_params,
        )
        print(exec_info)
        generated_text = exec_info.generated_texts
        generated_ids = _extract_generated_ids(exec_info, execution_mode)

    else:
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

        elif model_name in ModelConfig.MOLMO_MODELS:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
            img = requests.get(img_url, stream=True)
            image = Image.open(BytesIO(img.content)).convert("RGB")
            image = image.resize((536, 354))
            inputs = processor.process(images=[image], text=query)
            inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
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
            if model_name == "ibm-granite/granite-vision-3.2-2b":
                image = image.resize((1610, 1109))
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
            if hasattr(qeff_model.model.config, "model_type") and qeff_model.model.config.model_type in [
                "qwen2_5_vl",
                "qwen3_vl",
                "qwen3_vl_moe",
                "qwen3_5",
                "qwen3_5_moe",
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
        generated_text = processor.tokenizer.batch_decode(exec_info.generated_ids, skip_special_tokens=True)
        generated_ids = _extract_generated_ids(exec_info, execution_mode)

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
        "generated_ids": generated_ids,  # Converted to list by conftest serializer
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

    model_artifacts[model_name].update(artifacts_update)
