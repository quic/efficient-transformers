# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import copy
import json
import os
from io import BytesIO
from typing import Optional

import pytest
import requests
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
)

from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from QEfficient.utils.run_utils import ApiRunnerInternVL, ApiRunnerMolmo, ApiRunnerVlm
from QEfficient.utils.test_utils import (
    InternProcessor,
    ModelConfig,
    load_vlm_model,
    load_vlm_model_from_config,
    set_num_layers_vlm,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../configs/image_text_model_configs.json")
with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    multimodal_models = config_data["image_text_models"]
test_mm_models = [model_config["model_name"] for model_config in multimodal_models]
model_config_dict = {model["model_name"]: model for model in multimodal_models}

NEW_GENERATION_TOKENS = 10


def check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_CB(
    model_name: str,
    manual_cleanup: callable,
    num_hidden_layers: int = -1,
    kv_offload: bool = False,
    num_devices: int = 1,
    enable_qnn: Optional[bool] = False,
    qnn_config: Optional[str] = None,
    config: Optional[AutoConfig] = None,
):
    prompt_len = model_config_dict[model_name]["prompt_len"]
    ctx_len = model_config_dict[model_name]["ctx_len"]
    max_gen_len = (NEW_GENERATION_TOKENS,)
    img_size = model_config_dict[model_name].get("img_size")
    image_urls = model_config_dict[model_name]["img_url_list"]
    queries = model_config_dict[model_name]["text_prompt_list"]
    n_layer = num_hidden_layers
    batch_size = model_config_dict[model_name]["batch_size"]
    full_batch_size = model_config_dict[model_name]["full_batch_size"]
    max_gen_len = NEW_GENERATION_TOKENS

    if config is None:
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True, padding=model_name not in ModelConfig.MOLMO_MODELS
        )
        config = set_num_layers_vlm(config, n_layer=n_layer)
        if hasattr(config, "model_type") and config.model_type in ["gemma3"]:
            config.text_config._sliding_window_pattern = 2
            config.text_config.layer_types = ["sliding_attention", "full_attention"]
        if hasattr(config, "model_type") and config.model_type in [
            "qwen3_vl",
            "qwen3_vl_moe",
        ]:
            config.vision_config.depth = 9
            config.text_config.num_hidden_layers = 1
            config.vision_config.deepstack_visual_indexes = [8]
        if model_name in ModelConfig.INTERNVL_MODELS or model_name in ModelConfig.MOLMO_MODELS:
            config._attn_implementation = "eager"
            model_hf = load_vlm_model(config)
            qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
                model_name,
                kv_offload=kv_offload,
                config=config,
                continuous_batching=True,
            )
        else:
            model_hf = load_vlm_model(config)
            qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
                model_name,
                kv_offload=kv_offload,
                config=config,
                continuous_batching=True,
            )
    else:
        model_hf = load_vlm_model_from_config(config)
        qeff_model = QEFFAutoModelForImageTextToText(
            copy.deepcopy(model_hf),
            kv_offload=kv_offload,
            config=model_hf.config,
            continuous_batching=True,
        )

    compile_kwargs = {
        "num_cores": 16,
        "num_devices": num_devices,
        "prefill_seq_len": prompt_len,
        "ctx_len": ctx_len,
        "batch_size": batch_size,
        "full_batch_size": full_batch_size,
        "mxfp6_matmul": False,
    }

    images = []
    generation_config = None
    if model_name in ModelConfig.INTERNVL_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        processor = InternProcessor(model_hf, tokenizer)
        image_height = 448
        image_width = 448
        for img_url in image_urls:
            img = requests.get(img_url, stream=True)
            image = Image.open(BytesIO(img.content)).convert("RGB")
            image = image.resize((image_height, image_width))
            images.append(image)
        generation_config = dict(max_new_tokens=max_gen_len, do_sample=False)
        generation_config["eos_token_id"] = tokenizer.convert_tokens_to_ids("<|im_end|>\n".strip())
        api_runner = ApiRunnerInternVL(
            batch_size,
            processor,
            config,
            images[0],
            queries[0],
            prompt_len,
            ctx_len,
            max_gen_len,
            n_layer,
        )
        # For same prompt
        image_list = [images[0]] * full_batch_size
        prompt_list = [queries[0]] * full_batch_size
        pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch_CB(model_hf, image_list, prompt_list)
        compile_kwargs["num_patches"] = 1
    elif model_name in ModelConfig.MOLMO_MODELS:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        image_height = 536
        image_width = 354
        for img_url in image_urls:
            img = requests.get(img_url, stream=True)
            image = Image.open(BytesIO(img.content)).convert("RGB")
            image = image.resize((image_height, image_width))
            images.append(image)
        api_runner = ApiRunnerMolmo(
            batch_size,
            processor,
            config,
            images[0],
            queries[0],
            prompt_len,
            ctx_len,
            max_gen_len,
            n_layer,
        )
        generation_config = GenerationConfig(max_new_tokens=NEW_GENERATION_TOKENS, stop_strings="<|endoftext|>")
        image_list = [images[0]] * full_batch_size
        prompt_list = [queries[0]] * full_batch_size
        pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch_CB(
            model_hf, image_list, prompt_list, generation_config
        )
        compile_kwargs["img_size"] = img_size
    else:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        image_height = None
        image_width = None
        for img_url in image_urls:
            image = Image.open(requests.get(img_url, stream=True).raw)
            if model_name == "mistralai/Mistral-Small-3.1-24B-Instruct-2503":
                image_height = 1540
                image_width = 1540
                image = image.resize((image_height, image_width))
            images.append(image)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": queries[0]},
                    {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        api_runner = ApiRunnerVlm(
            batch_size,
            processor,
            config,
            images[0],
            conversation,
            prompt,
            prompt_len,
            ctx_len,
            max_gen_len,
            n_layer,
        )
        image_list = [images[0]] * full_batch_size
        prompt_list = [queries[0]] * full_batch_size
        pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch_CB(model_hf, image_list, prompt_list)
        compile_kwargs["img_size"] = img_size

    qeff_model.export()
    qeff_model.compile(**compile_kwargs)
    print("QPC Outputs (QAIC):")
    exec_info = qeff_model.generate(
        tokenizer=tokenizer,
        processor=processor,
        images=[image_urls[0]] * full_batch_size,
        prompts=prompt_list,
        generation_len=max_gen_len,
        image_height=image_height,
        image_width=image_width,
    )
    qpc_tokens = exec_info.generated_ids[:, :max_gen_len]
    print("QPC Outputs (QAIC) for Continuous Batching with same prompt:")
    print(exec_info.generated_texts)
    for i in range(full_batch_size):
        assert (pytorch_hf_tokens[i] == qpc_tokens[i]).all(), (
            f"Tokens don't match for prompt {i} between HF and QPC output for same prompts"
        )
    if model_name in ModelConfig.MOLMO_MODELS:
        pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch_CB(
            model_hf, images, queries, generation_config=generation_config
        )
    else:
        pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch_CB(model_hf, images, queries)

    print("QPC Outputs (QAIC):")
    exec_info = qeff_model.generate(
        tokenizer=tokenizer,
        processor=processor,
        images=image_urls,
        prompts=queries,
        generation_len=max_gen_len,
        image_height=image_height,
        image_width=image_width,
    )
    qpc_tokens = exec_info.generated_ids[:, :max_gen_len]
    print("QPC Outputs (QAIC) for Continuous Batching with different prompt:")
    print(exec_info.generated_texts)
    for i in range(full_batch_size):
        assert (pytorch_hf_tokens[i] == qpc_tokens[i]).all(), (
            f"Tokens don't match for prompt {i} between HF and QPC output for different prompts"
        )
    manual_cleanup(qeff_model.onnx_path)  # Clean up the model files after the tests are done.


@pytest.mark.skip("Token Mismatch for full models")
@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True])  # TODO: Add support for kv_offload=False
def test_full_image_text_to_text_pytorch_vs_ai100_continuous_batching(model_name, kv_offload, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    if model_name in ModelConfig.DUAL_QPC_MODELS and not kv_offload:
        pytest.skip("These models require kv_offload=True for testing.")

    torch.manual_seed(42)
    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_CB(
        model_name=model_name,
        kv_offload=kv_offload,
        manual_cleanup=manual_cleanup,
        num_devices=4,
    )


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True])  # TODO: Add support for kv_offload=False
def test_few_image_text_to_text_pytorch_vs_ai100_continuous_batching(model_name, kv_offload, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    if model_name in ModelConfig.DUAL_QPC_MODELS and not kv_offload:
        pytest.skip("These models require kv_offload=True for testing.")

    torch.manual_seed(42)
    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_CB(
        model_name=model_name,
        num_hidden_layers=model_config_dict[model_name]["num_layers"],
        kv_offload=kv_offload,
        manual_cleanup=manual_cleanup,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True])  # TODO: Add support for kv_offload=False
def test_dummy_image_text_to_text_pytorch_vs_ai100_continuous_batching(model_name, kv_offload, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    if model_name in ModelConfig.DUAL_QPC_MODELS and not kv_offload:
        pytest.skip("These models require kv_offload=True for testing.")

    torch.manual_seed(42)
    hf_config = None
    if model_name in ModelConfig.STANDARD_VLM_MODELS:
        model_type = model_config_dict[model_name].get("model_type", None)
        custom_config = model_config_dict[model_name].get("additional_params", {})
        hf_config = AutoConfig.for_model(model_type, trust_remote_code=True, **custom_config)
        hf_config.name_or_path = model_name
        check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_CB(
            model_name, kv_offload=kv_offload, config=hf_config, manual_cleanup=manual_cleanup
        )
    else:
        check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_CB(
            model_name,
            num_hidden_layers=model_config_dict[model_name]["num_layers"],
            kv_offload=kv_offload,
            manual_cleanup=manual_cleanup,
        )
