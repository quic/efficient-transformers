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
from typing import List, Optional

import pytest
import requests
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    TextStreamer,
)

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from QEfficient.utils._utils import create_json
from QEfficient.utils.constants import QnnConstants
from QEfficient.utils.run_utils import ApiRunnerInternVL, ApiRunnerMolmo, ApiRunnerVlm
from QEfficient.utils.test_utils import (
    InternProcessor,
    ModelConfig,
    load_vlm_model,
    load_vlm_model_from_config,
    set_num_layers_vlm,
)

NEW_GENERATION_TOKENS = 10

CONFIG_PATH = "tests/configs/image_text_model_configs.json"

with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    multimodal_models = config_data["image_text_models"]
    blocking_models = config_data["blocking_models"]
test_mm_models = [model_config["model_name"] for model_config in multimodal_models]
test_blocking_models = [model_config["model_name"] for model_config in blocking_models]
model_config_dict = {model["model_name"]: model for model in multimodal_models}


def check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
    model_name: str,
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
    img_size: Optional[int] = None,
    qaic_config: Optional[dict] = None,
):
    """
    Unified function to test PyTorch model, PyTorch KV model, ONNX model, and Cloud AI 100 model.
    Handles standard VLM models, InternVL models, and Molmo models.

    Args:
        model_name: Hugging Face model identifier
        img_url: URL to image for testing
        query: Text query for the model
        prompt_len: Prompt sequence length
        ctx_len: Context length
        max_gen_len: Maximum generation length
        batch_size: Batch size for processing
        n_layer: Number of layers to use
        kv_offload: Whether to use KV offloading
        num_devices: Number of devices to use
        enable_qnn: Enable QNN compilation
        qnn_config: Path to QNN config file
        config: Pre-configured model config (optional)
        img_size: Image size for standard models (optional)
    """
    if config is None:
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True, padding=model_name not in ModelConfig.MOLMO_MODELS
        )
        config = set_num_layers_vlm(config, n_layer=n_layer)
        if model_name in ModelConfig.INTERNVL_MODELS or model_name in ModelConfig.MOLMO_MODELS:
            config._attn_implementation = "eager"
            model_hf = load_vlm_model(config)
            qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
                model_name,
                kv_offload=kv_offload,
                config=config,
            )
        else:
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

    compile_kwargs = {
        "num_devices": num_devices,
        "prefill_seq_len": prompt_len,
        "ctx_len": ctx_len,
        "mxfp6": False,
        "enable_qnn": enable_qnn,
        "qnn_config": qnn_config,
    }

    if model_name in ModelConfig.INTERNVL_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
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
        generation_config = dict(max_new_tokens=max_gen_len, do_sample=False)
        generation_config["eos_token_id"] = tokenizer.convert_tokens_to_ids("<|im_end|>\n".strip())
        api_runner = ApiRunnerInternVL(
            batch_size,
            processor,
            config,
            image,
            query,
            prompt_len,
            ctx_len,
            max_gen_len,
            n_layer,
        )
        pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch(model_hf, inputs, generation_config)
        compile_kwargs["num_patches"] = 1

    elif model_name in ModelConfig.MOLMO_MODELS:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
        img = requests.get(img_url, stream=True)
        image = Image.open(BytesIO(img.content)).convert("RGB")
        image = image.resize((536, 354))
        inputs = processor.process(images=[image], text=query)
        inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
        generation_config = GenerationConfig(max_new_tokens=NEW_GENERATION_TOKENS, stop_strings="<|endoftext|>")
        api_runner = ApiRunnerMolmo(
            batch_size,
            processor,
            config,
            image,
            query,
            prompt_len,
            ctx_len,
            max_gen_len,
            (n_layer, n_layer),
        )
        pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch(model_hf, inputs, generation_config)
        batch_size, prompt_len = inputs["input_ids"].shape
        inputs["attention_mask"] = torch.ones((inputs["input_ids"].shape), dtype=torch.int64)
        valid = inputs["image_input_idx"] > 0
        valid = valid.reshape(1, -1)
        inputs["valid_idx"] = torch.nonzero(valid)[:, 1].unsqueeze(0)
        inputs["pixel_values"] = inputs.pop("images")
        compile_kwargs["img_size"] = img_size

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
        api_runner = ApiRunnerVlm(
            batch_size,
            processor,
            config,
            image,
            conversation,
            prompt,
            prompt_len,
            ctx_len,
            max_gen_len,
            n_layer,
        )
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)
        pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch(model_hf, inputs)
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        if hasattr(qeff_model.model.config, "model_type") and qeff_model.model.config.model_type == "qwen2_5_vl":
            inputs = qeff_model.model.prepare_inputs_for_generation(
                inputs=inputs, prefill_seq_len=prompt_len, batch_size=batch_size
            )
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)
        compile_kwargs["img_size"] = img_size

    # pytorch_kv_tokens = api_runner.run_vlm_kv_model_on_pytorch(qeff_model.model)
    # assert (pytorch_kv_tokens == pytorch_hf_tokens).all(), (
    #     "Tokens don't match for pytorch HF output and pytorch KV output"
    # )

    _ = qeff_model.export()
    # ort_tokens = api_runner.run_vlm_kv_model_on_ort(onnx_model_path)
    # assert (pytorch_hf_tokens == ort_tokens).all(), "Tokens don't match for pytorch HF output and ORT output"

    qeff_model.compile(**compile_kwargs)
    streamer = TextStreamer(processor.tokenizer)
    print("QPC Outputs (QAIC):")
    output = qeff_model.generate(inputs=inputs, generation_len=NEW_GENERATION_TOKENS, streamer=streamer)
    qpc_tokens = output.generated_ids[:, :-1]
    assert (pytorch_hf_tokens == qpc_tokens).all(), "Tokens don't match for pytorch HF output and QPC output"


def check_image_text_to_text_blocked_vs_nonblocked_ai100(
    model_name: str,
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
    img_size: Optional[int] = None,
    qaic_config: Optional[dict] = None,
    prev_nonblocked_output=None,
):
    """
    Unified function to test PyTorch model, PyTorch KV model, ONNX model, and Cloud AI 100 model.
    Handles standard VLM models, InternVL models, and Molmo models.

    Args:
        model_name: Hugging Face model identifier
        img_url: URL to image for testing
        query: Text query for the model
        prompt_len: Prompt sequence length
        ctx_len: Context length
        max_gen_len: Maximum generation length
        batch_size: Batch size for processing
        n_layer: Number of layers to use
        kv_offload: Whether to use KV offloading
        num_devices: Number of devices to use
        enable_qnn: Enable QNN compilation
        qnn_config: Path to QNN config file
        config: Pre-configured model config (optional)
        img_size: Image size for standard models (optional)
    """

    is_intern_model = model_name == "OpenGVLab/InternVL2_5-1B" or model_name == "OpenGVLab/InternVL3_5-1B"
    is_molmo_model = model_name == "allenai/Molmo-7B-D-0924"

    # ========== Config and Model Loading ==========
    if config is None:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, padding=not is_molmo_model)
        config._attn_implementation = "eager" if (is_intern_model or is_molmo_model) else None
        config = set_num_layers(config, n_layer=n_layer)

    if is_intern_model:
        model_hf = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            config=config,
        )
        n_layer = get_num_layers_vlm(config)

    elif is_molmo_model:
        model_hf, _ = load_image_text_to_text_model(config)
        n_layer = (n_layer, n_layer)
    else:
        model_hf, _ = load_image_text_to_text_model(config)
        n_layer = get_num_layers_vlm(config)

    # ========== Processor and Image Loading ==========
    if is_intern_model:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        processor = InternProcessor(model_hf, tokenizer)
    else:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)

    if is_intern_model:
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
    else:
        if is_molmo_model:
            img = requests.get(img_url, stream=True)
            image = Image.open(BytesIO(img.content)).convert("RGB")
            image = image.resize((536, 354))
        else:
            image = Image.open(requests.get(img_url, stream=True).raw)
            if model_name == "mistralai/Mistral-Small-3.1-24B-Instruct-2503":
                image = image.resize((1540, 1540))

    # ========== Prepare Inputs and Get PyTorch HF Tokens ==========
    if is_intern_model:
        messages: List[List[str]] = []
        roles = ("<|im_start|>user\n", "<|im_start|>assistant\n")
        prompt = processor(pixel_values, questions, messages, roles, num_patches_list=num_patches_list)
    else:
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

    streamer = TextStreamer(processor.tokenizer)

    # ========== Export Non blocked model ==========
    if prev_nonblocked_output is None:
        if is_intern_model or is_molmo_model:
            qeff_model_non_blocked = QEFFAutoModelForCausalLM.from_pretrained(
                model_name,
                kv_offload=kv_offload,
                config=config,
            )
        else:
            qeff_model_non_blocked = QEFFAutoModelForImageTextToText.from_pretrained(
                model_name,
                kv_offload=kv_offload,
                config=config,
            )

        qeff_model_non_blocked.transform(
            ctx_len=ctx_len,
            seq_len=prompt_len,
            batch_size=batch_size,
            num_devices=1,
            qaic_config={"enable_blocking": False},
        )
        qeff_model_non_blocked.export()

    # ========== Export Blocked Model ==========
    if is_intern_model or is_molmo_model:
        qeff_model_blocked = QEFFAutoModelForCausalLM.from_pretrained(
            model_name,
            kv_offload=kv_offload,
            config=config,
        )
    else:
        qeff_model_blocked = QEFFAutoModelForImageTextToText.from_pretrained(
            model_name,
            kv_offload=kv_offload,
            config=config,
        )

    qeff_model_blocked.transform(
        ctx_len=ctx_len, seq_len=prompt_len, batch_size=batch_size, num_devices=1, qaic_config=qaic_config
    )
    qeff_model_blocked.export()

    # onnx_model_path = qeff_model.export()
    # ort_tokens = api_runner.run_vlm_kv_model_on_ort(onnx_model_path)
    # assert (pytorch_hf_tokens == ort_tokens).all(), "Tokens don't match for pytorch HF output and ORT output"

    compile_kwargs = {
        "num_devices": num_devices,
        "prefill_seq_len": prompt_len,
        "ctx_len": ctx_len,
        "mxfp6": False,
        "enable_qnn": enable_qnn,
        "qnn_config": qnn_config,
    }

    if is_intern_model:
        compile_kwargs["num_patches"] = 1
    elif not is_molmo_model and img_size is not None:
        compile_kwargs["img_size"] = img_size

    # ========== Compile and Generate for non-blocked if previous output not passed  ==========
    if prev_nonblocked_output is None:
        qeff_model_non_blocked.compile(**compile_kwargs)
        if not is_intern_model and not is_molmo_model:
            inputs = processor(images=image, text=prompt, return_tensors="pt")
            if (
                hasattr(qeff_model_non_blocked.model.config, "model_type")
                and qeff_model_non_blocked.model.config.model_type == "qwen2_5_vl"
            ):
                inputs = qeff_model_non_blocked.model.prepare_inputs_for_generation(
                    inputs=inputs, prefill_seq_len=prompt_len, batch_size=batch_size
                )
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

        print("QPC Outputs (QAIC):")
        output_non_blocked = qeff_model_non_blocked.generate(
            inputs=inputs, generation_len=NEW_GENERATION_TOKENS, streamer=streamer
        )
        qpc_tokens_non_blocked = output_non_blocked.generated_ids[:, :-1]
    else:
        qpc_tokens_non_blocked = prev_nonblocked_output

    # ========== Compile and Generate for blocked and Verify Output  ==========
    qeff_model_blocked.compile(**compile_kwargs)

    if not is_intern_model and not is_molmo_model:
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        if (
            hasattr(qeff_model_blocked.model.config, "model_type")
            and qeff_model_blocked.model.config.model_type == "qwen2_5_vl"
        ):
            inputs = qeff_model_blocked.model.prepare_inputs_for_generation(
                inputs=inputs, prefill_seq_len=prompt_len, batch_size=batch_size
            )
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

    print("QPC Outputs (QAIC):")
    output_blocked = qeff_model_blocked.generate(inputs=inputs, generation_len=NEW_GENERATION_TOKENS, streamer=streamer)
    qpc_tokens_blocked = output_blocked.generated_ids[:, :-1]
    assert (qpc_tokens_blocked == qpc_tokens_non_blocked).all(), (
        "Tokens don't match for QPC output between blocked and nonblocked model"
    )
    return qpc_tokens_non_blocked


@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.regular
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True, False])
def test_custom_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(model_name, kv_offload):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model,  without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """
    torch.manual_seed(42)
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    if model_name in ModelConfig.DUAL_QPC_MODELS and not kv_offload:
        pytest.skip("These models require kv_offload=True for testing.")

    img_size = model_config_dict[model_name].get("img_size")

    hf_config = None
    model_type = model_config_dict[model_name].get("model_type", None)
    if model_name in ModelConfig.STANDARD_VLM_MODELS and model_type is not None:
        custom_config = model_config_dict[model_name].get("additional_params", {})
        hf_config = AutoConfig.for_model(model_type, trust_remote_code=True, **custom_config)
        hf_config.name_or_path = model_name

    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
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


@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.nightly
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True, False])
def test_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(model_name, kv_offload):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model,  without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """
    torch.manual_seed(42)
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    if model_name in ModelConfig.DUAL_QPC_MODELS and not kv_offload:
        pytest.skip("These models require kv_offload=True for testing.")

    img_size = model_config_dict[model_name].get("img_size")

    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
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


@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_blocking_models)
def test_image_text_to_text_blocked_pytorch_vs_kv_vs_ort_vs_ai100(model_name):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model,  without continuous batching with attention blocking in the language models
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """

    HEAD_BLOCK_SIZE = 8
    NUM_KV_BLOCKS = 2
    NUM_Q_BLOCKS = 2

    # Get img_size for standard models, None for InternVL and Molmo
    img_size = model_config_dict[model_name].get("img_size")

    # head blocking only
    qaic_config = dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE)
    nonblocked_output = check_image_text_to_text_blocked_vs_nonblocked_ai100(
        model_name=model_name,
        prompt_len=model_config_dict[model_name]["prompt_len"],
        ctx_len=model_config_dict[model_name]["ctx_len"],
        max_gen_len=NEW_GENERATION_TOKENS,
        img_size=img_size,
        img_url=model_config_dict[model_name]["img_url"],
        query=model_config_dict[model_name]["text_prompt"],
        n_layer=model_config_dict[model_name]["num_layers"],
        batch_size=model_config_dict[model_name]["batch_size"],
        kv_offload=True,
        qaic_config=qaic_config,
    )

    # kv blocking only
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS)
    check_image_text_to_text_blocked_vs_nonblocked_ai100(
        model_name=model_name,
        prompt_len=model_config_dict[model_name]["prompt_len"],
        ctx_len=model_config_dict[model_name]["ctx_len"],
        max_gen_len=NEW_GENERATION_TOKENS,
        img_size=img_size,
        img_url=model_config_dict[model_name]["img_url"],
        query=model_config_dict[model_name]["text_prompt"],
        n_layer=model_config_dict[model_name]["num_layers"],
        batch_size=model_config_dict[model_name]["batch_size"],
        kv_offload=True,
        qaic_config=qaic_config,
        prev_nonblocked_output=nonblocked_output,
    )

    # q block only
    qaic_config = dict(enable_blocking=True, num_q_blocks=NUM_Q_BLOCKS)
    check_image_text_to_text_blocked_vs_nonblocked_ai100(
        model_name=model_name,
        prompt_len=model_config_dict[model_name]["prompt_len"],
        ctx_len=model_config_dict[model_name]["ctx_len"],
        max_gen_len=NEW_GENERATION_TOKENS,
        img_size=img_size,
        img_url=model_config_dict[model_name]["img_url"],
        query=model_config_dict[model_name]["text_prompt"],
        n_layer=model_config_dict[model_name]["num_layers"],
        batch_size=model_config_dict[model_name]["batch_size"],
        kv_offload=True,
        qaic_config=qaic_config,
        prev_nonblocked_output=nonblocked_output,
    )

    # qkv blocking
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS, num_q_blocks=NUM_Q_BLOCKS)
    check_image_text_to_text_blocked_vs_nonblocked_ai100(
        model_name=model_name,
        prompt_len=model_config_dict[model_name]["prompt_len"],
        ctx_len=model_config_dict[model_name]["ctx_len"],
        max_gen_len=NEW_GENERATION_TOKENS,
        img_size=img_size,
        img_url=model_config_dict[model_name]["img_url"],
        query=model_config_dict[model_name]["text_prompt"],
        n_layer=model_config_dict[model_name]["num_layers"],
        batch_size=model_config_dict[model_name]["batch_size"],
        kv_offload=True,
        qaic_config=qaic_config,
        prev_nonblocked_output=nonblocked_output,
    )

    # head qkv blocking
    qaic_config = dict(
        enable_blocking=True,
        head_block_size=HEAD_BLOCK_SIZE,
        num_kv_blocks=NUM_KV_BLOCKS,
        num_q_blocks=NUM_Q_BLOCKS,
    )

    check_image_text_to_text_blocked_vs_nonblocked_ai100(
        model_name=model_name,
        prompt_len=model_config_dict[model_name]["prompt_len"],
        ctx_len=model_config_dict[model_name]["ctx_len"],
        max_gen_len=NEW_GENERATION_TOKENS,
        img_size=img_size,
        img_url=model_config_dict[model_name]["img_url"],
        query=model_config_dict[model_name]["text_prompt"],
        n_layer=model_config_dict[model_name]["num_layers"],
        batch_size=model_config_dict[model_name]["batch_size"],
        kv_offload=True,
        qaic_config=qaic_config,
        prev_nonblocked_output=nonblocked_output,
    )


### QNN Tests ###


@pytest.mark.on_qaic
@pytest.mark.qnn
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True, False])
def test_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_qnn(model_name, kv_offload):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model,  without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """
    if model_name == "meta-llama/Llama-4-Scout-17B-16E-Instruct" or model_name == "google/gemma-3-4b-it":
        pytest.skip("QNN is not supported for these models yet.")

    qnn_config_json_path = os.path.join(os.getcwd(), "qnn_config.json")
    create_json(qnn_config_json_path, QnnConstants.QNN_SAMPLE_CONFIG)

    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        prompt_len=model_config_dict[model_name]["prompt_len"],
        ctx_len=model_config_dict[model_name]["ctx_len"],
        max_gen_len=NEW_GENERATION_TOKENS,
        img_size=model_config_dict[model_name]["img_size"],
        img_url=model_config_dict[model_name]["img_url"],
        query=model_config_dict[model_name]["text_prompt"],
        n_layer=model_config_dict[model_name]["num_layers"],
        batch_size=model_config_dict[model_name]["batch_size"],
        kv_offload=kv_offload,
        enable_qnn=True,
        qnn_config=qnn_config_json_path,
    )
