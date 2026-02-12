# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

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
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    TextStreamer,
)

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from QEfficient.utils import hf_download
from QEfficient.utils._utils import create_json, get_num_layers_vlm
from QEfficient.utils.constants import QnnConstants
from QEfficient.utils.run_utils import ApiRunnerInternVL, ApiRunnerMolmo, ApiRunnerVlm
from QEfficient.utils.test_utils import InternProcessor

NEW_GENERATION_TOKENS = 10

CONFIG_PATH = "tests/configs/image_text_model_configs.json"

with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    multimodal_models = config_data["image_text_models"]
test_mm_models = [model_config["model_name"] for model_config in multimodal_models]
model_config_dict = {model["model_name"]: model for model in multimodal_models}


def load_image_text_to_text_model(model_config):
    model_path = hf_download(
        repo_id=model_config._name_or_path,
        ignore_patterns=["*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.h5", "*.msgpack"],
    )
    try:
        model_hf = AutoModelForImageTextToText.from_pretrained(
            model_path,
            low_cpu_mem_usage=False,
            config=model_config,
        )
    except ValueError:
        model_hf = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            config=model_config,
        )
    params = sum(p.numel() for p in model_hf.parameters())
    model_hf.eval()
    return model_hf, params


def load_image_text_to_text_model_from_config(model_name, config):
    torch.manual_seed(42)
    model_path = hf_download(
        repo_id=model_name,
        ignore_patterns=["*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.h5", "*.msgpack"],
    )
    try:
        model_hf = AutoModelForImageTextToText.from_config(
            config,
        )
    except ValueError:
        model_hf = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            config=config,
        )
    params = sum(p.numel() for p in model_hf.parameters())
    model_hf.eval()
    return model_hf, params


def set_num_layers(config, n_layer=1):
    ## -1 indicates use all the layers of the model.
    if n_layer == -1:
        return config
    elif hasattr(config, "model_type") and "mllama" in config.model_type:
        config.text_config.num_hidden_layers = n_layer
        config.text_config.cross_attention_layers = [
            x for x in config.text_config.cross_attention_layers if x < n_layer
        ]
    elif hasattr(config, "text_config"):
        config.text_config.num_hidden_layers = n_layer
        config.vision_config.num_hidden_layers = n_layer
    elif hasattr(config, "llm_config"):
        config.llm_config.num_hidden_layers = n_layer
        config.vision_config.num_hidden_layers = n_layer
    else:
        config.num_hidden_layers = n_layer
    return config


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
    elif is_molmo_model:
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
            n_layer,
        )
        pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch(model_hf, inputs, generation_config)
        batch_size, prompt_len = inputs["input_ids"].shape
        inputs["attention_mask"] = torch.ones((inputs["input_ids"].shape), dtype=torch.int64)
        valid = inputs["image_input_idx"] > 0
        valid = valid.reshape(1, -1)
        inputs["valid_idx"] = torch.nonzero(valid)[:, 1].unsqueeze(0)
        inputs["pixel_values"] = inputs.pop("images")
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

    # pytorch_kv_tokens = api_runner.run_vlm_kv_model_on_pytorch(qeff_model.model)
    # assert (pytorch_kv_tokens == pytorch_hf_tokens).all(), (
    #     "Tokens don't match for pytorch HF output and pytorch KV output"
    # )

    streamer = TextStreamer(processor.tokenizer)

    # ========== Export and Compile Model ==========
    if is_intern_model or is_molmo_model:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            model_name,
            kv_offload=kv_offload,
            config=config,
        )
    else:
        qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
            model_name,
            kv_offload=kv_offload,
            config=config,
        )

    qeff_model.export()

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

    qeff_model.compile(**compile_kwargs)

    # ========== Generate and Verify Output ==========

    if not is_intern_model and not is_molmo_model:
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        if hasattr(qeff_model.model.config, "model_type") and qeff_model.model.config.model_type == "qwen2_5_vl":
            inputs = qeff_model.model.prepare_inputs_for_generation(
                inputs=inputs, prefill_seq_len=prompt_len, batch_size=batch_size
            )
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

    print("QPC Outputs (QAIC):")
    output = qeff_model.generate(inputs=inputs, generation_len=NEW_GENERATION_TOKENS, streamer=streamer)
    qpc_tokens = output.generated_ids[:, :-1]
    assert (pytorch_hf_tokens == qpc_tokens).all(), "Tokens don't match for pytorch HF output and QPC output"
    return


@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True, False])
def test_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(model_name, kv_offload):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model,  without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """
    if model_name in [
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "allenai/Molmo-7B-D-0924",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
    ]:
        pytest.skip("Test skipped for this model due to some issues.")
    if (
        model_name in ["OpenGVLab/InternVL2_5-1B", "OpenGVLab/InternVL3_5-1B", "Qwen/Qwen2.5-VL-3B-Instruct"]
        and not kv_offload
    ):
        pytest.skip("These models require kv_offload=True for testing.")
    # Get img_size for standard models, None for InternVL and Molmo
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
