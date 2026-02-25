# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import json
from io import BytesIO
from typing import List, Optional

import pytest
import requests
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
)

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from QEfficient.utils import hf_download
from QEfficient.utils._utils import get_num_layers_vlm
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


def check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_CB(
    model_name: str,
    image_urls: List[str],
    queries: List[str],
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
    full_batch_size: Optional[int] = 4,
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
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True, padding=not is_intern_model and not is_molmo_model
        )
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
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    images = []
    if is_intern_model:
        image_height = 448
        image_width = 448
        for img_url in image_urls:
            img = requests.get(img_url, stream=True)
            image = Image.open(BytesIO(img.content)).convert("RGB")
            image = image.resize((image_height, image_width))
            images.append(image)
    else:
        if is_molmo_model:
            image_height = 536
            image_width = 354
            for img_url in image_urls:
                img = requests.get(img_url, stream=True)
                image = Image.open(BytesIO(img.content)).convert("RGB")
                image = image.resize((image_height, image_width))
                images.append(image)
        else:
            image_height = None
            image_width = None
            for img_url in image_urls:
                image = Image.open(requests.get(img_url, stream=True).raw)
                if model_name == "mistralai/Mistral-Small-3.1-24B-Instruct-2503":
                    image_height = 1540
                    image_width = 1540
                    image = image.resize((image_height, image_width))
                images.append(image)

    # ========== Prepare Inputs and Get PyTorch HF Tokens ==========
    generation_config = None
    if is_intern_model:
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
    elif is_molmo_model:
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

        # For same prompt
        image_list = [images[0]] * full_batch_size
        prompt_list = [queries[0]] * full_batch_size
        pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch_CB(
            model_hf, image_list, prompt_list, generation_config
        )

    else:
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
        # For same prompt
        image_list = [images[0]] * full_batch_size
        prompt_list = [queries[0]] * full_batch_size

        pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch_CB(model_hf, image_list, prompt_list)

    # ========== Export and Compile Model ==========
    if is_intern_model or is_molmo_model:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="eager",
            kv_offload=kv_offload,
            config=config,
            continuous_batching=True,
        )
    else:
        qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
            model_name,
            kv_offload=kv_offload,
            config=config,
            continuous_batching=True,
        )

    qeff_model.export()

    compile_kwargs = {
        "num_cores": 16,
        "num_devices": num_devices,
        "prefill_seq_len": prompt_len,
        "ctx_len": ctx_len,
        "batch_size": batch_size,
        "full_batch_size": full_batch_size,
        "mxfp6_matmul": False,
    }

    if is_intern_model:
        compile_kwargs["num_patches"] = 1
    elif not is_molmo_model and img_size is not None:
        compile_kwargs["img_size"] = img_size

    qeff_model.compile(**compile_kwargs)

    # ========== Generate and Verify Output ==========

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

    # For different prompts
    if is_molmo_model:
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
    return


@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True])  # TODO: Add support for kv_offload=False
def test_image_text_to_text_pytorch_vs_ai100_continuous_batching(model_name, kv_offload):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model,  with continuous batching.
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

    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_CB(
        model_name=model_name,
        prompt_len=model_config_dict[model_name]["prompt_len"],
        ctx_len=model_config_dict[model_name]["ctx_len"],
        max_gen_len=NEW_GENERATION_TOKENS,
        img_size=img_size,
        image_urls=model_config_dict[model_name]["img_url_list"],
        queries=model_config_dict[model_name]["text_prompt_list"],
        n_layer=model_config_dict[model_name]["num_layers"],
        batch_size=model_config_dict[model_name]["batch_size"],
        full_batch_size=model_config_dict[model_name]["full_batch_size"],
        kv_offload=kv_offload,
    )
