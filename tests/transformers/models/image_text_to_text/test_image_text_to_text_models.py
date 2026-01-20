# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

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
from QEfficient.utils.device_utils import get_available_device_id
from QEfficient.utils.run_utils import ApiRunnerInternVL, ApiRunnerMolmo, ApiRunnerVlm
from QEfficient.utils.test_utils import InternProcessor

NEW_GENERATION_TOKENS = 10
test_models_config = [
    # CONFIG PARAMS NEEDED FOR A MODEL TO BE TESTED
    # (
    # model_name,
    # kv_offload,
    # batch_size,
    # prompt_len,
    # ctx_len,
    # img_size,
    # img_url",
    # text_prompt,
    # number of layers of the model,
    # ),
    (
        "llava-hf/llava-1.5-7b-hf",
        True,
        1,
        784,
        1024,
        336,
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg",
        "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud",
        1,
    ),
    (
        "llava-hf/llava-1.5-7b-hf",
        False,
        1,
        784,
        1024,
        336,
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg",
        "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud",
        1,
    ),
    # Disabled in CI due to performance issues
    # (
    #     "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    #     True,
    #     1,
    #     128,
    #     3072,
    #     336,
    #     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg",
    #     "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud",
    #     4,
    # ),
    # (
    #     "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    #     False,
    #     1,
    #     128,
    #     3072,
    #     336,
    #     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg",
    #     "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud",
    #     4,
    # ),
    (
        "google/gemma-3-4b-it",
        True,
        1,
        128,
        3072,
        896,
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
        "Can you describe the image in detail.",
        6,
    ),
    (
        "google/gemma-3-4b-it",
        False,
        1,
        128,
        3072,
        896,
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
        "Can you describe the image in detail.",
        6,
    ),
    (
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        True,
        1,
        128,
        4096,
        1540,
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
        "Can you describe the image in detail.",
        1,
    ),
    (
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        False,
        1,
        128,
        4096,
        1540,
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
        "Can you describe the image in detail.",
        1,
    ),
    (
        "Qwen/Qwen2.5-VL-3B-Instruct",
        True,
        1,
        128,
        4096,
        1540,
        "https://picsum.photos/id/237/536/354",
        "Can you describe the image in detail.",
        1,
    ),
    # (
    #     "meta-llama/Llama-3.2-11B-Vision-Instruct",
    #     True,
    #     1,
    #     32,
    #     512,
    #     560,
    #     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
    #     "Explain this image",
    #     7,
    # ),
]

intern_model_config = [
    # (
    # "OpenGVLab/InternVL2_5-1B",
    # True,
    # 1,
    # 384,
    # 512,
    # "https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-1-2048.jpg",
    # "Please describe the image in detail.",
    # 2,
    # ),
    # (
    # "OpenGVLab/InternVL3_5-1B",
    # True,
    # 1,
    # 384,
    # 512,
    # "https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-1-2048.jpg",
    # "Please describe the image in detail.",
    # 2,
    # ),
    # (
    #     "OpenGVLab/InternVL2_5-1B",
    #     False,
    #     1,
    #     384,
    #     512,
    #     "https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-1-2048.jpg",
    #     "Please describe the image in detail.",
    #     2,
    # ), # commented becuase QNN Convertor is not supported for this model yet.
]

molmo_model_config = [
    # Disabled in CI due to HF issues
    # (
    #     "allenai/Molmo-7B-D-0924",
    #     True,
    #     1,
    #     128,
    #     4096,
    #     "https://picsum.photos/id/237/536/354",
    #     "Can you describe the image in detail.",
    #     2,
    # ),
]


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


def check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
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
):
    model_config = {"model_name": model_name}
    model_config["img_size"] = img_size
    config = AutoConfig.from_pretrained(model_config["model_name"], trust_remote_code=True, padding=True)
    config = set_num_layers(config, n_layer=n_layer)
    model_hf, _ = load_image_text_to_text_model(config)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)

    n_layer = get_num_layers_vlm(config)
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
    streamer = TextStreamer(processor.tokenizer)
    pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch(model_hf, inputs)
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_config["model_name"],
        kv_offload=kv_offload,
        config=config,
    )

    # pytorch_kv_tokens = api_runner.run_vlm_kv_model_on_pytorch(qeff_model.model)
    # assert (pytorch_kv_tokens == pytorch_hf_tokens).all(), (
    #     "Tokens don't match for pytorch HF output and pytorch KV output"
    # )

    qeff_model.export()
    # onnx_model_path = qeff_model.export()
    # ort_tokens = api_runner.run_vlm_kv_model_on_ort(onnx_model_path)
    # assert (pytorch_hf_tokens == ort_tokens).all(), "Tokens don't match for pytorch HF output and ORT output"
    if not get_available_device_id():
        pytest.skip("No available devices to run model on Cloud AI 100")
    qeff_model.compile(
        img_size=model_config["img_size"],
        num_devices=num_devices,
        prefill_seq_len=prompt_len,
        ctx_len=ctx_len,
        mxfp6=False,
        enable_qnn=enable_qnn,
        qnn_config=qnn_config,
    )
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


def check_molmo_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
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
):
    model_config = {"model_name": model_name}

    config = AutoConfig.from_pretrained(model_config["model_name"], trust_remote_code=True)
    config._attn_implementation = "eager"
    config = set_num_layers(config, n_layer=n_layer)
    model_hf, _ = load_image_text_to_text_model(config)
    n_layer = (n_layer, n_layer)

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
    img = requests.get(img_url, stream=True)
    image = Image.open(BytesIO(img.content)).convert("RGB")
    image = image.resize((536, 354))

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

    inputs = processor.process(images=[image], text=query)
    inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}

    generation_config = GenerationConfig(max_new_tokens=NEW_GENERATION_TOKENS, stop_strings="<|endoftext|>")
    pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch(model_hf, inputs, generation_config)

    batch_size, prompt_len = inputs["input_ids"].shape
    inputs["attention_mask"] = torch.ones((inputs["input_ids"].shape), dtype=torch.int64)
    valid = inputs["image_input_idx"] > 0
    valid = valid.reshape(1, -1)
    inputs["valid_idx"] = torch.nonzero(valid)[:, 1].unsqueeze(0)
    inputs["pixel_values"] = inputs.pop("images")

    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        model_config["model_name"],
        kv_offload=kv_offload,
        config=config,
    )

    streamer = TextStreamer(processor.tokenizer)
    qeff_model.export()

    if not get_available_device_id():
        pytest.skip("No available devices to run model on Cloud AI 100")
    qeff_model.compile(num_devices=num_devices, prefill_seq_len=prompt_len, ctx_len=ctx_len, mxfp6=False)
    print("QPC Outputs (QAIC):")
    output = qeff_model.generate(inputs=inputs, generation_len=NEW_GENERATION_TOKENS, streamer=streamer)
    qpc_tokens = output.generated_ids[:, :-1]
    assert (pytorch_hf_tokens == qpc_tokens).all(), "Tokens don't match for pytorch HF output and QPC output"
    return


def check_intern_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
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
):
    model_config = {"model_name": model_name}

    config = AutoConfig.from_pretrained(model_config["model_name"], trust_remote_code=True)
    config._attn_implementation = "eager"
    config = set_num_layers(config, n_layer=n_layer)
    model_hf = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
        config=config,
    )
    n_layer = get_num_layers_vlm(config)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    processor = InternProcessor(model_hf, tokenizer)

    prompt = [query]
    img_url = [img_url]
    pixel_values = []
    num_patches_list = []
    questions = []
    for i in range(len(prompt)):
        img = requests.get(img_url[i], stream=True)
        image = Image.open(BytesIO(img.content)).convert("RGB")

        image = image.resize((448, 448))

        # preprocess the resized image
        pixel_value = processor.load_image(image, max_num=12)
        num_patches_list.append(pixel_value.shape[0])
        pixel_values.append(pixel_value)

        question = "<image>\n" + prompt[i]
        questions.append(question)

    pixel_values = torch.cat(pixel_values, dim=0)

    # Chat Template information for prompt preprocessing
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

    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        model_config["model_name"],
        kv_offload=kv_offload,
        config=config,
    )
    # pytorch_kv_tokens = api_runner.run_vlm_kv_model_on_pytorch(qeff_model.model)
    # assert (pytorch_hf_tokens == pytorch_kv_tokens).all(), (
    #     "Tokens don't match for pytorch HF output and QEFF KV Model output"
    # )

    streamer = TextStreamer(processor.tokenizer)
    qeff_model.export()

    # onnx_model_path = qeff_model.export()
    # ort_tokens = api_runner.run_vlm_kv_model_on_ort(onnx_model_path)
    # assert (pytorch_hf_tokens == ort_tokens).all(), "Tokens don't match for pytorch HF output and ORT output"
    if not get_available_device_id():
        pytest.skip("No available devices to run model on Cloud AI 100")
    qeff_model.compile(
        num_patches=1,
        num_devices=num_devices,
        prefill_seq_len=prompt_len,
        ctx_len=ctx_len,
        mxfp6=False,
        enable_qnn=enable_qnn,
        qnn_config=qnn_config,
    )
    print("QPC Outputs (QAIC):")
    output = qeff_model.generate(inputs=inputs, generation_len=NEW_GENERATION_TOKENS, streamer=streamer)
    qpc_tokens = output.generated_ids[:, :-1]
    assert (pytorch_hf_tokens == qpc_tokens).all(), "Tokens don't match for pytorch HF output and QPC output"
    return


@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize(
    "model_name, kv_offload, batch_size, prompt_len, ctx_len, img_size, img_url, query, n_layer", test_models_config
)
def test_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
    model_name, kv_offload, batch_size, prompt_len, ctx_len, img_size, img_url, query, n_layer
):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model,  without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """
    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        max_gen_len=NEW_GENERATION_TOKENS,
        img_size=img_size,
        img_url=img_url,
        query=query,
        n_layer=n_layer,
        batch_size=batch_size,
        kv_offload=kv_offload,
    )


@pytest.mark.on_qaic
@pytest.mark.qnn
@pytest.mark.multimodal
@pytest.mark.parametrize(
    "model_name, kv_offload, batch_size, prompt_len, ctx_len, img_size, img_url, query, n_layer", test_models_config
)
def test_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_qnn(
    model_name, kv_offload, batch_size, prompt_len, ctx_len, img_size, img_url, query, n_layer
):
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
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        max_gen_len=NEW_GENERATION_TOKENS,
        img_size=img_size,
        img_url=img_url,
        query=query,
        n_layer=n_layer,
        batch_size=batch_size,
        kv_offload=kv_offload,
        enable_qnn=True,
        qnn_config=qnn_config_json_path,
    )


@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize(
    "model_name, kv_offload, batch_size, prompt_len, ctx_len, img_url, query, n_layer", molmo_model_config
)
def test_image_text_to_text_molmo_pytorch_vs_kv_vs_ort_vs_ai100(
    model_name, kv_offload, batch_size, prompt_len, ctx_len, img_url, query, n_layer
):
    check_molmo_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        max_gen_len=NEW_GENERATION_TOKENS,
        img_url=img_url,
        query=query,
        n_layer=n_layer,
        batch_size=batch_size,
        kv_offload=kv_offload,
    )


@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize(
    "model_name, kv_offload, batch_size, prompt_len, ctx_len, img_url, query, n_layer", intern_model_config
)
def test_image_text_to_text_intern_pytorch_vs_kv_vs_ort_vs_ai100(
    model_name, kv_offload, batch_size, prompt_len, ctx_len, img_url, query, n_layer
):
    check_intern_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        max_gen_len=NEW_GENERATION_TOKENS,
        img_url=img_url,
        query=query,
        n_layer=n_layer,
        batch_size=batch_size,
        kv_offload=kv_offload,
    )


@pytest.mark.on_qaic
@pytest.mark.qnn
@pytest.mark.multimodal
@pytest.mark.parametrize(
    "model_name, kv_offload, batch_size, prompt_len, ctx_len, img_url, query, n_layer", intern_model_config
)
def test_image_text_to_text_intern_pytorch_vs_kv_vs_ort_vs_ai100_qnn(
    model_name, kv_offload, batch_size, prompt_len, ctx_len, img_url, query, n_layer
):
    qnn_config_json_path = os.path.join(os.getcwd(), "qnn_config.json")
    create_json(qnn_config_json_path, QnnConstants.QNN_SAMPLE_CONFIG)

    check_intern_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        max_gen_len=NEW_GENERATION_TOKENS,
        img_url=img_url,
        query=query,
        n_layer=n_layer,
        batch_size=batch_size,
        kv_offload=kv_offload,
        enable_qnn=True,
        qnn_config=qnn_config_json_path,
    )
