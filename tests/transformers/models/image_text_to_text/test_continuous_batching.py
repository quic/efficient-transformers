# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from io import BytesIO
from typing import List

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
from QEfficient.utils.device_utils import get_available_device_id
from QEfficient.utils.run_utils import ApiRunnerInternVL, ApiRunnerMolmo, ApiRunnerVlm
from QEfficient.utils.test_utils import InternProcessor

NEW_GENERATION_TOKENS = 10

# TODO: Add CB support for kv_offload=False case
test_models_config = [
    # CONFIG PARAMS NEEDED FOR A MODEL TO BE TESTED
    # (
    # model_name,
    # kv_offload,
    # batch_size,
    # prompt_len,
    # ctx_len,
    # img_size,
    # img_url_list",
    # text_prompt_list,
    # number of layers of the model,
    # full_batch_size
    # ),
    (
        "llava-hf/llava-1.5-7b-hf",
        True,
        1,
        784,
        1024,
        336,
        [
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
        ],
        [
            "Can you describe the image in detail?",
            "What are the objects in the image?",
            "What is the main subject of the image?",
            "What colors are predominant in the image?",
        ],
        1,
        4,
    ),
    # Disabled in CI due to performance issues
    # (
    #     "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    #     True,
    #     1,
    #     128,
    #     3072,
    #     336,
    #     ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
    #      "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
    #      "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
    #      "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",],
    #     ["Can you describe the image in detail?",
    #      "What are the objects in the image?",
    #      "What is the main subject of the image?",
    #      "What colors are predominant in the image?"],
    #     4,
    #     4,
    # ),
    (
        "google/gemma-3-4b-it",
        True,
        1,
        128,
        3072,
        896,
        [
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
        ],
        [
            "Can you describe the image in detail?",
            "What are the objects in the image?",
            "What is the main subject of the image?",
            "What colors are predominant in the image?",
        ],
        1,
        4,
    ),
    (
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        True,
        1,
        128,
        4096,
        1540,
        [
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
        ],
        [
            "Can you describe the image in detail?",
            "What are the objects in the image?",
            "What is the main subject of the image?",
            "What colors are predominant in the image?",
        ],
        1,
        4,
    ),
    (
        "Qwen/Qwen2.5-VL-3B-Instruct",
        True,
        1,
        128,
        4096,
        1540,
        [
            "https://picsum.photos/id/237/536/354",
            "https://picsum.photos/id/237/536/354",
            "https://picsum.photos/id/237/536/354",
            "https://picsum.photos/id/237/536/354",
        ],
        [
            "Can you describe the image in detail?",
            "What are the objects in the image?",
            "What is the main subject of the image?",
            "What colors are predominant in the image?",
        ],
        2,
        4,
    ),
    # (
    #     "meta-llama/Llama-3.2-11B-Vision-Instruct",
    #     True,
    #     1,
    #     32,
    #     512,
    #     560,
    #     ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
    #      "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
    #      "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
    #      "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",],
    #     ["Can you describe the image in detail?",
    #      "What are the objects in the image?",
    #      "What is the main subject of the image?",
    #      "What colors are predominant in the image?"],
    #     7,
    #     4,
    # ),
]

intern_model_config = [
    (
        "OpenGVLab/InternVL2_5-1B",
        True,
        1,
        384,
        512,
        [
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
        ],
        [
            "Can you describe the image in detail?",
            "What are the objects in the image?",
            "What is the main subject of the image?",
            "What colors are predominant in the image?",
        ],
        2,
        4,
    ),
    (
        "OpenGVLab/InternVL3_5-1B",
        True,
        1,
        384,
        512,
        [
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
        ],
        [
            "Can you describe the image in detail?",
            "What are the objects in the image?",
            "What is the main subject of the image?",
            "What colors are predominant in the image?",
        ],
        2,
        4,
    ),
]

molmo_model_config = [
    # Disabled in CI due to HF issues
    # (
    #     "allenai/Molmo-7B-D-0924",
    #     True,
    #     1,
    #     128,
    #     4096,
    #     ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
    #      "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
    #      "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
    #      "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",],
    #     ["Can you describe the image in detail?",
    #      "What are the objects in the image?",
    #      "What is the main subject of the image?",
    #      "What colors are predominant in the image?"],
    #     2,
    #     4,
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


def check_image_text_to_text_pytorch_vs_ai100_continuous_batching(
    model_name: str,
    img_size: int,
    image_urls: List[str],
    queries: List[str],
    prompt_len: int,
    ctx_len: int,
    max_gen_len: int = 20,
    batch_size: int = 1,
    n_layer: int = 1,
    num_devices: int = 1,
    full_batch_size: int = 4,
    kv_offload: bool = True,
):
    model_config = {"model_name": model_name}
    model_config["img_size"] = img_size
    config = AutoConfig.from_pretrained(model_config["model_name"], trust_remote_code=True)
    config = set_num_layers(config, n_layer=n_layer)
    model_hf, _ = load_image_text_to_text_model(config)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)

    n_layer = get_num_layers_vlm(config)

    image_height = None
    image_width = None

    images = []
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

    # For same prompt
    image_list = [images[0]] * full_batch_size
    prompt_list = [queries[0]] * full_batch_size

    pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch_CB(model_hf, image_list, prompt_list)

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_config["model_name"],
        kv_offload=kv_offload,
        config=config,
        continuous_batching=True,
    )

    qeff_model.export()

    if not get_available_device_id():
        pytest.skip("No available devices to run model on Cloud AI 100")

    qeff_model.compile(
        img_size=model_config["img_size"],
        num_cores=16,
        num_devices=num_devices,
        prefill_seq_len=prompt_len,
        ctx_len=ctx_len,
        batch_size=batch_size,
        full_batch_size=full_batch_size,
        mxfp6_matmul=False,
    )

    print("QPC Outputs (QAIC):")
    exec_info = qeff_model.generate(
        tokenizer=processor.tokenizer,
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
    pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch_CB(model_hf, images, queries)

    print("QPC Outputs (QAIC):")
    exec_info = qeff_model.generate(
        tokenizer=processor.tokenizer,
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


def check_molmo_image_text_to_text_pytorch_vs_ai100_continuous_batching(
    model_name: str,
    image_urls: List[str],
    queries: List[str],
    prompt_len: int,
    ctx_len: int,
    max_gen_len: int = 20,
    batch_size: int = 1,
    n_layer: int = 1,
    num_devices: int = 1,
    full_batch_size: int = 4,
    kv_offload: bool = True,
):
    model_config = {"model_name": model_name}

    config = AutoConfig.from_pretrained(model_config["model_name"], trust_remote_code=True)
    config._attn_implementation = "eager"
    config = set_num_layers(config, n_layer=n_layer)
    model_hf, _ = load_image_text_to_text_model(config)
    n_layer = (n_layer, n_layer)

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    images = []
    for img_url in image_urls:
        img = requests.get(img_url, stream=True)
        image = Image.open(BytesIO(img.content)).convert("RGB")
        image = image.resize((536, 354))
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

    # For same prompt
    image_list = [images[0]] * full_batch_size
    prompt_list = [queries[0]] * full_batch_size
    pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch_CB(model_hf, image_list, prompt_list, generation_config)

    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation="eager",
        kv_offload=kv_offload,
        config=config,
        continuous_batching=True,
    )

    qeff_model.export()

    qeff_model.compile(
        prefill_seq_len=prompt_len,
        ctx_len=ctx_len,
        num_devices=4,
        batch_size=1,
        full_batch_size=full_batch_size,
        mxfp6_matmul=False,
        mxint8_kv_cache=True,
        aic_enable_depth_first=True,
        mos=1,
    )

    exec_info = qeff_model.generate(
        tokenizer=tokenizer,
        processor=processor,
        images=[image_urls[0]] * full_batch_size,
        prompts=prompt_list,
        generation_len=max_gen_len,
    )

    qpc_tokens = exec_info.generated_ids[:, :max_gen_len]
    print("QPC Outputs (QAIC) for Continuous Batching with same prompt:")
    print(exec_info.generated_texts)

    for i in range(full_batch_size):
        assert (pytorch_hf_tokens[i] == qpc_tokens[i]).all(), (
            f"Tokens don't match for prompt {i} between HF and QPC output for same prompts"
        )

    # For different prompts
    pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch_CB(model_hf, images, queries, generation_config)
    exec_info = qeff_model.generate(
        tokenizer=tokenizer,
        processor=processor,
        images=image_urls,
        prompts=queries,
        generation_len=max_gen_len,
    )

    qpc_tokens = exec_info.generated_ids[:, :max_gen_len]
    print("QPC Outputs (QAIC) for Continuous Batching with different prompt:")
    print(exec_info.generated_texts)

    for i in range(full_batch_size):
        assert (pytorch_hf_tokens[i] == qpc_tokens[i]).all(), (
            f"Tokens don't match for prompt {i} between HF and QPC output for different prompts"
        )
    return


def check_intern_image_text_to_text_pytorch_vs_ai100_continuous_batching(
    model_name: str,
    image_urls: str,
    queries: str,
    prompt_len: int,
    ctx_len: int,
    max_gen_len: int = 20,
    batch_size: int = 1,
    n_layer: int = 1,
    kv_offload: bool = True,
    num_devices: int = 1,
    full_batch_size: int = 4,
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

    generation_config = dict(max_new_tokens=max_gen_len, do_sample=False)
    generation_config["eos_token_id"] = tokenizer.convert_tokens_to_ids("<|im_end|>\n".strip())

    images = []
    for img_url in image_urls:
        img = requests.get(img_url, stream=True)
        image = Image.open(BytesIO(img.content)).convert("RGB")
        image = image.resize((448, 448))
        images.append(image)

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

    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        continuous_batching=True,
    )

    qeff_model.export()

    qeff_model.compile(
        num_patches=1,
        prefill_seq_len=prompt_len,
        ctx_len=ctx_len,
        num_devices=4,
        batch_size=1,
        full_batch_size=full_batch_size,
        mxfp6_matmul=False,
    )

    exec_info = qeff_model.generate(
        tokenizer=tokenizer,
        processor=processor,
        images=[image_urls[0]] * full_batch_size,
        prompts=prompt_list,
        generation_len=max_gen_len,
        image_height=448,
        image_width=448,
    )

    qpc_tokens = exec_info.generated_ids[:, :max_gen_len]
    print("QPC Outputs (QAIC) for Continuous Batching for same prompts:")
    print(exec_info.generated_texts)

    for i in range(full_batch_size):
        assert (pytorch_hf_tokens[i] == qpc_tokens[i]).all(), (
            f"Tokens don't match for prompt {i} between HF and QPC output for same prompts"
        )

    # For different prompts
    pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch_CB(model_hf, images, queries)

    exec_info = qeff_model.generate(
        tokenizer=tokenizer,
        processor=processor,
        images=image_urls,
        prompts=queries,
        generation_len=max_gen_len,
        image_height=448,
        image_width=448,
    )

    qpc_tokens = exec_info.generated_ids[:, :max_gen_len]
    print("QPC Outputs (QAIC) for Continuous Batching for different prompts:")
    print(exec_info.generated_texts)

    for i in range(full_batch_size):
        assert (pytorch_hf_tokens[i] == qpc_tokens[i]).all(), (
            f"Tokens don't match for prompt {i} between HF and QPC output for different prompts"
        )
    return


@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize(
    "model_name, kv_offload, batch_size, prompt_len, ctx_len, img_size, img_urls, queries, n_layer, full_batch_size",
    test_models_config,
)
def test_image_text_to_text_pytorch_vs_ai100_continuous_batching(
    model_name, kv_offload, batch_size, prompt_len, ctx_len, img_size, img_urls, queries, n_layer, full_batch_size
):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model,  without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """
    check_image_text_to_text_pytorch_vs_ai100_continuous_batching(
        model_name=model_name,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        max_gen_len=NEW_GENERATION_TOKENS,
        img_size=img_size,
        image_urls=img_urls,
        queries=queries,
        n_layer=n_layer,
        batch_size=batch_size,
        kv_offload=kv_offload,
        full_batch_size=full_batch_size,
    )


@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize(
    "model_name, kv_offload, batch_size, prompt_len, ctx_len, img_urls, queries, n_layer, full_batch_size",
    molmo_model_config,
)
def test_image_text_to_text_molmo_pytorch_vs_ai100_continuous_batching(
    model_name, kv_offload, batch_size, prompt_len, ctx_len, img_urls, queries, n_layer, full_batch_size
):
    check_molmo_image_text_to_text_pytorch_vs_ai100_continuous_batching(
        model_name=model_name,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        max_gen_len=NEW_GENERATION_TOKENS,
        image_urls=img_urls,
        queries=queries,
        n_layer=n_layer,
        batch_size=batch_size,
        kv_offload=kv_offload,
        full_batch_size=full_batch_size,
    )


@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize(
    "model_name, kv_offload, batch_size, prompt_len, ctx_len, img_url, queries, n_layer, full_batch_size",
    intern_model_config,
)
def test_image_text_to_text_intern_pytorch_vs_ai100_continuous_batching(
    model_name, kv_offload, batch_size, prompt_len, ctx_len, img_url, queries, n_layer, full_batch_size
):
    check_intern_image_text_to_text_pytorch_vs_ai100_continuous_batching(
        model_name=model_name,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        max_gen_len=NEW_GENERATION_TOKENS,
        image_urls=img_url,
        queries=queries,
        n_layer=n_layer,
        batch_size=batch_size,
        kv_offload=kv_offload,
        full_batch_size=full_batch_size,
    )
