# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from typing import Optional

import pytest
import requests
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    TextStreamer,
)

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText
from QEfficient.utils import hf_download
from QEfficient.utils._utils import get_num_layers_vlm
from QEfficient.utils.device_utils import get_available_device_id

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
        1,
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
        1,
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
    (
        "OpenGVLab/InternVL2_5-1B",
        True,
        1,
        384,
        512,
        "https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-1-2048.jpg",
        "Please describe the image in detail.",
        2,
    ),
    (
        "OpenGVLab/InternVL3_5-1B",
        True,
        1,
        384,
        512,
        "https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-1-2048.jpg",
        "Please describe the image in detail.",
        2,
    ),
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

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)
    streamer = TextStreamer(processor.tokenizer)
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_config["model_name"],
        kv_offload=kv_offload,
        config=config,
    )

    # pytorch_kv_tokens = api_runner.run_vlm_kv_model_on_pytorch(qeff_model.model)
    # assert (pytorch_kv_tokens == pytorch_hf_tokens).all(), (
    #     "Tokens don't match for pytorch HF output and pytorch KV output"
    # )

    with_sub_func_onnx = qeff_model.export(use_onnx_subfunctions=True, offload_pt_weights=False)
    without_sub_func_onnx = qeff_model.export(use_onnx_subfunctions=False)

    if not get_available_device_id():
        pytest.skip("No available devices to run model on Cloud AI 100")

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    if hasattr(qeff_model.model.config, "model_type") and qeff_model.model.config.model_type == "qwen2_5_vl":
        inputs = qeff_model.model.prepare_inputs_for_generation(
            inputs=inputs, prefill_seq_len=prompt_len, batch_size=batch_size
        )
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

    qeff_model.compile(
        img_size=model_config["img_size"],
        num_devices=num_devices,
        prefill_seq_len=prompt_len,
        ctx_len=ctx_len,
        mxfp6=False,
        enable_qnn=enable_qnn,
        qnn_config=qnn_config,
        onnx_path=with_sub_func_onnx,
    )

    print("Output With Subfunction Enabled:")
    output = qeff_model.generate(inputs=inputs, generation_len=NEW_GENERATION_TOKENS, streamer=streamer)
    tokens_sub = output.generated_ids[:, :-1]

    qeff_model.compile(
        img_size=model_config["img_size"],
        num_devices=num_devices,
        prefill_seq_len=prompt_len,
        ctx_len=ctx_len,
        mxfp6=False,
        enable_qnn=enable_qnn,
        qnn_config=qnn_config,
        onnx_path=without_sub_func_onnx,
    )

    print("Output With Subfunction Not Enabled:")
    output = qeff_model.generate(inputs=inputs, generation_len=NEW_GENERATION_TOKENS, streamer=streamer)
    tokens_no_sub = output.generated_ids[:, :-1]

    assert (tokens_sub == tokens_no_sub).all(), "Tokens don't match for pytorch HF output and QPC output"
    return
