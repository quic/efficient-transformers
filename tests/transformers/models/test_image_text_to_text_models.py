# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from io import BytesIO
from typing import List

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
    TextStreamer,
)

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from QEfficient.utils import hf_download
from QEfficient.utils._utils import get_num_layers_vlm
from QEfficient.utils.device_utils import get_available_device_id
from QEfficient.utils.run_utils import ApiRunnerVlm
from QEfficient.utils.test_utils import InternProcessor, InternVLModelWrapper

HF_TOKEN = ""
NEW_GENERATION_TOKENS = 2
test_models_config = [
    # (
    #     "meta-llama/Llama-3.2-11B-Vision-Instruct",
    #     1,
    #     32,
    #     512,
    #     560,
    #     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
    #     "Explain this image",
    # ),
    (
        "llava-hf/llava-1.5-7b-hf",
        1,
        784,
        1024,
        336,
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg",
        "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud",
    ),
]

intern_model_config = [
    # (
    #     "OpenGVLab/InternVL2_5-1B",
    #     1,
    #     3840,
    #     4096,
    #     "https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-1-2048.jpg",
    #     "Please describe the image in detail.",
    # )
]


### Seems no more changes would be required. Probably done for all.
def load_image_text_to_text_model(model_config):
    model_path = hf_download(
        repo_id=model_config._name_or_path,
        hf_token=HF_TOKEN,
        ignore_patterns=["*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.h5", "*.msgpack"],
    )
    try:
        model_hf = AutoModelForImageTextToText.from_pretrained(
            model_path,
            # _attn_implementation="eager",
            low_cpu_mem_usage=False,
            token=HF_TOKEN,
            config=model_config,
        )
    except ValueError:
        model_hf = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=False,
            token=HF_TOKEN,
            trust_remote_code=True,
            config=model_config,
        )
    params = sum(p.numel() for p in model_hf.parameters())
    model_hf.eval()
    return model_hf, params


### Add for InternVL and see if config.text_config could be used to decide layer names instead of architectures
def set_num_layers(config, n_layer=1):
    ## -1 indicates use all the layers of the model.
    if n_layer == -1:
        return config
    elif hasattr(config, "model_type") and "mllama" in config.model_type:
        config.text_config.num_hidden_layers = n_layer
        config.text_config.cross_attention_layers = [
            x for x in config.text_config.cross_attention_layers if x < n_layer
        ]
        # breakpoint()
    elif hasattr(config, "text_config"):
        config.text_config.num_hidden_layers = n_layer
        config.vision_config.num_hidden_layers = n_layer
    elif hasattr(config, "llm_config"):
        config.llm_config.num_hidden_layers = n_layer
        config.vision_config.num_hidden_layers = n_layer
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
):
    # breakpoint()
    model_config = {"model_name": model_name}
    model_config["img_size"] = img_size
    config = AutoConfig.from_pretrained(
        model_config["model_name"], token=HF_TOKEN, trust_remote_code=True, padding=True
    )
    config._attn_implementation = "eager"
    config = set_num_layers(config, n_layer=n_layer)
    model_hf, _ = load_image_text_to_text_model(config)
    processor = AutoProcessor.from_pretrained(model_name, token=HF_TOKEN, trust_remote_code=True, padding=True)

    n_layer = get_num_layers_vlm(config)
    image = Image.open(requests.get(img_url, stream=True).raw)
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
    streamer = TextStreamer(processor.tokenizer)
    # breakpoint()
    checks = []
    pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch(model_hf, inputs)
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_config["model_name"],
        kv_offload=kv_offload,
        config=config,
        token=HF_TOKEN,
    )
    ### Using only late fusion api for both llava and llama right now.
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    pytorch_kv_tokens = api_runner.run_late_fusion_vlm_kv_model_on_pytorch(qeff_model.model, inputs)
    # breakpoint()
    # assert (pytorch_kv_tokens == pytorch_hf_tokens).all(), (
    #     "Tokens don't match for pytorch HF output and pytorch KV output"
    # )
    checks.append(pytorch_hf_tokens == pytorch_kv_tokens)
    # breakpoint()
    onnx_model_path = qeff_model.export()

    ort_tokens = api_runner.run_late_fusion_vlm_kv_model_on_ort(onnx_model_path)
    breakpoint()
    # assert (pytorch_hf_tokens == ort_tokens).all(), "Tokens don't match for pytorch HF output and ORT output"
    checks.append(pytorch_hf_tokens == ort_tokens)
    if not get_available_device_id():
        pytest.skip("No available devices to run model on Cloud AI 100")
    qeff_model.compile(
        img_size=model_config["img_size"],
        num_devices=num_devices,
        prefill_seq_len=prompt_len,
        ctx_len=ctx_len,
        mxfp6=False,
    )
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    output = qeff_model.generate(inputs=inputs, generation_len=NEW_GENERATION_TOKENS, streamer=streamer)
    qpc_tokens = output.generated_ids[:, :-1]
    # breakpoint()
    # print(processor.tokenizer.batch_decode(output.generated_ids))
    # breakpoint()
    # assert (pytorch_hf_tokens == qpc_tokens).all(), "Tokens don't match for pytorch HF output and QPC output"
    checks.append(pytorch_hf_tokens == qpc_tokens)
    breakpoint()
    del model_hf
    del qeff_model
    del api_runner
    del processor
    return


def run_intern_model_on_pytorch(model, tokenizer, processor, img_file):
    pixel_values = processor.load_image(img_file, max_num=12)
    question = "<image>\n" + "Please describe the image shortly."
    inputs = tokenizer(question, return_tensors="pt")
    inputs["pixel_values"] = pixel_values.clone()
    # generation_config = dict(max_new_tokens=NEW_GENERATION_TOKENS, do_sample=True, bos_token_id=1)
    model_wrapper = InternVLModelWrapper(model)
    input_ids_len = len(inputs["input_ids"][0])
    outputs = model_wrapper(**inputs)
    logits = outputs.logits[:, -1, :]
    predicted_token_id = torch.argmax(logits, dim=-1)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], predicted_token_id.unsqueeze(1)], dim=-1)
    # change hardcode
    for _ in range(15):
        outputs = model_wrapper(inputs["input_ids"])
        logits = outputs.logits[:, -1, :]
        predicted_token_id = torch.argmax(logits, dim=-1)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], predicted_token_id.unsqueeze(1)], dim=-1)

    generated_ids = inputs["input_ids"][0][input_ids_len:].detach().numpy()
    tokenizer.decode(generated_ids, skip_special_tokens=True)
    return


def run_intern_kv_model_on_pytorch(model, inputs, processor, ctx_len, batch_size=1):
    head_dim = model.language_model.config.hidden_size // model.language_model.config.num_attention_heads
    inputs["past_key_values"] = [
        tuple(
            [
                torch.zeros(
                    batch_size,
                    model.language_model.config.num_key_value_heads,
                    ctx_len,
                    head_dim,
                    dtype=torch.float32,
                )
                for _ in range(2)
            ]
        )
        for _ in range(model.language_model.config.num_hidden_layers)
    ]

    streamer = TextStreamer(processor.tokenizer)
    generation_len = NEW_GENERATION_TOKENS
    generated_ids = torch.full((batch_size, generation_len + 1), processor.tokenizer.pad_token_id)
    pt_outputs = model(**inputs)
    inputs["input_ids"] = pt_outputs[0].argmax(2)
    inputs["position_ids"] = inputs["position_ids"].max(1, keepdim=True).values + 1
    streamer.put(inputs["input_ids"])
    generated_ids[:, 0] = inputs["input_ids"].squeeze(1)
    finished_sequences = inputs["input_ids"] == processor.tokenizer.eos_token_id
    for i in range(1, generation_len):
        outputs = model(**inputs)
        inputs["input_ids"] = outputs[0].argmax(2)
        streamer.put(inputs["input_ids"])
        inputs["position_ids"] += 1
        generated_ids[:, i] = inputs["input_ids"].squeeze(1)
        finished_sequences |= inputs["input_ids"] == processor.tokenizer.eos_token_id
        if finished_sequences.all():
            break

    streamer.end()

    generated_texts = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(generated_texts)
    return


def check_intern_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
    model_name: str,
    img_url: str,
    query: str,
    prompt_len: int,
    ctx_len: int,
    n_layer: int = 1,
    batch_size: int = 1,
):
    # model_config = {"model_name": model_name}
    # model_hf, _ = load_image_text_to_text_model(model_config)

    model_config = {"model_name": model_name}
    config = AutoConfig.from_pretrained(model_config["model_name"], token=HF_TOKEN)
    config._attn_implementation = "eager"
    config = set_num_layers(config, n_layer=n_layer)
    model_hf, _ = load_image_text_to_text_model(config)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    processor = InternProcessor(qeff_model.model, tokenizer)
    img = requests.get(img_url, stream=True)
    image = Image.open(BytesIO(img.content)).convert("RGB")
    image = image.resize((1000, 747))
    run_intern_model_on_pytorch(model_hf, tokenizer, processor, image)
    pixel_values = processor.load_image(image, max_num=12)
    question = "<image>\n" + query
    # Chat Template information for prompt preprocessing
    messages: List[List[str]] = []
    roles = ("<|im_start|>user\n", "<|im_start|>assistant\n")
    prompt = processor(pixel_values, question, messages, roles)
    inputs = tokenizer(prompt, return_tensors="pt")
    batch_size, prompt_len = inputs["input_ids"].shape
    inputs["pixel_values"] = pixel_values.clone()
    inputs["position_ids"] = torch.arange(prompt_len).view(1, -1)
    inputs.pop("attention_mask")
    run_intern_kv_model_on_pytorch(qeff_model.model, inputs, processor, ctx_len)

    return


@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name, batch_size, prompt_len, ctx_len, img_size, img_url, query", test_models_config)
def test_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
    model_name, batch_size, prompt_len, ctx_len, img_size, img_url, query
):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model,  without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """

    n_layer = 4
    # kv_offload = False
    # kv_offload = True
    # breakpoint()
    for offload in [True]:
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
            kv_offload=offload,
        )


@pytest.mark.parametrize("model_name, batch_size, prompt_len, ctx_len, img_url, query", intern_model_config)
def test_image_text_to_text_intern_pytorch_vs_kv_vs_ort_vs_ai100(
    model_name, batch_size, prompt_len, ctx_len, img_url, query
):
    check_intern_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        img_url=img_url,
        query=query,
        n_layer=1,
        batch_size=batch_size,
    )
