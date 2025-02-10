# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import requests
from PIL import Image
from transformers import AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForImageTextToText


def run_model(model_name, query):
    processor = AutoProcessor.from_pretrained(model_name, token="")
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    config.text_config.num_hidden_layers = 1
    model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_name, token="", attn_implementation="eager", kv_offload=False
    )
    prefill_seq_len = 32
    ctx_len = 512
    num_cores = 16
    num_devices = 4
    model.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        num_cores=num_cores,
        num_devices=num_devices,
        img_size=560,
        mxfp6_matmul=False,
    )

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    # image = Image.open("/home/ubuntu/amitraj/mllama_support/mllama_code/efficient-transformers/Image (3).jpg")
    query = query
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": query},
            ],
        }
    ]
    input_text = [processor.apply_chat_template(messages, add_generation_prompt=True)]

    split_inputs = processor(
        text=input_text,
        images=image,
        return_tensors="pt",
        add_special_tokens=False,
        # padding="max_length",
        # max_length=prefill_seq_len,
    )

    streamer = TextStreamer(processor.tokenizer)
    output = model.generate(inputs=split_inputs, device_ids=[0, 1, 2, 3], streamer=streamer)
    print(output)


run_model(model_name="meta-llama/Llama-3.2-11B-Vision-Instruct", query="explain this image")
