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

# Add HuggingFace Token to access the model
HF_TOKEN = ""


def run_model(
    model_name,
    token,
    query,
    image_url,
    kv_offload=False,
    prefill_seq_len=32,
    ctx_len=512,
    img_size=448,
    num_cores=16,
    num_devices=1,
):
    ## STEP - 1 Load the Processor and Model

    processor = AutoProcessor.from_pretrained(model_name, token=token)
    # `kv_offload` is used to decide if we wish to run Single QPC or 2 QPC setup
    model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_name, token=token, attn_implementation="eager", kv_offload=kv_offload
    )

    ## STEP - 2 Export & Compile the Model

    model.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        num_cores=num_cores,
        num_devices=num_devices,
        img_size=img_size,
        mxfp6_matmul=False,
    )

    ## STEP - 3 Load and process the inputs for Inference

    image = Image.open(requests.get(image_url, stream=True).raw)
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

    inputs = processor(
        text=input_text,
        images=image,
        return_tensors="pt",
        add_special_tokens=False,
        padding="max_length",
        max_length=prefill_seq_len,
    )

    ## STEP - 4 Run Inference on the compiled model

    streamer = TextStreamer(processor.tokenizer)
    model.generate(inputs=inputs, streamer=streamer)


if __name__ == "__main__":
    # Model name and Input parameters
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    query = "Describe this image"
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"

    # Compilation parameters for the model
    kv_offload = False
    prefill_seq_len = 32
    ctx_len = 512
    img_size = 448
    num_cores = 16
    num_devices = 1

    run_model(
        model_name=model_name,
        token=HF_TOKEN,
        query=query,
        kv_offload=kv_offload,
        image_url=image_url,
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        img_size=img_size,
        num_cores=num_cores,
        num_devices=num_devices,
    )
