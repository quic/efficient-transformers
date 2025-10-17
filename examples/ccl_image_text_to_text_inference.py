# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
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
    comp_ctx_lengths_prefill=None,
    comp_ctx_lengths_decode=None,
    generation_len=128,
    img_size=560,
    num_cores=16,
    num_devices=1,
):
    ## STEP - 1 Load the Processor and Model

    processor = AutoProcessor.from_pretrained(model_name, token=token)

    # `kv_offload` is used to compile the model in a Single QPC or 2 QPCs.
    # The Dual QPC approach splits the model to perform Image Encoding and Output generation in 2 different QPCs.
    # The outputs of the Vision Encoder are then passed to the Language model via host in this case.

    model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_name,
        token=token,
        attn_implementation="eager",
        kv_offload=kv_offload,
        comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
        comp_ctx_lengths_decode=comp_ctx_lengths_decode,
        ctx_len=ctx_len,
    )

    ## STEP - 2 Export & Compile the Model

    model.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        img_size=img_size,
        num_cores=num_cores,
        num_devices=num_devices,
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
    output_statistics = model.generate(inputs=inputs, streamer=streamer, generation_len=generation_len)
    print(output_statistics)


if __name__ == "__main__":
    # Model name and Input parameters
    model_name = "llava-hf/llava-1.5-7b-hf"
    # model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    query = "Describe this image."
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"

    # Compilation parameters for the model
    kv_offload = True
    prefill_seq_len = 32
    ctx_len = 8192
    generation_len = 128
    img_size = 336
    # img_size = 560
    num_cores = 16
    num_devices = 4
    comp_ctx_lengths_prefill = [4096]
    comp_ctx_lengths_decode = [6144, ctx_len]

    run_model(
        model_name=model_name,
        token=HF_TOKEN,
        query=query,
        kv_offload=kv_offload,
        image_url=image_url,
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
        comp_ctx_lengths_decode=comp_ctx_lengths_decode,
        generation_len=generation_len,
        img_size=img_size,
        num_cores=num_cores,
        num_devices=num_devices,
    )


"""
Expected Response:

This image depicts a charming anthropomorphic rabbit standing on a dirt path in front of a picturesque stone cottage, surrounded by a serene landscape.

The rabbit, with its light brown fur and distinctive long ears, is attired in a stylish blue coat, brown vest, and tan pants, exuding a sense of sophistication. The dirt path, flanked by vibrant flowers and lush greenery, leads to the cottage, which features a thatched roof and a chimney, adding to the rustic charm of the scene. In the background, rolling hills and trees create a breathtaking panorama, while the sky above is a brilliant blue with white clouds, completing the

"""
