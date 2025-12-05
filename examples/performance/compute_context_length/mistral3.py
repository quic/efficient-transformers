# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import requests
from PIL import Image
from transformers import AutoConfig, AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForImageTextToText


def run_model(
    model_name,
    query,
    image_url,
    kv_offload=False,
    prefill_seq_len=128,
    ctx_len=4096,
    ccl_enabled=False,
    comp_ctx_lengths_prefill=None,
    comp_ctx_lengths_decode=None,
    generation_len=128,
    img_size=1540,
    num_cores=16,
    num_devices=4,
):
    ## STEP - 1 Load the Processor and Model

    processor = AutoProcessor.from_pretrained(model_name)

    # `kv_offload` is used to compile the model in a 2 QPCs.Currently we are not supporting 1 qpc so the flag false is not allowed.
    # The `kv_offload` flag should always be set to True.
    # The Dual QPC approach splits the model to perform Image Encoding and Output generation in 2 different QPCs.
    # The outputs of the Vision Encoder are then passed to the Language model via host in this case.

    config = AutoConfig.from_pretrained(model_name)
    config.vision_config._attn_implementation = "eager"
    # For Testing Purpose Only
    config.text_config.num_hidden_layers = 4
    config.vision_config.num_hidden_layers = 2

    model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_name,
        kv_offload=kv_offload,
        config=config,
        qaic_config={
            "ccl_enabled": ccl_enabled,
        },
    )

    ## STEP - 2 Export & Compile the Model

    model.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        img_size=img_size,
        num_cores=num_cores,
        num_devices=num_devices,
        mxfp6_matmul=False,
        comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
        comp_ctx_lengths_decode=comp_ctx_lengths_decode,
    )

    ## STEP - 3 Load and process the inputs for Inference

    # We are resizing the image to (w x h) (1540 x 1540) so that any image can work on the model irrespective of image dimensssions
    # we have a fixed size of height 1540 and width 1540 as defined in the config

    image = Image.open(requests.get(image_url, stream=True).raw)
    image = image.resize((1540, 1540))

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt")

    ## STEP - 4 Run Inference on the compiled model

    streamer = TextStreamer(processor.tokenizer)
    output = model.generate(inputs=inputs, streamer=streamer, generation_len=generation_len)
    print(output)


if __name__ == "__main__":
    # Model name and Input parameters
    model_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

    # Please add prompt here
    query = "Describe the image"

    # Please pass image url or image path .The format of the image should be jpg.
    image_url = "https://www.ilankelman.org/stopsigns/australia.jpg"

    # Compilation parameters for the model
    kv_offload = True
    prefill_seq_len = 128
    ctx_len = 8192
    generation_len = 128
    num_cores = 16
    num_devices = 4
    ccl_enabled = True
    comp_ctx_lengths_prefill = [4096]
    comp_ctx_lengths_decode = [6144, ctx_len]

    run_model(
        model_name=model_name,
        query=query,
        kv_offload=kv_offload,
        image_url=image_url,
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        ccl_enabled=ccl_enabled,
        comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
        comp_ctx_lengths_decode=comp_ctx_lengths_decode,
        generation_len=generation_len,
        num_cores=num_cores,
        num_devices=num_devices,
    )


"""
Expected Response:
The image depicts a street scene in what appears to be a Chinatown district. The focal point is a traditional Chinese archway, known as a paifang, which is intricately designed with red columns and ornate details. The archway features Chinese characters at the top, which translate to "Chinatown Gate."
In the foreground, there is a red stop sign mounted on a pole. The street is relatively quiet, with a single dark-colored SUV driving through the archway. On either side of the archway, there are stone lion statues, which are common decorative elements in Chinese architecture and symbolize protection.


"""
