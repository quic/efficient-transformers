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


def run_model(
    model_name,
    token,
    query,
    image_url,
    kv_offload=False,
    prefill_seq_len=5500,
    ctx_len=6000,
    ccl_enabled=False,
    comp_ctx_lengths_prefill=None,
    comp_ctx_lengths_decode=None,
    generation_len=128,
    img_size=384,
    num_cores=16,
    num_devices=1,
):
    ## STEP - 1 Load the Processor and Model

    processor = AutoProcessor.from_pretrained(model_name, token=token)

    # `kv_offload` is used to compile the model in a 2 QPCs.Currently we are not supporting 1 qpc so the flag false is not allowed.
    # The `kv_offload` flag should always be set to True.
    # The Dual QPC approach splits the model to perform Image Encoding and Output generation in 2 different QPCs.
    # The outputs of the Vision Encoder are then passed to the Language model via host in this case.

    model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_name,
        token=token,
        kv_offload=kv_offload,
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

    # We are resizing the image to (w x h) (1610 x 1109) so that any image can work on the model irrespective of image dimensssions
    # we have a fixed size of height 1109 and width 1610

    image = Image.open(requests.get(image_url, stream=True).raw)
    image = image.resize((1610, 1109))

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt")

    ## STEP - 4 Run Inference on the compiled model

    streamer = TextStreamer(processor.tokenizer)
    output = model.generate(inputs=inputs, streamer=streamer, generation_len=generation_len)
    print(output)


if __name__ == "__main__":
    # Model name and Input parameters
    model_name = "ibm-granite/granite-vision-3.2-2b"

    # Please add prompt here
    query = "Describe the image"

    # Please pass image url or image path .The format of the image should be jpg.
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    # Compilation parameters for the model
    kv_offload = True
    prefill_seq_len = 5500
    ctx_len = 8192
    generation_len = 128
    img_size = 384
    num_cores = 16
    num_devices = 4
    ctx_len = 8192
    ccl_enabled = True
    comp_ctx_lengths_prefill = [5500]
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
        img_size=img_size,
        num_cores=num_cores,
        num_devices=num_devices,
    )


"""
Expected Response:

The image depicts two cats lying on a pink blanket that is spread out on a red couch. The cats are positioned in a relaxed manner, with their bodies stretched out and their heads resting on the blanket. 
The cat on the left is a smaller, tabby cat with a mix of black, gray, and white fur. It has a long, slender body and a distinctive tail that is curled up near its tail end. The cat on the right is a larger, 
tabby cat with a mix of gray, black, and brown fur. It has

"""
