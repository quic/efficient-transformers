# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from io import BytesIO
from typing import List

import requests
import torch
from PIL import Image
from transformers import AutoTokenizer, TextStreamer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils.test_utils import InternProcessor


def run_intern_on_aic(
    model_name,
    prompts,
    image_urls,
    messages,
    roles,
    kv_offload=False,
    prefill_seq_len=128,
    ctx_len=4096,
    batch_size=1,
    num_patches=1,
    num_devices=1,
    num_cores=16,
):
    ## STEP 1 -- LOAD THE MODEL

    # The original Intern-VL model, despite being multimodal, is loaded using `AutoModelForCausalLM` in Huggingface.
    # To maintain compatibility, we load this model using `QEFFAutoModelForCausalLM`.

    model = QEFFAutoModelForCausalLM.from_pretrained(model_name, kv_offload=kv_offload, trust_remote_code=True)

    ## STEP 2 -- EXPORT & COMPILE THE MODEL

    model.compile(
        num_cores=num_cores,
        num_devices=num_devices,
        num_patches=num_patches,
        batch_size=batch_size,
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
    )

    ## STEP 3 -- SETUP THE PROCESSOR

    # InternVL doesn't have an AutoProcessor yet, so we will use our own processor class "InternProcessor"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    internProcessor = InternProcessor(model.model, tokenizer)

    ## STEP 4 -- PREPROCESS THE INPUTS

    pixel_values = []
    num_patches_list = []
    questions = []
    for i in range(len(prompts)):
        img = requests.get(image_urls[i], stream=True)
        image = Image.open(BytesIO(img.content)).convert("RGB")

        # Images are resized to (1000, 747) for inference
        image = image.resize((1000, 747))

        # preprocess the resized image
        pixel_value = internProcessor.load_image(image, max_num=12)
        num_patches_list.append(pixel_value.shape[0])
        pixel_values.append(pixel_value)

        question = "<image>\n" + prompts[i]
        questions.append(question)

    pixel_values = torch.cat(pixel_values, dim=0)

    # Chat Template information for prompt preprocessing
    messages: List[List[str]] = []
    roles = ("<|im_start|>user\n", "<|im_start|>assistant\n")
    query = internProcessor(pixel_values, questions, messages, roles, num_patches_list=num_patches_list)
    inputs = tokenizer(
        query, return_tensors="pt", padding="max_length", max_length=prefill_seq_len, padding_side="right"
    )

    inputs["pixel_values"] = pixel_values

    ## STEP 5 -- RUN INFERENCE VIA GENERATE FUNCTION
    streamer = TextStreamer(tokenizer)
    output = model.generate(inputs=inputs, streamer=streamer, generation_len=128)
    return output


if __name__ == "__main__":
    model_name = "OpenGVLab/InternVL2_5-1B"

    # Chat Template information for prompt preprocessing
    messages: List[List[str]] = []
    roles = ("<|im_start|>user\n", "<|im_start|>assistant\n")

    # Inputs for the model

    # Add additional prompts and image urls to the respective lists for multi batch compilation and inference
    prompts = ["Please describe the image in detail."]
    image_urls = [
        "https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-1-2048.jpg"
    ]

    ## Compilation parameters

    # `kv_offload` is used to compile the model in a Single QPC or 2 QPCs.
    # The Dual QPC approach splits the model to perform Image Encoding and Output generation in 2 different QPCs.
    # The outputs of the Vision Encoder are then passed to the Language model via host in this case.

    kv_offload = True

    # InternVL is an Early-Fusion model that uses placeholder tokens within the input_ids to interleave text_embeddings with
    # Image embeddings and generate final input_embeds for outout generation. Hence we need very large prefill_seq_len (3840 in this case) to
    # incorporate the memory for the merged embeddings.

    prefill_seq_len = 128
    ctx_len = 4096
    num_devices = 4
    num_cores = 16
    num_patches = 13

    output = run_intern_on_aic(
        model_name=model_name,
        prompts=prompts,
        image_urls=image_urls,
        messages=messages,
        roles=roles,
        kv_offload=kv_offload,
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        batch_size=len(prompts),
        num_patches=num_patches,
        num_devices=num_devices,
        num_cores=num_cores,
    )


"""
Expected Response:

The image is a promotional graphic for Microsoft Azure. It features a blue background with a hexagonal pattern on the left side. The hexagons are white and are arranged in a way that suggests a network or connectivity theme. 

On the right side of the image, the Microsoft Azure logo is prominently displayed. The logo consists of the Azure name in white, with the Microsoft logo above it, which includes four colored squares (blue, green, yellow, and red). Below the logo, the word "Azure" is written in large white letters.

Below the logo, there is text that reads:
- "By Dinesh Kumar Wick
"""
