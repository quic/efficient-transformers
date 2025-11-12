# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Single Image Inference Example for Llama-4-Scout Vision Model

This example demonstrates two modes:
1. Text-only mode (skip_vision=True): Run language model without image processing
2. Vision+Text mode (skip_vision=False): Process image and text together
"""

import torch
import transformers
from transformers import AutoConfig, AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForImageTextToText

# Model configuration
model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

## STEP 1: Load Model Configuration and Processor
config = AutoConfig.from_pretrained(model_id)
# For Testing Purpose Only - reduce layers for faster testing
config.text_config.num_hidden_layers = 4
config.vision_config.num_hidden_layers = 2

## STEP 2: Initialize the Model
# Set kv_offload=True for Dual QPC mode (vision encoder + language model separately)
qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id, attn_implementation="eager", kv_offload=True, config=config
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

# Toggle between text-only and vision+text modes
# Set skip_vision=True for text-only execution (no image processing)
# Set skip_vision=False for vision+text execution (process images with text)
skip_vision = True

if skip_vision:
    ## TEXT-ONLY MODE ##

    ## STEP 3: Compile Model for Text-Only Execution
    # Set skip_vision=True to bypass image processing
    qeff_model.compile(
        prefill_seq_len=128,
        ctx_len=3072,
        img_size=336,
        num_cores=16,
        num_devices=8,
        max_num_tiles=17,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        aic_enable_depth_first=True,
        skip_vision=True,  # Skip vision encoder for text-only inference
        mos=1,
    )

    ## STEP 4: Prepare Text-Only Input
    # Create a text-only message without any image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Tell me about yourself."},
            ],
        },
    ]

    ## STEP 5: Process Input with Chat Template
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    ## STEP 6: Run Text-Only Inference
    streamer = TextStreamer(tokenizer)
    output = qeff_model.generate(inputs=inputs, device_ids=[0, 1, 2, 3, 4, 5, 6, 7], generation_len=100)

    ## STEP 7: Display Results
    print(output.generated_ids)
    print(tokenizer.batch_decode(output.generated_ids))
    print(output)

else:
    ## VISION + TEXT MODE ##

    ## STEP 3: Compile Model for Vision+Text Execution
    # Do not set skip_vision (defaults to False) to enable image processing
    qeff_model.compile(
        prefill_seq_len=128,
        ctx_len=3072,
        img_size=336,
        num_cores=16,
        num_devices=8,
        max_num_tiles=17,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        aic_enable_depth_first=True,
        mos=1,
    )

    ## STEP 4: Prepare Image and Text Input
    # Define the image URL to process
    image_url = (
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png"
    )

    # Create a message with both image and text
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_url},
                {"type": "text", "text": "Can you describe the image in detail."},
            ],
        },
    ]

    ## STEP 5: Process Input with Chat Template
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    # Convert pixel values to float32 for processing
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

    ## STEP 6: Run Vision+Text Inference
    streamer = TextStreamer(tokenizer)
    output = qeff_model.generate(inputs=inputs, device_ids=[0, 1, 2, 3, 4, 5, 6, 7], generation_len=100)

    ## STEP 7: Display Results
    print(output.generated_ids)
    print(tokenizer.batch_decode(output.generated_ids))
    print(output)
