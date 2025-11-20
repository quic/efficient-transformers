# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""
Continuous Batching Example for Llama-4-Scout Vision Model

This example demonstrates how to use continuous batching with vision-language models
to process multiple image-text pairs simultaneously in a single batch.
"""

import transformers
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText

# Model configuration
model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

## STEP 1: Load Model Configuration and Processor
config = AutoConfig.from_pretrained(model_id)
# For Testing Purpose Only - reduce layers for faster testing
config.text_config.num_hidden_layers = 4
config.vision_config.num_hidden_layers = 2

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

## STEP 2: Initialize Model with Continuous Batching
# Enable continuous batching to process multiple prompts in parallel
# Set kv_offload=True for Dual QPC mode (vision encoder + language model separately)
qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id,
    attn_implementation="eager",
    kv_offload=True,  # Dual QPC mode
    config=config,
    continuous_batching=True,  # Enable continuous batching
)

## STEP 3: Compile the Model for Cloud AI 100
# Configure compilation parameters for continuous batching
qeff_model.compile(
    prefill_seq_len=128,
    ctx_len=3072,
    img_size=336,
    num_cores=16,
    num_devices=4,
    max_num_tiles=17,
    batch_size=1,  # Batch size per request
    full_batch_size=4,  # Total batch size for continuous batching
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    aic_enable_depth_first=True,
    mos=1,
)

## STEP 4: Prepare Input Images and Prompts
# Define multiple images to process in the batch
image_urls = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
]

# Define corresponding prompts for each image
prompts = [
    "Can you describe the image in detail?",
    "What are the objects in the image?",
    "What is the main subject of the image?",
    "What colors are predominant in the image?",
]

## STEP 5: Run Inference with Continuous Batching
# Process all image-prompt pairs in a single batch
exec_info = qeff_model.generate(
    tokenizer=tokenizer,
    prompts=prompts,
    processor=processor,
    images=image_urls,  # Images are processed with their corresponding prompts
    device_ids=[0, 1, 2, 3],
    generation_len=100,
)

## STEP 6: Display Results
print("Generated IDs:", exec_info.generated_ids)
print("\nFull execution info:")
print(exec_info)
