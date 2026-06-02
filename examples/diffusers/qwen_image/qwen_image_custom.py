# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""
Qwen Image Custom Configuration Example

This example demonstrates how to customize the Qwen Image model with various options:
1. Custom image dimensions and aspect ratios
2. Optional scheduler customization
3. Optional reduced transformer layers for faster iteration
4. Optional explicit compilation with custom config
5. Runtime config usage via JSON config file

Use this example as a starting point for your own Qwen Image workflow.
"""

import torch

from QEfficient import QEffQwenImagePipeline

# ============================================================================
# PIPELINE INITIALIZATION WITH CUSTOM PARAMETERS
# ============================================================================

# Option 1: Basic initialization
pipeline = QEffQwenImagePipeline.from_pretrained("Qwen/Qwen-Image")

# Option 2: Advanced initialization with custom modules (example)
# pipeline = QEffQwenImagePipeline.from_pretrained(
#     "Qwen/Qwen-Image",
#     transformer=custom_transformer,
#     vae=custom_vae,
#     text_encoder=custom_text_encoder,
#     tokenizer=custom_tokenizer,
# )

# ============================================================================
# OPTIONAL: CUSTOM SCHEDULER CONFIGURATION
# ============================================================================
# pipeline.scheduler = custom_scheduler.from_config(pipeline.scheduler.config)

# ============================================================================
# OPTIONAL: REDUCE MODEL LAYERS FOR FASTER INFERENCE
# ============================================================================
# Trade-off: faster generation with possible quality drop
original_blocks = pipeline.transformer.model.transformer_blocks
pipeline.transformer.model.transformer_blocks = torch.nn.ModuleList([original_blocks[0], original_blocks[1]])
pipeline.transformer.model.config["num_layers"] = 2

# ============================================================================
# OPTIONAL: COMPILE WITH CUSTOM CONFIGURATION
# ============================================================================
# NOTE-1: If compile_config is not specified, default qwen config is used.
# pipeline.compile(
#     compile_config="examples/diffusers/qwen_image/qwen_config.json",
#     parallel=True,
#     height=928,
#     width=1664,
#     use_onnx_subfunctions=True,
# )

# ============================================================================
# IMAGE GENERATION WITH CUSTOM RUNTIME CONFIGURATION
# ============================================================================
# Generate an image using the configured pipeline.
#
# Note: Use of custom_config_path provides flexibility to set device_ids for each
#       module, so you can skip the separate pipeline.compile() step.

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",
}

prompt = (
    "A coffee shop entrance with a chalkboard sign reading 'Qwen Coffee $2 per cup', "
    "warm ambient lighting, realistic details"
)
negative_prompt = "low quality, blurry, distorted"

# Common Qwen image aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["16:9"]

output = pipeline(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cpu").manual_seed(42),
    custom_config_path="examples/diffusers/qwen_image/qwen_config.json",
    parallel_compile=True,
    max_sequence_length=128,
    use_onnx_subfunctions=False,
)

image = output.images[0]
image.save("qwen_image_custom.png")
print(output)
