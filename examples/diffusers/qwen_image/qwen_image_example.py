# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
"""
Qwen-Image Image Generation Example

This example demonstrates how to use the QEFFQwenImagePipeline to generate images
#TODO update docs
"""

import torch

from QEfficient import QEFFQwenImagePipeline

# Initialize the Qwen Image pipeline from pretrained weights
pipe = QEFFQwenImagePipeline.from_pretrained("Qwen/Qwen-Image")

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
}

# # # Config for two layers
# original_blocks = pipe.transformer.model.transformer_blocks
# pipe.transformer.model.transformer_blocks = torch.nn.ModuleList([original_blocks[0], original_blocks[1]])
# pipe.transformer.model.config.num_layers = 2

# Generate image
prompt = """A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197"."""
negative_prompt = ""

# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["16:9"]  # 512, 512

output = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=5,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cpu").manual_seed(42),
    parallel_compile=True,
    max_sequence_length=128,
)

# Extract the generated image from the output
image = output.images[0]

# Save the generated image to disk
image.save("qwen_image_t2v.png")  # working with neg prompt
print(output)
