# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
FLUX.1-schnell Image Generation Example

This example demonstrates how to use the QEFFFluxPipeline to generate images
using the FLUX.1-schnell model from Black Forest Labs. FLUX.1-schnell is a
fast, distilled version of the FLUX.1 text-to-image model optimized for
speed with minimal quality loss.

Key Features:
- Fast inference with only 4 steps
- High-quality image generation from text prompts
- Optimized for Qualcomm Cloud AI 100 using ONNX runtime
- Deterministic output using fixed random seed

Output:
- Generates an image based on the text prompt
- Saves the image as 'cat_with_sign.png' in the current directory
"""

import torch

from QEfficient import QEFFFluxPipeline

# Initialize the FLUX.1-schnell pipeline from pretrained weights
# use_onnx_function=True enables ONNX-based optimizations for faster compilation
pipeline = QEFFFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", use_onnx_function=False)

# Generate an image from a text prompt
output = pipeline(
    prompt="A cat holding a sign that says hello world",
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=torch.manual_seed(42),
)

# Extract the generated image from the output
image = output.images[0]

# Save the generated image to disk
image.save("cat_with_sign.png")

# Print the output object (contains perf info)
print(output)
