# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
FLUX.1-schnell Image Generation Example

This example demonstrates how to use the QEffFluxPipeline to generate images
using the FLUX.1-schnell model from Black Forest Labs. FLUX.1-schnell is a
fast, distilled version of the FLUX.1 text-to-image model optimized for
speed with minimal quality loss.
"""

import torch

from QEfficient import QEffFluxPipeline

# Initialize the FLUX.1-schnell pipeline from pretrained weights
pipeline = QEffFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")

# Generate an image from a text prompt
# use_onnx_subfunctions=True enables ONNX-based optimizations for faster compilation
output = pipeline(
    prompt="A laughing girl",
    height=1024,
    width=1024,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=torch.manual_seed(42),
    parallel_compile=True,
    use_onnx_subfunctions=False,
)

# Extract the generated image from the output
image = output.images[0]

# Save the generated image to disk
image.save("girl_laughing.png")

# Print the output object (contains perf info)
print(output)
