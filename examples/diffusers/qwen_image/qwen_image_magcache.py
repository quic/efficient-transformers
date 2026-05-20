# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
"""
Qwen-Image MagCache Example

This example demonstrates how to enable/disable runtime MagCache for Qwen-Image
while using a custom config file that can point to precompiled QPC paths.
"""

import torch

from QEfficient import QEffQwenImagePipeline

# Initialize the Qwen Image pipeline from pretrained weights
pipe = QEffQwenImagePipeline.from_pretrained("Qwen/Qwen-Image")

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",
}

prompt = """A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197"."""
negative_prompt = ""

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

# MagCache knobs
use_magcache = True
magcache_thresh = 0.06
magcache_K = 2
magcache_retention_ratio = 0.2
magcache_verbose = True

output = pipe(
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
    use_magcache=use_magcache,
    magcache_thresh=magcache_thresh,
    magcache_K=magcache_K,
    magcache_retention_ratio=magcache_retention_ratio,
    magcache_verbose=magcache_verbose,
)

image = output.images[0]
image.save("qwen_image_magcache.png")
print(output)
