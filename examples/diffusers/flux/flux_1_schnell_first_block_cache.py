# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
FLUX.1-schnell first-block-cache example.
"""

import torch

from QEfficient import QEffFluxPipeline

pipeline = QEffFluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    enable_first_block_cache=True,
    # Hidden-dimension downsampling used for first-block residual similarity check.
    first_block_cache_downsample_factor=4,
)

output = pipeline(
    prompt="A laughing girl",
    custom_config_path="examples/diffusers/flux/flux_config.json",
    height=256,
    width=256,
    guidance_scale=0.0,
    num_inference_steps=40,
    max_sequence_length=256,
    generator=torch.Generator("cpu").manual_seed(42),
    parallel_compile=True,
    use_onnx_subfunctions=False,
    cache_threshold=0.045,
)

image = output.images[0]
image.save("girl_laughing_first_block_cache.png")
print(output)
