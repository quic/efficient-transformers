# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import torch
from diffusers.utils import export_to_video

from QEfficient import QEffWanPipeline

# Non-unified WAN + first-block-cache (patch-based activation at load time).
pipeline = QEffWanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    use_unified=False,
    enable_first_block_cache=True,
    # Hidden-dimension downsampling used for first-block residual similarity check.
    first_block_cache_downsample_factor=4,
)

# ============================================================================
# OPTIONAL: REDUCE MODEL LAYERS FOR FASTER INFERENCE
# ============================================================================
# pipeline.transformer_high.model.config["num_layers"] = 2
# pipeline.transformer_low.model.config["num_layers"] = 2
#
# high_blocks = pipeline.transformer_high.model.blocks
# pipeline.transformer_high.model.blocks = torch.nn.ModuleList(
#     [high_blocks[i] for i in range(0, pipeline.transformer_high.model.config["num_layers"])]
# )
#
# low_blocks = pipeline.transformer_low.model.blocks
# pipeline.transformer_low.model.blocks = torch.nn.ModuleList(
#     [low_blocks[i] for i in range(0, pipeline.transformer_low.model.config["num_layers"])]
# )

prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=81,
    guidance_scale=4.0,
    guidance_scale_2=3.0,
    num_inference_steps=40,
    generator=torch.Generator().manual_seed(42),
    custom_config_path="examples/diffusers/wan/wan_non_unified_config.json",
    height=96,
    width=160,
    use_onnx_subfunctions=False,
    parallel_compile=True,
    cache_threshold_high=0.1,
    cache_threshold_low=0.065,
)

frames = output.images[0]
export_to_video(frames, "wan_first_block_cache.mp4", fps=16)
print(output)
