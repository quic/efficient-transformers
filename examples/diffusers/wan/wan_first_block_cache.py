# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import torch
from diffusers.utils import export_to_video

from QEfficient import QEffWanPipeline

# Load the pipeline with first-block cache enabled.
pipeline = QEffWanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    enable_first_cache=True,
)


# ============================================================================
# OPTIONAL: REDUCE MODEL LAYERS FOR FASTER INFERENCE
# ============================================================================
# Reduce the number of transformer blocks to speed up video generation.
#
# Trade-off: Faster inference but potentially lower video quality
# Use case: Quick testing, prototyping, or when speed is critical
#
# Uncomment the following lines to use only a subset of transformer layers:
#
# Configure for 2-layer model (faster inference)
# pipeline.transformer_high.model.config["num_layers"] = 2
# pipeline.transformer_low.model.config["num_layers"] = 2

# # # Reduce high noise transformer blocks
# original_blocks = pipeline.transformer_high.model.blocks
# pipeline.transformer_high.model.blocks = torch.nn.ModuleList(
#     [original_blocks[i] for i in range(0, pipeline.transformer_high.model.config["num_layers"])]
# )

# # Reduce low noise transformer blocks
# org_blocks = pipeline.transformer_low.model.blocks
# pipeline.transformer_low.model.blocks = torch.nn.ModuleList(
#     [org_blocks[i] for i in range(0, pipeline.transformer_low.model.config["num_layers"])]
# )


prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

# Alternative video dimensions for different use cases, corresponding default blocking
# height=192, width=320    # ATTENTION_BLOCKING_MODE=kv head_block_size=16 num_kv_blocks=3 python3  examples/diffusers/wan/wan_lightning.py
# height=480, width=832    #  ATTENTION_BLOCKING_MODE=qkv head_block_size=16 num_kv_blocks=21 num_q_blocks=2 python3  examples/diffusers/wan/wan_lightning.py
# height=720, width=1280   #  ATTENTION_BLOCKING_MODE=qkv head_block_size=16 num_kv_blocks=48 num_q_blocks=5 python3  examples/diffusers/wan/wan_lightning.py

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=81,  # Number of video frames to generate
    guidance_scale=4.0,  # Primary guidance scale
    guidance_scale_2=3.0,  # Secondary guidance scale for dual guidance
    num_inference_steps=40,  # Lightning model uses fewer steps
    generator=torch.Generator().manual_seed(42),  # For reproducible results
    custom_config_path="examples/diffusers/wan/wan_config.json",
    height=96,
    width=160,
    use_onnx_subfunctions=False,  # Enable ONNX optimizations
    parallel_compile=True,  # Set to True for parallel compilation
    cache_threshold_high=0.01,
    cache_threshold_low=0.01,
)

# Extract generated frames and export to video
frames = output.images[0]
export_to_video(frames, "wa.mp4", fps=16)
print(output)
