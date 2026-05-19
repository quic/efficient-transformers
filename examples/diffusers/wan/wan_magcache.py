# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import torch
from diffusers.utils import export_to_video

from QEfficient import QEffWanPipeline

# Non-unified WAN + MagCache runtime acceleration (no ONNX/QPC signature changes).
pipeline = QEffWanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    use_unified=False,
)

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
    parallel_compile=True,
    use_onnx_subfunctions=True,
    use_magcache=True,
    magcache_thresh=0.06,
    magcache_K=2,
    magcache_retention_ratio=0.4,
    magcache_verbose=False,
)

frames = output.images[0]
export_to_video(frames, "wan_magcache.mp4", fps=16)
print(output)
