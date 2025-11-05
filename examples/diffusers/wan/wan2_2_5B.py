# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import torch
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from QEfficient import QEFFWanPipeline
model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipeline = QEFFWanPipeline.from_pretrained(model_id, vae=vae)

# ######## for single layer
# original_blocks = pipeline.transformer.model.blocks
# pipeline.transformer.model.blocks = torch.nn.ModuleList([original_blocks[0]])
# pipeline.transformer.config.num_layers = 1
# import pdb; pdb.set_trace()
pipeline.compile(num_devices_text_encoder=1, num_devices_transformer=4, num_devices_vae_decoder=1)
flow_shift = 3.0  # 5.0 for 720P, 3.0 for 480P
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, flow_shift=flow_shift)
pipeline.to("cpu")
prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_frames=61,
    guidance_scale=3.0,
    ).frames[0]
export_to_video(output, "output.mp4", fps=12)
