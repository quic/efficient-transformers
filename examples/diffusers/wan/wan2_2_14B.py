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
# vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
# pipeline = QEFFWanPipeline.from_pretrained(model_id, vae=vae)


model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipeline = QEFFWanPipeline.from_pretrained(model_id, vae=vae)
import pdb; pdb.set_trace()
pipeline.transformer.model.config.num_layers = 2
pipeline.transformer_2.model.config.num_layers = 2
original_blocks = pipeline.transformer.model.blocks
org_blocks = pipeline.transformer_2.model.blocks
pipeline.transformer.model.blocks = torch.nn.ModuleList([original_blocks[i] for i in range(0,pipeline.transformer.model.config.num_layers)]) # 2 layers
pipeline.transformer_2.model.blocks = torch.nn.ModuleList([org_blocks[i] for i in range(0,pipeline.transformer_2.model.config.num_layers)]) # 2 layers
pipeline.to("cpu")


pipeline.compile(num_devices_transformer=2,num_devices_transformer_2=2)
# flow_shift = 3.0  # 5.0 for 720P, 3.0 for 480P
height = 480 # for 480-P
width = 832

generator = torch.manual_seed(42)
prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=61,
    guidance_scale=3.0,
    num_inference_steps = 50,
    generator = generator,
    ).frames[0]
export_to_video(output, "output.mp4", fps=12)
