# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import safetensors.torch
import torch
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_wan_lora_to_diffusers
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download

from QEfficient import QEffWanPipeline

# Load the pipeline
pipeline = QEffWanPipeline.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers")

# Download the LoRAs
high_noise_lora_path = hf_hub_download(
    repo_id="lightx2v/Wan2.2-Lightning",
    filename="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors",
)
low_noise_lora_path = hf_hub_download(
    repo_id="lightx2v/Wan2.2-Lightning",
    filename="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors",
)


# LoRA conversion
def load_wan_lora(path: str):
    return _convert_non_diffusers_wan_lora_to_diffusers(safetensors.torch.load_file(path))


# Load into the transformers
pipeline.transformer.model.transformer_high.load_lora_adapter(
    load_wan_lora(high_noise_lora_path), adapter_name="high_noise"
)
pipeline.transformer.model.transformer_high.set_adapters(["high_noise"], weights=[1.0])
pipeline.transformer.model.transformer_low.load_lora_adapter(
    load_wan_lora(low_noise_lora_path), adapter_name="low_noise"
)
pipeline.transformer.model.transformer_low.set_adapters(["low_noise"], weights=[1.0])

prompt = "In a warmly lit living room, an elderly man with gray hair sits in a wooden armchair adorned with a blue cushion. He wears a gray cardigan over a white shirt, engrossed in reading a book. As he turns the pages, he subtly adjusts his posture, ensuring his glasses stay in place. He then removes his glasses, holding them in his hand, and turns his head to the right, maintaining his grip on the book. The soft glow of a bedside lamp bathes the scene, creating a calm and serene atmosphere, with gentle shadows enhancing the intimate setting."

output = pipeline(
    prompt=prompt,
    num_frames=81,
    guidance_scale=1.0,
    guidance_scale_2=1.0,
    num_inference_steps=4,
    generator=torch.manual_seed(0),
    height=480,
    width=832,
    use_onnx_subfunctions=True,
    parallel_compile=True,
)
frames = output.images[0]
export_to_video(frames, "output_t2v.mp4", fps=16)
print(output)
