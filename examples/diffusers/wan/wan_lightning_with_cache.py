# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
WAN Lightning Example with First Block Cache

This example demonstrates how to use the first block cache optimization
with WAN 2.2 Lightning for faster video generation on QAIC hardware.

First block cache can provide 30-50% speedup with minimal quality loss
by reusing computations from previous denoising steps.
"""

import safetensors.torch
import torch
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_wan_lora_to_diffusers
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download

from QEfficient import QEffWanPipeline

# Load the pipeline
print("Loading WAN 2.2 pipeline...")
pipeline = QEffWanPipeline.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers", enable_first_cache=True)

# Download the LoRAs for Lightning (4-step inference)
# print("Downloading Lightning LoRAs...")
# high_noise_lora_path = hf_hub_download(
#     repo_id="lightx2v/Wan2.2-Lightning",
#     filename="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors",
# )
# low_noise_lora_path = hf_hub_download(
#     repo_id="lightx2v/Wan2.2-Lightning",
#     filename="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors",
# )


# # LoRA conversion helper
# def load_wan_lora(path: str):
#     return _convert_non_diffusers_wan_lora_to_diffusers(safetensors.torch.load_file(path))


# # Load LoRAs into the transformers
# print("Loading LoRAs into transformers...")
# pipeline.transformer.model.transformer_high.load_lora_adapter(
#     load_wan_lora(high_noise_lora_path), adapter_name="high_noise"
# )
# pipeline.transformer.model.transformer_high.set_adapters(["high_noise"], weights=[1.0])

# pipeline.transformer.model.transformer_low.load_lora_adapter(
#     load_wan_lora(low_noise_lora_path), adapter_name="low_noise"
# )
# pipeline.transformer.model.transformer_low.set_adapters(["low_noise"], weights=[1.0])


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
# pipeline.transformer.model.transformer_high.config['num_layers'] = 10
# pipeline.transformer.model.transformer_low.config['num_layers']= 10

# # Reduce high noise transformer blocks
# original_blocks = pipeline.transformer.model.transformer_high.blocks
# pipeline.transformer.model.transformer_high.blocks = torch.nn.ModuleList(
#     [original_blocks[i] for i in range(0, pipeline.transformer.model.transformer_high.config['num_layers'])]
# )

# # Reduce low noise transformer blocks
# org_blocks = pipeline.transformer.model.transformer_low.blocks
# pipeline.transformer.model.transformer_low.blocks = torch.nn.ModuleList(
#     [org_blocks[i] for i in range(0, pipeline.transformer.model.transformer_low.config['num_layers'])]
# )

# Define the prompt
prompt = "In a warmly lit living room."

print("\n" + "="*80)
print("GENERATING VIDEO WITH FIRST BLOCK CACHE")
print("="*80)
print(f"Prompt: {prompt[:100]}...")
print(f"Resolution: 832x480, 81 frames")
print(f"Inference steps: 4")
print(f"Cache enabled: True")
print(f"Cache threshold: 0.08")
print(f"Cache warmup steps: 2")
print("="*80 + "\n")

# Generate video with first block cache enabled
output = pipeline(
    prompt=prompt,
    num_frames=81,
    guidance_scale=1.0,
    guidance_scale_2=1.0,
    num_inference_steps=40,
    generator=torch.manual_seed(0),
    height=96,
    width=160,
    use_onnx_subfunctions=False,
    parallel_compile=True,
    custom_config_path="examples/diffusers/wan/wan_config.json",
    cache_threshold=0.1,  # Cache similarity threshold (lower = more aggressive caching)
    cache_warmup_steps=3,  # Number of initial steps to run without caching
    # First block cache parameters)
)

# Save the generated video
frames = output.images[0]
export_to_video(frames, "output_t2v_with_cache.mp4", fps=16)

# Print performance metrics
print("\n" + "="*80)
print("GENERATION COMPLETE")
print("="*80)
print(f"Output saved to: output_t2v_with_cache.mp4")
print(f"\nPerformance Metrics:")
for module_perf in output.pipeline_module:
    if module_perf.module_name == "transformer":
        avg_time = sum(module_perf.perf) / len(module_perf.perf)
        print(f"  Transformer average step time: {avg_time:.3f}s")
        print(f"  Total transformer time: {sum(module_perf.perf):.3f}s")
    elif module_perf.module_name == "vae_decoder":
        print(f"  VAE decoder time: {module_perf.perf:.3f}s")
print("="*80)

print("\n💡 Tips for optimizing cache performance:")
print("  - Lower threshold (0.05-0.07): More aggressive caching, higher speedup")
print("  - Higher threshold (0.10-0.15): Conservative caching, better quality")
print("  - Warmup steps (1-3): Balance between stability and speedup")
print("  - For 4-step inference, warmup_steps=2 is recommended")
