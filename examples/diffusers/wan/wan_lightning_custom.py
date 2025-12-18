# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Wan2.2-Lightning Custom Configuration Example

This example demonstrates how to customize the Wan2.2-Lightning model with various options:
1. Custom video dimensions (height/width) and frame count
2. Custom transformer model and text encoder
3. Custom scheduler configuration
4. Reduced model layers for faster inference
5. Custom compilation settings
6. Custom runtime configuration via JSON config file
7. LoRA adapter loading and configuration

Use this example to learn how to tune Wan2.2-Lightning for your specific video generation needs.
"""

import safetensors.torch
import torch
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_wan_lora_to_diffusers
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download

from QEfficient import QEffWanPipeline

# ============================================================================
# PIPELINE INITIALIZATION WITH CUSTOM PARAMETERS
# ============================================================================

# Option 1: Basic initialization with default parameters
pipeline = QEffWanPipeline.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers")

# Option 2: Advanced initialization with custom modules
# Uncomment and modify to use your own custom components:
#
# pipeline = QEffWanPipeline.from_pretrained(
#     "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
#     text_encoder=custom_text_encoder,      # Your custom text encoder
#     transformer=custom_transformer,         # Your custom transformer model
#     tokenizer=custom_tokenizer,             # Your custom tokenizer
# )

# ============================================================================
# LORA ADAPTER LOADING FOR LIGHTNING MODEL
# ============================================================================
# Download and load Lightning LoRA adapters for faster inference

# Download the LoRAs from Hugging Face Hub
high_noise_lora_path = hf_hub_download(
    repo_id="lightx2v/Wan2.2-Lightning",
    filename="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors",
)
low_noise_lora_path = hf_hub_download(
    repo_id="lightx2v/Wan2.2-Lightning",
    filename="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors",
)


# LoRA conversion utility function
def load_wan_lora(path: str):
    """Convert and load WAN LoRA weights from safetensors format."""
    return _convert_non_diffusers_wan_lora_to_diffusers(safetensors.torch.load_file(path))


# Load LoRA adapters into the high and low noise transformers
pipeline.transformer.model.transformer_high.load_lora_adapter(
    load_wan_lora(high_noise_lora_path), adapter_name="high_noise"
)
pipeline.transformer.model.transformer_high.set_adapters(["high_noise"], weights=[1.0])

pipeline.transformer.model.transformer_low.load_lora_adapter(
    load_wan_lora(low_noise_lora_path), adapter_name="low_noise"
)
pipeline.transformer.model.transformer_low.set_adapters(["low_noise"], weights=[1.0])

# ============================================================================
# OPTIONAL: CUSTOM SCHEDULER CONFIGURATION
# ============================================================================
# Uncomment to use a custom scheduler (e.g., different sampling methods):
#
# pipeline.scheduler = custom_scheduler.from_config(pipeline.scheduler.config)

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
# # Configure for 2-layer model (faster inference)
# pipeline.transformer.model.transformer_high.config.num_layers = 1
# pipeline.transformer.model.transformer_low.config.num_layers = 1
#
# # Reduce high noise transformer blocks
# original_blocks = pipeline.transformer.model.transformer_high.blocks
# pipeline.transformer.model.transformer_high.blocks = torch.nn.ModuleList(
#     [original_blocks[i] for i in range(0, pipeline.transformer.model.transformer_high.config.num_layers)]
# )
#
# # Reduce low noise transformer blocks
# org_blocks = pipeline.transformer.model.transformer_low.blocks
# pipeline.transformer.model.transformer_low.blocks = torch.nn.ModuleList(
#     [org_blocks[i] for i in range(0, pipeline.transformer.model.transformer_low.config.num_layers)]
# )

# ============================================================================
# OPTIONAL: COMPILE WITH CUSTOM CONFIGURATION
# ============================================================================
# Pre-compile the model for optimized performance on target hardware.
#
# When to use:
# - When you want to compile the model separately before generation
# - When you need to skip video generation and only prepare the model
#
# NOTE-1: If compile_config is not specified, the default configuration from
#       QEfficient/diffusers/pipelines/wan/wan_config.json will be used
#
# NOTE-2: use_onnx_subfunctions=True enables modular ONNX export optimizations
#       This feature improves export performance by breaking down the model into smaller,
#       more manageable ONNX functions, which can lead to improved compile time.
#
# Uncomment to compile with a custom configuration:
# pipeline.compile(
#     compile_config="examples/diffusers/wan/wan_config.json",
#     height=480,
#     width=832,
#     num_frames=81,
#     use_onnx_subfunctions=True
# )

# ============================================================================
# VIDEO GENERATION WITH CUSTOM RUNTIME CONFIGURATION
# ============================================================================
# Generate a video using the configured pipeline.
#
# Note: Use of custom_config_path provides flexibility to set device_ids for each
#       module, so you can skip the separate pipeline.compile() step.

# Custom prompt for video generation
prompt = "A cat wearing a hat walking through a magical forest with glowing mushrooms and fireflies dancing around, cinematic lighting, high quality"

# Alternative video dimensions for different use cases:
# height=192, width=320
# height=480, width=832
# height=720, width=1280

output = pipeline(
    prompt=prompt,
    num_frames=81,  # Number of video frames to generate
    guidance_scale=1.0,  # Primary guidance scale
    guidance_scale_2=1.0,  # Secondary guidance scale for dual guidance
    num_inference_steps=4,  # Lightning model uses fewer steps
    generator=torch.manual_seed(42),  # For reproducible results
    custom_config_path="examples/diffusers/wan/wan_config.json",
    height=480,
    width=832,
    use_onnx_subfunctions=True,  # Enable ONNX optimizations
    parallel_compile=False,  # Set to True for parallel compilation
)

# Extract generated frames and export to video
frames = output.images[0]
export_to_video(frames, "custom_wan_lightning_output.mp4", fps=16)
print(output)
