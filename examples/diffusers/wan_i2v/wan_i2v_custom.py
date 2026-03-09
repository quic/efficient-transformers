# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Wan2.2-Lightning Image-to-Video Custom Configuration Example

This example demonstrates how to customize the Wan2.2-Lightning I2V model with various options:
1. Custom image input handling (local files, URLs, different formats)
2. Custom video dimensions (height/width) and frame count
3. Custom scheduler configuration
4. Reduced model layers for faster inference
5. Custom compilation settings
6. Custom runtime configuration via JSON config file
7. LoRA adapter loading and configuration
8. Image preprocessing and aspect ratio handling

Use this example to learn how to tune Wan2.2-Lightning I2V for your specific image-to-video generation needs.
"""

import numpy as np
import safetensors.torch
import torch
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_wan_lora_to_diffusers
from diffusers.utils import export_to_video, load_image
from huggingface_hub import hf_hub_download

from QEfficient import QEffWanImageToVideoPipeline

# ============================================================================
# PIPELINE INITIALIZATION WITH CUSTOM PARAMETERS
# ============================================================================

# Option 1: Basic initialization with default parameters
pipeline = QEffWanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers")

# ============================================================================
# LORA ADAPTER LOADING FOR LIGHTNING MODEL
# ============================================================================
# Download and load Lightning LoRA adapters for faster inference

# Download the LoRAs from Hugging Face Hub
high_noise_lora_path = hf_hub_download(
    repo_id="lightx2v/Wan2.2-Lightning",
    filename="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
)
low_noise_lora_path = hf_hub_download(
    repo_id="lightx2v/Wan2.2-Lightning",
    filename="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
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
#
# Uncomment the following lines to use only a subset of transformer layers:
#
# # Configure for 2-layer model (faster inference)
# pipeline.transformer.model.transformer_high.config['num_layers'] = 2
# pipeline.transformer.model.transformer_low.config['num_layers']= 2
#
# # Reduce high noise transformer blocks
# original_blocks = pipeline.transformer.model.transformer_high.blocks
# pipeline.transformer.model.transformer_high.blocks = torch.nn.ModuleList(
#     [original_blocks[i] for i in range(0, pipeline.transformer.model.transformer_high.config['num_layers'])]
# )
#
# # Reduce low noise transformer blocks
# org_blocks = pipeline.transformer.model.transformer_low.blocks
# pipeline.transformer.model.transformer_low.blocks = torch.nn.ModuleList(
#     [org_blocks[i] for i in range(0, pipeline.transformer.model.transformer_low.config['num_layers'])]
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
#       QEfficient/diffusers/pipelines/wan/wan_i2v_config.json will be used
#
# NOTE-2: use_onnx_subfunctions=True enables modular ONNX export optimizations
#       This feature improves export performance by breaking down the model into smaller,
#       more manageable ONNX functions, which can lead to improved compile time.
#
# Uncomment to compile with a custom configuration:
# pipeline.compile(
#     compile_config="examples/diffusers/wan_i2v/wan_i2v_config.json",
#     parallel=True,
#     height=480,
#     width=832,
#     num_frames=81,
#     use_onnx_subfunctions=True
# )

# ============================================================================
# OPTIONAL: Skip Export, Compilation
# ============================================================================
#
# Use this when you want to skip export and compilation if you have already compiled QPC.
#
# Changes needed in config.json: update qpc_path of desired module
#
# "execute":
#          {
#           "device_ids": null,
#           "qpc_path" : "<QPC_PATH>"
#          }

# ============================================================================
# IMAGE INPUT CONFIGURATION
# ============================================================================
# Configure input image with various options

# Option 1: Load from URL (default example)
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
image = load_image(image_url)

# Option 2: Load from local file
# image = load_image("path/to/your/image.png")


# ============================================================================
# AUTOMATIC RESOLUTION CALCULATION
# ============================================================================
# Calculate optimal dimensions based on image aspect ratio and target resolution

# Choose target resolution preset
max_area = (
    190 * 320
)  # 180p - ATTENTION_BLOCKING_MODE=kv head_block_size=16 num_kv_blocks=3 python3 examples/diffusers/wan_i2v/wan_i2v_custom.py
# max_area = 480 * 832    # 480p - ATTENTION_BLOCKING_MODE=qkv head_block_size=16 num_kv_blocks=21 num_q_blocks=2 python3 examples/diffusers/wan_i2v/wan_i2v_custom.py
# max_area = 720 * 1280   # 720p - ATTENTION_BLOCKING_MODE=qkv head_block_size=16 num_kv_blocks=48 num_q_blocks=5 python3 examples/diffusers/wan_i2v/wan_i2v_custom.py

# Calculate dimensions preserving aspect ratio
aspect_ratio = image.height / image.width
mod_value = pipeline.model.vae.config.scale_factor_spatial * pipeline.model.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value

# Resize image to calculated dimensions
image = image.resize((width, height))

print(f"Image resized to: {width}x{height} (aspect ratio: {aspect_ratio:.2f})")

# ============================================================================
# IMAGE-TO-VIDEO GENERATION WITH CUSTOM RUNTIME CONFIGURATION
# ============================================================================
# Generate a video using the configured pipeline and input image.
#
# Note: Use of custom_config_path provides flexibility to set device_ids for each
#       module, so you can skip the separate pipeline.compile() step.

# Custom prompt for image-to-video generation
prompt = (
    "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "
    "the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
)


output = pipeline(
    image=image,  # Input image for I2V generation
    prompt=prompt,  # Text prompt to guide video generation
    num_frames=81,  # Number of video frames to generate
    height=height,  # Video height (calculated from image)
    width=width,  # Video width (calculated from image)
    guidance_scale=1.0,  # Guidance scale for Lightning model
    num_inference_steps=4,  # Lightning model uses fewer steps
    generator=torch.manual_seed(42),  # For reproducible results
    custom_config_path="wan_i2v_config.json",  # I2V-specific config : examples/diffusers/wan_i2v/wan_i2v_config.json
    use_onnx_subfunctions=True,  # Enable ONNX optimizations
    parallel_compile=True,  # Set to False for sequential compilation
)

# Extract generated frames and export to video
frames = output.images[0]
export_to_video(frames, "custom_wan_i2v_output.mp4", fps=16)
print("Video saved as: custom_wan_i2v_output.mp4")
print(f"Video details: {len(frames)} frames, {width}x{height} resolution")
print(output)

# ============================================================================
# OPTIONAL: CALLBACK FUNCTIONS FOR INTERMEDIATE RESULTS
# ============================================================================
# Use callbacks to save intermediate video drafts during generation process.
# This is useful for monitoring progress and debugging generation quality.
#
# Uncomment the following section to enable callback-based draft saving:

# def save_draft_callback(pipeline, step, callback_kwargs, num_frames = 81):
#     """
#     Callback function to save intermediate video drafts during generation.

#     This function is called at each denoising step and saves a draft video
#     showing the current state of generation.

#     Args:
#         pipeline: The I2V pipeline instance
#         step: Current denoising step
#         callback_kwargs: Dictionary containing latents and other tensors
#         num_frames :  Number of video frames to generate (deafult = 81)

#     Returns:
#         Dictionary with updated latents (required for pipeline continuation)
#     """
#     latents = callback_kwargs["latents"]

#     # Convert latents to appropriate dtype for VAE decoder
#     latents = latents.to(pipeline.vae_decoder.model.dtype)

#     # Apply VAE decoder normalization
#     latents_mean = torch.tensor(
#         pipeline.vae_decoder.model.config.latents_mean
#     ).view(1, pipeline.vae_decoder.model.config.z_dim, 1, 1, 1)

#     latents_std = 1.0 / torch.tensor(
#         pipeline.vae_decoder.model.config.latents_std
#     ).view(1, pipeline.vae_decoder.model.config.z_dim, 1, 1, 1)

#     latents = latents / latents_std + latents_mean

#     # Check if VAE decoder session exists and is properly initialized
#     if pipeline.vae_decoder.qpc_session is None:
#         from QEfficient.generation.cloud_infer import QAICInferenceSession
#         pipeline.vae_decoder.qpc_session = QAICInferenceSession(
#             str(pipeline.vae_decoder.qpc_path),
#             device_ids=pipeline.vae_decoder.device_ids,
#         )
#         # Only set buffers once when initializing the session
#         output_buffer = {"sample": np.random.rand(latents.shape[0], 3, num_frames, height, width).astype(np.int32)}
#         pipeline.vae_decoder.qpc_session.set_buffers(output_buffer)

#     # Run VAE decoder to get video frames
#     inputs = {"latent_sample": latents.numpy()}
#     try:
#         video = pipeline.vae_decoder.qpc_session.run(inputs)
#         video_tensor = torch.from_numpy(video["sample"])
#         video = pipeline.model.video_processor.postprocess_video(video_tensor)

#         # Save intermediate draft video
#         export_to_video(video[0], f"draft_i2v_step_{step}.mp4", fps=16)
#         print(f"Saved intermediate draft: draft_i2v_step_{step}.mp4")
#     except Exception as e:
#         print(f"Warning: Callback failed to generate intermediate video at step {step}: {e}")
#         # Continue without saving intermediate video

#     return {"latents": callback_kwargs["latents"]}

# # Example usage with callbacks enabled:
# # Uncomment the following code to generate video with intermediate draft saving

# output_with_callbacks = pipeline(
#     image=image,  # Input image for I2V generation
#     prompt=prompt,  # Text prompt to guide video generation
#     num_frames=81,  # Number of video frames to generate
#     height=height,  # Video height (calculated from image)
#     width=width,   # Video width (calculated from image)
#     guidance_scale=1.0,  # Guidance scale for Lightning model
#     num_inference_steps=4,  # Lightning model uses fewer steps
#     generator=torch.manual_seed(42),  # For reproducible results
#     custom_config_path="examples/diffusers/wan_i2v/wan_i2v_config.json",
#     use_onnx_subfunctions=True,  # Enable ONNX optimizations
#     parallel_compile=True,  # Set to False for sequential compilation
#     callback_on_step_end=save_draft_callback,  # Enable callback function
#     callback_on_step_end_tensor_inputs=["latents"],  # Pass latents to callback
# )

# frames_with_callbacks = output_with_callbacks.images[0]
# export_to_video(frames_with_callbacks, "final_i2v_with_callbacks.mp4", fps=16)
# print("Final video with callbacks saved as: final_i2v_with_callbacks.mp4")
