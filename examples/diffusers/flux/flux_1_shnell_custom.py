# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
FLUX.1 Schnell Custom Configuration Example

This example demonstrates various customization options for the FLUX.1 model:
1. Custom image dimensions (height/width)
2. Using a different transformer model
3. Custom scheduler configuration
4. Reducing model layers for faster inference
5. Custom compilation configuration
6. Custom runtime configuration via JSON config file
"""

import torch

from QEfficient import QEFFFluxPipeline

# ============================================================================
# PIPELINE INITIALIZATION WITH CUSTOM PARAMETERS
# ============================================================================
# Initialize the FLUX pipeline with custom settings:
# - Base model: FLUX.1-schnell (fast inference variant)
# - height/width: Custom output image dimensions (256x256 instead of default 1024x1024)
# - text_encoder: Optional custom CLIP text encoder (uncomment to use)

# Option 1: Basic initialization with custom dimensions
pipeline = QEFFFluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    height=256,
    width=256,
)

# Option 2: Initialize with custom text encoder passed directly in from_pretrained
# Uncomment to use a custom text encoder instead of the default one
# This allows you to use a different CLIP model or a fine-tuned text encoder
# custom_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
# pipeline = QEFFFluxPipeline.from_pretrained(
#     "black-forest-labs/FLUX.1-schnell",
#     height=256,
#     width=256,
#     text_encoder=custom_text_encoder,
# )

# ============================================================================
# OPTIONAL: CUSTOM SCHEDULER CONFIGURATION
# ============================================================================
# Uncomment to use EulerAncestralDiscreteScheduler instead of the default scheduler
# This scheduler can provide different sampling characteristics and quality trade-offs

# pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

# ============================================================================
# OPTIONAL: REDUCE MODEL LAYERS FOR FASTER INFERENCE
# ============================================================================
# Uncomment to reduce the number of transformer blocks for faster inference
# This significantly reduces computation but may impact output quality
# Useful for testing or when speed is more critical than quality
original_blocks = pipeline.transformer.model.transformer_blocks
org_single_blocks = pipeline.transformer.model.single_transformer_blocks
pipeline.transformer.model.transformer_blocks = torch.nn.ModuleList([original_blocks[0]])
pipeline.transformer.model.single_transformer_blocks = torch.nn.ModuleList([org_single_blocks[0]])

# ============================================================================
# OPTIONAL: UPDATE CONFIG TO REFLECT LAYER REDUCTION
# ============================================================================
# If you reduced the layers above, update the config to match
# This ensures the model knows about the architectural changes
pipeline.transformer.model.config.num_layers = 1
pipeline.transformer.model.config.num_single_layers = 1

# ============================================================================
# OPTIONAL: COMPILE WITH CUSTOM CONFIGURATION
# ============================================================================
# Uncomment to compile the model with a specific compilation configuration
# This optimizes the model for Cloud AI 100 hardware execution
# pipeline.compile(compile_config="QEfficient/diffusers/pipelines/flux/config/default_flux_compile_config.json")


# ============================================================================
# IMAGE GENERATION WITH CUSTOM RUNTIME CONFIGURATION
# ============================================================================
# Generate an image using the configured pipeline with custom parameters:
# - prompt: Text description of the desired image
# - custom_config_path: Path to JSON file with custom runtime configurations
#   (e.g., device settings, memory optimizations, execution parameters)
# - guidance_scale: 0.0 for FLUX.1-schnell (this model doesn't use guidance)
# - num_inference_steps: Number of denoising steps (4 is typical for schnell variant)
# - max_sequence_length: Maximum length for text encoding (256 for efficiency)
# - generator: Random seed for reproducible results
image = pipeline(
    prompt="A cat holding a sign that says hello world",
    custom_config_path="examples/diffusers/flux/flux_config.json",
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=torch.manual_seed(42),
).images[0]

# Save the generated image to disk
image.save("flux-schnell_aic_1024.png")
