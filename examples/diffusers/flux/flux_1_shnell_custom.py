# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
FLUX.1 Schnell Custom Configuration Example

This example demonstrates how to customize the FLUX.1 model with various options:
1. Custom image dimensions (height/width)
2. Custom transformer model and text encoder
3. Custom scheduler configuration
4. Reduced model layers for faster inference
5. Custom compilation settings
6. Custom runtime configuration via JSON config file

Use this example to learn how to fine-tune FLUX.1 for your specific needs.
"""

import torch

from QEfficient import QEffFluxPipeline

# ============================================================================
# PIPELINE INITIALIZATION WITH CUSTOM PARAMETERS
# ============================================================================

# Option 1: Basic initialization with default parameters
pipeline = QEffFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")
# Option 2: Advanced initialization with custom modules
# Uncomment and modify to use your own custom components:
#
# pipeline = QEffFluxPipeline.from_pretrained(
#     "black-forest-labs/FLUX.1-schnell",
#     text_encoder=custom_text_encoder,      # Your custom CLIP text encoder
#     transformer=custom_transformer,         # Your custom transformer model
#     tokenizer=custom_tokenizer,             # Your custom tokenizer
# )

# ============================================================================
# OPTIONAL: CUSTOM SCHEDULER CONFIGURATION
# ============================================================================
# Uncomment to use a custom scheduler (e.g., different sampling methods):
#
# pipeline.scheduler = custom_scheduler.from_config(pipeline.scheduler.config)

# ============================================================================
# OPTIONAL: REDUCE MODEL LAYERS FOR FASTER INFERENCE
# ============================================================================
# Reduce the number of transformer blocks to speed up image generation.
#
# Trade-off: Faster inference but potentially lower image quality
# Use case: Quick testing, prototyping, or when speed is critical
#
# Uncomment the following lines to use only the first transformer block:
#
# original_blocks = pipeline.transformer.model.transformer_blocks
# org_single_blocks = pipeline.transformer.model.single_transformer_blocks
# pipeline.transformer.model.transformer_blocks = torch.nn.ModuleList([original_blocks[0]])
# pipeline.transformer.model.single_transformer_blocks = torch.nn.ModuleList([org_single_blocks[0]])
# pipeline.transformer.model.config['num_layers'] = 1
# pipeline.transformer.model.config['num_single_layers'] = 1

# ============================================================================
# OPTIONAL: COMPILE WITH CUSTOM CONFIGURATION
# ============================================================================
# Pre-compile the model for optimized performance on target hardware.
#
# When to use:
# - When you want to compile the model separately before generation
# - When you need to skip image generation and only prepare the model
#
# NOTE-1: If compile_config is not specified, the default configuration from
#       QEfficient/diffusers/pipelines/flux/flux_config.json will be used
#
# NOTE-2: use_onnx_subfunctions=True enables modular ONNX export optimizations (Experimental so not recommended)
#       This feature improves export performance by breaking down the model into smaller,
#       more manageable ONNX functions, which can lead to improve compile time.
# Uncomment to compile with a custom configuration:
# pipeline.compile(
#     compile_config="examples/diffusers/flux/flux_config.json",
#     height=512,
#     width=512,
#     use_onnx_subfunctions=False
#     )

# ============================================================================
# IMAGE GENERATION WITH CUSTOM RUNTIME CONFIGURATION
# ============================================================================
# Generate an image using the configured pipeline.
#
# Note: Use of custom_config_path provides flexibility to set device_ids for each
#       module, so you can skip the separate pipeline.compile() step.

output = pipeline(
    prompt="A laughing girl",
    custom_config_path="examples/diffusers/flux/flux_config.json",
    height=1024,
    width=1024,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=torch.manual_seed(42),
    parallel_compile=True,
    use_onnx_subfunctions=False,
)

image = output.images[0]
# Save the generated image to disk
image.save("laughing_girl.png")
print(output)
