# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import torch

from QEfficient import QEFFFluxPipeline

pipeline = QEFFFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", height=256, width=256)

######## for single layer
original_blocks = pipeline.transformer.model.transformer_blocks
org_single_blocks = pipeline.transformer.model.single_transformer_blocks
pipeline.transformer.model.transformer_blocks = torch.nn.ModuleList([original_blocks[0]])
pipeline.transformer.model.single_transformer_blocks = torch.nn.ModuleList([org_single_blocks[0]])
# Update num_layers to reflect the change
pipeline.transformer.model.config.num_layers = 1
pipeline.transformer.model.config.num_single_layers = 1

pipeline.compile(compile_config="QEfficient/diffusers/pipelines/flux/config/default_flux_compile_config.json")

generator = torch.manual_seed(42)
# NOTE: guidance_scale <=1 is not supported
image = pipeline(
    "A cat holding a sign that says hello world",
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=generator,
).images[0]
image.save("flux-schnell_aic_1024.png")
