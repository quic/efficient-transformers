# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import torch
from QEfficient import QEFFFluxPipeline

pipeline = QEFFFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")
pipeline.compile(num_devices_text_encoder=1, num_devices_transformer=4, num_devices_vae_decoder=1)

generator = torch.manual_seed(42)
# NOTE: guidance_scale <=1 is not supported
image = pipeline("A cat holding a sign that says hello world",
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=generator).images[0]
image.save("flux-schnell_aic.png")
