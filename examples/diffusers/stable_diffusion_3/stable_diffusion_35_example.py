# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from QEfficient import QEFFStableDiffusion3Pipeline

pipeline = QEFFStableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large-turbo")
pipeline.compile(num_devices_text_encoder=1, num_devices_transformer=4, num_devices_vae_decoder=1)

# NOTE: guidance_scale <=1 is not supported
image = pipeline("A girl laughing", num_inference_steps=28, guidance_scale=2.0).images[0]
image.save("girl_laughing_turbo.png")
