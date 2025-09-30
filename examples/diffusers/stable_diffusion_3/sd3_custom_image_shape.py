from QEfficient import QEFFStableDiffusion3Pipeline
pipeline = QEFFStableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large-turbo",  height = 1152, width=768)
# for single layer check
# original_blocks = pipeline.transformer.model.transformer_blocks
# pipeline.transformer.model.transformer_blocks = torch.nn.ModuleList([original_blocks[0]])
# pipeline.transformer.model.config.num_layers = 1


pipeline.compile(num_devices_text_encoder=1, num_devices_transformer=4, num_devices_vae_decoder=1, mdts_mos=1, height = 1152, width=768 )
# NOTE: guidance_scale <=1 is not supported
image = pipeline("A girl laughing", num_inference_steps=4, guidance_scale=2.0, height = 1152, width=768).images[0]
image.save("girl_laughing_turbo_new.png")