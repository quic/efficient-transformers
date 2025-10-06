from QEfficient import QEFFStableDiffusion3Pipeline
import torch
pipeline = QEFFStableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large-turbo",  height = 1152, width=768)
# for single layer check
# original_blocks = pipeline.transformer.model.transformer_blocks
# pipeline.transformer.model.transformer_blocks = torch.nn.ModuleList([original_blocks[0]])
# pipeline.transformer.model.config.num_layers = 1


pipeline.compile(num_devices_text_encoder=1, num_devices_transformer=4, num_devices_vae_decoder=1, mdts_mos=1, height = 1152, width=768 )
# NOTE: guidance_scale <=1 is not supported

def call_back_to_save_interim_results(pipeline, i, t, callback_kwargs, batch_size, output_type, device_ids):
    try:
        img  =  pipeline._convert_latents_to_image(callback_kwargs['latents'], batch_size, output_type, device_ids )[0]
        img.save(f"img_generated_at_step_{i}.png")
    except Exception as e:
        print(f"Exception occured in call_back_to_save_interim_results : {e}")
    return True

image = pipeline("A girl laughing", num_inference_steps=2, guidance_scale=2.0, height = 1152, width=768, callback_on_step_end=call_back_to_save_interim_results, callback_on_step_end_tensor_inputs=["latents"], device_ids_text_encoder_1=[40], device_ids_text_encoder_2=[41], device_ids_text_encoder_3=[42], device_ids_vae_decoder=[43], device_ids_transformer=[44,45,46,47]).images[0]
image.save("girl_laughing_turbo_final.png")

image = pipeline("A boy laughing ", num_inference_steps=4, guidance_scale=2.0, height = 1152, width=768, callback_on_step_end=call_back_to_save_interim_results, callback_on_step_end_tensor_inputs=["latents"], device_ids_text_encoder_1=[48], device_ids_text_encoder_2=[49], device_ids_text_encoder_3=[50], device_ids_vae_decoder=[51], device_ids_transformer=[52,53,54,55]).images[0]
image.save("boy_laughing_turbo_new_save.png")


image = pipeline("A boy cycling ", num_inference_steps=3, guidance_scale=2.0, height = 1152, width=768, callback_on_step_end=call_back_to_save_interim_results, callback_on_step_end_tensor_inputs=["latents"]).images[0]
image.save("boy_cycling_save3.png")