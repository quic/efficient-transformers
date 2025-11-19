import torch

from QEfficient import QEFFQwenImagePipeline

model_name = "Qwen/Qwen-Image"

pipe = QEFFQwenImagePipeline.from_pretrained(model_name)
positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
}

# Generate image
prompt = """A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition"""

# negative_prompt = " "  # using an empty string if you do not have specific concept to remove
negative_prompt = "do not use green color" * 24 + " "


# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["16:9"]
original_blocks = pipe.transformer.model.transformer_blocks
pipe.transformer.model.transformer_blocks = torch.nn.ModuleList([original_blocks[0]])
pipe.transformer.model.config.num_layers = 1


pipe.compile()


image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=5,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cpu").manual_seed(42),
).images[0]

image.save("example.png")
