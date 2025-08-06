from QEfficient import QEFFStableDiffusion3Pipeline

pipeline = QEFFStableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large-turbo", cache_dir="/home/amitraj/forked_repo/sd35_turbo/cache/hub"
)
pipeline.export()
pipeline.compile()
x = pipeline("A man holding a sign that reads Efficient Transformers").images[0]
x.save("man_image.png")
