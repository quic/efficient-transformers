# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import requests
from PIL import Image
from transformers import AutoConfig, AutoProcessor, TextStreamer
from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration

from QEfficient import QEFFAutoModelForImageTextToText  # noqa: E402

model_id = "llava-hf/llava-1.5-7b-hf"

config = AutoConfig.from_pretrained(model_id)
config.text_config.num_hidden_layers = 1
config.vision_config.num_hidden_layers = 1
py_model = LlavaForConditionalGeneration.from_pretrained(model_id, low_cpu_mem_usage=True, config=config)

processor = AutoProcessor.from_pretrained(model_id)

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image")
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What are these?"},
            {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(images=raw_image, text=prompt, return_tensors="pt")

# streamer = TextStreamer(processor.tokenizer)
# output = model.generate(inputs=inputs, device_ids=[0], generation_len=128)

output = py_model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))
print(output)

model = QEFFAutoModelForImageTextToText.from_pretrained(model_id, config=config, kv_offload=False)
model.compile(num_devices=1, img_size=336, prefill_seq_len=1024, ctx_len=2048)

processor = AutoProcessor.from_pretrained(model_id)

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image")
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What are these?"},
            {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(images=raw_image, text=prompt, return_tensors="pt")

streamer = TextStreamer(processor.tokenizer)
output = model.generate(inputs=inputs, device_ids=[0], generation_len=128)
print(output.generated_ids)
print(processor.tokenizer.batch_decode(output.generated_ids))
print(output)
