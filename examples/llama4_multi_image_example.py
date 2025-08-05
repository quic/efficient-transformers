# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
import transformers
from transformers import AutoConfig, AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForImageTextToText

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
config = AutoConfig.from_pretrained(model_id)
# For Testing Purpose Only
config.text_config.num_hidden_layers = 4
config.vision_config.num_hidden_layers = 2

qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id, attn_implementation="eager", kv_offload=True, config=config
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id)

### For multi-image, the value of max_num_tiles should be the sum of the num_tiles values across all the images ###
qeff_model.compile(
    prefill_seq_len=128,
    ctx_len=5376,
    img_size=336,
    num_cores=16,
    num_devices=8,
    max_num_tiles=34,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    aic_enable_depth_first=True,
    mos=1,
)

### Multi_image Prompt ###
image_url_1 = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png"
)


image_url_2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": image_url_1},
            {"type": "image", "url": image_url_2},
            {
                "type": "text",
                "text": "Analyze the key elements, colors, and objects in the two images. Discuss their similarities, differences, and how they complement or contrast each other. Reflect on the emotions or ideas they convey, considering the context, light, shadow, and composition.",
            },
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)
streamer = TextStreamer(tokenizer)
output = qeff_model.generate(inputs=inputs, device_ids=[0, 1, 2, 3, 4, 5, 6, 7], generation_len=100)
print(output.generated_ids)
print(tokenizer.batch_decode(output.generated_ids))
print(output)
