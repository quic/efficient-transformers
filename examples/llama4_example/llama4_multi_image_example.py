# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
import transformers
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForImageTextToText

model_id = "/local/mnt/workspace/aditjadh/aisyssol/models--meta-llama--Llama-4-Scout-17B-16E-Instruct/snapshots/7dab2f5f854fe665b6b2f1eccbd3c48e5f627ad8"
config = AutoConfig.from_pretrained(model_id)

model = AutoModelForImageTextToText.from_pretrained(model_id, attn_implementation="eager", config=config)
model.eval()
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id)

### For running the model in single QPC approach use kv_offload=False. For Dual QPC approach use kv_offload=True ###
qeff_model = QEFFAutoModelForImageTextToText(model, kv_offload=True)

### For multi-image, the value of max_num_tiles should be the sum of the num_tiles values across all the images ###
qeff_model.compile(
    prefill_seq_len=128,
    ctx_len=8192,
    img_size=336,
    num_cores=16,
    num_devices=4,
    max_num_tiles=45,
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
output = qeff_model.generate(inputs=inputs, device_id_vision=[32,33,34,35], device_id_lang=[36,37,38,39], generation_len=100)
print(output)
