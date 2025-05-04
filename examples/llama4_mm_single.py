# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
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
# config.text_config.num_hidden_layers = 1
# config.vision_config.num_hidden_layers = 2

model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id, attn_implementation="eager", kv_offload=True, config=config
)

# TODO: Map the Vision Encoder to FP16 Only and Disable MXFP6 For Better Accuracy.
model.compile(
    prefill_seq_len=128,
    ctx_len=3072,
    img_size=336,
    num_cores=16,
    num_devices=8,
    batch_size_times_num_tiles=17,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    aic_enable_depth_first=True,
    mos=1,
)

image_url = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png"
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": image_url},
            {"type": "text", "text": "Can you describe the image in detail."},
        ],
    },
]

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id)
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)
streamer = TextStreamer(tokenizer)
output = model.generate(inputs=inputs, device_ids=[0, 1, 2, 3, 4, 5, 6, 7], generation_len=100)
print(output.generated_ids)
print(tokenizer.batch_decode(output.generated_ids))
print(output)
