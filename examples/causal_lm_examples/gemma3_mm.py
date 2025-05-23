# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
import transformers
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForImageTextToText

model_id = "google/gemma-3-4b-it"
config = AutoConfig.from_pretrained(model_id)
# For Testing Purpose Only
# config.text_config.num_hidden_layers = 1
# config.vision_config.num_hidden_layers = 2

model = AutoModelForImageTextToText.from_pretrained(model_id, attn_implementation="eager", config=config)
model.eval()

qeff_model = QEFFAutoModelForImageTextToText(model, kv_offload=True)
# TODO: Map the Vision Encoder to FP16 Only and Disable MXFP6 For Better Accuracy.
qeff_model.compile(
    prefill_seq_len=128,
    ctx_len=3072,
    img_size=896,
    num_cores=16,
    num_devices=1,
    mxfp6_matmul=False,
    mxint8_kv_cache=False,
    aic_enable_depth_first=True,
    mos=1,
    node_precision_info="fp32_nodes_gemma3_4b_text.yaml",
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
for key, value in inputs.items():
    print(f"key : {key} and value shape is {value.shape}")

inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)
streamer = TextStreamer(tokenizer)
output = qeff_model.generate(inputs=inputs, device_ids=[0], generation_len=100)
print(output.generated_ids)
print(tokenizer.batch_decode(output.generated_ids))
print(output)
