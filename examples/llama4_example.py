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

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
config = AutoConfig.from_pretrained(model_id)
# For Testing Purpose Only
config.text_config.num_hidden_layers = 4
config.vision_config.num_hidden_layers = 2

model = AutoModelForImageTextToText.from_pretrained(model_id, attn_implementation="eager", config=config)
model.eval()
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id)

### For running the model in single QPC approach use kv_offload=False. For Dual QPC approach use kv_offload=True ###
qeff_model = QEFFAutoModelForImageTextToText(model, kv_offload=True)

### use skip_vision=Ture, if want to run only text, ow false ###
skip_vision = True

if skip_vision:
    ## Only Text ##
    qeff_model.compile(
        prefill_seq_len=128,
        ctx_len=3072,
        img_size=336,
        num_cores=16,
        num_devices=8,
        max_num_tiles=17,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        aic_enable_depth_first=True,
        skip_vision=True,
        mos=1,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Tell me about yourself."},
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

    streamer = TextStreamer(tokenizer)
    output = qeff_model.generate(inputs=inputs, device_ids=[0, 1, 2, 3, 4, 5, 6, 7], generation_len=100)
    print(output.generated_ids)
    print(tokenizer.batch_decode(output.generated_ids))
    print(output)

else:
    ## Vision + Text ##
    qeff_model.compile(
        prefill_seq_len=128,
        ctx_len=3072,
        img_size=336,
        num_cores=16,
        num_devices=8,
        max_num_tiles=17,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        aic_enable_depth_first=True,
        mos=1,
    )

    ### IMAGE + TEXT ###
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
    print()
