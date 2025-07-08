# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
import transformers
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText

model_id = "google/gemma-3-4b-it"
config = AutoConfig.from_pretrained(model_id)
# For Testing Purpose Only
config.text_config.num_hidden_layers = 1
config.vision_config.num_hidden_layers = 2
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id)

# pass HF_TOKEN if gated model
# For running the model in single QPC approach use kv_offload=False. For Dual QPC approach use kv_offload=True ###
qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id, config=config, attn_implementation="eager", kv_offload=True
)

### use skip_vision=Ture, if want to run only text, or false ###
skip_vision = True

if skip_vision:
    ## Only Text ##
    qeff_model.compile(
        prefill_seq_len=128,
        ctx_len=3072,
        img_size=896,
        num_cores=16,
        num_devices=1,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        aic_enable_depth_first=True,
        skip_vision=True,
        mos=1,
        node_precision_info="examples/gemma3_example/fp32_mm.yaml",
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the transformers architecture in LLMs."},
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

    output = qeff_model.generate(inputs=inputs, generation_len=100)
    print(tokenizer.batch_decode(output.generated_ids))
    print(output)

else:
    ## Vision + Text ##
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
        node_precision_info="examples/gemma3_example/fp32_mm.yaml",
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
    output = qeff_model.generate(inputs=inputs, generation_len=100)
    print(tokenizer.batch_decode(output.generated_ids))
    print(output)
