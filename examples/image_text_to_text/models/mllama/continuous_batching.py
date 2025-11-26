# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import transformers
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText


def set_num_layers(config, n_layer=1):
    ## -1 indicates use all the layers of the model.
    if n_layer == -1:
        return config
    elif hasattr(config, "model_type") and "mllama" in config.model_type:
        config.text_config.num_hidden_layers = n_layer
        config.text_config.cross_attention_layers = [
            x for x in config.text_config.cross_attention_layers if x < n_layer
        ]
    elif hasattr(config, "text_config"):
        config.text_config.num_hidden_layers = n_layer
        config.vision_config.num_hidden_layers = n_layer
    elif hasattr(config, "llm_config"):
        config.llm_config.num_hidden_layers = n_layer
        config.vision_config.num_hidden_layers = n_layer
    else:
        config.num_hidden_layers = n_layer
    return config


model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
config = AutoConfig.from_pretrained(model_id)
# For Testing Purpose Only
config = set_num_layers(config, n_layer=7)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

continious_batching = True
if continious_batching:
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_id,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        continuous_batching=True,
    )

    qeff_model.compile(
        prefill_seq_len=32,
        ctx_len=512,
        img_size=560,
        num_cores=16,
        num_devices=4,
        batch_size=1,
        full_batch_size=4,
        mxfp6_matmul=False,
    )
else:
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_id,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
    )

    qeff_model.compile(
        prefill_seq_len=32,
        ctx_len=512,
        img_size=560,
        num_cores=16,
        num_devices=4,
        batch_size=1,
        mxfp6_matmul=False,
        # mxint8_kv_cache=True,
        # aic_enable_depth_first=True,
        # mos=1,
    )

image_urls = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
]

prompts = [
    "Can you describe the image in detail?",
    "What are the objects in the image?",
    "What is the main subject of the image?",
    "What colors are predominant in the image?",
]

exec_info = qeff_model.generate(
    tokenizer=tokenizer,
    prompts=prompts,
    processor=processor,
    images=image_urls,
    device_ids=[0, 1, 2, 3],
    generation_len=10,
)

print("Generated texts:", exec_info.generated_texts)
print("Generated IDs:", exec_info.generated_ids)
print(exec_info)
