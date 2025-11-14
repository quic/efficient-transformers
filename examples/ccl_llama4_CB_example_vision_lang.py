# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import transformers
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
config = AutoConfig.from_pretrained(model_id)
# For Testing Purpose Only
config.text_config.num_hidden_layers = 4
config.vision_config.num_hidden_layers = 2

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

ctx_len = 4096
# Set the list of ccl during prefilling process
comp_ctx_lengths_prefill = [3072]
# Set the list of ccl during decoding process
comp_ctx_lengths_decode = [ctx_len]

continious_batching = True
if continious_batching:
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_id,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        continuous_batching=True,
        qaic_config={
            "comp_ctx_lengths_prefill": comp_ctx_lengths_prefill,
            "comp_ctx_lengths_decode": comp_ctx_lengths_decode,
            "ctx_len": ctx_len,
        },
    )

    qeff_model.compile(
        prefill_seq_len=128,
        ctx_len=ctx_len,
        img_size=336,
        num_cores=16,
        num_devices=4,
        max_num_tiles=17,
        batch_size=1,
        full_batch_size=4,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        aic_enable_depth_first=True,
        mos=1,
    )
else:
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_id,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        qaic_config={
            "comp_ctx_lengths_prefill": comp_ctx_lengths_prefill,
            "comp_ctx_lengths_decode": comp_ctx_lengths_decode,
            "ctx_len": ctx_len,
        },
    )

    qeff_model.compile(
        prefill_seq_len=128,
        ctx_len=ctx_len,
        img_size=336,
        num_cores=16,
        num_devices=4,
        max_num_tiles=17,
        batch_size=1,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        aic_enable_depth_first=True,
        mos=1,
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
    device_ids=[32, 33, 34, 35],
    generation_len=100,
)

# print("Generated texts:", exec_info.generated_texts)
print("Generated IDs:", exec_info.generated_ids)
print(exec_info)
