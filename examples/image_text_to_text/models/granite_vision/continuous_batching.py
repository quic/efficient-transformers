# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import transformers
from transformers import AutoConfig, AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForImageTextToText

## For AWQ model update pytorch version to 2.8.*
model_id = "ibm-granite/granite-vision-3.2-2b"
config = AutoConfig.from_pretrained(model_id)
config.text_config.num_hidden_layers = 2

qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id,
    attn_implementation="eager",
    kv_offload=True,
    config=config,
    continuous_batching=True,
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

batch_size = 1
## Vision + Text ##
qeff_model.compile(
    batch_size=batch_size,
    full_batch_size=4,
    prefill_seq_len=5500,
    ctx_len=6000,
    num_cores=16,
    num_devices=4,
    img_size=384,
    mxfp6_matmul=False,
)

image_urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/val2017/000000039769.jpg",
]

prompts = [
    "Describe the image",
    "What are the objects in the image?",
    "What is the main subject of the image?",
    "What colors are predominant in the image?",
]

streamer = TextStreamer(tokenizer)
output = qeff_model.generate(
    tokenizer=tokenizer,
    prompts=prompts,
    processor=processor,
    images=image_urls,
    generation_len=10,
    image_height=1610,
    image_width=1109,
)
print(output.generated_ids)
print(tokenizer.batch_decode(output.generated_ids))
print(output.generated_texts)
