# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

# If we want to enable QBlocking Run below command:, default is without blocking
# ATTENTION_BLOCKING_MODE=q num_q_blocks=2 python -W ignore qwen2_5_vl_example.py

import transformers
from transformers import AutoConfig, AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForImageTextToText

## For AWQ model update pytorch version to 2.8.*
model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
config = AutoConfig.from_pretrained(model_id)
# For Testing Purpose Only
config.text_config.num_hidden_layers = 4

## Activate Compute-Context-Length (CCL) feature by setting ccl_enabled=True when loading the model with from_pretrained().
## Use the optional comp_ctx_lengths argument to provide two lists of context lengths for the prefilling and decoding processes. If comp_ctx_lengths=None, the model will run with its default context length.
##   - The first list, comp_ctx_lengths_prefill, defines the compute-context-length values for the prefilling process.
##           -- The process starts with the first value in the list and gradually increases the context length based on the position_id of the current prompt chunk.
##   - The second list, comp_ctx_lengths_decode, defines the compute-context-length values for the decoding process.
##           -- During decoding, the model selects an appropriate context length from the list based on the input prompt length and cache index.
##           -- It starts from the correct value in the list and increases the context length dynamically when the cache index exceeds the current threshold.

ctx_len = 8192
comp_ctx_lengths_prefill = [4096]
comp_ctx_lengths_decode = [6144, ctx_len]

qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id,
    attn_implementation="eager",
    kv_offload=True,
    config=config,
    continuous_batching=True,
    qaic_config={
        "ccl_enabled": True,
    },
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

batch_size = 1
## Vision + Text ##
qeff_model.compile(
    batch_size=batch_size,
    full_batch_size=4,
    prefill_seq_len=128,
    ctx_len=8192,
    num_cores=16,
    num_devices=4,
    height=354,
    width=536,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    aic_enable_depth_first=True,
    mos=1,
    comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
    comp_ctx_lengths_decode=comp_ctx_lengths_decode,
)

image_urls = [
    "https://picsum.photos/id/237/536/354",
    "https://picsum.photos/id/237/536/354",
    "https://picsum.photos/id/237/536/354",
    "https://picsum.photos/id/237/536/354",
]

prompts = [
    "Can you describe the image in detail?",
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
    generation_len=100,
    device_ids=[0, 1, 2, 3],
)
print(output.generated_ids)
print(tokenizer.batch_decode(output.generated_ids))
print(output)
