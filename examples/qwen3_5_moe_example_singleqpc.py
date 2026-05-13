# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import transformers
from transformers import AutoConfig, AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForImageTextToText

model_id = "Qwen/Qwen3.6-35B-A3B"
# model_id = "Qwen/Qwen3.5-122B-A10B"
config = AutoConfig.from_pretrained(model_id)
config.torch_dtype = "float32"
# config.text_config.num_hidden_layers = 4

qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id, attn_implementation="eager", kv_offload=False, config=config
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)


## Only Text ##
## Set Batch_Size ##
batch_size = 1
qeff_model.compile(
    batch_size=batch_size,
    prefill_seq_len=32,
    ctx_len=4 * 1024,
    num_cores=16,
    num_devices=16,
    mxfp6_matmul=True,
    mxint8_kv_cache=False,
    aic_enable_depth_first=True,
    prefill_only=True,
    enable_chunking=True,
    # convert_to_fp16=False,
    # skip_vision=True,
    mos=1,
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Tell me about yourself."},
            # {"type": "text", "text": "Write an elaborate story of around 20000 words about a robot learning to paint with watercolors in a bustling city market. Include sensory details and emotional development of the robot character. What challenges does it face, and how does it overcome them?"},
        ],
    },
]


messages = [messages] * batch_size

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

inputs.pop("mm_token_type_ids")
streamer = TextStreamer(tokenizer)
output = qeff_model.generate(inputs=inputs, generation_len=100, streamer=streamer)
print(output.generated_ids)
print(tokenizer.batch_decode(output.generated_ids))
print(output)
