# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import transformers
from transformers import AutoConfig, AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForImageTextToText

# from QEfficient import QEFFAutoModelForCausalLM

## For AWQ model update pytorch version to 2.8.*
model_id = "Qwen/Qwen3.5-0.8B"
config = AutoConfig.from_pretrained(model_id)
config.torch_dtype = "float32"
# config.text_config.num_hidden_layers = 2

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
    prefill_seq_len=64,
    ctx_len=4096,
    num_cores=16,
    num_devices=1,
    # mxfp6_matmul=False,
    mxfp6_matmul=False,
    mxint8_kv_cache=False,
    aic_enable_depth_first=True,
    # convert_to_fp16=False,
    # skip_vision=True,
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


messages = [messages] * batch_size

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

# inputs = qeff_model.model.prepare_inputs_for_generation(inputs=inputs, prefill_seq_len=128, batch_size=batch_size)

inputs.pop("mm_token_type_ids")
# import ipdb; ipdb.set_trace()
streamer = TextStreamer(tokenizer)
output = qeff_model.generate(inputs=inputs, generation_len=100)
print(output.generated_ids)
print(tokenizer.batch_decode(output.generated_ids))
print(output)
