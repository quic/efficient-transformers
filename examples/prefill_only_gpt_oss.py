# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import time

import numpy as np
import torch
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession

model_id = "openai/gpt-oss-120b"  # weights are not required to convert to fp32

prompt = """
Once upon a time, in a small town, there lived a young boy named Alex. Alex was a curious and adventurous child, always eager to explore the world around him. One day, while playing in the park, Alex stumbled upon a mysterious old book hidden beneath a pile of leaves. The book was filled with stories of distant lands, magical creatures, and extraordinary adventures.

As Alex flipped through the pages, he discovered a map that led to a hidden treasure. Excited by the prospect of a real-life treasure hunt, Alex decided to embark on a thrilling journey. He packed his backpack with snacks, a flashlight, and a compass, and set off into the unknown.

The path to the treasure was not an easy one. Alex had to navigate through dense forests, cross rickety bridges, and solve riddles that guarded the treasure's location.
"""
all_outputs = []
# Run prefill
tokenizer = AutoTokenizer.from_pretrained(model_id)
PREFILL_SEQ_LEN = 8192
CTX_LEN = 8192
inputs = tokenizer(prompt, return_tensors="np", padding=True)
position_ids = inputs["attention_mask"].sum(1, keepdims=True)
padded_len = inputs["input_ids"].shape[1]
num_chunks = -(padded_len // -PREFILL_SEQ_LEN)  # ceil divide without float
padded_len = num_chunks * PREFILL_SEQ_LEN  # Convert to a multiple of prompt_len

# Initialize variables specific to request
# Calculate the max generation length.
max_gen_len = CTX_LEN - position_ids.max()
generation_len = max_gen_len

# config = AutoConfig.from_pretrained(model_id)
# config.num_hidden_layers=2
# model = AutoModelForCausalLM.from_config(config)
# qeff_model = QEFFAutoModelForCausalLM(model)
qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, num_hidden_layers=2)
config = qeff_model.model.config
inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
inputs.pop("token_type_ids", None)
inputs = {k: torch.from_numpy(v) for k, v in inputs.items()}
past_key_values = []
for i in range(config.num_hidden_layers):
    cache_len = config.sliding_window if i % 2 == 0 else PREFILL_SEQ_LEN
    pad_shape = (1, 8, cache_len, 64)
    past_key = torch.zeros((pad_shape), dtype=torch.float32)
    past_value = torch.zeros((pad_shape), dtype=torch.float32)
    pkv = (past_key, past_value)
    past_key_values.append(pkv)
inputs["past_key_values"] = past_key_values

# DECODE ONLY MODEL
prefill_qpc_path = qeff_model.compile(
    prefill_seq_len=1,
    ctx_len=CTX_LEN,
    num_cores=16,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    num_devices=8,
    mos=1,
    aic_enable_depth_first=True,
    num_speculative_tokens=None,
    # prefill_only=True,
)

# PREFILL ONLY MODEL
prefill_qpc_path = qeff_model.compile(
    prefill_seq_len=PREFILL_SEQ_LEN,
    ctx_len=CTX_LEN,
    num_cores=16,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    num_devices=8,
    mos=1,
    aic_enable_depth_first=True,
    num_speculative_tokens=None,
    prefill_only=True,
)
prefill_session = QAICInferenceSession(prefill_qpc_path)

logits_out_placeholder = np.zeros((1, 1, 201088), dtype=np.float32)
prefill_session.set_buffers({"logits": logits_out_placeholder})
inputs.pop("past_key_values")
inputs = {k: v.detach().numpy() for k, v in inputs.items()}
st = time.time()
qpc_out = prefill_session.run(inputs)
print(f"time for prefill_run={time.time() - st} sec\n")
