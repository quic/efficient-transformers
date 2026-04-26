# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

# parameters to be configured
prompt = "Once upon a time,"
num_hidden_layers = 2
TS = 4
mla_absorption_config = {"cache_compressed": False, "absorption": False, "online": False}
# qaic_config = None #for orig_forward
# qaic_config = {"num_kv_heads_repeat": TS}  #with  head replication for orig_forward
# qaic_config = {"enable_blocking": True, "blocking_mode": "h", "num_kv_heads_repeat": TS} # for h blocking, it internally sets head_block_size equal to num_devices/num_kv_heads_repeat
qaic_config = {"enable_blocking": True, "blocking_mode": "kv"}  # for KV blocking

# model_path = "/home/ochougul/.cache/huggingface/hub/models--moonshotai--Kimi-K2-Thinking/snapshots/a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55/"
model_path = (
    "/home/huggingface_hub/models--moonshotai--Kimi-K2-Thinking/snapshots/612681931a8c906ddb349f8ad0f582cb552189cd"
)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float32, num_hidden_layers=num_hidden_layers, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("moonshotai/Kimi-K2-Thinking", trust_remote_code=True)

PREFILL_SEQ_LEN = 32
CTX_LEN = 8192
generation_len = 10
generated_ids = []

inputs = tokenizer(prompt, return_tensors="pt", padding=True)
padded_len = inputs["input_ids"].shape[1]
num_chunks = -(padded_len // -PREFILL_SEQ_LEN)  # ceil divide without float
padded_len = num_chunks * PREFILL_SEQ_LEN  # Convert to a multiple of prompt_len

# with torch.no_grad():
#    out = model(**inputs)
#    predictions = torch.argmax(out.logits, dim=-1)

qeff_model = QEFFAutoModelForCausalLM(model)
qeff_model.transform(ctx_len=CTX_LEN, seq_len=PREFILL_SEQ_LEN, bs=1, num_devices=TS, qaic_config=qaic_config)
qeff_model.mla(mla_absorption_config=mla_absorption_config)

inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
inputs.pop("token_type_ids", None)
inputs = {k: torch.from_numpy(v) for k, v in inputs.items()}

pad_shape_k = (
    1,
    model.config.num_attention_heads,
    CTX_LEN,
    model.config.qk_nope_head_dim + model.config.qk_rope_head_dim,
)
pad_shape_v = (1, model.config.num_attention_heads, CTX_LEN, model.config.v_head_dim)

num_heads = model.model.layers[0].self_attn.kv_a_proj_with_mqa.weight.shape[0] // (
    model.config.kv_lora_rank + model.config.qk_rope_head_dim
)
pad_shape_ckv = (1, num_heads, CTX_LEN, model.config.kv_lora_rank)
pad_shape_k_pe = (1, num_heads, CTX_LEN, model.config.qk_rope_head_dim)

past_key_values = []
compressed_kvs = []

for i in range(model.config.num_hidden_layers):
    past_key = torch.zeros((pad_shape_k), dtype=torch.float32)
    past_value = torch.zeros((pad_shape_v), dtype=torch.float32)
    pkv = (past_key, past_value)
    past_key_values.append(pkv)

    ckv = torch.zeros((pad_shape_ckv), dtype=torch.float32)
    k_pe = torch.zeros((pad_shape_k_pe), dtype=torch.float32)
    x = (ckv, k_pe)
    compressed_kvs.append(x)

cache_compressed = mla_absorption_config.get("cache_compressed", False)
if cache_compressed:
    inputs["compressed_kvs"] = compressed_kvs
else:
    inputs["past_key_values"] = past_key_values

prefill_qeff_out = qeff_model.model(**inputs)

position_ids = inputs["position_ids"]
qeff_out = prefill_qeff_out
qeff_generated_ids = []
for _ in range(1, generation_len):
    next_token_id = qeff_out["logits"][:, -1, :].argmax(-1).reshape(-1, 1)
    qeff_generated_ids.append(next_token_id)
    position_ids = position_ids.max(1, keepdim=True).values + 1
    decode_inputs = {
        "input_ids": next_token_id,
        "position_ids": position_ids,
    }
    if cache_compressed:
        decode_inputs["compressed_kvs"] = qeff_out["past_key_values"]
    else:
        decode_inputs["past_key_values"] = qeff_out["past_key_values"]

    qeff_out = qeff_model.model(**decode_inputs)

qeff_generated_ids.append(qeff_out["logits"][:, -1, :].argmax(-1).reshape(-1, 1))
qeff_generated_ids = np.concatenate(qeff_generated_ids, axis=1)
predicted_string = tokenizer.batch_decode(qeff_generated_ids, skip_special_tokens=True)
print("QEFF Transformed Model Outputs (Torch CPU): \n")
print("Prompt:", repr(prompt))
print("Completion:", repr(predicted_string))
