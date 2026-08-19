# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import time

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession

model_id = "tiny-random/glm-4-moe"
prompt = """
Explain quantum computing in simple terms.
"""
config = AutoConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
PREFILL_SEQ_LEN = 512
CTX_LEN = 1024
NUM_CORES = 4
MOE_PREFILL_PACKED_CHUNK_SIZE = 256

qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id)
decode_qpc_path = qeff_model.compile(
    prefill_seq_len=1,
    ctx_len=CTX_LEN,
    num_cores=NUM_CORES,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    num_devices=1,
    mos=1,
    aic_enable_depth_first=False,
    user_tiled=True,
    num_speculative_tokens=None,
    offload_pt_weights=False,
    retain_full_kv=True,
    use_onnx_subfunctions=True,
    qaic_config={"enable_blocking": True, "blocking_mode": "kv", "num_kv_blocks": 2},
)

prefill_qpc_path = qeff_model.compile(
    prefill_seq_len=PREFILL_SEQ_LEN,
    ctx_len=CTX_LEN,
    num_cores=NUM_CORES,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    num_devices=1,
    # split_retained_state_io=True,
    mos=1,
    aic_enable_depth_first=False,
    user_tiled=True,
    num_speculative_tokens=None,
    prefill_only=True,
    moe_prefill_packed_chunk_size=MOE_PREFILL_PACKED_CHUNK_SIZE,
    enable_chunking=True,
    use_onnx_subfunctions=True,
    qaic_config={"enable_blocking": True, "blocking_mode": "kv", "num_kv_blocks": 2},
)

inputs = tokenizer(prompt, return_tensors="np", padding=True)
position_ids = inputs["attention_mask"].sum(1, keepdims=True)
generation_len = CTX_LEN - position_ids.max()
padded_len = inputs["input_ids"].shape[1]
num_chunks = -(padded_len // -PREFILL_SEQ_LEN)
padded_len = num_chunks * PREFILL_SEQ_LEN
inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
inputs.pop("token_type_ids", None)
inputs = {k: torch.from_numpy(v) for k, v in inputs.items()}
inputs.pop("past_key_values", None)
inputs = {k: v.detach().numpy() for k, v in inputs.items()}

prefill_session = QAICInferenceSession(prefill_qpc_path)
decode_session = QAICInferenceSession(decode_qpc_path)

all_outputs = []
for chunk_id in range(num_chunks):
    chunk_inputs = inputs.copy()
    chunk_inputs["input_ids"] = inputs["input_ids"][:, chunk_id * PREFILL_SEQ_LEN : (chunk_id + 1) * PREFILL_SEQ_LEN]
    chunk_inputs["position_ids"] = inputs["position_ids"][
        :, chunk_id * PREFILL_SEQ_LEN : (chunk_id + 1) * PREFILL_SEQ_LEN
    ]
    ins = time.time()
    qpc_out = prefill_session.run(chunk_inputs)
    print(f"time for this run={time.time() - ins}")
    for layer_idx in range(config.num_hidden_layers):
        inputs[f"past_key.{layer_idx}"] = qpc_out[f"past_key.{layer_idx}_RetainedState"]
        inputs[f"past_value.{layer_idx}"] = qpc_out[f"past_value.{layer_idx}_RetainedState"]

all_outputs.append(np.argmax(qpc_out["logits"]))

decode_inputs = {
    "input_ids": np.argmax(qpc_out["logits"]).reshape(1, 1),
    "position_ids": np.max(inputs["position_ids"]).reshape(1, 1) + 1,
}
for layer_idx in range(config.num_hidden_layers):
    decode_inputs[f"past_key.{layer_idx}"] = qpc_out[f"past_key.{layer_idx}_RetainedState"]
    decode_inputs[f"past_value.{layer_idx}"] = qpc_out[f"past_value.{layer_idx}_RetainedState"]

st = time.time()
decode_out = decode_session.run(decode_inputs)
print(f"time for first run of decode with KV as input = {time.time() - st} sec\n")
all_outputs.append(np.argmax(decode_out["logits"]))
pos_id = np.max(decode_inputs["position_ids"]).reshape(1, 1) + 1
loop_decode_inputs = {
    "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
    "position_ids": pos_id,
}

for layer_idx in range(config.num_hidden_layers):
    loop_decode_inputs[f"past_key.{layer_idx}"] = decode_out[f"past_key.{layer_idx}_RetainedState"]
    loop_decode_inputs[f"past_value.{layer_idx}"] = decode_out[f"past_value.{layer_idx}_RetainedState"]

st = time.time()
for _ in range(generation_len - 2):
    decode_out = decode_session.run(loop_decode_inputs)
    all_outputs.append(np.argmax(decode_out["logits"]))
    pos_id += 1
    for layer_idx in range(config.num_hidden_layers):
        loop_decode_inputs[f"past_key.{layer_idx}"] = decode_out[f"past_key.{layer_idx}_RetainedState"]
        loop_decode_inputs[f"past_value.{layer_idx}"] = decode_out[f"past_value.{layer_idx}_RetainedState"]

    loop_decode_inputs.update(
        {
            "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
            "position_ids": pos_id,
        }
    )
ft = time.time()

print(f"decode tok/sec={(generation_len - 2) / (ft - st)}")
print(f"input\n{prompt}\noutput\n{tokenizer.decode(all_outputs)}")
