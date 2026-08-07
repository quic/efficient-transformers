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
from QEfficient.utils import constants

model_id = "mistralai/Mixtral-8x7B-v0.1"
prompt = """
Tell me about yourself.
"""

config = AutoConfig.from_pretrained(model_id)
# config.num_hidden_layers=2
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

PREFILL_SEQ_LEN = 512
CTX_LEN = 4096
NUM_CORES = 16
MOE_PREFILL_PACKED_CHUNK_SIZE = 256

qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, config=config)

try:
    constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN = 1
    decode_qpc_path = qeff_model.compile(
        prefill_seq_len=1,
        prefill_only=False,
        ctx_len=CTX_LEN,
        num_cores=NUM_CORES,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        num_devices=2,
        mos=1,
        aic_enable_depth_first=True,
        # num_speculative_tokens=None,
        offload_pt_weights=False,  # keep weights resident for prefill export/compile
        retain_full_kv=True,
        split_model_io=True,
        use_onnx_subfunctions=True,
    )
finally:
    constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN = 32

prefill_qpc_path = qeff_model.compile(
    prefill_seq_len=PREFILL_SEQ_LEN,
    ctx_len=CTX_LEN,
    num_cores=NUM_CORES,
    moe_prefill_packed_chunk_size=MOE_PREFILL_PACKED_CHUNK_SIZE,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    num_devices=2,
    mdp_num_partitions=2,
    split_retained_state_io=True,
    mos=1,
    user_tiled=True,
    aic_enable_depth_first=False,
    num_speculative_tokens=None,
    prefill_only=True,
    enable_chunking=True,
    use_onnx_subfunctions=True,
)

inputs = tokenizer(prompt, return_tensors="np", padding=True)
position_ids = inputs["attention_mask"].sum(1, keepdims=True)
generation_len = CTX_LEN - position_ids.max()
padded_len = inputs["input_ids"].shape[1]
num_chunks = -(padded_len // -PREFILL_SEQ_LEN)  # ceil divide without float
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
for chunk_idx in range(num_chunks):
    chunk_inputs = inputs.copy()
    chunk_inputs["input_ids"] = inputs["input_ids"][:, chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN]
    chunk_inputs["position_ids"] = inputs["position_ids"][
        :, chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
    ]
    start = time.time()
    qpc_out = prefill_session.run(chunk_inputs)
    print(f"time for this prefill chunk={time.time() - start}")
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

start = time.time()
decode_out = decode_session.run(decode_inputs)
print(f"time for first decode run with KV as input = {time.time() - start} sec\n")
all_outputs.append(np.argmax(decode_out["logits"]))
pos_id = np.max(decode_inputs["position_ids"]).reshape(1, 1) + 1
loop_decode_inputs = {
    "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
    "position_ids": pos_id,
}


for layer_idx in range(config.num_hidden_layers):
    loop_decode_inputs[f"past_key.{layer_idx}"] = decode_out[f"past_key.{layer_idx}_RetainedState"]
    loop_decode_inputs[f"past_value.{layer_idx}"] = decode_out[f"past_value.{layer_idx}_RetainedState"]

start = time.time()
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
finish = time.time()

print(f"decode tok/sec={(generation_len - 2) / (finish - start)}")
print(f"input\n{prompt}\noutput\n{tokenizer.decode(all_outputs)}")
