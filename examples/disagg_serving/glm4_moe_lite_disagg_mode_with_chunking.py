# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
#
# Disaggregated prefill + chunked inference example for:
#   cerebras/GLM-4.7-Flash-REAP-23B-A3B  (glm4_moe_lite, 47 layers, 48 experts, 3B active)
#
# Two QPC binaries are compiled:
#   decode_qpc  – prefill_seq_len=1,               used for autoregressive decode
#   prefill_qpc – prefill_seq_len=PREFILL_SEQ_LEN, prefill_only + enable_chunking
#
# Usage:
#   python glm4_moe_lite_disagg_mode_with_chunking.py
#
# The first run of prefill_qpc will fail and print the compile command.
# Copy the generated QPC path into `prefill_qpc_path` and re-run.
#
# For disaggregated serving via vLLM set split_retained_state_io=True in both
# compile() calls.
# -----------------------------------------------------------------------------

import time

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession

# ---------------------------------------------------------------------------
# Model / prompt configuration
# ---------------------------------------------------------------------------
model_id = "cerebras/GLM-4.7-Flash-REAP-23B-A3B"

# GLM-4.7 uses role tokens baked into the vocabulary; apply_chat_template or
# format manually as shown below.
system_prompt = "You are a helpful AI assistant."
user_message = "Write a Python function that checks if a number is prime."

prompt = f"<|system|>\n{system_prompt}<|user|>\n{user_message}<|assistant|>\n"

# ---------------------------------------------------------------------------
# Compile-time shapes
# ---------------------------------------------------------------------------
PREFILL_SEQ_LEN = 128   # chunk size; must divide evenly into the padded prompt
CTX_LEN = 4096          # total KV-cache context window

# ---------------------------------------------------------------------------
# Load model (weights not required for ONNX export / compilation)
# ---------------------------------------------------------------------------
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

# ---------------------------------------------------------------------------
# Step 1 — compile decode QPC  (prefill_seq_len=1, retain_full_kv=True)
# offload_pt_weights=False keeps weights in CPU RAM so the prefill export
# reuses them in Step 2 without a second download.
# ---------------------------------------------------------------------------
decode_qpc_path = qeff_model.compile(
    prefill_seq_len=1,
    ctx_len=CTX_LEN,
    num_cores=16,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    num_devices=1,
    mos=1,
    aic_enable_depth_first=True,
    num_speculative_tokens=None,
    offload_pt_weights=False,  # keep weights for Step 2
    retain_full_kv=True,
    # split_retained_state_io=True,  # enable for disagg vLLM serving
)
print(f"decode QPC path: {decode_qpc_path}")

# ---------------------------------------------------------------------------
# Step 2 — compile prefill QPC  (chunked, prefill_only)
# On the very first run this call will error and print the compile command.
# Provide the resulting path via the `prefill_qpc_path` variable below and
# comment out this compile() call.
# ---------------------------------------------------------------------------

# prefill_qpc_path = ""  # uncomment and fill after first run

prefill_qpc_path = qeff_model.compile(
    prefill_seq_len=PREFILL_SEQ_LEN,
    ctx_len=CTX_LEN,
    num_cores=16,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    num_devices=1,
    mos=1,
    aic_enable_depth_first=True,
    num_speculative_tokens=None,
    prefill_only=True,
    enable_chunking=True,
    retain_full_kv=True,
    # split_retained_state_io=True,  # enable for disagg vLLM serving
)

print(f"prefill QPC path: {prefill_qpc_path}")

# ---------------------------------------------------------------------------
# Tokenise & prepare chunked inputs
# ---------------------------------------------------------------------------
inputs = tokenizer(prompt, return_tensors="np", padding=True)
position_ids = inputs["attention_mask"].sum(1, keepdims=True)
generation_len = CTX_LEN - int(position_ids.max())
padded_len = inputs["input_ids"].shape[1]
num_chunks = -(padded_len // -PREFILL_SEQ_LEN)   # ceil divide
padded_len = num_chunks * PREFILL_SEQ_LEN         # round up to chunk boundary

inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
inputs.pop("token_type_ids", None)
inputs = {k: torch.from_numpy(v) for k, v in inputs.items()}
inputs.pop("past_key_values", None)
inputs = {k: v.detach().numpy() for k, v in inputs.items()}

# ---------------------------------------------------------------------------
# Create inference sessions
# ---------------------------------------------------------------------------
prefill_session = QAICInferenceSession(prefill_qpc_path)
decode_session = QAICInferenceSession(decode_qpc_path)

# ---------------------------------------------------------------------------
# Chunked prefill — iterate over PREFILL_SEQ_LEN-token chunks
# ---------------------------------------------------------------------------
all_outputs = []
qpc_out = None
for chunk_idx in range(num_chunks):
    chunk_inputs = inputs.copy()
    chunk_inputs["input_ids"] = inputs["input_ids"][:, chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN]
    chunk_inputs["position_ids"] = inputs["position_ids"][:, chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN]

    t0 = time.time()
    qpc_out = prefill_session.run(chunk_inputs)
    print(f"prefill chunk {chunk_idx + 1}/{num_chunks}: {time.time() - t0:.3f}s")

    # Transfer retained KV state from prefill to decode inputs buffer
    for layer_idx in range(config.num_hidden_layers):
        inputs[f"past_key.{layer_idx}"] = qpc_out[f"past_key.{layer_idx}_RetainedState"]
        inputs[f"past_value.{layer_idx}"] = qpc_out[f"past_value.{layer_idx}_RetainedState"]

# First generated token comes from the last prefill chunk's logits
all_outputs.append(int(np.argmax(qpc_out["logits"])))

# ---------------------------------------------------------------------------
# First decode step — hand off KV cache from prefill
# ---------------------------------------------------------------------------
decode_inputs = {
    "input_ids": np.argmax(qpc_out["logits"]).reshape(1, 1),
    "position_ids": (np.max(inputs["position_ids"]) + 1).reshape(1, 1),
}
for layer_idx in range(config.num_hidden_layers):
    decode_inputs[f"past_key.{layer_idx}"] = qpc_out[f"past_key.{layer_idx}_RetainedState"]
    decode_inputs[f"past_value.{layer_idx}"] = qpc_out[f"past_value.{layer_idx}_RetainedState"]

t0 = time.time()
decode_out = decode_session.run(decode_inputs)
print(f"first decode step (KV hand-off): {time.time() - t0:.3f}s")
all_outputs.append(int(np.argmax(decode_out["logits"])))

# ---------------------------------------------------------------------------
# Autoregressive decode loop
# ---------------------------------------------------------------------------
pos_id = decode_inputs["position_ids"] + 1
loop_inputs = {
    "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
    "position_ids": pos_id,
}
for layer_idx in range(config.num_hidden_layers):
    loop_inputs[f"past_key.{layer_idx}"] = decode_out[f"past_key.{layer_idx}_RetainedState"]
    loop_inputs[f"past_value.{layer_idx}"] = decode_out[f"past_value.{layer_idx}_RetainedState"]

eos_token_ids = {154820, 154827, 154829}  # GLM-4.7 multi-EOS tokens
t0 = time.time()
for _ in range(generation_len - 2):
    decode_out = decode_session.run(loop_inputs)
    next_token = int(np.argmax(decode_out["logits"]))
    all_outputs.append(next_token)

    pos_id = pos_id + 1
    loop_inputs.update(
        {
            "input_ids": np.array([[next_token]]),
            "position_ids": pos_id,
        }
    )
    for layer_idx in range(config.num_hidden_layers):
        loop_inputs[f"past_key.{layer_idx}"] = decode_out[f"past_key.{layer_idx}_RetainedState"]
        loop_inputs[f"past_value.{layer_idx}"] = decode_out[f"past_value.{layer_idx}_RetainedState"]

    if next_token in eos_token_ids:
        break

elapsed = time.time() - t0
tokens_generated = len(all_outputs) - 2
print(f"\ndecode throughput: {tokens_generated / elapsed:.2f} tok/s ({tokens_generated} tokens)")
print(f"\n--- Input ---\n{prompt}")
print(f"--- Output ---\n{tokenizer.decode(all_outputs, skip_special_tokens=True)}")
