# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Session layout
--------------
  prefill_session : cluster_id="prefill", stages=1
      exec-obj pool: [slot-1]  (prefill only, no decode slot)

  decode_session  : cluster_id="decode"
      exec-obj pool: [slot-0]  (decode only, no prefill pool)

Shared KV buffers
-----------------
  kv_cache : list[np.ndarray]  — one array per (layer, key/value) pair,
             shape = prefill_session.kv_shape, dtype = prefill_session.kv_size
             Allocated once; written by prefill via DMA slice, read by decode
             via the same pointer — no copy at the prefill→decode boundary.
"""

import math
import time

import numpy as np
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer_KV_share import QAICInferenceSession

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
PREFILL_SEQ_LEN = 128
CTX_LEN = 256  # = PREFILL_SEQ_LEN * 2
STAGES = 4  # prefill pp stages

qeff_model = QEFFAutoModelForCausalLM.from_pretrained(MODEL_ID)
PREFILL_QPC_PATH = qeff_model.compile(
    prefill_seq_len=PREFILL_SEQ_LEN,
    ctx_len=CTX_LEN,
    num_cores=16,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    num_devices=8,  #  TS=2 num_devices/STAGES
    split_retained_state_io=True,
    mos=1,
    aic_enable_depth_first=True,
    num_speculative_tokens=None,
    prefill_only=True,
    enable_chunking=True,
    stages=STAGES,
    # use_onnx_subfunctions=True,
)

DECODE_QPC_PATH = qeff_model.compile(
    prefill_seq_len=1,
    ctx_len=CTX_LEN,
    num_cores=16,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    num_devices=2,
    split_retained_state_io=True,
    mos=1,
    aic_enable_depth_first=True,
    num_speculative_tokens=None,
    offload_pt_weights=False,
    retain_full_kv=True,
)

PROMPT = """
Explain quantum computing in simple terms.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Load tokenizer and model config
# ─────────────────────────────────────────────────────────────────────────────
print("Loading tokenizer and config …")
config = AutoConfig.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
num_layers = config.num_hidden_layers
print(f"  num_hidden_layers = {num_layers}")

# ─────────────────────────────────────────────────────────────────────────────
# Tokenise and build chunked inputs
# ─────────────────────────────────────────────────────────────────────────────
raw_inputs = tokenizer(PROMPT, return_tensors="np", padding=True)
generation_len = CTX_LEN - int(raw_inputs["attention_mask"].sum(1, keepdims=True).max())

padded_len = raw_inputs["input_ids"].shape[1]
num_chunks = math.ceil(padded_len / PREFILL_SEQ_LEN)  # ceil divide
padded_len = num_chunks * PREFILL_SEQ_LEN

inputs = tokenizer(
    PROMPT,
    return_tensors="np",
    padding="max_length",
    max_length=padded_len,
)
inputs["position_ids"] = np.where(
    inputs.pop("attention_mask"),
    np.arange(padded_len),
    -1,
)
inputs.pop("token_type_ids", None)
inputs.pop("past_key_values", None)

print(f"  prompt tokens = {padded_len}  num_chunks = {num_chunks}  generation_len = {generation_len}")

# ─────────────────────────────────────────────────────────────────────────────
# Load sessions
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading prefill session (cluster_id='prefill', stages) …")
prefill_session = QAICInferenceSession(
    qpc_path=PREFILL_QPC_PATH,
    full_batch_size=1,
    device_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    cluster_id="prefill",
    stages=STAGES,
)
print("  prefill_session loaded ✓")
print(f"  prefill exec-obj pool size : {prefill_session.prefill_num_execObj}")
print(f"  kv_shape : {prefill_session.kv_shape}  kv_dtype : {prefill_session.kv_size}")

print("\nLoading decode session (cluster_id='decode') …")
decode_session = QAICInferenceSession(
    qpc_path=DECODE_QPC_PATH,
    full_batch_size=1,
    device_ids=[8, 9],  # 2-device MQ decode
    cluster_id="decode",
)
print("  decode_session loaded ✓")

# ─────────────────────────────────────────────────────────────────────────────
# Allocate shared KV cache buffers
#
# One numpy array per (layer, key/value) pair.
# These arrays are shared between prefill (written via DMA slice) and decode
# (read as direct inputs) — no copy at the boundary.
# ─────────────────────────────────────────────────────────────────────────────
kv_cache: list[np.ndarray] = [
    np.zeros(prefill_session.kv_shape, dtype=prefill_session.kv_size)
    for _ in prefill_session.prefill_buff_map[:-1]  # KV entries only, no logits
]
# Logits output buffer for the last prefill chunk (written in-place by the QPC)
logits_buf = np.zeros((1, 1, config.vocab_size), dtype=np.float32)

print(f"\nAllocated {len(kv_cache)} shared KV buffers  shape={prefill_session.kv_shape}")

# ─────────────────────────────────────────────────────────────────────────────
# Chunked prefill
#
# Non-last chunks: np_run(is_prefill=True)
# Last chunk:      np_run_pipeline(last_chunk=True, kv_cache_buffers=kv_cache)
#                  set_data_for_kv_handoff() wires RetainedState outputs into
#                  kv_cache via setDataWithSlices before enqueue — zero-copy.
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Chunked prefill ──")
prefill_logits = None

for chunk_idx in range(num_chunks):
    start = chunk_idx * PREFILL_SEQ_LEN
    end = start + PREFILL_SEQ_LEN
    is_last = chunk_idx == num_chunks - 1

    chunk_inputs = {
        "input_ids": inputs["input_ids"][:, start:end],
        "position_ids": inputs["position_ids"][:, start:end],
        "batch_index": np.array([[0]], dtype=np.int64),
    }
    if is_last:
        chunk_inputs["logits"] = logits_buf

    t0 = time.perf_counter()

    if is_last:
        # Last chunk: wire KV outputs into shared kv_cache via DMA slice,
        # then enqueue.  No numpy copy of KV at the prefill→decode boundary.
        exec_idx = prefill_session.np_run_pipeline(
            inputs=chunk_inputs,
            last_chunk=True,
            kv_cache_buffers=kv_cache,
        )
    else:
        # Non-last chunks:
        # KV RetainedState outputs are skipped (not needed mid-pipeline).
        exec_idx = prefill_session.np_run(
            inputs=chunk_inputs,
            is_prefill=True,
        )

    prefill_session.complete_inf(exec_idx, is_prefill=True)
    elapsed = time.perf_counter() - t0

    print(f"  chunk {chunk_idx + 1}/{num_chunks}  last={is_last}  time={elapsed * 1000:.1f} ms")

# After the last chunk, logits_buf has been written in-place by the QPC
prefill_logits = logits_buf
first_token = int(np.argmax(prefill_logits))
print(f"\n  Prefill done.  First generated token id = {first_token}")

# ─────────────────────────────────────────────────────────────────────────────
# Decode — shared DMA KV buffer design
#
# kv_cache is the single shared numpy buffer across all decode steps:
#   - prefill wrote into it via setDataWithSlices (zero-copy DMA slice)
#   - each decode step reads KV from it as input (decode_buff_map INPUT slots)
#   - each decode step writes updated KV back into it via set_data_for_kv_handoff
#     (setDataWithSlices on RetainedState OUTPUT slots → same kv_cache arrays)
#
# decode_inputs[kv_name] always points at kv_cache[i] — never reassigned.
# After complete_inf, kv_cache is updated in-place with the new token's KV.
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Decode ──")

next_pos = int(np.max(inputs["position_ids"])) + 1

decode_inputs: dict[str, np.ndarray] = {
    "input_ids": np.array([[first_token]], dtype=np.int64),
    "position_ids": np.array([[next_pos]], dtype=np.int64),
    "logits": logits_buf,
}
for (kv_name, _), kv_buf in zip(decode_session.decode_buff_map, kv_cache):
    decode_inputs[kv_name] = kv_buf  # same kv_cache array, no copy

all_tokens = [first_token]


def run_decode_step():
    # Wire RetainedState OUTPUT slots → kv_cache via setDataWithSlices.
    # Runtime DMA-writes updated KV directly into kv_cache after inference.
    # decode_rs_buff_map holds OUTPUT binding indices.
    decode_session.set_data_for_kv_handoff(
        kv_cache,
        [("batch_index", 0), ("ctx_start", 0)],
        decode_session.decode_execObj_idx,
        decode_session.decode_rs_buff_map,
    )
    exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
    decode_session.complete_inf(exec_idx, is_prefill=False)
    return decode_session.get_outputs(exec_idx)


t0 = time.perf_counter()
dec_outputs = run_decode_step()
print(f"  First decode step  time={(time.perf_counter() - t0) * 1000:.1f} ms")

next_token = int(np.argmax(dec_outputs["logits"]))
all_tokens.append(next_token)
next_pos += 1

# ─────────────────────────────────────────────────────────────────────────────
# Decode loop — kv_cache updated in-place each step, no KV copy or reassignment
# ─────────────────────────────────────────────────────────────────────────────
t_loop_start = time.perf_counter()

for step in range(generation_len - 2):
    decode_inputs["input_ids"] = np.array([[next_token]], dtype=np.int64)
    decode_inputs["position_ids"] = np.array([[next_pos]], dtype=np.int64)

    dec_outputs = run_decode_step()

    next_token = int(np.argmax(dec_outputs["logits"]))
    all_tokens.append(next_token)
    next_pos += 1

t_loop_end = time.perf_counter()

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
decode_steps = generation_len - 2
tok_per_sec = decode_steps / (t_loop_end - t_loop_start) if decode_steps > 0 else 0.0

print(f"\n  decode tok/sec = {tok_per_sec:.2f}  ({decode_steps} steps)")
print(f"\ninput\n{PROMPT}\noutput\n{tokenizer.decode(all_tokens)}")

prefill_session.deactivate()
decode_session.deactivate()
