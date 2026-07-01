# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
GPT-OSS disaggregated serving with chunked prefill and KV-slice DMA handoff.

Shared KV buffers
-----------------
  kv_cache : list[np.ndarray]  — one array per (layer, key/value) pair,
             shape = prefill_session.kv_shape, dtype = prefill_session.kv_size
             Allocated once; written by prefill via DMA slice, read by decode
             via the same pointer — no copy at the prefill->decode boundary.
"""

import math
import os
import time

import numpy as np
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer_KV_share import QAICInferenceSession

dir_path = os.path.dirname(os.path.realpath(__file__))
subfunc_npi_file_path = os.path.join(dir_path, "subfunction_120b_npi.yaml")
non_subfunc_npi_file_path = os.path.join(dir_path, "non_subfunction_120b_npi.yaml")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID = "tiny-random/gpt-oss-bf16"
# MODEL_ID = "openai/gpt-oss-20b"
PREFILL_SEQ_LEN = 512
CTX_LEN = 1024
NUM_CORES = 16
MOE_PREFILL_PACKED_CHUNK_SIZE = 256
STAGES = 2
PREFILL_NUM_DEVICES = 4
DECODE_NUM_DEVICES = 4
LAYERWISE = False

PROMPT = """
Once upon a time, in a small town, there lived a young boy named Alex. Alex was a curious and adventurous child, always eager to explore the world around him. One day, while playing in the park, Alex stumbled upon a mysterious old book hidden beneath a pile of leaves. The book was filled with stories of distant lands, magical creatures, and extraordinary adventures.

As Alex flipped through the pages, he discovered a map that led to a hidden treasure. Excited by the prospect of a real-life treasure hunt, Alex decided to embark on a thrilling journey. He packed his backpack with snacks, a flashlight, and a compass, and set off into the unknown.

The path to the treasure was not an easy one. Alex had to navigate through dense forests, cross rickety bridges, and solve riddles that guarded the treasure's location.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Compile decode and prefill QPCs
# ─────────────────────────────────────────────────────────────────────────────
qeff_model = QEFFAutoModelForCausalLM.from_pretrained(MODEL_ID, layerwise=LAYERWISE)

decode_qpc_path = qeff_model.compile(
    prefill_seq_len=1,
    ctx_len=CTX_LEN,
    num_cores=NUM_CORES,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    num_devices=DECODE_NUM_DEVICES,
    mos=1,
    aic_enable_depth_first=True,
    num_speculative_tokens=None,
    use_onnx_subfunctions=True,
    offload_pt_weights=False,
    retain_full_kv=True,
    split_model_io=True,
    split_retained_state_io=True,
    # node_precision_info=non_subfunc_npi_file_path,
)

prefill_qpc_path = qeff_model.compile(
    prefill_seq_len=PREFILL_SEQ_LEN,
    ctx_len=CTX_LEN,
    num_cores=NUM_CORES,
    moe_prefill_packed_chunk_size=MOE_PREFILL_PACKED_CHUNK_SIZE,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    num_devices=PREFILL_NUM_DEVICES,
    mos=1,
    user_tiled=True,
    aic_enable_depth_first=False,
    num_speculative_tokens=None,
    prefill_only=True,
    enable_chunking=True,
    use_onnx_subfunctions=True,
    split_model_io=True,
    split_retained_state_io=True,
    # node_precision_info=subfunc_npi_file_path,
    mdp_num_partitions=STAGES,
)

# ─────────────────────────────────────────────────────────────────────────────
# Load tokenizer and model config
# ─────────────────────────────────────────────────────────────────────────────
print("Loading tokenizer and config ...")
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
num_chunks = math.ceil(padded_len / PREFILL_SEQ_LEN)
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
# Load sessions (KV-share sessions with cluster_id separation)
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading prefill session (cluster_id='prefill') ...")
prefill_session = QAICInferenceSession(
    qpc_path=prefill_qpc_path,
    full_batch_size=1,
    cluster_id="prefill",
    stages=STAGES,
)
print("  prefill_session loaded")
print(f"  prefill exec-obj pool size : {prefill_session.prefill_num_execObj}")
print(f"  kv_shape : {prefill_session.kv_shape}  kv_dtype : {prefill_session.kv_size}")

print("\nLoading decode session (cluster_id='decode') ...")
decode_session = QAICInferenceSession(
    qpc_path=decode_qpc_path,
    full_batch_size=1,
    cluster_id="decode",
)
print("  decode_session loaded")

# ─────────────────────────────────────────────────────────────────────────────
# Allocate shared KV cache buffers
# ─────────────────────────────────────────────────────────────────────────────
kv_cache: list[np.ndarray] = [
    np.zeros(prefill_session.kv_shape, dtype=prefill_session.kv_size) for _ in prefill_session.kv_only_buff_map
]
logits_buf = np.zeros((1, 1, config.vocab_size), dtype=np.float32)

print(f"\nAllocated {len(kv_cache)} shared KV buffers  shape={prefill_session.kv_shape}")

# ─────────────────────────────────────────────────────────────────────────────
# Chunked prefill with DMA KV-slice handoff
# ─────────────────────────────────────────────────────────────────────────────
print("\n-- Chunked prefill --")

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
        exec_idx = prefill_session.np_run_pipeline(
            inputs=chunk_inputs,
            last_chunk=True,
            kv_cache_buffers=kv_cache,
        )
    else:
        exec_idx = prefill_session.np_run(
            inputs=chunk_inputs,
            is_prefill=True,
        )

    prefill_session.complete_inf(exec_idx, is_prefill=True)
    elapsed = time.perf_counter() - t0

    print(f"  chunk {chunk_idx + 1}/{num_chunks}  last={is_last}  time={elapsed * 1000:.1f} ms")

first_token = int(np.argmax(logits_buf))
print(f"\n  Prefill done.  First generated token id = {first_token}")

# ─────────────────────────────────────────────────────────────────────────────
# Run decode loop kv_cache is updated in-place with the new token's KV.
# ─────────────────────────────────────────────────────────────────────────────
print("\n-- Decode --")

next_pos = int(np.max(inputs["position_ids"])) + 1

decode_inputs: dict[str, np.ndarray] = {
    "input_ids": np.array([[first_token]], dtype=np.int64),
    "position_ids": np.array([[next_pos]], dtype=np.int64),
    "logits": logits_buf,
}
for (kv_name, _), kv_buf in zip(decode_session.decode_buff_map, kv_cache):
    decode_inputs[kv_name] = kv_buf

all_tokens = [first_token]


def run_decode_step():
    """Execute one decode step with DMA KV handoff (zero-copy in-place update)."""
    decode_session.set_data_for_kv_handoff(
        kv_cache,
        [("batch_index", 0), ("ctx_start", 0)],
        decode_session.decode_execObj_idx,
        decode_session.decode_rs_kv_only_buff_map,
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
# Decode loop -- kv_cache updated in-place each step, no KV copy or reassignment
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
