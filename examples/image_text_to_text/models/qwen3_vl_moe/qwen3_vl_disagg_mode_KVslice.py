# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Disaggregated serving for Qwen3-VL-MoE with zero-copy DMA KV-slice handoff.

Session layout
--------------
  vision_session      :  Runs once per request; produces vision_embeds + deepstack_features.

  lang_prefill_session : QAICInferenceSession (cloud_infer_KV_share)
      cluster_id="prefill", stages=STAGES
      exec-obj pool: [slot-0 .. slot-STAGES]  (prefill only)

  lang_decode_session  : QAICInferenceSession (cloud_infer_KV_share)
      cluster_id="decode"
      exec-obj pool: [slot-0]  (decode only)

Shared KV buffers
-----------------
  kv_cache : list[np.ndarray]  — one array per (layer, key/value) pair,
             shape = lang_prefill_session.kv_shape  (per-batch-slot)
             dtype = lang_prefill_session.kv_size

  Allocated once before prefill.  On the last prefill chunk the runtime DMA-
  writes the RetainedState outputs directly into kv_cache via
  setDataWithSlices (zero-copy).  The decode session reads the same arrays as
  inputs and writes updated KV back into them each step — also via
  setDataWithSlices — so no numpy copy ever crosses the prefill→decode or
  decode→decode boundary.

Chunked prefill
---------------
  Non-last chunks : np_run(is_prefill=True) + complete_inf()
                    KV RetainedState outputs are skipped (on-device retained
                    state carries KV across chunks automatically).
  Last chunk      : np_run_pipeline(last_chunk=True, kv_cache_buffers=kv_cache)
                    set_data_for_kv_handoff() wires RetainedState outputs into
                    kv_cache via setDataWithSlices before enqueue — zero-copy.

Decode loop
-----------
  Each step calls run_decode_step() which:
    1. Wires RetainedState OUTPUT slots → kv_cache via set_data_for_kv_handoff
       (so the runtime DMA-writes updated KV into kv_cache after inference).
    2. np_run(is_prefill=False) + complete_inf() + get_outputs()
  decode_inputs[kv_name] always points at kv_cache[i] — never reassigned.
"""

import time
from time import perf_counter

import numpy as np
import requests
import torch
import transformers
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession as _StandardSession
from QEfficient.generation.cloud_infer_KV_share import AIC_TO_NP as _AIC_TO_NP
from QEfficient.generation.cloud_infer_KV_share import QAICInferenceSession as _KVShareSession

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct"
# model_id = "tiny-random/qwen3-vl-moe"
PREFILL_SEQ_LEN = 128
CTX_LEN = 4096
BS = 1
STAGES = 2  # pipeline stages for lang prefill; set >1 for pipelined PP prefill
PREFILL_NUM_DEVICES = 2
DECODE_NUM_DEVICES = 2
config = AutoConfig.from_pretrained(model_id)
config.dtype = "float16"

# For faster execution user can run with lesser layers, For Testing Purpose Only
config.vision_config.depth = 9
config.text_config.num_hidden_layers = 6
config.vision_config.deepstack_visual_indexes = [8]

qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id, attn_implementation="eager", kv_offload=True, config=config, dtype=torch.float16, layerwise=False
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

# ─────────────────────────────────────────────────────────────────────────────
# Compile
# ─────────────────────────────────────────────────────────────────────────────
skip_vision = False

if not skip_vision:
    vision_qpc_path = qeff_model.compile(
        batch_size=BS,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        height=354,
        width=536,
        num_cores=16,
        num_devices=1,
        mos=1,
        mxfp6_matmul=True,
        aic_enable_depth_first=True,
        skip_vision=skip_vision,
        split_model_io=True,
        skip_lang=True,
        use_onnx_subfunctions=True,
        layerwise=False,
    )

prefill_qpc_path = qeff_model.compile(
    batch_size=BS,
    prefill_seq_len=PREFILL_SEQ_LEN,
    ctx_len=CTX_LEN,
    height=354,
    width=536,
    num_cores=16,
    num_devices=PREFILL_NUM_DEVICES,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    split_retained_state_io=True,  # enables DMA KV-slice handoff
    split_model_io=True,
    mos=1,
    aic_enable_depth_first=True,
    prefill_only=True,
    enable_chunking=True,
    skip_vision=True,
    use_onnx_subfunctions=True,
    layerwise=False,
    layerwise_window_size=1,
    mdp_num_partitions=STAGES,
    mdp_strategy="onnx",
)

decode_qpc_path = qeff_model.compile(
    batch_size=BS,
    prefill_seq_len=1,
    ctx_len=CTX_LEN,
    height=354,
    width=536,
    num_cores=16,
    num_devices=DECODE_NUM_DEVICES,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    split_retained_state_io=True,  # enables DMA KV-slice handoff
    split_model_io=True,
    mos=1,
    aic_enable_depth_first=True,
    prefill_only=False,
    skip_vision=True,
    use_onnx_subfunctions=True,
    layerwise=False,
    layerwise_window_size=1,
)

# ─────────────────────────────────────────────────────────────────────────────
# Load sessions
# ─────────────────────────────────────────────────────────────────────────────
# Vision session: no KV cache, use the standard session
if not skip_vision:
    vision_session = _StandardSession(vision_qpc_path.get("vision_qpc_path"))

# Lang prefill session: pipelined, KV-share capable
print("\nLoading lang prefill session (cluster_id='prefill') …")
lang_prefill_session = _KVShareSession(
    qpc_path=prefill_qpc_path.get("lang_prefill_qpc_path"),
    full_batch_size=BS,
    cluster_id="prefill",
    stages=STAGES,
)
print("  lang_prefill_session loaded ✓")
print(f"  prefill exec-obj pool size : {lang_prefill_session.prefill_num_execObj}")
print(f"  kv_shape : {lang_prefill_session.kv_shape}  kv_dtype : {lang_prefill_session.kv_size}")

# Lang decode session: single decode exec-obj, KV-share capable
print("\nLoading lang decode session (cluster_id='decode') …")
lang_decode_session = _KVShareSession(
    qpc_path=decode_qpc_path.get("lang_decode_qpc_path"),
    full_batch_size=BS,
    cluster_id="decode",
)
print("  lang_decode_session loaded ✓")

# ─────────────────────────────────────────────────────────────────────────────
# Build inputs
# ─────────────────────────────────────────────────────────────────────────────
if skip_vision:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Tell me about yourself."},
            ],
        },
    ]
else:
    image_url = "https://picsum.photos/id/237/536/354"
    image = Image.open(requests.get(image_url, stream=True).raw)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe all the colors seen in the image."},
            ],
        },
    ]

messages = [messages] * BS

texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=texts,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = qeff_model.model.prepare_inputs_for_generation(inputs=inputs, prefill_seq_len=PREFILL_SEQ_LEN, batch_size=BS)

pad_token_id = 1
input_len = inputs["attention_mask"].sum(1, keepdims=True)
input_ids_length = inputs["input_ids"].shape[1]
num_chunks = -(input_ids_length // -PREFILL_SEQ_LEN)  # ceil divide
padded_len = num_chunks * PREFILL_SEQ_LEN
generation_len = CTX_LEN - int(input_len.max())
print(
    f"\nPrompt tokens = {input_ids_length}  padded = {padded_len}  "
    f"num_chunks = {num_chunks}  generation_len = {generation_len}"
)

inputs["input_ids"] = torch.nn.functional.pad(
    inputs["input_ids"], (0, padded_len - input_ids_length), "constant", pad_token_id
)
inputs["attention_mask"] = torch.nn.functional.pad(
    inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
)

for k, v in inputs.items():
    inputs[k] = np.array(v)

# ─────────────────────────────────────────────────────────────────────────────
# Vision inference (produces vision_embeds + deepstack_features)
# ─────────────────────────────────────────────────────────────────────────────
vision_inputs = {
    k: v
    for k, v in inputs.items()
    if k in {"pixel_values", "image_masks", "image_input_idx", "valid_idx", "aspect_ratio_ids", "aspect_ratio_mask"}
}
vision_inputs_fp16 = {"pixel_values", "image_masks"}
vision_inputs.update({k: vision_inputs[k].astype("float16") for k in vision_inputs_fp16 if k in vision_inputs})

vision_outputs = {}
if vision_inputs and not skip_vision:
    vision_start = perf_counter()
    vision_outputs = vision_session.run(vision_inputs)
    vision_elapsed = perf_counter() - vision_start
    print(f"Vision inference done  time={vision_elapsed * 1000:.1f} ms")

# ─────────────────────────────────────────────────────────────────────────────
# Build lang inputs
# ─────────────────────────────────────────────────────────────────────────────
lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}
if "position_ids" in inputs:
    lang_inputs["position_ids"] = inputs["position_ids"]
    lang_inputs.pop("attention_mask", None)
else:
    lang_inputs["position_ids"] = np.where(lang_inputs.pop("attention_mask"), np.arange(padded_len), -1)

lang_inputs["image_idx"] = np.array([[0]], dtype=np.int64)

if not skip_vision:
    for _slot in range(lang_prefill_session.queue_len):
        lang_prefill_session.set_buffers(
            {
                "vision_embeds": vision_outputs["vision_embeds"],
                "deepstack_features": vision_outputs["deepstack_features"],
            },
            _slot,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Allocate shared KV cache buffers
#
# One numpy array per (layer, key/value) pair.
# lang_prefill_session.kv_only_buff_map contains only past_key.*/past_value.*
# RetainedState entries — for both text-only and VLM QPCs.
# ─────────────────────────────────────────────────────────────────────────────
kv_cache: list[np.ndarray] = [
    np.zeros(lang_prefill_session.kv_shape, dtype=lang_prefill_session.kv_size)
    for _ in lang_prefill_session.kv_only_buff_map  # past_key.*/past_value.* only
]
# Allocate logits_buf
_logits_binding = lang_prefill_session.bindings[lang_prefill_session.binding_index_map["logits"]]
logits_buf = np.zeros(
    list(_logits_binding.dims),
    dtype=_AIC_TO_NP[_logits_binding.type],
)

print(f"\nAllocated {len(kv_cache)} shared KV buffers  shape={lang_prefill_session.kv_shape}")

# ─────────────────────────────────────────────────────────────────────────────
# Chunked lang prefill
#
# Non-last chunks : np_run(is_prefill=True) — KV retained on-device across
#                   chunks; image_idx_output fed back for next chunk.
# Last chunk      : np_run_pipeline(last_chunk=True, kv_cache_buffers=kv_cache)
#                   Wires RetainedState outputs into kv_cache via DMA slice
#                   (zero-copy) before enqueue.
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Chunked lang prefill ──")

# vision_embeds and deepstack_features are registered on qbuffers above;
# exclude them from chunk_inputs so they are never passed through
# get_tuple_list_from_dict() in np_run_pipeline's setData(tuple_list) path.
_vision_keys = {"vision_embeds", "deepstack_features"}
chunk_inputs = {k: v for k, v in lang_inputs.items() if k not in _vision_keys}
chunk_inputs["batch_index"] = np.array([[0]], dtype=np.int64)
prefill_outputs = None

lang_start = perf_counter()

for chunk_idx in range(num_chunks):
    start = chunk_idx * PREFILL_SEQ_LEN
    end = start + PREFILL_SEQ_LEN
    is_last = chunk_idx == num_chunks - 1

    chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, start:end]
    chunk_inputs["position_ids"] = lang_inputs["position_ids"][..., start:end]

    if is_last:
        # Provide logits output buffer so the QPC writes the first token logits
        # into logits_buf in-place
        chunk_inputs["logits"] = logits_buf

    t0 = time.perf_counter()

    if is_last:
        # Last chunk: wire KV RetainedState outputs → kv_cache via DMA slice,
        # then enqueue.  No numpy copy of KV at the prefill→decode boundary.
        exec_idx = lang_prefill_session.np_run_pipeline(
            inputs=chunk_inputs,
            last_chunk=True,
            kv_cache_buffers=kv_cache,
        )
    else:
        # Non-last chunks: KV is retained on-device; no KV output needed here.
        exec_idx = lang_prefill_session.np_run(
            inputs=chunk_inputs,
            is_prefill=True,
        )

    lang_prefill_session.complete_inf(exec_idx, is_prefill=True)
    elapsed = time.perf_counter() - t0

    # Retrieve outputs for non-last chunks to get image_idx_output
    if not is_last:
        prefill_outputs = lang_prefill_session.get_outputs(exec_idx)
        if "image_idx_output" in prefill_outputs:
            chunk_inputs["image_idx"] = prefill_outputs["image_idx_output"]
    else:
        prefill_outputs = lang_prefill_session.get_outputs(exec_idx)

    print(f"  chunk {chunk_idx + 1}/{num_chunks}  last={is_last}  time={elapsed * 1000:.1f} ms")

lang_prefill_elapsed = perf_counter() - lang_start
prefill_time = lang_prefill_elapsed + (vision_elapsed if not skip_vision else 0.0)
print(f"\nPrefill done  lang={lang_prefill_elapsed * 1000:.1f} ms  total={prefill_time * 1000:.1f} ms")

# After the last chunk, logits_buf has been written in-place by the QPC
first_token = int(np.argmax(logits_buf))
print(f"First generated token id = {first_token}")

# ─────────────────────────────────────────────────────────────────────────────
# Build decode inputs
#
# KV slots point directly at kv_cache arrays — never reassigned.
# decode_buff_map is sorted by (layer, key/value) and matches the order in
# which kv_cache was allocated from kv_only_buff_map.
# ─────────────────────────────────────────────────────────────────────────────
next_pos = np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1  # shape [3, BS, 1]

decode_inputs: dict[str, np.ndarray] = {
    "input_ids": np.array([[first_token]], dtype=np.int64),
    "position_ids": next_pos,
    "logits": logits_buf,
}
for (kv_name, _), kv_buf in zip(lang_decode_session.decode_buff_map, kv_cache):
    decode_inputs[kv_name] = kv_buf  # shared array — no copy

all_tokens = [first_token]


# ─────────────────────────────────────────────────────────────────────────────
# Decode helper
#
# 1. Wire RetainedState OUTPUT slots → kv_cache via setDataWithSlices so the
#    runtime DMA-writes updated KV directly into kv_cache after inference.
# 2. np_run + complete_inf + get_outputs.
# ─────────────────────────────────────────────────────────────────────────────
def run_decode_step() -> dict[str, np.ndarray]:
    lang_decode_session.set_data_for_kv_handoff(
        kv_cache,
        [("batch_index", 0), ("ctx_start", 0)],
        lang_decode_session.decode_execObj_idx,
        lang_decode_session.decode_rs_kv_only_buff_map,  # past_key.*/past_value.* only, safe for VLMs
    )
    exec_idx = lang_decode_session.np_run(decode_inputs, is_prefill=False)
    lang_decode_session.complete_inf(exec_idx, is_prefill=False)
    return lang_decode_session.get_outputs(exec_idx)


# ─────────────────────────────────────────────────────────────────────────────
# First decode step (prefill→decode KV handoff)
# ─────────────────────────────────────────────────────────────────────────────
t0 = perf_counter()
dec_outputs = run_decode_step()
print(f"\nFirst decode step  time={(perf_counter() - t0) * 1000:.1f} ms")

next_token = int(np.argmax(dec_outputs["logits"]))
all_tokens.append(next_token)
next_pos = next_pos + 1  # shape [3, BS, 1] — broadcast increment

# ─────────────────────────────────────────────────────────────────────────────
# Decode loop
# ─────────────────────────────────────────────────────────────────────────────
t_loop_start = perf_counter()

for step in range(generation_len - 2):
    decode_inputs["input_ids"] = np.array([[next_token]], dtype=np.int64)
    decode_inputs["position_ids"] = next_pos  # shape [3, BS, 1]

    dec_outputs = run_decode_step()

    next_token = int(np.argmax(dec_outputs["logits"]))
    all_tokens.append(next_token)
    next_pos = next_pos + 1

t_loop_end = perf_counter()

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
decode_steps = generation_len - 2
tok_per_sec = decode_steps / (t_loop_end - t_loop_start) if decode_steps > 0 else 0.0

print(f"\ndecode tok/sec = {tok_per_sec:.2f}  ({decode_steps} steps)")
print(f"\ninput\n{texts[0]}\noutput\n{tokenizer.decode(all_tokens)}")

lang_prefill_session.deactivate()
lang_decode_session.deactivate()
if not skip_vision:
    vision_session.deactivate()
