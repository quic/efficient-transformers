# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Disaggregated serving for Qwen3.5-MoE with zero-copy DMA KV-slice handoff.

Qwen3.5-MoE is a *hybrid* model: each transformer layer is either
``full_attention`` (standard KV-cache) or ``linear_attention`` (Mamba-style
conv_state + recurrent_state).  The two state types require different handoff
strategies:

full_attention layers  →  past_key.N / past_value.N
    Handled by ``setDataWithSlices`` (zero-copy DMA slice).
    Allocated once in ``kv_cache[b]``; written by prefill via DMA, read by
    decode as direct numpy inputs, updated in-place each decode step via
    ``set_data_for_kv_handoff``.

linear_attention layers  →  conv_state.N / recurrent_state.N
    Small fixed-size states; NOT covered by ``kv_only_buff_map``.
    ``conv_state.*`` inputs are NOT auto-skipped by the session (only
    ``past_*`` inputs are skipped).  They must be provided on every prefill
    chunk (zero-initialised for non-last chunks; the QPC updates them
    on-device via retained state across chunks).
    After the last prefill chunk, ``conv_state.N_RetainedState`` and
    ``recurrent_state.N_RetainedState`` outputs are read and stored in
    ``recurrent_cache[b]``; fed as decode inputs each step and updated
    in-place from decode outputs.

Session layout
--------------
  vision_session       : standard QAICInferenceSession (cloud_infer)
                         Runs once per request; produces vision_embeds.

  lang_prefill_session : QAICInferenceSession (cloud_infer_KV_share)
      cluster_id="prefill", stages=STAGES
      exec-obj pool: [slot-0 .. slot-STAGES]  (prefill only)

  lang_decode_session  : QAICInferenceSession (cloud_infer_KV_share)
      cluster_id="decode"
      exec-obj pool: [slot-0]  (decode only)

Shared KV buffers (full_attention layers only)
----------------------------------------------
  kv_cache[b]  : list[np.ndarray]
      One array per (layer, key/value) pair from ``kv_only_buff_map``.
      shape = lang_prefill_session.kv_shape  (kv_shape[0]=1, per-slot)
      dtype = lang_prefill_session.kv_size
      Written by prefill via DMA slice (zero-copy), read by decode as
      direct inputs, updated in-place each decode step via
      ``set_data_for_kv_handoff``.

Recurrent state buffers (linear_attention layers)
-------------------------------------------------
  recurrent_cache[b]  : dict[str, np.ndarray]
      Keys: conv_state.N, recurrent_state.N  (for each linear_attention layer N)
      Populated by copying ``conv_state.N_RetainedState`` /
      ``recurrent_state.N_RetainedState`` outputs after the last prefill chunk.
      Fed as decode inputs each step; updated in-place from decode outputs.
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
MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
PREFILL_SEQ_LEN = 64
CTX_LEN = 4096
LAYERWISE = False
LAYERWISE_WINDOW_SIZE = 1
BS = 1
STAGES = 4  # pipeline-parallel prefill stages; set >1 for pipelined PP prefill

PREFILL_NUM_DEVICES = 8
DECODE_NUM_DEVICES = 4


skip_vision = True
generation_len = 100

# ─────────────────────────────────────────────────────────────────────────────
# Model config — build layer_types list for the hybrid model
# ─────────────────────────────────────────────────────────────────────────────
config = AutoConfig.from_pretrained(MODEL_ID)
config.torch_dtype = "float16"
# For faster execution user can run with lesser layers, For Testing Purpose Only
# config.vision_config.depth = 5
# config.text_config.num_hidden_layers = 4
# Ensure layer_types covers all hidden layers (pad with full_attention if short)
layer_types = list(getattr(config.text_config, "layer_types", []))
if len(layer_types) < config.text_config.num_hidden_layers:
    layer_types.extend(["full_attention"] * (config.text_config.num_hidden_layers - len(layer_types)))
config.text_config.layer_types = layer_types[: config.text_config.num_hidden_layers]

_FULL_ATTN_LAYERS = [i for i, lt in enumerate(config.text_config.layer_types) if lt == "full_attention"]
_LINEAR_ATTN_LAYERS = [i for i, lt in enumerate(config.text_config.layer_types) if lt != "full_attention"]

print(f"Model: {MODEL_ID}")
print(f"  num_hidden_layers : {config.text_config.num_hidden_layers}")
print(f"  full_attention    : {len(_FULL_ATTN_LAYERS)} layers")
print(f"  linear_attention  : {len(_LINEAR_ATTN_LAYERS)} layers")

# ─────────────────────────────────────────────────────────────────────────────
# Build model + processor
# ─────────────────────────────────────────────────────────────────────────────
qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    attn_implementation="eager",
    kv_offload=True,
    config=config,
    dtype=torch.float16,
    layerwise=LAYERWISE,
)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# ─────────────────────────────────────────────────────────────────────────────
# Compile QPCs
# ─────────────────────────────────────────────────────────────────────────────

# Vision encoder (optional; skip_vision=True skips this)
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
        mxfp6_matmul=False,
        aic_enable_depth_first=True,
        skip_vision=False,
        split_model_io=True,
        skip_lang=True,
        use_onnx_subfunctions=True,
        layerwise=LAYERWISE,
        layerwise_window_size=1,
    )

# Lang prefill QPC
prefill_qpc_path = qeff_model.compile(
    batch_size=BS,
    prefill_seq_len=PREFILL_SEQ_LEN,
    ctx_len=CTX_LEN,
    height=354,
    width=536,
    num_cores=16,
    num_devices=PREFILL_NUM_DEVICES,
    mxfp6_matmul=False,
    mxint8_kv_cache=False,
    retain_full_kv=True,
    split_retained_state_io=True,
    split_model_io=True,
    mos=1,
    user_tiled=True,
    aic_enable_depth_first=False,
    prefill_only=True,
    enable_chunking=True,
    skip_vision=True,
    use_onnx_subfunctions=True,
    layerwise=LAYERWISE,
    layerwise_window_size=LAYERWISE_WINDOW_SIZE,
    mdp_num_partitions=STAGES,
    mdp_strategy="onnx",
    offload_pt_weights=False,
)

# Lang decode QPC
decode_qpc_path = qeff_model.compile(
    batch_size=BS,
    prefill_seq_len=1,
    ctx_len=CTX_LEN,
    height=354,
    width=536,
    num_cores=16,
    num_devices=DECODE_NUM_DEVICES,
    mxfp6_matmul=False,
    mxint8_kv_cache=False,
    split_retained_state_io=True,
    split_model_io=True,
    mos=1,
    aic_enable_depth_first=True,
    prefill_only=False,
    skip_vision=True,
    use_onnx_subfunctions=True,
    layerwise=LAYERWISE,
    layerwise_window_size=LAYERWISE_WINDOW_SIZE,
    offload_pt_weights=False,
)

# ─────────────────────────────────────────────────────────────────────────────
# Load sessions
# ─────────────────────────────────────────────────────────────────────────────
if not skip_vision:
    vision_session = _StandardSession(vision_qpc_path.get("vision_qpc_path"))

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

print("\nLoading lang decode session (cluster_id='decode') …")
lang_decode_session = _KVShareSession(
    qpc_path=decode_qpc_path.get("lang_decode_qpc_path"),
    full_batch_size=BS,
    cluster_id="decode",
)
print("  lang_decode_session loaded ✓")

# ─────────────────────────────────────────────────────────────────────────────
# Session sanity checks
# ─────────────────────────────────────────────────────────────────────────────
assert any("_RetainedState" in n for n in lang_prefill_session.output_names), (
    "split_retained_state_io=True not effective on prefill QPC — no RetainedState outputs found.\n"
    "Recompile with split_retained_state_io=True."
)
assert any(n.startswith("past_key") for n in lang_decode_session.input_names), (
    "Decode QPC missing past_key.* inputs — check retain_full_kv=True and split_retained_state_io=True."
)
assert any("_RetainedState" in n for n in lang_decode_session.output_names), (
    "Decode QPC missing RetainedState outputs — check split_retained_state_io=True."
)
assert lang_prefill_session.kv_shape[0] == 1, (
    f"kv_shape[0] should be 1 (per-batch-slot), got {lang_prefill_session.kv_shape[0]}"
)

# Identify conv_state / recurrent_state RetainedState output names from prefill session
# These are the linear_attention layer states that need explicit numpy copy handoff
_PREFILL_CONV_RS_NAMES = [
    n for n in lang_prefill_session.output_names if n.startswith("conv_state.") and n.endswith("_RetainedState")
]
_PREFILL_RECURRENT_RS_NAMES = [
    n for n in lang_prefill_session.output_names if n.startswith("recurrent_state.") and n.endswith("_RetainedState")
]
# Corresponding decode INPUT names (strip _RetainedState suffix)
_DECODE_CONV_INPUT_NAMES = [n.replace("_RetainedState", "") for n in _PREFILL_CONV_RS_NAMES]
_DECODE_RECURRENT_INPUT_NAMES = [n.replace("_RetainedState", "") for n in _PREFILL_RECURRENT_RS_NAMES]

# Decode RetainedState OUTPUT names for conv/recurrent (to update recurrent_cache each step)
_DECODE_CONV_RS_NAMES = [
    n for n in lang_decode_session.output_names if n.startswith("conv_state.") and n.endswith("_RetainedState")
]
_DECODE_RECURRENT_RS_NAMES = [
    n for n in lang_decode_session.output_names if n.startswith("recurrent_state.") and n.endswith("_RetainedState")
]

print(f"\n  conv_state  RetainedState outputs (prefill) : {len(_PREFILL_CONV_RS_NAMES)}")
print(f"  recurrent_state RetainedState outputs (prefill) : {len(_PREFILL_RECURRENT_RS_NAMES)}")

# ─────────────────────────────────────────────────────────────────────────────
# Build inputs
# ─────────────────────────────────────────────────────────────────────────────
if skip_vision:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Tell me about yourself."}],
        }
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
        }
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
generation_len = min(generation_len, CTX_LEN - int(input_len.max()))

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
# Vision inference (optional)
# ─────────────────────────────────────────────────────────────────────────────
vision_inputs = {
    k: v
    for k, v in inputs.items()
    if k in {"pixel_values", "image_masks", "image_input_idx", "valid_idx", "aspect_ratio_ids", "aspect_ratio_mask"}
}
vision_inputs.update(
    {k: vision_inputs[k].astype("float16") for k in {"pixel_values", "image_masks"} if k in vision_inputs}
)

vision_outputs: dict[str, np.ndarray] = {}
vision_elapsed = 0.0
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

# Pre-register vision outputs on all prefill exec-obj slots (VLM path)
if not skip_vision and vision_outputs:
    for _slot in range(lang_prefill_session.queue_len):
        lang_prefill_session.set_buffers(
            {
                "vision_embeds": vision_outputs["vision_embeds"],
            },
            _slot,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Allocate shared KV cache buffers (full_attention layers — zero-copy DMA)
#
# kv_cache[b]  : one list per batch slot.
# Each inner list has one np.ndarray per (layer, key/value) pair from
# kv_only_buff_map (past_key.*/past_value.* RetainedState outputs only).
# kv_shape[0] = 1 (per-batch-slot shape set by the session at init).
# ─────────────────────────────────────────────────────────────────────────────
kv_cache: list[list[np.ndarray]] = [
    [
        np.zeros(lang_prefill_session.kv_shape, dtype=lang_prefill_session.kv_size)
        for _ in lang_prefill_session.kv_only_buff_map  # past_key.*/past_value.* only
    ]
    for _ in range(BS)
]

# Logits output buffer per batch slot (written in-place by the QPC on last chunk)
_logits_binding = lang_prefill_session.bindings[lang_prefill_session.binding_index_map["logits"]]
logits_bufs: list[np.ndarray] = [
    np.zeros(list(_logits_binding.dims), dtype=_AIC_TO_NP[_logits_binding.type]) for _ in range(BS)
]

print(f"\nAllocated {BS} x {len(kv_cache[0])} shared KV buffers  shape={lang_prefill_session.kv_shape}")


# ─────────────────────────────────────────────────────────────────────────────
# Allocate recurrent state cache (linear_attention layers — numpy copy handoff)
#
# recurrent_cache[b]  : dict[str, np.ndarray]
#   Keys: conv_state.N, recurrent_state.N  for each linear_attention layer N.
#   Populated after the last prefill chunk by copying RetainedState outputs.
#   Fed as decode inputs each step; updated in-place from decode outputs.
#
# We pre-allocate zero arrays using the binding shapes from the prefill session.
# ─────────────────────────────────────────────────────────────────────────────
def _alloc_recurrent_cache_slot() -> dict[str, np.ndarray]:
    """Allocate zero-filled recurrent state arrays for one batch slot."""
    slot: dict[str, np.ndarray] = {}
    for rs_name in _PREFILL_CONV_RS_NAMES + _PREFILL_RECURRENT_RS_NAMES:
        if rs_name not in lang_prefill_session.binding_index_map:
            continue
        binding = lang_prefill_session.bindings[lang_prefill_session.binding_index_map[rs_name]]
        dtype = _AIC_TO_NP[binding.type]
        # Key is the input name (strip _RetainedState suffix)
        input_name = rs_name.replace("_RetainedState", "")
        slot[input_name] = np.zeros(list(binding.dims), dtype=dtype)
    return slot


recurrent_cache: list[dict[str, np.ndarray]] = [_alloc_recurrent_cache_slot() for _ in range(BS)]
print(f"Allocated {BS} x {len(recurrent_cache[0])} recurrent state buffers (conv+recurrent)")

# ─────────────────────────────────────────────────────────────────────────────
# Chunked lang prefill
#
# For each batch slot b in [0, BS):
#
#   Non-last chunks:
#     - conv_state.N / recurrent_state.N inputs: zero-initialised (on-device
#       retained state carries them across chunks automatically).
#     - np_run(is_prefill=True) — KV retained on-device; no KV output needed.
#
#   Last chunk:
#     - np_run_pipeline(last_chunk=True, kv_cache_buffers=kv_cache[b])
#       Wires past_key.*/past_value.* RetainedState outputs into kv_cache[b]
#       via DMA slice (batch_index=b, zero-copy) before enqueue.
#     - After complete_inf, read conv_state.N_RetainedState /
#       recurrent_state.N_RetainedState outputs and store in recurrent_cache[b].
# ─────────────────────────────────────────────────────────────────────────────
_VISION_KEYS = {"vision_embeds", "deepstack_features"}

print("\n── Chunked lang prefill ──")
lang_start = perf_counter()

# Per-slot storage for last-chunk outputs (needed for decode input building)
_slot_last_out: list[dict] = [{} for _ in range(BS)]
_slot_image_idx_out: list[np.ndarray] = [np.array([[0]], dtype=np.int64) for _ in range(BS)]

for batch_idx in range(BS):
    chunk_inputs: dict[str, np.ndarray] = {k: v for k, v in lang_inputs.items() if k not in _VISION_KEYS}
    chunk_inputs["batch_index"] = np.array([[batch_idx]], dtype=np.int64)

    # Initialise conv_state / recurrent_state inputs to zero for the first chunk.
    # The QPC carries them on-device via retained state across non-last chunks.
    for input_name, arr in recurrent_cache[batch_idx].items():
        chunk_inputs[input_name] = np.zeros_like(arr)

    print(f"\n  [slot {batch_idx}] prefill …")
    for chunk_idx in range(num_chunks):
        start = chunk_idx * PREFILL_SEQ_LEN
        end = start + PREFILL_SEQ_LEN
        is_last = chunk_idx == num_chunks - 1

        chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, start:end]
        chunk_inputs["position_ids"] = lang_inputs["position_ids"][..., start:end]

        if is_last:
            # Provide per-slot logits buffer so the QPC writes the first-token
            # logits for this slot in-place.
            chunk_inputs["logits"] = logits_bufs[batch_idx]
            # Register conv/recurrent RS output buffers on the execObj via tuple_list.
            # setData(tuple_list) used by np_run_pipeline bypasses qbuffers[], so
            # unskip_buffers cannot work here.  Passing RS output names with their
            # pre-allocated recurrent_cache arrays registers them as write destinations
            # on the execObj — identical to how logits_bufs[b] is registered above.
            for rs_name in _PREFILL_CONV_RS_NAMES + _PREFILL_RECURRENT_RS_NAMES:
                chunk_inputs[rs_name] = recurrent_cache[batch_idx][rs_name.replace("_RetainedState", "")]

        t0 = time.perf_counter()

        if is_last:
            # Last chunk: wire past_key.*/past_value.* RetainedState outputs →
            # kv_cache[batch_idx] via DMA slice (batch_index=batch_idx, zero-copy).
            # np_run_pipeline uses setData(tuple_list) — overload 4 — so the
            # DMA wiring on KV slots is preserved.
            exec_idx = lang_prefill_session.np_run_pipeline(
                inputs=chunk_inputs,
                last_chunk=True,
                kv_cache_buffers=kv_cache[batch_idx],
            )
        else:
            # Non-last chunks: KV retained on-device; no KV output needed.
            exec_idx = lang_prefill_session.np_run(
                inputs=chunk_inputs,
                is_prefill=True,
            )

        lang_prefill_session.complete_inf(exec_idx, is_prefill=True)
        elapsed = time.perf_counter() - t0

        # Feed image_idx_output back for the next chunk (VLM path)
        if not is_last:
            _chunk_out = lang_prefill_session.get_outputs(exec_idx)
            if "image_idx_output" in _chunk_out:
                chunk_inputs["image_idx"] = _chunk_out["image_idx_output"]
        else:
            # Last chunk: read conv_state / recurrent_state RetainedState outputs
            # and store in recurrent_cache[batch_idx] for decode.
            _last_out = lang_prefill_session.get_outputs(exec_idx)
            _slot_last_out[batch_idx] = _last_out

            for rs_name in _PREFILL_CONV_RS_NAMES + _PREFILL_RECURRENT_RS_NAMES:
                if rs_name in _last_out:
                    input_name = rs_name.replace("_RetainedState", "")
                    np.copyto(recurrent_cache[batch_idx][input_name], _last_out[rs_name])

            # Also capture image_idx_output for decode (VLM path)
            _slot_image_idx_out[batch_idx] = _last_out.get(
                "image_idx_output", chunk_inputs.get("image_idx", np.array([[0]], dtype=np.int64))
            )

        print(f"    chunk {chunk_idx + 1}/{num_chunks}  last={is_last}  time={elapsed * 1000:.1f} ms")

lang_prefill_elapsed = perf_counter() - lang_start
prefill_time = lang_prefill_elapsed + vision_elapsed
print(f"\nPrefill done  lang={lang_prefill_elapsed * 1000:.1f} ms  total={prefill_time * 1000:.1f} ms")

# ─────────────────────────────────────────────────────────────────────────────
# Verify KV handoff correctness
# ─────────────────────────────────────────────────────────────────────────────
for b in range(BS):
    for i, arr in enumerate(kv_cache[b]):
        if not np.any(arr != 0):
            print(
                f"  WARNING: kv_cache[{b}][{i}] is all zeros after prefill — "
                "DMA KV handoff may have failed. Check split_retained_state_io=True on prefill QPC."
            )
            break

# Extract first generated token per slot from in-place logits buffers
first_tokens: list[int] = [int(np.argmax(logits_bufs[b])) for b in range(BS)]
print(f"First generated token ids = {first_tokens}")

# ─────────────────────────────────────────────────────────────────────────────
# Build per-slot decode inputs
#
# full_attention KV:
#   kv_cache[b][i] arrays are wired directly as decode inputs via
#   decode_buff_map — no copy.  decode_buff_map is sorted by (layer, key/value)
#   and matches the order in which kv_cache[b] was allocated from kv_only_buff_map.
#
# linear_attention states:
#   recurrent_cache[b] arrays are wired directly as decode inputs — no copy.
#   Updated in-place from decode outputs each step.
#
# position_ids: max valid position from prefill + 1.
# ─────────────────────────────────────────────────────────────────────────────
next_positions: list[np.ndarray] = [
    np.max(lang_inputs["position_ids"][:, b : b + 1, :], axis=-1, keepdims=True) + 1
    if lang_inputs["position_ids"].ndim == 3
    else np.max(lang_inputs["position_ids"][b : b + 1, :], axis=-1, keepdims=True) + 1
    for b in range(BS)
]

slot_decode_inputs: list[dict[str, np.ndarray]] = []
for b in range(BS):
    d: dict[str, np.ndarray] = {
        "input_ids": np.array([[first_tokens[b]]], dtype=np.int64),
        "position_ids": next_positions[b],
        "logits": logits_bufs[b],
        "image_idx": _slot_image_idx_out[b] if not skip_vision else np.array([[0]], dtype=np.int64),
    }
    # Wire full_attention KV arrays — no copy
    for (kv_name, _), kv_buf in zip(lang_decode_session.decode_buff_map, kv_cache[b]):
        d[kv_name] = kv_buf
    # Wire linear_attention recurrent state arrays — no copy
    for input_name, arr in recurrent_cache[b].items():
        d[input_name] = arr
    # Register conv/recurrent RS output buffers on the decode execObj via tuple_list.
    # Same mechanism as logits and prefill: passing RS output names with their
    # recurrent_cache arrays registers them as write destinations so the QPC writes
    # updated state in-place each decode step.
    for rs_name in _DECODE_CONV_RS_NAMES + _DECODE_RECURRENT_RS_NAMES:
        d[rs_name] = recurrent_cache[b][rs_name.replace("_RetainedState", "")]
    # VLM: wire vision_embeds RetainedState if present
    if not skip_vision and "vision_embeds_RetainedState" in _slot_last_out[b]:
        d["vision_embeds"] = _slot_last_out[b]["vision_embeds_RetainedState"]
    slot_decode_inputs.append(d)

# Per-slot generated token lists
all_tokens: list[list[int]] = [[first_tokens[b]] for b in range(BS)]


# ─────────────────────────────────────────────────────────────────────────────
# Decode helper
#
# full_attention KV (zero-copy DMA):
#   1. Wire past_key.*/past_value.* RetainedState OUTPUT slots → kv_cache[b]
#      via set_data_for_kv_handoff (setDataWithSlices).
#      Runtime DMA-writes updated KV directly into kv_cache[b] after inference.
#   2. np_run(is_prefill=False) + complete_inf + get_outputs.
#
# linear_attention states (numpy copy):
#   3. Read conv_state.N_RetainedState / recurrent_state.N_RetainedState from
#      decode outputs and update recurrent_cache[b] in-place.
#      slot_decode_inputs[b][input_name] already points at recurrent_cache[b][input_name]
#      so the next step automatically uses the updated values.
# ─────────────────────────────────────────────────────────────────────────────
def run_decode_step(batch_idx: int) -> dict[str, np.ndarray]:
    # ── full_attention KV: wire RetainedState OUTPUT → kv_cache via DMA slice ──
    lang_decode_session.set_data_for_kv_handoff(
        kv_cache[batch_idx],
        [("batch_index", batch_idx % BS), ("ctx_start", 0)],
        lang_decode_session.decode_execObj_idx,
        lang_decode_session.decode_rs_kv_only_buff_map,  # past_key.*/past_value.* only, VLM-safe
    )

    exec_idx = lang_decode_session.np_run(slot_decode_inputs[batch_idx], is_prefill=False)
    lang_decode_session.complete_inf(exec_idx, is_prefill=False)
    outputs = lang_decode_session.get_outputs(exec_idx)

    # ── linear_attention states: update recurrent_cache in-place ──
    for rs_name in _DECODE_CONV_RS_NAMES + _DECODE_RECURRENT_RS_NAMES:
        if rs_name in outputs:
            input_name = rs_name.replace("_RetainedState", "")
            if input_name in recurrent_cache[batch_idx]:
                np.copyto(recurrent_cache[batch_idx][input_name], outputs[rs_name])
                # slot_decode_inputs[batch_idx][input_name] already points at
                # recurrent_cache[batch_idx][input_name] — no reassignment needed

    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# First decode step for all slots (prefill → decode KV handoff verification)
# ─────────────────────────────────────────────────────────────────────────────
t0 = perf_counter()
for b in range(BS):
    dec_outputs = run_decode_step(b)
    next_token = int(np.argmax(dec_outputs["logits"]))
    all_tokens[b].append(next_token)
    slot_decode_inputs[b]["input_ids"] = np.array([[next_token]], dtype=np.int64)
    next_positions[b] = next_positions[b] + 1
    slot_decode_inputs[b]["position_ids"] = next_positions[b]
    if not skip_vision and "image_idx_output" in dec_outputs:
        slot_decode_inputs[b]["image_idx"] = dec_outputs["image_idx_output"]
print(f"\nFirst decode step (all slots)  time={(perf_counter() - t0) * 1000:.1f} ms")

# ─────────────────────────────────────────────────────────────────────────────
# Decode loop — all BS slots decoded each step
# ─────────────────────────────────────────────────────────────────────────────
t_loop_start = perf_counter()

for step in range(generation_len - 2):
    for b in range(BS):
        dec_outputs = run_decode_step(b)
        next_token = int(np.argmax(dec_outputs["logits"]))
        all_tokens[b].append(next_token)
        slot_decode_inputs[b]["input_ids"] = np.array([[next_token]], dtype=np.int64)
        next_positions[b] = next_positions[b] + 1
        slot_decode_inputs[b]["position_ids"] = next_positions[b]
        if not skip_vision and "image_idx_output" in dec_outputs:
            slot_decode_inputs[b]["image_idx"] = dec_outputs["image_idx_output"]

t_loop_end = perf_counter()

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
decode_steps = generation_len - 2
tok_per_sec = (decode_steps * BS) / (t_loop_end - t_loop_start) if decode_steps > 0 else 0.0

print(f"\ndecode tok/sec = {tok_per_sec:.2f}  ({decode_steps} steps x {BS} slots)")
for b in range(BS):
    print(f"\n[slot {b}] input\n{texts[b]}\noutput\n{tokenizer.decode(all_tokens[b])}")

lang_prefill_session.deactivate()
lang_decode_session.deactivate()
if not skip_vision:
    vision_session.deactivate()
