# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Unit tests for disaggregated serving with DMA KV-slice handoff
(cloud_infer_KV_share.QAICInferenceSession).

Tests validate that the zero-copy prefill->decode KV handoff via
setDataWithSlices produces token outputs that match a reference file.

Reference token IDs are stored in:
    tests/transformers/disaggregated/kv_share_references.json

Format:
    {
        "<model_id>|<prompt_key>": [token_id, token_id, ...]
    }

All tests are marked @pytest.mark.on_qaic and require QAIC hardware.
"""

import json
import math
import os

import numpy as np
import pytest
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer_KV_share import QAICInferenceSession

# ---------------------------------------------------------------------------
# Reference file
# ---------------------------------------------------------------------------
_REF_FILE = os.path.join(os.path.dirname(__file__), "kv_share_references.json")


def _load_refs() -> dict:
    if not os.path.exists(_REF_FILE):
        return {}
    with open(_REF_FILE) as f:
        return json.load(f)


_REFS = _load_refs()

# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------
# Each entry: (model_id, dummy_config_overrides, tokenizer_id, prompt_key, prompt)
# dummy_config_overrides shrink the model so it compiles on a single device
# without downloading real weights.
_KV_SHARE_CASES = [
    (
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        {
            "model_type": "qwen3_moe",
            "hidden_size": 256,
            "intermediate_size": 256,
            "max_position_embeddings": 512,
            "max_window_layers": 48,
            "moe_intermediate_size": 768,
            "num_attention_heads": 2,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "vocab_size": 151936,
        },
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "short",
        "Once upon a time",
    ),
    (
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        {
            "model_type": "qwen3_moe",
            "hidden_size": 256,
            "intermediate_size": 256,
            "max_position_embeddings": 512,
            "max_window_layers": 48,
            "moe_intermediate_size": 768,
            "num_attention_heads": 2,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "vocab_size": 151936,
        },
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "long",
        (
            "Once upon a time, in a small town, there lived a young boy named Alex. "
            "Alex was a curious and adventurous child, always eager to explore the world around him. "
            "One day, while playing in the park, Alex stumbled upon a mysterious old book."
        ),
    ),
]

_PREFILL_SEQ_LEN = 128
_CTX_LEN = 256
_GENERATION_LEN = 10
_STAGES = 1


def _ref_key(model_id: str, prompt_key: str) -> str:
    return f"{model_id}|{prompt_key}"


def _build_dummy_model(model_id: str, overrides: dict) -> QEFFAutoModelForCausalLM:
    """Build a tiny QEff model from config overrides — no weight download."""
    import torch
    from transformers import AutoModelForCausalLM

    config = AutoConfig.for_model(overrides["model_type"], **{k: v for k, v in overrides.items() if k != "model_type"})
    torch.manual_seed(42)
    hf_model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    with torch.no_grad():
        for param in hf_model.parameters():
            param.mul_(0.02)
    return QEFFAutoModelForCausalLM(hf_model)


def _run_kv_share_disagg(
    model_id: str,
    overrides: dict,
    tokenizer_id: str,
    prompt: str,
    generation_len: int = _GENERATION_LEN,
) -> list[int]:
    """
    Full disaggregated prefill+decode pipeline using _KVShareSession.
    Returns list of generated token IDs (length = generation_len).
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Tokenise ──────────────────────────────────────────────────────────
    raw = tokenizer(prompt, return_tensors="np", padding=True)
    input_ids_length = raw["input_ids"].shape[1]
    num_chunks = math.ceil(input_ids_length / _PREFILL_SEQ_LEN)
    padded_len = num_chunks * _PREFILL_SEQ_LEN

    raw = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    raw["position_ids"] = np.where(raw.pop("attention_mask"), np.arange(padded_len), -1)
    raw.pop("token_type_ids", None)

    # ── Build + compile ───────────────────────────────────────────────────
    qeff_model = _build_dummy_model(model_id, overrides)
    config = qeff_model.model.config

    prefill_qpc_path = qeff_model.compile(
        prefill_seq_len=_PREFILL_SEQ_LEN,
        ctx_len=_CTX_LEN,
        num_cores=getattr(config, "num_experts", 16),
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        num_devices=1,
        mos=1,
        aic_enable_depth_first=True,
        prefill_only=True,
        enable_chunking=True,
        split_retained_state_io=True,
        stages=_STAGES,
        offload_pt_weights=False,
    )

    decode_qpc_path = qeff_model.compile(
        prefill_seq_len=1,
        ctx_len=_CTX_LEN,
        num_cores=getattr(config, "num_experts", 16),
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        num_devices=1,
        mos=1,
        aic_enable_depth_first=True,
        prefill_only=False,
        split_retained_state_io=True,
        retain_full_kv=True,
    )

    # ── Load KVShare sessions ─────────────────────────────────────────────
    prefill_session = QAICInferenceSession(
        qpc_path=prefill_qpc_path,
        full_batch_size=1,
        device_ids=[0],
        cluster_id="prefill",
        stages=_STAGES,
    )
    decode_session = QAICInferenceSession(
        qpc_path=decode_qpc_path,
        full_batch_size=1,
        device_ids=[0],
        cluster_id="decode",
    )

    # ── Allocate shared KV cache ──────────────────────────────────────────
    kv_cache: list[np.ndarray] = [
        np.zeros(prefill_session.kv_shape, dtype=prefill_session.kv_size) for _ in prefill_session.kv_only_buff_map
    ]

    _logits_b = prefill_session.bindings[prefill_session.binding_index_map["logits"]]
    from QEfficient.generation.cloud_infer_KV_share import AIC_TO_NP

    logits_buf = np.zeros(list(_logits_b.dims), dtype=AIC_TO_NP[_logits_b.type])

    # ── Chunked prefill ───────────────────────────────────────────────────
    chunk_inputs = {k: v for k, v in raw.items()}
    chunk_inputs["batch_index"] = np.array([[0]], dtype=np.int64)

    for chunk_idx in range(num_chunks):
        start = chunk_idx * _PREFILL_SEQ_LEN
        end = start + _PREFILL_SEQ_LEN
        is_last = chunk_idx == num_chunks - 1

        chunk_inputs["input_ids"] = raw["input_ids"][:, start:end]
        chunk_inputs["position_ids"] = raw["position_ids"][:, start:end]

        if is_last:
            chunk_inputs["logits"] = logits_buf
            exec_idx = prefill_session.np_run_pipeline(
                inputs=chunk_inputs,
                last_chunk=True,
                kv_cache_buffers=kv_cache,
            )
        else:
            exec_idx = prefill_session.np_run(inputs=chunk_inputs, is_prefill=True)

        prefill_session.complete_inf(exec_idx, is_prefill=True)

    prefill_session.deactivate()

    # ── Build decode inputs ───────────────────────────────────────────────
    first_token = int(np.argmax(logits_buf))
    next_pos = np.max(raw["position_ids"], axis=-1, keepdims=True) + 1  # [1, 1]

    decode_inputs: dict[str, np.ndarray] = {
        "input_ids": np.array([[first_token]], dtype=np.int64),
        "position_ids": next_pos,
        "logits": logits_buf,
    }
    for (kv_name, _), kv_buf in zip(decode_session.decode_buff_map, kv_cache):
        decode_inputs[kv_name] = kv_buf

    all_tokens = [first_token]

    def _decode_step() -> dict:
        decode_session.set_data_for_kv_handoff(
            kv_cache,
            [("batch_index", 0), ("ctx_start", 0)],
            decode_session.decode_execObj_idx,
            decode_session.decode_rs_kv_only_buff_map,
        )
        idx = decode_session.np_run(decode_inputs, is_prefill=False)
        decode_session.complete_inf(idx, is_prefill=False)
        return decode_session.get_outputs(idx)

    # First decode step
    dec_out = _decode_step()
    next_token = int(np.argmax(dec_out["logits"]))
    all_tokens.append(next_token)
    decode_inputs["input_ids"] = np.array([[next_token]], dtype=np.int64)
    next_pos = next_pos + 1
    decode_inputs["position_ids"] = next_pos

    # Decode loop
    for _ in range(generation_len - 2):
        dec_out = _decode_step()
        next_token = int(np.argmax(dec_out["logits"]))
        all_tokens.append(next_token)
        decode_inputs["input_ids"] = np.array([[next_token]], dtype=np.int64)
        next_pos = next_pos + 1
        decode_inputs["position_ids"] = next_pos

    decode_session.deactivate()
    return all_tokens


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.on_qaic
@pytest.mark.parametrize(
    "model_id,overrides,tokenizer_id,prompt_key,prompt",
    _KV_SHARE_CASES,
    ids=[f"{c[0].split('/')[1]}_{c[3]}" for c in _KV_SHARE_CASES],
)
def test_kv_share_disagg_token_match(model_id, overrides, tokenizer_id, prompt_key, prompt):
    """
    Run the full KV-slice disaggregated pipeline and assert generated tokens
    match the reference stored in kv_share_references.json.

    If no reference exists for this (model_id, prompt_key) pair the test is
    skipped with a message instructing the user to generate references first.
    """
    ref_key = _ref_key(model_id, prompt_key)
    if ref_key not in _REFS:
        pytest.skip(
            f"No reference found for key '{ref_key}'. Run generate_kv_share_references.py to populate {_REF_FILE}."
        )

    ref_tokens = _REFS[ref_key]
    gen_tokens = _run_kv_share_disagg(model_id, overrides, tokenizer_id, prompt, generation_len=len(ref_tokens))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    ref_text = tokenizer.decode(ref_tokens, skip_special_tokens=True)
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    assert gen_tokens == ref_tokens, (
        f"Token mismatch for model={model_id} prompt={prompt_key!r}\n"
        f"  ref ({len(ref_tokens)} tokens): {ref_tokens}\n"
        f"  got ({len(gen_tokens)} tokens): {gen_tokens}\n"
        f"  ref text: {ref_text!r}\n"
        f"  got text: {gen_text!r}"
    )


@pytest.mark.on_qaic
@pytest.mark.parametrize(
    "model_id,overrides,tokenizer_id,prompt_key,prompt",
    _KV_SHARE_CASES,
    ids=[f"{c[0].split('/')[1]}_{c[3]}" for c in _KV_SHARE_CASES],
)
def test_kv_share_disagg_kv_buffer_written(model_id, overrides, tokenizer_id, prompt_key, prompt):
    """
    Sanity check: after prefill the shared KV cache must be non-zero.
    Verifies that the DMA slice handoff actually wrote KV data into the
    shared numpy buffers (i.e. set_data_for_kv_handoff worked).
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = tokenizer(prompt, return_tensors="np", padding=True)
    input_ids_length = raw["input_ids"].shape[1]
    num_chunks = math.ceil(input_ids_length / _PREFILL_SEQ_LEN)
    padded_len = num_chunks * _PREFILL_SEQ_LEN

    raw = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    raw["position_ids"] = np.where(raw.pop("attention_mask"), np.arange(padded_len), -1)
    raw.pop("token_type_ids", None)

    qeff_model = _build_dummy_model(model_id, overrides)
    config = qeff_model.model.config

    prefill_qpc_path = qeff_model.compile(
        prefill_seq_len=_PREFILL_SEQ_LEN,
        ctx_len=_CTX_LEN,
        num_cores=getattr(config, "num_experts", 16),
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        num_devices=1,
        mos=1,
        aic_enable_depth_first=True,
        prefill_only=True,
        enable_chunking=True,
        split_retained_state_io=True,
        stages=_STAGES,
    )

    prefill_session = QAICInferenceSession(
        qpc_path=prefill_qpc_path,
        full_batch_size=1,
        device_ids=[0],
        cluster_id="prefill",
        stages=_STAGES,
    )

    kv_cache: list[np.ndarray] = [
        np.zeros(prefill_session.kv_shape, dtype=prefill_session.kv_size) for _ in prefill_session.kv_only_buff_map
    ]

    _logits_b = prefill_session.bindings[prefill_session.binding_index_map["logits"]]
    from QEfficient.generation.cloud_infer_KV_share import AIC_TO_NP

    logits_buf = np.zeros(list(_logits_b.dims), dtype=AIC_TO_NP[_logits_b.type])

    chunk_inputs = {k: v for k, v in raw.items()}
    chunk_inputs["batch_index"] = np.array([[0]], dtype=np.int64)

    for chunk_idx in range(num_chunks):
        start = chunk_idx * _PREFILL_SEQ_LEN
        end = start + _PREFILL_SEQ_LEN
        is_last = chunk_idx == num_chunks - 1
        chunk_inputs["input_ids"] = raw["input_ids"][:, start:end]
        chunk_inputs["position_ids"] = raw["position_ids"][:, start:end]
        if is_last:
            chunk_inputs["logits"] = logits_buf
            exec_idx = prefill_session.np_run_pipeline(inputs=chunk_inputs, last_chunk=True, kv_cache_buffers=kv_cache)
        else:
            exec_idx = prefill_session.np_run(inputs=chunk_inputs, is_prefill=True)
        prefill_session.complete_inf(exec_idx, is_prefill=True)

    prefill_session.deactivate()

    # At least one KV buffer must be non-zero after prefill
    any_nonzero = any(np.any(buf != 0) for buf in kv_cache)
    assert any_nonzero, (
        "All KV cache buffers are zero after prefill — "
        "DMA KV-slice handoff (set_data_for_kv_handoff) did not write any data."
    )
