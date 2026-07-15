# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""Token-level parity test for the gpt-oss disaggregated prefill/decode DMA path.
pytest -m "on_qaic" tests/transformers/disaggregated/test_gpt_oss_disagg_kv_share_w_hf_fp32.py
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession

MODEL_NAME = "openai/gpt-oss-20b"
TOKENIZER_ID = MODEL_NAME
NUM_HIDDEN_LAYERS = 4
PREFILL_SEQ_LEN = 32
CTX_LEN = 256
BATCH_SIZE = 1
GENERATION_LEN = 50
SLIDING_WINDOW = 128
TEXT_PROMPT = "Explain quantum computing in simple terms."

NUM_CORES = 16
MOE_PREFILL_PACKED_CHUNK_SIZE = 16
STAGES = 2
PREFILL_NUM_DEVICES = 2
DECODE_NUM_DEVICES = 1


def _assert_onnx_path(onnx_path, label: str) -> Path:
    assert onnx_path is not None, f"{label} compile did not set an ONNX path"
    onnx_path = Path(onnx_path)
    assert onnx_path.is_file(), f"{label} ONNX path does not exist: {onnx_path}"
    assert onnx_path.suffix == ".onnx", f"{label} path is not an ONNX file: {onnx_path}"
    return onnx_path.resolve()


def _build_config(dtype: str = "float32"):
    """Load the real config; optionally truncate depth to a shallow sliding/full layer mix.

    When ``NUM_HIDDEN_LAYERS`` is an int, run a shallow model (cheap compile/run) and
    re-derive ``layer_types`` for the reduced depth so gpt-oss still exercises BOTH a
    sliding-window and a full-attention layer (the sliding layer is the interesting case for
    the DMA handoff: retain_full_kv promotes it to full ctx_len). When it is ``None`` the
    checkpoint's own depth, ``layer_types`` and ``sliding_window`` are kept unchanged.
    """
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if NUM_HIDDEN_LAYERS is not None:
        config.num_hidden_layers = NUM_HIDDEN_LAYERS
        config.sliding_window = SLIDING_WINDOW
        # Alternate sliding / full so the truncated model still has at least one of each.
        config.layer_types = ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(NUM_HIDDEN_LAYERS)]
    config.dtype = dtype
    config.torch_dtype = getattr(torch, dtype)
    return config


def _load_hf_model(config) -> AutoModelForCausalLM:
    torch.manual_seed(42)
    if NUM_HIDDEN_LAYERS is None:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            config=config,
            attn_implementation="eager",
            torch_dtype=config.torch_dtype,
            trust_remote_code=True,
        )
        return model.eval()
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    # Scale weights down so fp32 activations stay small; keeps HF and QAIC numerics close.
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(0.02)
    return model.eval()


def _get_next_token_ids(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    return logits[:, -1, :].argmax(axis=-1).astype(np.int64)


def _prompt_input_ids(tokenizer) -> torch.Tensor:
    return tokenizer(TEXT_PROMPT, return_tensors="pt")["input_ids"]


def _prepare_inputs(tokenizer) -> dict:
    """Tokenize the prompt, right-pad to a multiple of PREFILL_SEQ_LEN, build position_ids."""
    ids = _prompt_input_ids(tokenizer)
    input_len = ids.shape[1]
    num_chunks = -(input_len // -PREFILL_SEQ_LEN)  # ceil divide without float
    padded_len = num_chunks * PREFILL_SEQ_LEN
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = np.full((BATCH_SIZE, padded_len), pad_id, dtype=np.int64)
    input_ids[:, :input_len] = ids.numpy()
    attention_mask = np.zeros((BATCH_SIZE, padded_len), dtype=np.int64)
    attention_mask[:, :input_len] = 1
    position_ids = np.where(attention_mask, np.arange(padded_len), -1)
    return {
        "input_ids": input_ids,
        "position_ids": position_ids.astype(np.int64),
        "attention_mask": attention_mask,
        "num_chunks": num_chunks,
        "input_len": input_len,
    }


def _run_hf_torch_fp32(model, tokenizer) -> np.ndarray:
    model = model.to(dtype=torch.float32).eval()
    input_ids = _prompt_input_ids(tokenizer)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=GENERATION_LEN,
            min_new_tokens=GENERATION_LEN,
            do_sample=False,
        )
    prompt_len = input_ids.shape[-1]
    return outputs[:, prompt_len:].detach().cpu().numpy()


def _run_disagg_kv_share_qaic_generation(
    tokenizer,
    prefill_session: QAICInferenceSession,
    decode_session: QAICInferenceSession,
) -> np.ndarray:
    prepared = _prepare_inputs(tokenizer)
    num_chunks = prepared["num_chunks"]
    input_ids = prepared["input_ids"]
    position_ids = prepared["position_ids"]
    kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode_session.kv_cache_info]
    chunk_inputs = {}
    exec_idx = None
    for chunk_idx in range(num_chunks):
        chunk_inputs["input_ids"] = input_ids[:, chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN]
        chunk_inputs["position_ids"] = position_ids[:, chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN]
        last_chunk = chunk_idx == num_chunks - 1
        exec_idx = prefill_session.np_run_pipeline(
            chunk_inputs,
            last_chunk=last_chunk,
            kv_cache_buffers=kv_caches if last_chunk else None,
        )
        prefill_session.complete_inf(exec_idx, is_prefill=True)

    prefill_out = prefill_session.get_outputs(index=exec_idx)
    generated_ids = [_get_next_token_ids(prefill_out["logits"])]
    decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map
    pos = np.max(position_ids, axis=-1, keepdims=True) + 1
    decode_inputs = {
        "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
        "position_ids": pos,
    }
    for _ in range(GENERATION_LEN - 1):
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
            [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        decode_outputs = decode_session.get_outputs(index=exec_idx)
        generated_ids.append(_get_next_token_ids(decode_outputs["logits"]))
        pos = pos + 1
        decode_inputs = {
            "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
            "position_ids": pos,
        }

    return np.stack(generated_ids, axis=1)


@pytest.mark.on_qaic
def test_gpt_oss_disagg_kv_share_qaic_vs_hf_fp32(manual_cleanup):
    torch.manual_seed(42)

    config = _build_config(dtype="float32")
    hf_model = _load_hf_model(config)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_tokens = _run_hf_torch_fp32(hf_model, tokenizer)

    qeff_model = QEFFAutoModelForCausalLM(hf_model)

    sessions = []
    compiled_onnx_paths = {}
    try:
        decode_qpc_path = qeff_model.compile(
            prefill_seq_len=1,
            ctx_len=CTX_LEN,
            num_cores=NUM_CORES,
            num_devices=DECODE_NUM_DEVICES,
            mos=1,
            aic_enable_depth_first=True,
            num_speculative_tokens=None,
            offload_pt_weights=False,
            split_retained_state_io=True,
            retain_full_kv=True,
            use_onnx_subfunctions=True,
        )
        compiled_onnx_paths["decode"] = _assert_onnx_path(qeff_model.onnx_path, "decode")

        prefill_qpc_path = qeff_model.compile(
            prefill_seq_len=PREFILL_SEQ_LEN,
            ctx_len=CTX_LEN,
            num_cores=NUM_CORES,
            moe_prefill_packed_chunk_size=MOE_PREFILL_PACKED_CHUNK_SIZE,
            num_devices=PREFILL_NUM_DEVICES,
            mdp_num_partitions=STAGES,
            split_retained_state_io=True,
            mos=1,
            aic_enable_depth_first=False,
            num_speculative_tokens=None,
            prefill_only=True,
            enable_chunking=True,
            retain_full_kv=True,
            use_onnx_subfunctions=True,
        )
        compiled_onnx_paths["prefill"] = _assert_onnx_path(qeff_model.onnx_path, "prefill")
        print(f"Disagg ONNX paths: {compiled_onnx_paths}")

        prefill_session = QAICInferenceSession(prefill_qpc_path, kv_dma_share=True, stages=STAGES)
        decode_session = QAICInferenceSession(decode_qpc_path, kv_dma_share=True)
        sessions.extend([prefill_session, decode_session])

        qaic_tokens = _run_disagg_kv_share_qaic_generation(
            tokenizer=tokenizer,
            prefill_session=prefill_session,
            decode_session=decode_session,
        )
    finally:
        for session in sessions:
            session.deactivate()
        # cleanup_paths = list(compiled_onnx_paths.values()) or [getattr(qeff_model, "onnx_path", None)]
        # manual_cleanup([path for path in cleanup_paths if path is not None])

    assert qaic_tokens.shape == (BATCH_SIZE, GENERATION_LEN)
    assert hf_tokens.shape == (BATCH_SIZE, GENERATION_LEN)
    assert np.issubdtype(qaic_tokens.dtype, np.integer)
    assert np.issubdtype(hf_tokens.dtype, np.integer)

    matches = hf_tokens == qaic_tokens
    num_matched = int(matches.all(axis=0).cumprod().sum())  # leading run matched across all rows
    hf_text = tokenizer.batch_decode(hf_tokens, skip_special_tokens=True)
    qaic_text = tokenizer.batch_decode(qaic_tokens, skip_special_tokens=True)
    print(f"HF Torch fp32 tokens   : {hf_tokens.tolist()}")
    print(f"Disagg QAIC DMA tokens : {qaic_tokens.tolist()}")
    print(f"HF Torch fp32 text     : {hf_text}")
    print(f"Disagg QAIC DMA text   : {qaic_text}")
    print(f"Matched leading tokens : {num_matched}/{GENERATION_LEN}")

    if not matches.all():
        first_mismatch = int(np.argmin(matches.all(axis=0)))
        raise AssertionError(
            "Tokens don't match for HF Torch fp32 output and disagg QAIC DMA output; "
            f"first mismatch at token index {first_mismatch} "
            f"(matched {num_matched}/{GENERATION_LEN} leading tokens): "
            f"HF={hf_tokens[:, first_mismatch].tolist()} vs QAIC={qaic_tokens[:, first_mismatch].tolist()}"
        )
