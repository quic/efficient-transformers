# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Disaggregated prefill/decode with chunked prefill — DMA-based KV handoff."""

import time
from queue import Queue

import numpy as np
import torch
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession

# DEFAULT_MODEL_ID = "yujiepan/qwen3-moe-tiny-random"
DEFAULT_MODEL_ID = "Qwen/Qwen3-30B-A3B"
DEFAULT_PROMPT = "Explain quantum computing in simple terms."
DEFAULT_PREFILL_SEQ_LEN = 256
DEFAULT_CTX_LEN = DEFAULT_PREFILL_SEQ_LEN * 2
NUM_CORES = 4
MOE_PREFILL_PACKED_CHUNK_SIZE = 128
# Prefill pipeline depth: >1 chunk can be in flight on the device at once.
STAGES = 4
DECODE_NUM_DEVICES = 4
PREFILL_NUM_DEVICES = 8


def _compile_sessions(qeff_model, prefill_seq_len, ctx_len, stages, prefill_num_devices, decode_num_devices):
    """Compile decode/prefill QPCs (spec §5 flags) and open kv_dma_share sessions."""
    decode_qpc_path = qeff_model.compile(
        prefill_seq_len=1,
        ctx_len=ctx_len,
        num_cores=NUM_CORES,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        num_devices=decode_num_devices,
        mos=1,
        aic_enable_depth_first=True,
        num_speculative_tokens=None,
        offload_pt_weights=False,  # Need the weights in memory for prefill-model export/compilation
        split_retained_state_io=True,
        retain_full_kv=True,  # required for DMA slice writes into full KV
    )
    prefill_qpc_path = qeff_model.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        num_cores=NUM_CORES,
        moe_prefill_packed_chunk_size=MOE_PREFILL_PACKED_CHUNK_SIZE,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        num_devices=prefill_num_devices,
        mdp_num_partitions=stages,
        split_retained_state_io=True,
        mos=1,
        user_tiled=True,
        aic_enable_depth_first=False,
        num_speculative_tokens=None,
        prefill_only=True,
        enable_chunking=True,
        use_onnx_subfunctions=True,
    )
    # `stages` sizes the prefill execObj pool (pipeline depth); the decode session
    # keeps a single decode slot.
    prefill_session = QAICInferenceSession(prefill_qpc_path, kv_dma_share=True, stages=stages)
    decode_session = QAICInferenceSession(decode_qpc_path, kv_dma_share=True)
    return prefill_session, decode_session


def run(
    model_id: str = DEFAULT_MODEL_ID,
    prompt: str = DEFAULT_PROMPT,
    prefill_seq_len: int = DEFAULT_PREFILL_SEQ_LEN,
    ctx_len: int = DEFAULT_CTX_LEN,
    stages: int = STAGES,
    prefill_num_devices: int = PREFILL_NUM_DEVICES,
    decode_num_devices: int = DECODE_NUM_DEVICES,
):
    """Run chunked-prefill + decode with the DMA-based KV handoff.

    Returns a dict with the prefill ``logits``, the ``first_token`` (argmax over
    those logits) and the full decoded ``tokens`` list, for parity comparison.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id)
    prefill_session, decode_session = _compile_sessions(
        qeff_model, prefill_seq_len, ctx_len, stages, prefill_num_devices, decode_num_devices
    )

    inputs = tokenizer(prompt, return_tensors="np", padding=True)
    position_ids = inputs["attention_mask"].sum(1, keepdims=True)
    generation_len = ctx_len - position_ids.max()
    padded_len = inputs["input_ids"].shape[1]
    num_chunks = -(padded_len // -prefill_seq_len)  # ceil divide without float
    padded_len = num_chunks * prefill_seq_len  # Convert to a multiple of prompt_len
    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
    inputs.pop("token_type_ids", None)
    inputs = {k: torch.from_numpy(v) for k, v in inputs.items()}
    inputs.pop("past_key_values", None)
    inputs = {k: v.detach().numpy() for k, v in inputs.items()}

    # Shared host KV arrays, allocated once in decode-map order (per-slot shape[0] == 1).
    kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode_session.kv_cache_info]

    # ---- Prefill (producer, PIPELINED): keep up to `stages` chunks in flight ----
    # Submit chunks without blocking; when the execObj pool is full, drain the
    # oldest completion before submitting the next. Only the LAST chunk wires the
    # DMA handoff into kv_caches (earlier chunks just accumulate KV on-device).
    pending: Queue = Queue()
    in_flight = 0
    exec_idx = None
    prefill_start = time.time()
    for i in range(num_chunks):
        chunk_inputs = dict(inputs)
        chunk_inputs["input_ids"] = inputs["input_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
        chunk_inputs["position_ids"] = inputs["position_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
        last_chunk = i == num_chunks - 1
        if in_flight == prefill_session.prefill_num_execObj:
            prefill_session.complete_inf(pending.get(timeout=120), is_prefill=True)
            in_flight -= 1
        exec_idx = prefill_session.np_run_pipeline(
            chunk_inputs,
            last_chunk=last_chunk,
            kv_cache_buffers=kv_caches if last_chunk else None,
        )
        pending.put(exec_idx)
        in_flight += 1
    # Drain remaining in-flight chunks (barrier before decode reads the slot).
    last_exec_idx = exec_idx
    while not pending.empty():
        exec_idx = pending.get(timeout=120)
        prefill_session.complete_inf(exec_idx, is_prefill=True)
    print(f"prefill time={time.time() - prefill_start} sec")

    prefill_logits = prefill_session.get_outputs(index=last_exec_idx)["logits"]
    first_token = int(np.argmax(prefill_logits))
    all_outputs = [first_token]

    # ---- Decode (consumer): re-point DMA descriptor at kv_caches EVERY step ----
    # The decode QPC is compiled with split_retained_state_io=True (spec §5), so the
    # KV *input* binding (past_key.N) and the *_RetainedState output* (past_key.N_
    # RetainedState) are distinct device buffers — exactly the two sides the baseline
    # bridges with `inputs[past_key.N] = out[past_key.N_RetainedState]` each step.
    # Wire BOTH at kv_caches: the input for the read, the RS output for the write-back
    # of the newly-decoded position. kv_caches is allocated in decode_buff_map order,
    # and decode_rs_kv_only_buff_map is the same layer-sorted family order, so the two
    # concatenated maps line up with kv_caches repeated.
    decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map
    bidx = 0
    pos = int(np.max(inputs["position_ids"])) + 1
    decode_inputs = {
        "input_ids": np.array(first_token, dtype=np.int64).reshape(1, 1),
        "position_ids": np.array([[pos]], dtype=np.int64),
    }
    st = time.time()
    for _ in range(generation_len - 1):
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
            [("batch_index", bidx), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        out = decode_session.get_outputs(index=exec_idx)
        tok = int(np.argmax(out["logits"]))
        all_outputs.append(tok)
        pos += 1
        decode_inputs = {
            "input_ids": np.array(tok, dtype=np.int64).reshape(1, 1),
            "position_ids": np.array([[pos]], dtype=np.int64),
        }
    ft = time.time()

    print(f"decode tok/sec={(generation_len - 1) / (ft - st)}")
    print(f"input\n{prompt}\noutput\n{tokenizer.decode(all_outputs)}")

    return {"logits": prefill_logits, "first_token": first_token, "tokens": all_outputs}


if __name__ == "__main__":
    run()
