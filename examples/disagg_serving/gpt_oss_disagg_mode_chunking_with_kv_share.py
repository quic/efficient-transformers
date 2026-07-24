# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Disaggregated prefill/decode with chunked prefill for gpt-oss — DMA slicing-based KV handoff."""

import argparse
import time
from queue import Queue

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession

# NOTE: 120b-requires NPI yaml
DEFAULT_SUBFUNC_NPI = None
DEFAULT_NON_SUBFUNC_NPI = None
DEFAULT_MODEL_ID = "openai/gpt-oss-20b"
DEFAULT_PROMPT = "Explain quantum computing in simple terms."
DEFAULT_PREFILL_SEQ_LEN = 512
DEFAULT_CTX_LEN = DEFAULT_PREFILL_SEQ_LEN * 2
NUM_CORES = 16
MOE_PREFILL_PACKED_CHUNK_SIZE = 256
# Prefill pipeline depth: >1 chunk can be in flight on the device at once.
STAGES = 4
PREFILL_NUM_DEVICES = 8
DECODE_NUM_DEVICES = 4


def _build_config(model_id: str, num_hidden_layers: int = None):
    """Load the model config, optionally reducing ``num_hidden_layers``."""
    if num_hidden_layers is None:
        return None
    config = AutoConfig.from_pretrained(model_id)
    config.num_hidden_layers = num_hidden_layers
    return config


def _compile_sessions(
    qeff_model,
    prefill_seq_len,
    ctx_len,
    stages,
    prefill_num_devices,
    decode_num_devices,
):
    decode_qpc_path = qeff_model.compile(
        prefill_seq_len=1,
        ctx_len=ctx_len,
        num_cores=NUM_CORES,
        num_devices=decode_num_devices,
        mos=1,
        aic_enable_depth_first=True,
        num_speculative_tokens=None,
        offload_pt_weights=False,  # Need the weights in memory for prefill-model export/compilation
        split_retained_state_io=True,
        retain_full_kv=True,
        use_onnx_subfunctions=True,
    )
    prefill_qpc_path = qeff_model.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        num_cores=NUM_CORES,
        moe_prefill_packed_chunk_size=MOE_PREFILL_PACKED_CHUNK_SIZE,
        num_devices=prefill_num_devices,
        mdp_num_partitions=stages,
        split_retained_state_io=True,
        mos=1,
        aic_enable_depth_first=True,
        num_speculative_tokens=None,
        prefill_only=True,
        enable_chunking=True,
        retain_full_kv=True,  # promote sliding layers to full ctx_len
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
    num_hidden_layers: int = None,
    subfunc_npi: str = DEFAULT_SUBFUNC_NPI,
    non_subfunc_npi: str = DEFAULT_NON_SUBFUNC_NPI,
):
    """Run chunked-prefill + decode with the DMA-based KV handoff for gpt-oss.

    ``num_hidden_layers`` (testing only) reduces the model depth for a fast compile;
    leave it ``None`` to use the full model. Returns a dict with the prefill ``logits``,
    the ``first_token`` (argmax over those logits) and the full decoded ``tokens`` list,
    for parity comparison.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    config = _build_config(model_id, num_hidden_layers)
    from_pretrained_kwargs = {"config": config} if config is not None else {}
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, **from_pretrained_kwargs)
    prefill_session, decode_session = _compile_sessions(
        qeff_model,
        prefill_seq_len,
        ctx_len,
        stages,
        prefill_num_devices,
        decode_num_devices,
        subfunc_npi,
        non_subfunc_npi,
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

    # ---- Decode (consumer): re-point DMA descriptor at kv_caches----
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="text prompt")
    parser.add_argument("--prefill-seq-len", type=int, default=DEFAULT_PREFILL_SEQ_LEN)
    parser.add_argument("--ctx-len", type=int, default=DEFAULT_CTX_LEN)
    parser.add_argument("--stages", type=int, default=STAGES, help="prefill pipeline depth (mdp_num_partitions)")
    parser.add_argument(
        "--prefill-num-devices", type=int, default=PREFILL_NUM_DEVICES, help="num devices for the prefill QPC"
    )
    parser.add_argument(
        "--decode-num-devices", type=int, default=DECODE_NUM_DEVICES, help="num devices for the decode QPC"
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=None,
        help="reduce model depth for a fast compile (testing only; outputs not meaningful)",
    )
    parser.add_argument("--subfunc-npi", default=DEFAULT_SUBFUNC_NPI, help="prefill (subfunction) NPI yaml path")
    parser.add_argument(
        "--non-subfunc-npi", default=DEFAULT_NON_SUBFUNC_NPI, help="decode (non-subfunction) NPI yaml path"
    )
    args = parser.parse_args()

    run(
        model_id=args.model_id,
        prompt=args.prompt,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        stages=args.stages,
        prefill_num_devices=args.prefill_num_devices,
        decode_num_devices=args.decode_num_devices,
        num_hidden_layers=args.num_hidden_layers,
        subfunc_npi=args.subfunc_npi,
        non_subfunc_npi=args.non_subfunc_npi,
    )
