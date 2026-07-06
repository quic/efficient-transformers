# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Disaggregated prefill/decode with chunked prefill — numpy-copy KV handoff.

This is the baseline path: after every prefill chunk and every decode step the
``*_RetainedState`` outputs are copied back into the next input dict on the host.
The DMA-share counterpart lives in ``qwen3moe_disagg_mode_chunking_with_kv_share.py``.

The body is exposed as ``run(...)`` so the on_qaic parity test
(``tests/transformers/disaggregated/test_kv_share_parity.py``) can import and
compare it against the KV-DMA-share path in one process.
"""

import time

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession

# model_id = "Qwen/Qwen3-30B-A3B-Instruct-2507"  # weights are not required to convert to fp32
DEFAULT_MODEL_ID = "yujiepan/qwen3-moe-tiny-random"
DEFAULT_PROMPT = "Explain quantum computing in simple terms."
DEFAULT_PREFILL_SEQ_LEN = 512
DEFAULT_CTX_LEN = DEFAULT_PREFILL_SEQ_LEN * 3
NUM_CORES = 4
MOE_PREFILL_PACKED_CHUNK_SIZE = 256
# Prefill pipeline depth (mdp_num_partitions) for the chunked prefill QPC.
STAGES = 2
PREFILL_NUM_DEVICES = 2
DECODE_NUM_DEVICES = 1


def _compile_sessions(qeff_model, prefill_seq_len, ctx_len, stages, prefill_num_devices, decode_num_devices):
    """Compile the decode and (chunked) prefill QPCs and open sessions on each."""
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
        retain_full_kv=True,
    )
    prefill_qpc_path = qeff_model.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        num_cores=NUM_CORES,
        moe_prefill_packed_chunk_size=MOE_PREFILL_PACKED_CHUNK_SIZE,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        num_devices=prefill_num_devices,
        split_retained_state_io=True,
        mos=1,
        user_tiled=True,
        aic_enable_depth_first=False,
        num_speculative_tokens=None,
        prefill_only=True,
        enable_chunking=True,
        use_onnx_subfunctions=True,
        mdp_num_partitions=stages,
    )
    return QAICInferenceSession(prefill_qpc_path), QAICInferenceSession(decode_qpc_path)


def run(
    model_id: str = DEFAULT_MODEL_ID,
    prompt: str = DEFAULT_PROMPT,
    prefill_seq_len: int = DEFAULT_PREFILL_SEQ_LEN,
    ctx_len: int = DEFAULT_CTX_LEN,
    stages: int = STAGES,
    prefill_num_devices: int = PREFILL_NUM_DEVICES,
    decode_num_devices: int = DECODE_NUM_DEVICES,
):
    """Run chunked-prefill + decode with the numpy-copy KV handoff.

    Returns a dict with the prefill ``logits``, the ``first_token`` (argmax over
    those logits) and the full decoded ``tokens`` list, for parity comparison.
    """
    config = AutoConfig.from_pretrained(model_id)
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

    all_outputs = []
    qpc_out = None
    for i in range(num_chunks):
        chunk_inputs = inputs.copy()
        chunk_inputs["input_ids"] = inputs["input_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
        chunk_inputs["position_ids"] = inputs["position_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
        ins = time.time()
        qpc_out = prefill_session.run(chunk_inputs)
        print(f"time for this run={time.time() - ins}")
        for layer in range(config.num_hidden_layers):
            inputs[f"past_key.{layer}"] = qpc_out[f"past_key.{layer}_RetainedState"]
            inputs[f"past_value.{layer}"] = qpc_out[f"past_value.{layer}_RetainedState"]

    prefill_logits = qpc_out["logits"]
    first_token = int(np.argmax(prefill_logits))
    all_outputs.append(first_token)

    decode_inputs = {
        "input_ids": np.array(first_token).reshape(1, 1),
        "position_ids": np.max(inputs["position_ids"]).reshape(1, 1) + 1,
    }
    for layer in range(config.num_hidden_layers):
        decode_inputs[f"past_key.{layer}"] = qpc_out[f"past_key.{layer}_RetainedState"]
        decode_inputs[f"past_value.{layer}"] = qpc_out[f"past_value.{layer}_RetainedState"]

    st = time.time()
    decode_out = decode_session.run(decode_inputs)
    print(f"time for first run of decode with KV as input = {time.time() - st} sec\n")
    all_outputs.append(int(np.argmax(decode_out["logits"])))
    pos_id = np.max(decode_inputs["position_ids"]).reshape(1, 1) + 1
    loop_decode_inputs = {
        "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
        "position_ids": pos_id,
    }
    for layer in range(config.num_hidden_layers):
        loop_decode_inputs[f"past_key.{layer}"] = decode_out[f"past_key.{layer}_RetainedState"]
        loop_decode_inputs[f"past_value.{layer}"] = decode_out[f"past_value.{layer}_RetainedState"]

    st = time.time()
    for _ in range(generation_len - 2):
        decode_out = decode_session.run(loop_decode_inputs)
        all_outputs.append(int(np.argmax(decode_out["logits"])))
        pos_id += 1
        for layer in range(config.num_hidden_layers):
            loop_decode_inputs[f"past_key.{layer}"] = decode_out[f"past_key.{layer}_RetainedState"]
            loop_decode_inputs[f"past_value.{layer}"] = decode_out[f"past_value.{layer}_RetainedState"]
        loop_decode_inputs.update(
            {
                "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
                "position_ids": pos_id,
            }
        )
    ft = time.time()

    print(f"decode tok/sec={(generation_len - 2) / (ft - st)}")
    print(f"input\n{prompt}\noutput\n{tokenizer.decode(all_outputs)}")

    return {"logits": prefill_logits, "first_token": first_token, "tokens": all_outputs}


if __name__ == "__main__":
    run()
