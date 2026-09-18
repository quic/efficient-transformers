# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import math
import numpy as np
import os
import tempfile
import time

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

MODEL_ID = "MiniMaxAI/MiniMax-M3"


def _expand_batch(inputs, batch_size: int):
    """Repeat single-prompt tokenizer tensors for the compiled execution batch."""
    expanded = {}
    for name, value in inputs.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 1:
            expanded[name] = value.repeat((batch_size,) + (1,) * (value.ndim - 1))
        else:
            expanded[name] = value
    return expanded





def _execution_batch_size(batch_size: int, msa_indexer_dp: int, msa_attn_dp: int) -> int:
    return batch_size * math.lcm(msa_indexer_dp, msa_attn_dp)


def _run_pytorch_parity_test(
    model_id: str,
    prompt: str,
    export_dir: str,
    ctx_len: int = 128,
    num_cores: int = 16,
    num_devices: int = 1,
    expert_parallel_chunk_size: int = 256,
    cores_per_expert: int = 2,
    tree_reduce: bool = True,
    msa_indexer_dp: int = 1,
    msa_indexer_cp: int = 1,
    msa_attn_dp: int = 1,
    indexer_n_head: int = 1,
    num_cores_per_device: int = 16,
    msa_q_chunk: int = 64,
    batch_size: int = 1,
) -> None:
    """Compare HF PyTorch vs AIC on the last decode token of the prompt (prefill_seq_len=1)."""
    full_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    full_config.text_config.num_hidden_layers = 4

    torch.manual_seed(42)
    model_hf = AutoModelForImageTextToText.from_config(full_config).eval()
    model_dir = os.path.join(export_dir, "minimax-m3-parity")
    model_hf.save_pretrained(model_dir)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    messages = [[{"role": "user", "content": [{"type": "text", "text": prompt}]}]]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    last_token_ids = inputs["input_ids"][:, -1:]
    execution_batch_size = _execution_batch_size(batch_size, msa_indexer_dp, msa_attn_dp)
    last_token_ids = last_token_ids.repeat(execution_batch_size, 1)

    with torch.no_grad():
        hf_logits = model_hf.language_model(input_ids=last_token_ids, use_cache=False).logits[:, -1:, :]
    expected_token = int(hf_logits.argmax(-1)[0, 0])

    qaic_config = {
        "moe_config": {
            "flavour": "expert_parallel",
            "expert_parallel_chunk_size": expert_parallel_chunk_size,
            "cores_per_expert": cores_per_expert,
            "tree_reduce": tree_reduce,
        }
    }
    if msa_indexer_dp > 1 or msa_attn_dp > 1:
        qaic_config["blocking_mode"] = "kv_headpar"
        qaic_config["num_kv_blocks"] = 2
        if msa_indexer_dp > 1 or msa_indexer_cp > 1:
            qaic_config["msa_indexer_dp"] = msa_indexer_dp
            qaic_config["msa_indexer_cp"] = msa_indexer_cp
            qaic_config["indexer_n_head"] = indexer_n_head
            qaic_config["num_cores_per_device"] = num_cores_per_device
        if msa_attn_dp > 1:
            qaic_config["msa_attn_dp"] = msa_attn_dp

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(model_dir, torch_dtype=torch.float32)
    qeff_model.compile(
        batch_size=execution_batch_size,
        prefill_seq_len=1,
        ctx_len=ctx_len,
        num_cores=num_cores,
        num_devices=num_devices,
        use_onnx_subfunctions=False,
        skip_vision=True,
        offload_pt_weights=False,
        weight_free=False,
        qaic_config=qaic_config,
    )

    aic_inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs={"input_ids": last_token_ids},
        prefill_seq_len=1,
        batch_size=execution_batch_size,
    )
    output = qeff_model.generate(inputs=aic_inputs, generation_len=1)
    aic_token = int(output.generated_ids[0, 0])
    assert aic_token == expected_token, f"Parity check FAILED: expected {expected_token}, got {aic_token}"
    print(f"[PASS] PyTorch vs AIC parity check passed (token={aic_token})")


def main():
    parser = argparse.ArgumentParser(
        description="Export and compile separate MiniMax-M3 prefill and decode QPCs for disaggregated serving."
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--ctx-len", type=int, default=4096)
    parser.add_argument("--generation-len", type=int, default=32)
    parser.add_argument("--num-devices", type=int, default=16)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument(
        "--prefill-seq-len",
        type=int,
        default=128,
        help="Prompt-token specialization length; must be greater than 1 for MSA prefill.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Logical prompt batch size; QEfficient expands it by the DP LCM for execution.",
    )
    parser.add_argument("--prompt", default="Tell me about yourself.")
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument(
        "--expert-parallel-chunk-size",
        type=int,
        default=256,
        help="MoE expert-parallel chunk size (expert_parallel_chunk_size in moe_config).",
    )
    parser.add_argument(
        "--cores-per-expert",
        type=int,
        default=2,
        help="Number of NSP cores assigned to each expert during decode.",
    )
    parser.add_argument(
        "--no-tree-reduce",
        dest="tree_reduce",
        action="store_false",
        default=True,
        help="Disable tree-reduce for MoE expert-parallel dispatch.",
    )
    parser.add_argument(
        "--msa-indexer-dp",
        type=int,
        default=1,
        help="DP factor for the MSA indexer; prefill currently requires 1.",
    )
    parser.add_argument(
        "--msa-indexer-cp",
        type=int,
        default=1,
        help="CP factor for the MSA indexer; prefill currently requires 1.",
    )
    parser.add_argument(
        "--msa-attn-dp",
        type=int,
        default=1,
        help="DP factor for MSA attention; prefill currently requires 1.",
    )
    parser.add_argument(
        "--indexer-n-head",
        type=int,
        default=1,
        help="Number of KV heads used by the MSA indexer in the DP path.",
    )
    parser.add_argument(
        "--msa-q-chunk",
        type=int,
        default=64,
        help="MSA prefill attention query chunk size.",
    )
    parser.add_argument(
        "--num-cores-per-device",
        type=int,
        default=8,
        help="Number of NSP cores per device for MSA indexer DP block-scoring.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run PyTorch vs ONNX parity check using a tiny random model.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.prefill_seq_len <= 1:
        parser.error("--prefill-seq-len must be greater than 1")
    if args.msa_indexer_dp != 1 or args.msa_indexer_cp != 1 or args.msa_attn_dp != 1:
        parser.error("MSA prefill currently requires --msa-indexer-dp 1, --msa-indexer-cp 1, and --msa-attn-dp 1")
    execution_batch_size = _execution_batch_size(args.batch_size, args.msa_indexer_dp, args.msa_attn_dp)

    if args.test:
        with tempfile.TemporaryDirectory() as tmp_dir:
            _run_pytorch_parity_test(
                model_id=args.model_id,
                prompt=args.prompt,
                export_dir=tmp_dir,
                ctx_len=args.ctx_len,
                num_cores=args.num_cores,
                num_devices=args.num_devices,
                expert_parallel_chunk_size=args.expert_parallel_chunk_size,
                cores_per_expert=args.cores_per_expert,
                tree_reduce=args.tree_reduce,
                msa_indexer_dp=args.msa_indexer_dp,
                msa_indexer_cp=args.msa_indexer_cp,
                msa_attn_dp=args.msa_attn_dp,
                indexer_n_head=args.indexer_n_head,
                num_cores_per_device=args.num_cores_per_device,
                msa_q_chunk=args.msa_q_chunk,
                batch_size=args.batch_size,
            )
        return

    factory_kwargs = dict(kv_offload=True, dtype=torch.float16)
    config = AutoConfig.from_pretrained(args.model_id)
    if args.num_layers is not None:
        config.text_config.num_hidden_layers = args.num_layers
    factory_kwargs["config"] = config

    t0 = time.perf_counter()
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(args.model_id, **factory_kwargs)
    print(f"[timing] model load:          {time.perf_counter() - t0:.2f}s")

    common_compile_kwargs = dict(
        batch_size=execution_batch_size,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        use_onnx_subfunctions=True,
        skip_vision=True,
        offload_pt_weights=False,
        node_precision_info=True,
        log_times=True,
        retain_full_kv=True,
        split_model_io=True,
        qaic_config={
            "blocking_mode": "kv_headpar",
            "num_kv_blocks": 2,
            "msa_indexer_dp": args.msa_indexer_dp,
            "msa_indexer_cp": args.msa_indexer_cp,
            "msa_attn_dp": args.msa_attn_dp,
            "indexer_n_head": args.indexer_n_head,
            "num_cores_per_device": args.num_cores_per_device,
            "msa_q_chunk": args.msa_q_chunk,
            "moe_config": {
                "flavour": "expert_parallel",
                "expert_parallel_chunk_size": args.expert_parallel_chunk_size,
                "cores_per_expert": args.cores_per_expert,
                "tree_reduce": args.tree_reduce,
            },
        },
    )

    t0 = time.perf_counter()
    prefill_qpc_paths = qeff_model.compile(
        prefill_seq_len=args.prefill_seq_len,
        prefill_only=True,
        enable_chunking=True,
        **common_compile_kwargs,
    )
    prefill_qpc_path = prefill_qpc_paths["lang_prefill_qpc_path"]
    print(f"[timing] prefill export + compile: {time.perf_counter() - t0:.2f}s")
    print(f"Prefill QPC path: {prefill_qpc_path}")

    t0 = time.perf_counter()
    decode_qpc_paths = qeff_model.compile(
        prefill_seq_len=1,
        prefill_only=False,
        **common_compile_kwargs,
    )
    decode_qpc_path = decode_qpc_paths["lang_decode_qpc_path"]
    print(f"[timing] decode export + compile:  {time.perf_counter() - t0:.2f}s")
    print(f"Decode QPC path: {decode_qpc_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    messages = [[{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]]
    model_inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    model_inputs = _expand_batch(model_inputs, execution_batch_size)
    model_inputs = qeff_model.model.prepare_inputs_for_generation(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs.get("attention_mask"),
        position_ids=model_inputs.get("position_ids"),
        prefill_seq_len=args.prefill_seq_len,
        batch_size=execution_batch_size,
    )
    input_len = model_inputs["input_ids"].shape[1]
    num_chunks = (input_len + args.prefill_seq_len - 1) // args.prefill_seq_len
    padded_len = num_chunks * args.prefill_seq_len
    pad_len = padded_len - input_len
    model_inputs["input_ids"] = torch.nn.functional.pad(model_inputs["input_ids"], (0, pad_len), value=0)
    if "attention_mask" in model_inputs:
        model_inputs["attention_mask"] = torch.nn.functional.pad(
            model_inputs["attention_mask"], (0, pad_len), value=0
        )
    if "attention_mask" in model_inputs:
        model_inputs["position_ids"] = torch.where(
            model_inputs["attention_mask"].bool(),
            torch.arange(padded_len).unsqueeze(0),
            torch.full((execution_batch_size, padded_len), -1),
        )
    elif "position_ids" not in model_inputs:
        model_inputs["position_ids"] = torch.arange(padded_len).unsqueeze(0).expand(execution_batch_size, -1)
    elif model_inputs["position_ids"].shape[-1] != padded_len:
        model_inputs["position_ids"] = torch.nn.functional.pad(model_inputs["position_ids"], (0, pad_len), value=-1)
    np_inputs = {key: value.detach().cpu().numpy() for key, value in model_inputs.items() if torch.is_tensor(value)}
    np_inputs.pop("attention_mask", None)
    prefill_session = QAICInferenceSession(prefill_qpc_path)
    prefill_state = np_inputs.copy()
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * args.prefill_seq_len
        chunk_end = chunk_start + args.prefill_seq_len
        prefill_state["input_ids"] = np_inputs["input_ids"][:, chunk_start:chunk_end]
        prefill_state["position_ids"] = np_inputs["position_ids"][:, chunk_start:chunk_end]
        prefill_output = prefill_session.run(prefill_state)
        for layer_idx in range(config.text_config.num_hidden_layers):
            prefill_state[f"past_key.{layer_idx}"] = prefill_output[f"past_key.{layer_idx}_RetainedState"]
            prefill_state[f"past_value.{layer_idx}"] = prefill_output[f"past_value.{layer_idx}_RetainedState"]
    prefill_session.deactivate()
    exit(0)
    decode_session = QAICInferenceSession(decode_qpc_path)
    decode_session.activate()
    last_position = np.max(np_inputs["position_ids"], axis=-1, keepdims=True)
    last_indices = (last_position[:, 0] % args.prefill_seq_len).astype(np.int64)
    batch_indices = np.arange(execution_batch_size)
    next_tokens = np.argmax(prefill_output["logits"][batch_indices, last_indices], axis=-1, keepdims=True).astype(np_inputs["input_ids"].dtype)
    decode_inputs = {"input_ids": next_tokens, "position_ids": last_position + 1}
    for layer_idx in range(config.text_config.num_hidden_layers):
        decode_inputs[f"past_key.{layer_idx}"] = prefill_output[f"past_key.{layer_idx}_RetainedState"]
        decode_inputs[f"past_value.{layer_idx}"] = prefill_output[f"past_value.{layer_idx}_RetainedState"]
    generated = [next_tokens]
    for _ in range(max(0, args.generation_len - 1)):
        decode_output = decode_session.run(decode_inputs)
        next_tokens = np.argmax(decode_output["logits"], axis=-1).astype(np_inputs["input_ids"].dtype)
        generated.append(next_tokens)
        decode_inputs["input_ids"] = next_tokens
        decode_inputs["position_ids"] = decode_inputs["position_ids"] + 1
        for layer_idx in range(config.text_config.num_hidden_layers):
            decode_inputs[f"past_key.{layer_idx}"] = decode_output[f"past_key.{layer_idx}_RetainedState"]
            decode_inputs[f"past_value.{layer_idx}"] = decode_output[f"past_value.{layer_idx}_RetainedState"]
    generated_ids = np.concatenate(generated, axis=1)
    print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))



if __name__ == "__main__":
    main()
