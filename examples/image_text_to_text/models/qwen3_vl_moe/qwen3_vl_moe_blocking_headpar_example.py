# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Qwen3-VL-MoE blocked attention example with head-parallel support.

Tests three blocking configurations for the Qwen3-VL-MoE text (language
decoder) path, comparing AIC output against a vanilla HuggingFace PyTorch
reference. A dummy 1-layer model (random weights, reduced config) is used
for fast execution.

Blocking modes (selected via --blocking-mode):

  decode          HQKV blocking with kv_blocking_headpar_split=0 (auto-set
                  to num_cores). Single prefill+decode QPC, uses the
                  high-level generate() API.

  prefill_par     Separate prefill QPC (prefill_blocking_mode="qkv") plus
                  a decode QPC (HQKV). Manual chunked-prefill + decode loop
                  via QAICInferenceSession.

  prefill_online  Same structure as prefill_par but uses
                  prefill_blocking_mode="online".

  all             Run all three modes back-to-back against the same PyTorch
                  reference.
"""

import argparse
import copy
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"

# ── Compile / inference defaults ──────────────────────────────────────────────
PREFILL_SEQ_LEN = 256
CTX_LEN = 8192
GEN_LEN = 2048
DEFAULT_NUM_CORES = 16
DEFAULT_NUM_DEVICES = 4
NUM_KV_BLOCKS = 2
NUM_Q_BLOCKS = 2
HEAD_BLOCK_SIZE = 8
PREFILL_BLOCK_CHUNKS = 2

PROMPT = "Tell me about yourself."


# ── Config builders ────────────────────────────────────────────────────────────


def _decode_qaic_config() -> dict:
    return {
        "blocking_mode": "hqkv",
        "num_kv_blocks": NUM_KV_BLOCKS,
        "num_q_blocks": NUM_Q_BLOCKS,
        "head_block_size": HEAD_BLOCK_SIZE,
        "kv_blocking_headpar_split": 0,  # 0 → resolved to num_cores at compile time
    }


def _prefill_qaic_config(prefill_mode: str) -> dict:
    cfg = _decode_qaic_config()
    cfg["prefill_block_chunks"] = PREFILL_BLOCK_CHUNKS
    cfg["prefill_blocking_mode"] = prefill_mode
    cfg["ctx_len"] = CTX_LEN
    return cfg


# ── Dummy model builder ────────────────────────────────────────────────────────


def build_dummy_config() -> AutoConfig:
    """Shrink the Qwen3-VL-MoE config to 1 text layer for fast testing."""
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    config.vision_config.depth = 9
    config.text_config.num_hidden_layers = 1
    config.vision_config.deepstack_visual_indexes = [8]
    return config


def build_hf_model(config: AutoConfig):
    torch.manual_seed(42)
    model_hf = AutoModelForImageTextToText.from_config(
        config,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    if getattr(model_hf.config, "torch_dtype", None) in (torch.bfloat16, torch.float16):
        model_hf = model_hf.to(torch.float32)
    model_hf.eval()
    return model_hf


# ── Shared compile kwargs ─────────────────────────────────────────────────────


def _base_compile_kwargs(num_cores: int, num_devices: int) -> dict:
    return dict(
        batch_size=1,
        ctx_len=CTX_LEN,
        num_cores=num_cores,
        num_devices=num_devices,
        mxfp6_matmul=False,
        mxint8_kv_cache=True,
        retain_full_kv=True,
        split_model_io=True,  # required for disagg KV transfer via VLLM
        user_tiled=True,
        skip_vision=True,
        use_onnx_subfunctions=True,
        layerwise=False,
    )


# ── Input helpers ─────────────────────────────────────────────────────────────


def build_processor_inputs(processor, prompt: str):
    messages = [[{"role": "user", "content": [{"type": "text", "text": prompt}]}]]
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


# ── PyTorch reference ─────────────────────────────────────────────────────────


@torch.no_grad()
def run_pytorch_reference(model_hf, raw_inputs, gen_len: int, tokenizer=None) -> np.ndarray:
    output = model_hf.generate(**raw_inputs, max_new_tokens=gen_len, do_sample=False)
    generated = output[0, raw_inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True) if tokenizer is not None else str(generated.tolist())
    print(f"  [pytorch] {text}")
    return generated.numpy()


# ── Config 1: decode (single QPC, high-level generate()) ──────────────────────


def run_decode_config(
    model_hf, config, processor, raw_inputs, gen_len: int, num_cores: int, num_devices: int
) -> np.ndarray:
    print("\n=== Config: decode (HQKV, kv_blocking_headpar_split=0) ===")

    qeff_model = QEFFAutoModelForImageTextToText(
        copy.deepcopy(model_hf),
        kv_offload=True,
        config=config,
    )

    ckwargs = _base_compile_kwargs(num_cores, num_devices)
    ckwargs["qaic_config"] = _decode_qaic_config()
    ckwargs["prefill_seq_len"] = PREFILL_SEQ_LEN
    ckwargs["offload_pt_weights"] = False
    print("  Exporting + compiling...")
    qpc_paths = qeff_model.compile(**ckwargs)
    print(f"  QPC path: {qpc_paths}")

    qeff_inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs=copy.deepcopy(raw_inputs),
        prefill_seq_len=PREFILL_SEQ_LEN,
        batch_size=1,
    )

    print(f"  Running {gen_len} tokens on AIC...")
    t0 = time.time()
    output = qeff_model.generate(inputs=qeff_inputs, generation_len=gen_len)
    elapsed = time.time() - t0

    aic_tokens = output.generated_ids[0, : gen_len + 1]
    text = processor.tokenizer.decode(aic_tokens, skip_special_tokens=True)
    print(f"  [aic/decode] {text} ({elapsed:.2f}s)")
    return aic_tokens


# ── Configs 2 & 3: prefill_par / prefill_online (two QPCs) ───────────────────


def run_prefill_config(
    model_hf,
    config,
    processor,
    raw_inputs,
    gen_len: int,
    prefill_mode: str,
    num_cores: int,
    num_devices: int,
) -> np.ndarray:
    label = f"prefill_{prefill_mode}"
    num_layers = config.text_config.num_hidden_layers
    print(f"\n=== Config: {label} (HQKV decode + {prefill_mode} prefill) ===")

    ckwargs = _base_compile_kwargs(num_cores, num_devices)

    # ── Compile decode model (prefill_seq_len=1) ──────────────────────────────
    print("  Compiling decode model (prefill_seq_len=1)...")
    decode_model = QEFFAutoModelForImageTextToText(
        copy.deepcopy(model_hf),
        kv_offload=True,
        config=config,
    )
    decode_qpc_paths = decode_model.compile(
        prefill_seq_len=1,
        qaic_config=_decode_qaic_config(),
        offload_pt_weights=False,
        **ckwargs,
    )
    decode_qpc = decode_qpc_paths.get("lang_decode_qpc_path")
    print(f"  Decode QPC: {decode_qpc}")

    # ── Compile prefill model ─────────────────────────────────────────────────
    print(f"  Compiling prefill model (prefill_blocking_mode={prefill_mode})...")
    prefill_model = QEFFAutoModelForImageTextToText(
        copy.deepcopy(model_hf),
        kv_offload=True,
        config=config,
    )
    prefill_qpc_paths = prefill_model.compile(
        prefill_seq_len=PREFILL_SEQ_LEN,
        prefill_only=True,
        enable_chunking=True,
        qaic_config=_prefill_qaic_config(prefill_mode),
        offload_pt_weights=True,
        **ckwargs,
    )
    prefill_qpc = prefill_qpc_paths.get("lang_prefill_qpc_path")
    print(f"  Prefill QPC: {prefill_qpc}")

    # ── Prepare padded inputs ─────────────────────────────────────────────────
    qeff_inputs = decode_model.model.prepare_inputs_for_generation(
        inputs=copy.deepcopy(raw_inputs),
        prefill_seq_len=PREFILL_SEQ_LEN,
        batch_size=1,
    )
    for k, v in qeff_inputs.items():
        qeff_inputs[k] = np.array(v)

    input_ids = qeff_inputs["input_ids"]
    position_ids = qeff_inputs["position_ids"]
    seq_len = input_ids.shape[1]
    num_chunks = -(seq_len // -PREFILL_SEQ_LEN)
    padded_len = num_chunks * PREFILL_SEQ_LEN

    if padded_len > seq_len:
        pad = padded_len - seq_len
        input_ids = np.pad(input_ids, ((0, 0), (0, pad)), constant_values=0)
        # position_ids shape is [4, batch, seq_len]; pad the last dim
        position_ids = np.pad(position_ids, [(0, 0)] * (position_ids.ndim - 1) + [(0, pad)], constant_values=-1)

    # ── Run chunked prefill ───────────────────────────────────────────────────
    prefill_session = QAICInferenceSession(prefill_qpc)
    decode_session = QAICInferenceSession(decode_qpc)

    chunk_inputs = {"image_idx": np.array([[0]])}
    prefill_out = None
    t0 = time.time()
    for i in range(num_chunks):
        chunk_inputs["input_ids"] = input_ids[:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
        chunk_inputs["position_ids"] = position_ids[..., i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
        prefill_out = prefill_session.run(chunk_inputs)
        for layer in range(num_layers):
            chunk_inputs[f"past_key.{layer}"] = prefill_out[f"past_key.{layer}_RetainedState"]
            chunk_inputs[f"past_value.{layer}"] = prefill_out[f"past_value.{layer}_RetainedState"]
        chunk_inputs["image_idx"] = prefill_out["image_idx_output"]
    t_prefill = time.time() - t0
    print(f"  Prefill done: {num_chunks} chunk(s) in {t_prefill:.3f}s")

    # # ── Build decode seed inputs from last prefill output ─────────────────────
    # # position_ids max over last dim for the decode start position
    # decode_pos = np.max(position_ids, axis=-1, keepdims=True) + 1  # [..., 1]
    # decode_inputs = {
    #     "input_ids": np.argmax(prefill_out["logits"]).reshape(1, 1),
    #     "position_ids": decode_pos,
    #     "image_idx": prefill_out["image_idx_output"],
    # }
    # for layer in range(num_layers):
    #     decode_inputs[f"past_key.{layer}"] = prefill_out[f"past_key.{layer}_RetainedState"]
    #     decode_inputs[f"past_value.{layer}"] = prefill_out[f"past_value.{layer}_RetainedState"]

    # # ── Decode loop ───────────────────────────────────────────────────────────
    # all_tokens = [decode_inputs["input_ids"].flatten()]
    # t0 = time.time()
    # for _ in range(gen_len - 1):
    #     out = decode_session.run(decode_inputs)
    #     next_tok = np.argmax(out["logits"], axis=-1).reshape(1, 1)
    #     all_tokens.append(next_tok.flatten())
    #     decode_inputs["input_ids"] = next_tok
    #     decode_inputs["position_ids"] = decode_inputs["position_ids"] + 1
    #     decode_inputs["image_idx"] = out["image_idx_output"]
    #     for layer in range(num_layers):
    #         decode_inputs[f"past_key.{layer}"] = out[f"past_key.{layer}_RetainedState"]
    #         decode_inputs[f"past_value.{layer}"] = out[f"past_value.{layer}_RetainedState"]
    # t_decode = time.time() - t0
    # print(f"  Decode {gen_len} tokens in {t_decode:.2f}s ({gen_len / t_decode:.1f} tok/s)")

    # aic_tokens = np.concatenate(all_tokens)
    # text = processor.tokenizer.decode(aic_tokens, skip_special_tokens=True)
    # print(f"  [aic/{label}] {text}")
    # return aic_tokens


# ── Comparison helper ─────────────────────────────────────────────────────────


def compare_tokens(pt_tokens: np.ndarray, aic_tokens: np.ndarray, label: str):
    n = min(len(pt_tokens), len(aic_tokens))
    match = np.array_equal(pt_tokens[:n], aic_tokens[:n])
    status = "PASS" if match else "FAIL"
    print(f"\n[{status}] {label}: PyTorch vs AIC token match ({n} tokens)")
    if not match:
        mismatches = np.where(pt_tokens[:n] != aic_tokens[:n])[0]
        print(
            f"  First mismatch at position {mismatches[0]}: "
            f"PyTorch={pt_tokens[mismatches[0]]} AIC={aic_tokens[mismatches[0]]}"
        )
    return match


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-VL-MoE blocking attention example")
    parser.add_argument(
        "--blocking-mode",
        choices=["decode", "prefill_par", "prefill_online", "all"],
        default="all",
        help="Blocking configuration to test (default: all)",
    )
    parser.add_argument("--gen-len", type=int, default=GEN_LEN, help="Number of tokens to generate")
    parser.add_argument("--num-cores", type=int, default=DEFAULT_NUM_CORES, help="Number of AIC cores")
    parser.add_argument("--num-devices", type=int, default=DEFAULT_NUM_DEVICES, help="Number of AIC devices")
    parser.add_argument("--prompt", type=str, default=PROMPT, help="Text prompt for inference")
    return parser.parse_args()


def main():
    args = parse_args()

    num_cores = args.num_cores
    num_devices = args.num_devices

    modes_to_run = ["decode", "prefill_par", "prefill_online"] if args.blocking_mode == "all" else [args.blocking_mode]

    print(f"\nModel ID : {MODEL_ID}")
    print(f"Modes    : {modes_to_run}")
    print(f"Gen len  : {args.gen_len}")
    print(f"Devices  : {num_devices} x {num_cores} cores\n")

    # ── Build shared dummy model + processor ──────────────────────────────────
    print("Building dummy 1-layer model (from_config, random weights)...")
    config = build_dummy_config()
    model_hf = build_hf_model(config)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    raw_inputs = build_processor_inputs(processor, args.prompt)

    # # ── PyTorch reference ─────────────────────────────────────────────────────
    # print("\n--- PyTorch reference (vanilla HF model) ---")
    # pt_tokens = run_pytorch_reference(model_hf, raw_inputs, args.gen_len, tokenizer=processor.tokenizer)

    # ── Run selected blocking config(s) ──────────────────────────────────────
    results = {}
    for mode in modes_to_run:
        if mode == "decode":
            aic_tokens = run_decode_config(
                model_hf, config, processor, raw_inputs, args.gen_len, num_cores=num_cores, num_devices=num_devices
            )
        else:
            prefill_mode = "qkv" if mode == "prefill_par" else "online"
            aic_tokens = run_prefill_config(
                model_hf, config, processor, raw_inputs, args.gen_len, prefill_mode, num_cores, num_devices
            )
        results[mode] = aic_tokens

    # ── Summary ───────────────────────────────────────────────────────────────
    # print("\n" + "=" * 60)
    # print("RESULTS")
    # print("=" * 60)
    # all_pass = True
    # for mode, aic_tokens in results.items():
    #     passed = compare_tokens(pt_tokens, aic_tokens, mode)
    #     all_pass = all_pass and passed

    # print("=" * 60)
    # print(f"Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    # print("=" * 60)


if __name__ == "__main__":
    main()
