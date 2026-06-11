# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
GPT-OSS blocked prefill with head-parallel attention.

Demonstrates disaggregated-style inference where:
  - Prefill uses blocked_kv_attention_forward_prefill_headpar_offline via
    prefill_blocked_attention_interface, running chunked prefill over the
    prompt in slices of --prefill-seq-len tokens.
  - Decode uses blocked_kv_attention_forward_headpar_offline (standard
    blocked head-par KV attention) with the KV states produced by prefill.

Optionally compares against a non-blocked baseline with --compare-non-blocked.
"""

import argparse
import time

import numpy as np
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession


def parse_args():
    parser = argparse.ArgumentParser(description="GPT-OSS blocked prefill with head-parallel attention")
    parser.add_argument("--model-name", type=str, default="openai/gpt-oss-20b", help="HuggingFace model ID")
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Once upon a time, in a small town, there lived a young boy named Alex. Alex was a curious and adventurous "
            "child, always eager to explore the world around him. One day, while playing in the park, Alex stumbled "
            "upon a mysterious old book hidden beneath a pile of leaves. The book was filled with stories of distant "
            "lands, magical creatures, and extraordinary adventures.\n\nAs Alex flipped through the pages, he "
            "discovered a map that led to a hidden treasure. Excited by the prospect of a real-life treasure hunt, "
            "Alex decided to embark on a thrilling journey. He packed his backpack with snacks, a flashlight, and a "
            "compass, and set off into the unknown.\n\nThe path to the treasure was not an easy one. Alex had to "
            "navigate through dense forests, cross rickety bridges, and solve riddles that guarded the treasure's "
            "location."
        ),
        help="Input prompt text.",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=None,
        help=(
            "Pad the tokenized prompt to exactly this many tokens. "
            "If --prompt is omitted, a synthetic input of this length is generated."
        ),
    )
    parser.add_argument("--prefill-seq-len", type=int, default=128, help="Prefill chunk sequence length")
    parser.add_argument("--ctx-len", type=int, default=8192, help="Context length")
    parser.add_argument("--generation-len", type=int, default=100, help="Number of decode tokens to generate")
    parser.add_argument("--num-cores", type=int, default=16, help="Number of cores per device")
    parser.add_argument("--num-layers", type=int, default=None, help="Override number of layers (for quick testing)")
    parser.add_argument("--num-kv-blocks", type=int, default=2, help="Number of KV blocks for blocked attention")
    parser.add_argument(
        "--headpar-split",
        type=int,
        default=None,
        help="Head-parallel split factor (defaults to --num-cores)",
    )
    parser.add_argument(
        "--num-devices",
        type=int,
        default=1,
        help="Number of devices to compile for",
    )
    parser.add_argument(
        "--compare-non-blocked",
        action="store_true",
        help="Also compile and run a non-blocked baseline for comparison",
    )
    parser.add_argument(
        "--subf",
        action="store_true",
        help="Use ONNX subfunctions during export",
    )
    return parser.parse_args()


def prepare_chunked_inputs(tokenizer, prompt, prefill_seq_len, prompt_len=None):
    """Tokenize prompt and pad to a multiple of prefill_seq_len.

    If prompt_len is given the input is padded to at least that many tokens.
    """
    inputs = tokenizer(prompt, return_tensors="np", padding=True)
    actual_len = int(inputs["attention_mask"].sum(1).max())
    if prompt_len is not None and prompt_len < actual_len:
        raise ValueError(
            f"--prompt-len {prompt_len} is shorter than the tokenized prompt ({actual_len} tokens)"
        )
    effective_len = prompt_len if prompt_len is not None else actual_len
    num_chunks = -(effective_len // -prefill_seq_len)  # ceil divide
    padded_len = num_chunks * prefill_seq_len
    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
    inputs.pop("token_type_ids", None)
    return inputs, effective_len, num_chunks


def run_chunked_prefill(prefill_session, inputs, num_chunks, prefill_seq_len, num_hidden_layers):
    """Run chunked prefill, accumulating KV states into inputs. Returns final chunk output."""
    qpc_out = None
    for i in range(num_chunks):
        chunk = {
            "input_ids": inputs["input_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len],
            "position_ids": inputs["position_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len],
        }
        t0 = time.time()
        qpc_out = prefill_session.run(chunk)
        print(f"    chunk {i + 1}/{num_chunks}: {time.time() - t0:.3f}s")
        for layer in range(num_hidden_layers):
            inputs[f"past_key.{layer}"] = qpc_out[f"past_key.{layer}_RetainedState"]
            inputs[f"past_value.{layer}"] = qpc_out[f"past_value.{layer}_RetainedState"]
    return qpc_out


def run_decode_loop(decode_session, decode_inputs, generation_len, num_hidden_layers):
    """Autoregressive decode loop. Returns (token list, elapsed seconds)."""
    all_tokens = []
    st = time.time()
    for _ in range(generation_len):
        out = decode_session.run(decode_inputs)
        next_token = int(np.argmax(out["logits"]))
        all_tokens.append(next_token)
        decode_inputs["input_ids"] = np.array([[next_token]])
        decode_inputs["position_ids"] = decode_inputs["position_ids"] + 1
        for layer in range(num_hidden_layers):
            decode_inputs[f"past_key.{layer}"] = out[f"past_key.{layer}_RetainedState"]
            decode_inputs[f"past_value.{layer}"] = out[f"past_value.{layer}_RetainedState"]
    return all_tokens, time.time() - st


def build_decode_inputs(qpc_out, inputs, num_hidden_layers):
    """Assemble decode inputs from the last prefill chunk output."""
    decode_inputs = {
        "input_ids": np.argmax(qpc_out["logits"]).reshape(1, 1),
        "position_ids": np.max(inputs["position_ids"]).reshape(1, 1) + 1,
    }
    for layer in range(num_hidden_layers):
        decode_inputs[f"past_key.{layer}"] = qpc_out[f"past_key.{layer}_RetainedState"]
        decode_inputs[f"past_value.{layer}"] = qpc_out[f"past_value.{layer}_RetainedState"]
    return decode_inputs


def main():
    args = parse_args()

    headpar_split = args.headpar_split if args.headpar_split is not None else args.num_cores

    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    num_hidden_layers = args.num_layers if args.num_layers else config.num_hidden_layers
    from_pretrained_kwargs = {"num_hidden_layers": num_hidden_layers} if args.num_layers else {}

    generation_len = args.generation_len

    # ── qaic configs ──────────────────────────────────────────────────────────
    # Decode: standard blocked head-parallel KV attention
    decode_qaic_config = {
        "blocking_mode": "kv",
        "num_kv_blocks": args.num_kv_blocks,
        "kv_blocking_headpar_split": headpar_split,
    }
    # Prefill: blocked head-parallel prefill attention
    # prefill_headpar=True routes the prefill attention through
    # prefill_blocked_attention_interface ->
    # blocked_kv_attention_forward_prefill_headpar_offline
    prefill_qaic_config = {
        "blocking_mode": "kv",
        "num_kv_blocks": args.num_kv_blocks,
        "kv_blocking_headpar_split": headpar_split,
        "prefill_block_chunks": 2,
    }

    compile_kwargs = dict(
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        use_onnx_subfunctions=args.subf,
        retain_full_kv=True,
    )

    # ── Compile decode model ──────────────────────────────────────────────────
    print("\n[1/2] Compiling decode model (blocked head-par KV)...")
    decode_model = QEFFAutoModelForCausalLM.from_pretrained(args.model_name, **from_pretrained_kwargs)
    decode_qpc_path = decode_model.compile(
        prefill_seq_len=1,
        ctx_len=args.ctx_len,
        qaic_config=decode_qaic_config,
        user_tiled=True,
        **compile_kwargs,
    )
    print(f"  -> {decode_qpc_path}")

    # ── Compile prefill model ─────────────────────────────────────────────────
    print("\n[2/2] Compiling prefill model (blocked head-par prefill)...")
    prefill_model = QEFFAutoModelForCausalLM.from_pretrained(args.model_name, **from_pretrained_kwargs)
    prefill_qpc_path = prefill_model.compile(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        qaic_config=prefill_qaic_config,
        prefill_only=True,
        enable_chunking=True,
        user_tiled=True,
        moe_prefill_packed_chunk_size=256,
        **compile_kwargs,
    )
    print(f"  -> {prefill_qpc_path}")

    # ── Optionally compile non-blocked baseline ───────────────────────────────
    baseline_decode_qpc = baseline_prefill_qpc = None
    if args.compare_non_blocked:
        print("\n[2b] Compiling non-blocked baseline...")
        baseline_decode_model = QEFFAutoModelForCausalLM.from_pretrained(args.model_name, **from_pretrained_kwargs)
        baseline_decode_qpc = baseline_decode_model.compile(
            prefill_seq_len=1,
            ctx_len=args.ctx_len,
            aic_enable_depth_first=True,
            **compile_kwargs,
        )
        baseline_prefill_model = QEFFAutoModelForCausalLM.from_pretrained(args.model_name, **from_pretrained_kwargs)
        baseline_prefill_qpc = baseline_prefill_model.compile(
            prefill_seq_len=args.prefill_seq_len,
            ctx_len=args.ctx_len,
            prefill_only=True,
            enable_chunking=True,
            aic_enable_depth_first=True,
            **compile_kwargs,
        )
        print(f"  -> decode:  {baseline_decode_qpc}")
        print(f"  -> prefill: {baseline_prefill_qpc}")

    # ── Prepare inputs ────────────────────────────────────────────────────────
    inputs, prompt_len, num_chunks = prepare_chunked_inputs(
        tokenizer, args.prompt, args.prefill_seq_len, prompt_len=args.prompt_len
    )
    generation_len = min(generation_len, args.ctx_len - prompt_len - 1)
    print(f"\nPrompt tokens: {prompt_len}, chunks: {num_chunks}, decode steps: {generation_len}")

    # ── Run blocked head-par prefill + decode ─────────────────────────────────
    print("\n--- Blocked head-par prefill ---")
    decode_session = QAICInferenceSession(decode_qpc_path)
    prefill_session = QAICInferenceSession(prefill_qpc_path)

    t0 = time.time()
    qpc_out = run_chunked_prefill(prefill_session, inputs, num_chunks, args.prefill_seq_len, num_hidden_layers)
    t_prefill = time.time() - t0

    decode_inputs = build_decode_inputs(qpc_out, inputs, num_hidden_layers)
    print(f"\n--- Blocked head-par decode ({generation_len} tokens) ---")
    tokens, t_decode = run_decode_loop(decode_session, decode_inputs, generation_len, num_hidden_layers)
    blocked_text = tokenizer.decode([int(np.argmax(qpc_out["logits"]))] + tokens)

    # ── Optionally run baseline ───────────────────────────────────────────────
    t_baseline_prefill = t_baseline_decode = None
    baseline_text = None
    if args.compare_non_blocked:
        inputs_bl, _, _ = prepare_chunked_inputs(
            tokenizer, args.prompt, args.prefill_seq_len, prompt_len=args.prompt_len
        )
        baseline_decode_session = QAICInferenceSession(baseline_decode_qpc)
        baseline_prefill_session = QAICInferenceSession(baseline_prefill_qpc)

        print("\n--- Baseline prefill ---")
        t0 = time.time()
        qpc_out_bl = run_chunked_prefill(
            baseline_prefill_session, inputs_bl, num_chunks, args.prefill_seq_len, num_hidden_layers
        )
        t_baseline_prefill = time.time() - t0

        decode_inputs_bl = build_decode_inputs(qpc_out_bl, inputs_bl, num_hidden_layers)
        print(f"\n--- Baseline decode ({generation_len} tokens) ---")
        tokens_bl, t_baseline_decode = run_decode_loop(
            baseline_decode_session, decode_inputs_bl, generation_len, num_hidden_layers
        )
        baseline_text = tokenizer.decode([int(np.argmax(qpc_out_bl["logits"]))] + tokens_bl)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Prompt: {args.prompt}\n")
    print(f"[Blocked head-par]  prefill {t_prefill:.3f}s | decode {generation_len / t_decode:.1f} tok/s")
    print(f"Output: {blocked_text}")
    if args.compare_non_blocked:
        print(f"\n[Baseline]          prefill {t_baseline_prefill:.3f}s | decode {generation_len / t_baseline_decode:.1f} tok/s")
        print(f"Output: {baseline_text}")
        print(f"\nPrefill speedup: {t_baseline_prefill / t_prefill:.2f}x")
        print(f"Decode  speedup: {(generation_len / t_decode) / (generation_len / t_baseline_decode):.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
