# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Weight-Free Continuous Batching Verification Script
====================================================
End-to-end smoke-test for weight-free dynamo export with continuous batching (CB):

  1. Build QEFFAutoModelForCausalLM from config on meta device (no weights in RAM)
  2. compile() — weight-free ONNX export + QAIC compile with full_batch_size
  3. generate() — continuous batching inference across multiple prompts

Usage
-----
# Weight-free + continuous batching (default):
python examples/text_generation/continuous_batching_weightfree.py \\
    --model_name Qwen/Qwen2-1.5B-Instruct \\
    --full_batch_size 4

# Smaller test (fewer layers):
python examples/text_generation/continuous_batching_weightfree.py \\
    --model_name Qwen/Qwen2-1.5B-Instruct \\
    --layers 4 \\
    --full_batch_size 4

# Regular dynamo (load real weights) + continuous batching:
python examples/text_generation/continuous_batching_weightfree.py \\
    --model_name Qwen/Qwen2-1.5B-Instruct \\
    --no_weight_free \\
    --full_batch_size 4
"""

import argparse
import time
from pathlib import Path

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM


DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a short poem about the ocean.",
    "What are the benefits of exercise?",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", default="tiiuae/falcon-7b", help="HuggingFace model ID or local path.")
    p.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Pipe-separated prompts. Defaults to 4 built-in prompts.",
    )
    p.add_argument("--prefill_seq_len", type=int, default=32, help="Prefill sequence length.")
    p.add_argument("--ctx_len", type=int, default=256, help="KV cache context length.")
    p.add_argument("--full_batch_size", type=int, default=4, help="CB pool size (number of concurrent slots).")
    p.add_argument("--generation_len", type=int, default=50, help="Tokens to generate per prompt.")
    p.add_argument("--num_cores", type=int, default=16, help="Number of QAIC cores.")
    p.add_argument("--num_devices", type=int, default=1, help="Number of QAIC devices.")
    p.add_argument("--layers", type=int, default=None, help="Override num_hidden_layers for fast testing.")
    p.add_argument("--output_dir", default="test_models/cb_weightfree", help="Output directory for ONNX and QPC.")
    p.add_argument("--no_weight_free", action="store_true", help="Load real weights (regular dynamo, no weight-free).")
    p.add_argument("--no_subfunctions", action="store_true", help="Disable ONNX subfunction extraction.")
    p.add_argument("--mxfp6_matmul", action="store_true", help="Enable MXFP6 matmul quantisation.")
    p.add_argument("--mxint8_kv_cache", action="store_true", help="Enable MXINT8 KV cache quantisation.")
    return p.parse_args()


def main():
    args = parse_args()

    use_weight_free = not args.no_weight_free
    use_subfunctions = not args.no_subfunctions
    prompts = args.prompts.split("|") if args.prompts else DEFAULT_PROMPTS

    print(f"\nModel       : {args.model_name}")
    print(f"Mode        : {'weight-free' if use_weight_free else 'regular'} + dynamo")
    print(f"CB pool     : full_batch_size={args.full_batch_size}")
    print(f"Prompts     : {len(prompts)}")
    print(f"Subfunctions: {'enabled' if use_subfunctions else 'disabled'}")
    print(f"Layers      : {args.layers or 'all'}")

    # ── Tokenizer + config ────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Prefer loading config WITHOUT trust_remote_code so we get the standard
    # transformers config with all required attributes. Legacy configs (e.g. Falcon-7B)
    # loaded via trust_remote_code are missing attributes like ffn_hidden_size and
    # activation that the built-in implementation expects — affects both
    # from_config() (weight-free) and from_pretrained() (regular) paths.
    config = None
    try:
        config = AutoConfig.from_pretrained(args.model_name)
    except Exception:
        pass

    if config is None:
        config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)

    config.dtype = torch.float32

    if not hasattr(config, "max_position_embeddings"):
        config.max_position_embeddings = getattr(config, "n_positions", 2048)

    if args.layers is not None:
        print(f"\nOverriding num_hidden_layers → {args.layers}")
        config.num_hidden_layers = args.layers

    # ── Build QEff model ─────────────────────────────────────────────────────
    if use_weight_free:
        print("\nBuilding meta-device model for weight-free export ...")
        with init_empty_weights():
            meta_model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
        model = QEFFAutoModelForCausalLM(
            meta_model,
            pretrained_model_name_or_path=args.model_name,
            continuous_batching=True,
        )
    else:
        print("\nLoading model via from_pretrained (regular dynamo) ...")
        model = QEFFAutoModelForCausalLM.from_pretrained(
            args.model_name,
            attn_implementation="eager",
            config=config,
            continuous_batching=True,
        )

    # ── Compile ───────────────────────────────────────────────────────────────
    print(f"\nCompiling (export + QAIC compile) ...")
    print(f"  prefill_seq_len={args.prefill_seq_len}, ctx_len={args.ctx_len}, "
          f"full_batch_size={args.full_batch_size}")

    t_start = time.perf_counter()

    qpc_path = model.compile(
        compile_dir=args.output_dir,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        full_batch_size=args.full_batch_size,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        mxfp6_matmul=args.mxfp6_matmul,
        mxint8_kv_cache=args.mxint8_kv_cache,
        use_dynamo=True,
        use_onnx_subfunctions=use_subfunctions,
        use_weight_free_export=use_weight_free,
    )

    t_compile = time.perf_counter() - t_start
    print(f"\nCompile time : {t_compile:.1f} sec  ({t_compile/60:.1f} min)")
    print(f"QPC path     : {qpc_path}")

    # ── Continuous batching inference ─────────────────────────────────────────
    print(f"\n--- Continuous batching inference ({len(prompts)} prompts) ---")
    for i, p in enumerate(prompts):
        print(f"  Prompt {i+1}: {p!r}")

    try:
        t_gen = time.perf_counter()
        exec_info = model.generate(
            tokenizer=tokenizer,
            prompts=prompts,
            generation_len=args.generation_len,
            automation=True,
        )
        t_gen = time.perf_counter() - t_gen

        print(f"\nGeneration time: {t_gen:.2f} sec")
        print("\n" + "=" * 70)
        for i, (prompt, generated) in enumerate(zip(prompts, exec_info.generated_texts)):
            print(f"\nPrompt {i+1}: {prompt}")
            print(f"Output  : {generated}")
            print("-" * 70)

    except RuntimeError as exc:
        print(f"\nQPC inference failed: {exc}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Model          : {args.model_name}")
    print(f"  Layers         : {config.num_hidden_layers}")
    print(f"  Mode           : {'weight-free' if use_weight_free else 'regular'} + dynamo")
    print(f"  full_batch_size: {args.full_batch_size}")
    print(f"  ctx_len        : {args.ctx_len}")
    print(f"  Compile time   : {t_compile:.1f} sec")
    print(f"  QPC            : {qpc_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
