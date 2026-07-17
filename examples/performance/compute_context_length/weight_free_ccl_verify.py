#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Single-model CCL (Compute Context Length) verification.
Called by run_ccl_verification.py for each model.

Steps:
  1. Load model (weight-free meta device OR from_pretrained)
  2. compile() with CCL specializations
  3. generate() and print output
  4. Print specializations.json to confirm CCL slots were compiled

Usage
-----
# Weight-free + CCL:
python examples/text_generation/ccl_verify_single.py \\
    --model_name Qwen/Qwen2-1.5B-Instruct \\
    --layers 4 \\
    --ctx_len 256 \\
    --ccl_values 64,128,256

# Regular from_pretrained + CCL:
python examples/text_generation/ccl_verify_single.py \\
    --model_name Qwen/Qwen2-1.5B-Instruct \\
    --no_weight_free \\
    --layers 4
"""

import argparse
import json
import time
from pathlib import Path

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils.constants import Constants


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", required=True)
    p.add_argument("--layers", type=int, default=None, help="Override num_hidden_layers.")
    p.add_argument("--ctx_len", type=int, default=4096)
    p.add_argument("--prefill_seq_len", type=int, default=1)
    p.add_argument(
        "--ccl_values",
        type=str,
        default="1024,2048,4096",
        help="Comma-separated CCL values, e.g. '1024,2048,4096'.",
    )
    p.add_argument("--num_cores", type=int, default=16)
    p.add_argument("--num_devices", type=int, default=1)
    p.add_argument("--no_weight_free", action="store_true")
    p.add_argument("--no_subfunctions", action="store_true")
    p.add_argument("--mxfp6_matmul", action="store_true")
    p.add_argument("--mxint8_kv_cache", action="store_true")
    p.add_argument("--output_dir", default="test_models/ccl_verify")
    return p.parse_args()


def main():
    args = parse_args()

    use_weight_free = not args.no_weight_free
    use_subfunctions = not args.no_subfunctions

    # Parse CCL values
    if args.ccl_values:
        ccl_list = [int(x) for x in args.ccl_values.split(",")]
    else:
        ctx = args.ctx_len
        ccl_list = sorted({max(ctx // 4, 1024), ctx // 2, ctx})

    print(f"\nModel        : {args.model_name}")
    print(f"Mode         : {'weight-free' if use_weight_free else 'regular'} + dynamo")
    print(f"ctx_len      : {args.ctx_len}")
    print(f"CCL values   : {ccl_list}")
    print(f"Layers       : {args.layers or 'all'}")

    # ── Tokenizer + config ────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

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
        print(f"Overriding num_hidden_layers → {args.layers}")
        config.num_hidden_layers = args.layers

    # ── Build QEff model ─────────────────────────────────────────────────────
    if use_weight_free:
        print("\nBuilding meta-device model (weight-free + CCL) ...")
        with init_empty_weights():
            meta_model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
        model = QEFFAutoModelForCausalLM(
            meta_model,
            pretrained_model_name_or_path=args.model_name,
            qaic_config={"ccl_enabled": True},
        )
    else:
        print("\nLoading model via from_pretrained (regular + CCL) ...")
        model = QEFFAutoModelForCausalLM.from_pretrained(
            args.model_name,
            attn_implementation="eager",
            config=config,
            ccl_enabled=True,
        )

    # ── Compile with CCL ─────────────────────────────────────────────────────
    print(f"\nCompiling with CCL specializations {ccl_list} ...")
    t_start = time.perf_counter()

    qpc_path = model.compile(
        compile_dir=args.output_dir,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        mxfp6_matmul=args.mxfp6_matmul,
        mxint8_kv_cache=args.mxint8_kv_cache,
        use_dynamo=True,
        use_onnx_subfunctions=use_subfunctions,
        use_weight_free_export=use_weight_free,
        comp_ctx_lengths_prefill=ccl_list,
        comp_ctx_lengths_decode=ccl_list,
    )

    t_compile = time.perf_counter() - t_start
    print(f"\nCompile time : {t_compile:.1f}s  ({t_compile / 60:.1f} min)")
    print(f"QPC path     : {qpc_path}")

    # ── Verify specializations.json ───────────────────────────────────────────
    spec_files = list(Path(args.output_dir).glob("**/specializations.json"))
    if spec_files:
        print(f"\nSpecializations ({spec_files[0]}):")
        specs = json.loads(spec_files[0].read_text())
        for s in specs.get("specializations", []):
            print(f"  {s['name']:15} → {s['symbols']}")
    else:
        print("\nWARNING: specializations.json not found")

    # ── Inference ─────────────────────────────────────────────────────────────
    print("\n--- Inference ---")
    prompts = Constants.INPUT_STR if isinstance(Constants.INPUT_STR, list) else [Constants.INPUT_STR]

    try:
        t_gen = time.perf_counter()
        exec_info = model.generate(
            tokenizer=tokenizer,
            prompts=prompts,
            generation_len=50,
            automation=True,
        )
        t_gen = time.perf_counter() - t_gen

        print(f"Generation time: {t_gen:.2f}s")
        for i, (prompt, text) in enumerate(zip(prompts, exec_info.generated_texts)):
            print(f"\nPrompt : {prompt!r}")
            print(f"Output : {text!r}")

    except RuntimeError as exc:
        print(f"Inference failed: {exc}")
        raise  # propagate so run_ccl_verification.py sees failure

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"CCL PASS: {args.model_name}")
    print(f"  ctx_len    : {args.ctx_len}")
    print(f"  CCL values : {ccl_list}")
    print(f"  Compile    : {t_compile:.1f}s")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
