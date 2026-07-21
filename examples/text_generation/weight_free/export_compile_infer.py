# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Weight-Free Export Verification Script
=======================================
End-to-end smoke-test for dynamo export (weight-free or regular):

  1. Build QEFFAutoModelForCausalLM
       - weight-free: from config on meta device  (use_weight_free_export=True)
       - regular dynamo: from_pretrained with real weights (use_weight_free_export=False)
  2. compile() — calls export + QAIC compile in one step
  3. QPC inference on device

Timing is printed for compile and inference.
ORT and PyTorch baseline are intentionally omitted (use compare.py for parity checks).

Usage
-----
# Weight-free export (default)
python examples/text_generation/weight_free/export_compile_infer.py \\
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \\
    --layers 2 \\
    --output_dir llama_test

# Regular dynamo export (loads real weights)
python examples/text_generation/weight_free/export_compile_infer.py \\
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \\
    --no_weight_free \\
    --layers 2 \\
    --output_dir llama_dynamo_test
"""

import argparse
import time
from pathlib import Path

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", required=True, help="HuggingFace model ID or local checkpoint path.")
    p.add_argument("--prompt", default="What is faith?", help="Prompt string for inference.")
    p.add_argument("--prompt_len", type=int, default=32, help="Padded prompt length (tokens).")
    p.add_argument("--ctx_len", type=int, default=256, help="KV cache context length (tokens).")
    p.add_argument("--prefill_seq_len", type=int, default=1, help="Prefill sequence length for compile.")
    p.add_argument("--num_cores", type=int, default=16, help="Number of QAIC cores.")
    p.add_argument("--num_devices", type=int, default=1, help="Number of QAIC devices.")
    p.add_argument("--output_dir", default="test_models/weightfree_verify", help="Output directory for ONNX and QPC.")
    p.add_argument("--layers", type=int, default=None, help="Override num_hidden_layers for fast testing.")
    p.add_argument("--no_weight_free", action="store_true", help="Use regular dynamo export instead of weight-free.")
    p.add_argument("--no_dynamo", action="store_true", help="Use TorchScript export (no dynamo).")
    p.add_argument("--no_subfunctions", action="store_true", help="Disable ONNX subfunction extraction.")
    p.add_argument("--mxfp6_matmul", action="store_true", help="Enable MXFP6 matmul quantisation.")
    p.add_argument("--mxint8_kv_cache", action="store_true", help="Enable MXINT8 KV cache quantisation.")
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16", help="Dtype for model weights (float16 or float32).")
    return p.parse_args()


def main():
    args = parse_args()

    use_weight_free = not args.no_weight_free
    use_dynamo = not args.no_dynamo
    use_subfunctions = not args.no_subfunctions
    output_dir = Path(args.output_dir)

    # ── Config + tokenizer ────────────────────────────────────────────────────
    print(f"\nModel: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Prefer loading config WITHOUT trust_remote_code so we get the standard
    # transformers config with all required attributes. Legacy configs (e.g. Falcon-7B)
    # loaded via trust_remote_code are missing attributes like ffn_hidden_size and
    # activation that the built-in implementation expects — this affects both
    # from_config() (weight-free) and from_pretrained() (regular) paths.
    # Fall back to trust_remote_code for models not yet in mainline transformers.
    config = None
    try:
        config = AutoConfig.from_pretrained(args.model_name)
    except Exception:
        pass

    if config is None:
        config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)

    config.dtype = getattr(torch, args.dtype)

    # Some older models (e.g. Falcon-7B) predate the max_position_embeddings attribute.
    # Newer transformers requires it — set a safe default if missing.
    if not hasattr(config, "max_position_embeddings"):
        config.max_position_embeddings = getattr(config, "n_positions", 2048)
        print(f"Note: max_position_embeddings not in config, defaulting to {config.max_position_embeddings}")

    if args.layers is not None:
        print(f"Overriding num_hidden_layers: {config.num_hidden_layers} → {args.layers}")
        config.num_hidden_layers = args.layers

    print(config)

    # ── Build QEff model ─────────────────────────────────────────────────────
    # Weight-free: build from config on meta device (no weights in memory)
    # Regular dynamo: load real weights via from_pretrained
    if use_weight_free:
        print("\nBuilding meta-device model for weight-free export ...")
        with init_empty_weights():
            meta_model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
        qeff_model = QEFFAutoModelForCausalLM(
            meta_model,
            pretrained_model_name_or_path=args.model_name,
            enable_proxy=False,
        )
    else:
        print("\nLoading model via from_pretrained for regular dynamo export ...")
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            args.model_name,
            attn_implementation="eager",
            config=config,
        )

    export_mode = "weight-free" if use_weight_free else "regular"
    dynamo_mode = "dynamo" if use_dynamo else "torchscript"
    print(f"\nExport mode : {export_mode} + {dynamo_mode}")
    print(f"Subfunctions: {'enabled' if use_subfunctions else 'disabled'}")

    # ── Compile (export + compile in one step) ────────────────────────────────
    print("\nCompiling (export + QAIC compile) ...")
    t_start = time.perf_counter()
    qpc_path = qeff_model.compile(
        compile_dir=str(output_dir),
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_devices=args.num_devices,
        num_cores=args.num_cores,
        mxfp6_matmul=args.mxfp6_matmul,
        mxint8_kv_cache=args.mxint8_kv_cache,
        use_dynamo=use_dynamo,
        use_onnx_subfunctions=use_subfunctions,
        use_weight_free_export=use_weight_free,
    )

    t_compile = time.perf_counter() - t_start
    print(f"\nCompile time : {t_compile:.1f} sec  ({t_compile / 60:.1f} min)")
    print(f"QPC path     : {qpc_path}")

    # ── QPC inference ─────────────────────────────────────────────────────────
    print("\n--- QPC inference ---")
    print(f"Prompt: {args.prompt!r}")

    try:
        t_gen = time.perf_counter()
        exec_info = qeff_model.generate(
            prompts=[args.prompt],
            tokenizer=tokenizer,
            automation=True,
            generation_len=args.ctx_len - args.prompt_len,
        )
        t_gen = time.perf_counter() - t_gen

        # exec_info.generated_ids[0] is shape (batch, tokens) — flatten to get token list
        import numpy as np

        generated_ids_array = np.asarray(exec_info.generated_ids[0]).flatten()
        generated_text = tokenizer.decode(generated_ids_array.tolist(), skip_special_tokens=True)

        print(f"Generation time : {t_gen:.2f} sec")
        print(f"Generated tokens: {len(generated_ids_array)}")
        print(f"Output: {generated_text}")

    except RuntimeError as exc:
        print(f"QPC inference failed: {exc}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Model       : {args.model_name}")
    print(f"  Layers      : {config.num_hidden_layers}")
    print(f"  Export mode : {export_mode} + {dynamo_mode}")
    print(f"  ctx_len     : {args.ctx_len}")
    print(f"  Compile time: {t_compile:.1f} sec")
    print(f"  QPC         : {qpc_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
