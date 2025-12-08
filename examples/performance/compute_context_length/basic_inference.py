# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Basic Compute Context Length (CCL) inference example.

This example demonstrates how to use CCL optimization for text generation models.
CCL allows using different context lengths during prefill and decode phases,
reducing memory footprint and computation for shorter sequences.
"""

import argparse

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Text generation with Compute Context Length (CCL) optimization")
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="My name is ",
        help="Input prompt for text generation",
    )
    parser.add_argument(
        "--prefill-seq-len",
        type=int,
        default=128,
        help="Prefill sequence length",
    )
    parser.add_argument(
        "--ctx-len",
        type=int,
        default=1024,
        help="Maximum context length",
    )
    parser.add_argument(
        "--ccl-enabled",
        action="store_true",
        help="Enable compute-context-length (CCL) feature",
    )
    parser.add_argument(
        "--comp-ctx-lengths-prefill",
        type=lambda x: [int(i) for i in x.split(",")],
        default="256,500",
        help="Comma-separated list of context lengths for prefill phase (e.g., '256,500')",
    )
    parser.add_argument(
        "--comp-ctx-lengths-decode",
        type=lambda x: [int(i) for i in x.split(",")],
        default="512,1024",
        help="Comma-separated list of context lengths for decode phase (e.g., '512,1024')",
    )
    parser.add_argument(
        "--generation-len",
        type=int,
        default=128,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=16,
        help="Number of cores for compilation",
    )
    parser.add_argument(
        "--num-devices",
        type=int,
        default=1,
        help="Number of devices to use",
    )
    parser.add_argument(
        "--continuous-batching",
        action="store_true",
        help="Enable continuous batching mode",
    )
    parser.add_argument(
        "--full-batch-size",
        type=int,
        default=1,
        help="Full batch size for continuous batching",
    )
    parser.add_argument(
        "--mxint8-kv-cache",
        action="store_true",
        default=True,
        help="Enable MX INT8 KV cache",
    )
    parser.add_argument(
        "--mxfp6-matmul",
        action="store_true",
        default=True,
        help="Enable MX FP6 matrix multiplication",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    print("CCL Configuration:")
    print(f"  - Prefill context lengths: {args.comp_ctx_lengths_prefill}")
    print(f"  - Decode context lengths: {args.comp_ctx_lengths_decode}")
    print(f"  - Max context length: {args.ctx_len}")
    print(f"  - Continuous batching: {args.continuous_batching}")

    # Load model with CCL configuration
    model = QEFFAutoModelForCausalLM.from_pretrained(
        args.model_name,
        continuous_batching=args.continuous_batching,
        qaic_config={
            "ccl_enabled": args.ccl_enabled,
        },
    )

    # Compile the model
    print("\nCompiling model...")
    compile_kwargs = {
        "prefill_seq_len": args.prefill_seq_len,
        "ctx_len": args.ctx_len,
        "num_cores": args.num_cores,
        "num_devices": args.num_devices,
        "mxint8_kv_cache": args.mxint8_kv_cache,
        "mxfp6_matmul": args.mxfp6_matmul,
    }

    if args.continuous_batching:
        compile_kwargs["full_batch_size"] = args.full_batch_size
    if args.ccl_enabled:
        compile_kwargs["comp_ctx_lengths_prefill"] = args.comp_ctx_lengths_prefill
        compile_kwargs["comp_ctx_lengths_decode"] = args.comp_ctx_lengths_decode

    qpc_path = model.compile(**compile_kwargs)
    print(f"Model compiled successfully to: {qpc_path}")

    # Load tokenizer and generate
    print("\nGenerating text...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    exec_info = model.generate(
        prompts=[args.prompt],
        tokenizer=tokenizer,
        generation_len=args.generation_len,
    )

    print(f"\nPrompt: {args.prompt}")
    print(f"Generated: {exec_info.generated_texts[0]}")


if __name__ == "__main__":
    main()
