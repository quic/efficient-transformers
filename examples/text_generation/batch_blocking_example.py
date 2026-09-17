# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse

import numpy as np
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM


def assert_generated_tokens_match(reference_exec_info, blocked_exec_info, label):
    reference_ids = np.asarray(reference_exec_info.generated_ids)
    blocked_ids = np.asarray(blocked_exec_info.generated_ids)
    if np.array_equal(reference_ids, blocked_ids):
        print(f"Token comparison ({label} vs non-blocked): MATCH")
        print(f"Generated token IDs: {blocked_ids.tolist()}")
        return

    mismatch = np.argwhere(reference_ids != blocked_ids)
    first_mismatch = tuple(mismatch[0].tolist()) if mismatch.size else None
    raise AssertionError(
        f"Token comparison ({label} vs non-blocked): MISMATCH at {first_mismatch}; "
        f"non-blocked={reference_ids.tolist()}, blocked={blocked_ids.tolist()}"
    )


def main():
    parser = argparse.ArgumentParser(description="Basic text generation inference")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B", help="HuggingFace model ID")
    parser.add_argument("--prompt", type=str, default="Hello", help="Input prompt")
    parser.add_argument("--prefill-seq-len", type=int, default=1, help="Prefill sequence length")
    parser.add_argument(
        "--ctx-len", type=int, default=32768, help="Context length high enough to force blocking computation"
    )
    parser.add_argument("--generation-len", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--num-cores", type=int, default=16, help="Number of cores")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--num-devices", type=int, default=8, help="Number of devices to compile for")
    parser.add_argument("--num-q-blocks", type=int, default=2, help="Number of query blocks for q/qkv/hqkv/bhqkv modes")
    parser.add_argument(
        "--num-kv-blocks", type=int, default=2, help="Number of KV blocks for kv/qkv/hkv/hqkv/bhqkv modes"
    )
    parser.add_argument("--head-block-size", type=int, default=8, help="Number of attention heads per head block")
    parser.add_argument("--num-batch-blocks", type=int, default=2, help="Number of batch blocks for bhqkv mode")
    parser.add_argument(
        "--device-group",
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        default=[36, 37, 38, 39, 40, 41, 42, 43],
        help="Device IDs (comma-separated) e.g. [0,1]",
    )
    parser.add_argument(
        "--blocking-mode",
        type=str,
        default="hqkv",
        help="Blocking mode, valid options: kv, q, h, qkv, hqkv, bhqkv",
    )
    parser.add_argument(
        "--compare-non-blocking",
        action="store_true",
        help="Compile and print results for non-blocked version of model as well",
    )
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = QEFFAutoModelForCausalLM.from_pretrained(args.model_name, num_hidden_layers=args.num_layers)

    if args.compare_non_blocking:
        # Compile the model
        qpc_path = model.compile(
            prefill_seq_len=args.prefill_seq_len,
            ctx_len=args.ctx_len,
            num_cores=args.num_cores,
            num_devices=args.num_devices,
        )
        print(f"Model compiled to: {qpc_path}")

        # Generate text
        exec_info = model.generate(
            tokenizer=tokenizer,
            prompts=[args.prompt],
            generation_len=args.generation_len,
        )

        print(f"\nPrompt: {args.prompt}")
        print(f"Generated: {exec_info.generated_texts[0]}")

    # setup qaic config to enable blocking, ensure 4 or more device ids are passed
    qaic_config = {
        "blocking_mode": args.blocking_mode,
        "num_batch_blocks": args.num_batch_blocks,
        "num_kv_blocks": args.num_kv_blocks,
        "num_q_blocks": args.num_q_blocks,
        "head_block_size": args.head_block_size,
    }
    model_blocked = QEFFAutoModelForCausalLM.from_pretrained(args.model_name, num_hidden_layers=args.num_layers)

    # Compile the model
    qpc_path_blocked = model_blocked.compile(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        qaic_config=qaic_config,
    )
    print(f"Model compiled to: {qpc_path_blocked}")

    # Generate text
    exec_info_blocked = model_blocked.generate(
        tokenizer=tokenizer,
        prompts=[args.prompt],
        generation_len=args.generation_len,
    )

    print(f"\nPrompt: {args.prompt}")
    print(f"Generated: {exec_info_blocked.generated_texts[0]}")

    if args.compare_non_blocking:
        assert_generated_tokens_match(exec_info, exec_info_blocked, qaic_config["blocking_mode"])
        print("Performance non-blocked:")
        print(exec_info)

    print("Performance blocked:")
    print(exec_info_blocked)


if __name__ == "__main__":
    main()
