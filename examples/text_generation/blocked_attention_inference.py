# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM


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
        help="Blocking mode, valid options: kv, q, h, qkv, hqkv",
    )
    parser.add_argument(
        "--compare-non-blocking",
        action="store_true",
        help="Compile and print results for non-blocked version of model as well",
    )
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.compare_non_blocking:
        model = QEFFAutoModelForCausalLM.from_pretrained(args.model_name, num_hidden_layers=2)

        # Compile the model
        qpc_path = model.compile(
            prefill_seq_len=args.prefill_seq_len,
            ctx_len=args.ctx_len,
            num_cores=args.num_cores,
            num_devices=8,
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
    qaic_config = {"enable_blocking": True, "blocking_mode": args.blocking_mode}
    model_blocked = QEFFAutoModelForCausalLM.from_pretrained(args.model_name, num_hidden_layers=2)

    # Compile the model
    qpc_path_blocked = model_blocked.compile(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=8,
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
        print("Performance non-blocked:")
        print(exec_info)

    print("Performance blocked:")
    print(exec_info_blocked)


if __name__ == "__main__":
    main()
