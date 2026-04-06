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
    parser = argparse.ArgumentParser(description="MoE model inference")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="HuggingFace MoE model ID",
    )
    parser.add_argument("--prompt", type=str, default="Explain quantum computing", help="Input prompt")
    parser.add_argument("--prefill-seq-len", type=int, default=32, help="Prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=256, help="Context length")
    parser.add_argument("--generation-len", type=int, default=None, help="Number of tokens to generate")
    parser.add_argument("--num-cores", type=int, default=16, help="Number of cores")
    parser.add_argument(
        "--device-group",
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        default=None,
        help="Device IDs (comma-separated) e.g. [0,1]",
    )
    args = parser.parse_args()

    print(f"Loading MoE model: {args.model_name}")
    print("Note: MoE models use sparse expert activation for efficient inference")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = QEFFAutoModelForCausalLM.from_pretrained(args.model_name)

    # Compile the model
    qpc_path = model.compile(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=(1 if args.device_group is None else len(args.device_group)),
    )
    print(f"Model compiled to: {qpc_path}")

    # Generate text
    exec_info = model.generate(
        tokenizer=tokenizer,
        prompts=[args.prompt],
        device_id=args.device_group,
        generation_len=args.generation_len,
    )

    print(f"\nPrompt: {args.prompt}")
    print(f"Generated: {exec_info.generated_texts[0]}")


if __name__ == "__main__":
    main()
