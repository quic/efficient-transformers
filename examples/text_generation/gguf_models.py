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
    parser = argparse.ArgumentParser(description="GGUF model inference")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2-1.5B-Instruct-GGUF",
        help="HuggingFace model ID for GGUF model",
    )
    parser.add_argument(
        "--gguf-file",
        type=str,
        default="qwen2-1_5b-instruct-q8_0.gguf",
        help="GGUF file name within the model repository",
    )
    parser.add_argument("--prompt", type=str, default="Hello! How are you?", help="Input prompt")
    parser.add_argument("--prefill-seq-len", type=int, default=32, help="Prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=128, help="Context length")
    parser.add_argument("--num-cores", type=int, default=16, help="Number of cores")
    parser.add_argument("--num-devices", type=int, default=1, help="Number of devices")
    args = parser.parse_args()

    # Load the model and tokenizer
    print(f"Loading GGUF model: {args.model_name}")
    print(f"GGUF file: {args.gguf_file}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, gguf_file=args.gguf_file)
    model = QEFFAutoModelForCausalLM.from_pretrained(args.model_name, gguf_file=args.gguf_file)

    # Compile the model
    generated_qpc_path = model.compile(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
    )
    print(f"Model compiled to: {generated_qpc_path}")

    # Generate text
    exec_info = model.generate(prompts=[args.prompt], tokenizer=tokenizer)
    print(f"\nPrompt: {args.prompt}")
    print(f"Generated: {exec_info.generated_texts[0]}")


if __name__ == "__main__":
    main()
