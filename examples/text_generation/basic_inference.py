# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Basic text generation inference")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-1.5B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--num-hidden-layers", type=int, default=-1, help="Num hidden layers in the model")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Input prompt")
    parser.add_argument("--prefill-seq-len", type=int, default=32, help="Prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=128, help="Context length")
    parser.add_argument("--dynamo", action="store_true", help="Export via dynamo")
    parser.add_argument(
        "--use-weight-free-export",
        action="store_true",
        help="Export via weight-free (meta-device, no weights in RAM during export)",
    )
    parser.add_argument("--use-onnx-subfunctions", action="store_true", help="Use subfunctions while exporting")
    parser.add_argument("--generation-len", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--num-cores", type=int, default=16, help="Number of cores")
    parser.add_argument("--aic-hw-version", type=str, default="ai100", help="Version of aic hardware")
    parser.add_argument(
        "--device-group",
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        default=None,
        help="Device IDs (comma-separated) e.g. [0,1]",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    if args.num_hidden_layers > 0:
        config.num_hidden_layers = args.num_hidden_layers

    if args.use_weight_free_export:
        # Build on meta device — no weights loaded into RAM.
        # The QAIC compiler loads weights from the checkpoint at compile time.
        config.dtype = torch.float16
        with init_empty_weights():
            hf_model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
        model = QEFFAutoModelForCausalLM(hf_model, pretrained_model_name_or_path=args.model_name)
    else:
        model = QEFFAutoModelForCausalLM.from_pretrained(args.model_name, config=config)

    qpc_path = model.compile(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        aic_hw_version=args.aic_hw_version,
        num_devices=(1 if args.device_group is None else len(args.device_group)),
        dynamo=args.dynamo,
        use_weight_free_export=args.use_weight_free_export,
        use_onnx_subfunctions=args.use_onnx_subfunctions,
    )
    print(f"Model compiled to: {qpc_path}")

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
