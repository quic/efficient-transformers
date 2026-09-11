# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils import constants


def main():
    parser = argparse.ArgumentParser(description="Basic text generation inference")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-1.5B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--num-hidden-layers", type=int, default=-1, help="Num hidden layers in the model")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Input prompt")
    parser.add_argument("--prefill-seq-len", type=int, default=32, help="Prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=128, help="Context length")
    parser.add_argument("--dynamo", action="store_true", help="Export via dynamo")
    parser.add_argument("--use-onnx-subfunctions", action="store_true", help="Use subfunctions while exporting")
    parser.add_argument("--generation-len", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--num-cores", type=int, default=constants.DEFAULT_AIC_NUM_CORES, help="Number of cores")
    parser.add_argument(
        "--aic-hw-version", type=str, default=constants.DEFAULT_AIC_HW_VERSION, help="Version of aic hardware"
    )
    parser.add_argument(
        "--device-group",
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        default=None,
        help="Device IDs (comma-separated) e.g. [0,1]",
    )
    parser.add_argument(
        "--profiling-type",
        dest="profiling_type",
        type=str,
        default=None,
        choices=["latency", "trace", "raw_device_stats", "stats"],
        help="Enable QAIC device profiling via the runtime Program/ExecObj profiling API and capture a "
        "report for the generate() call. Default: disabled",
    )
    parser.add_argument(
        "--profiling-output-dir",
        dest="profiling_output_dir",
        type=str,
        default=None,
        help="Directory to write the profiling report to. Default: <qpc dir>/profiling_output",
    )
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    if args.num_hidden_layers > 0:
        config.num_hidden_layers = args.num_hidden_layers
    model = QEFFAutoModelForCausalLM.from_pretrained(args.model_name, config=config)

    # Compile the model
    qpc_path = model.compile(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        stats_level=100,
        aic_hw_version=args.aic_hw_version,
        num_devices=(1 if args.device_group is None else len(args.device_group)),
        dynamo=args.dynamo,
        use_onnx_subfunctions=args.use_onnx_subfunctions,
    )
    print(f"Model compiled to: {qpc_path}")

    # Generate text
    generate_kwargs = {
        "tokenizer": tokenizer,
        "prompts": [args.prompt],
        "device_id": args.device_group,
        "generation_len": args.generation_len,
    }
    if args.profiling_type is not None:
        generate_kwargs["profiling_type"] = args.profiling_type
        if args.profiling_output_dir is not None:
            generate_kwargs["profiling_output_dir"] = args.profiling_output_dir
    exec_info = model.generate(**generate_kwargs)

    print(f"\nPrompt: {args.prompt}")
    print(f"Generated: {exec_info.generated_texts[0]}")

    if args.profiling_type is not None:
        output_dir = (
            Path(args.profiling_output_dir)
            if args.profiling_output_dir
            else (Path(qpc_path) if Path(qpc_path).is_dir() else Path(qpc_path).parent) / "profiling_output"
        )
        report_files = sorted(p.name for p in output_dir.glob("*") if p.is_file())
        print(f"\nProfiling type: {args.profiling_type}")
        print(f"Profiling report directory: {output_dir}")
        print(f"Profiling report files: {report_files if report_files else '(none found)'}")


if __name__ == "__main__":
    main()
