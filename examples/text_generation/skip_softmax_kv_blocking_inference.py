# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
from pprint import pprint

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM


def _parse_device_group(device_ids):
    if device_ids is None:
        return None
    return [int(x) for x in device_ids.strip("[]").split(",") if x]


def main():
    parser = argparse.ArgumentParser(description="KV-blocked text generation with skip-softmax attention")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B", help="HuggingFace model ID")
    parser.add_argument("--prompt", type=str, default="Hello", help="Input prompt")
    parser.add_argument("--prefill-seq-len", type=int, default=1, help="Prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=32768, help="Context length used for compilation")
    parser.add_argument("--generation-len", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--num-cores", type=int, default=16, help="Number of cores")
    parser.add_argument("--aic-hw-version", type=str, default="ai100", help="AIC hardware version")
    parser.add_argument(
        "--device-group",
        type=_parse_device_group,
        default=None,
        help="Device IDs, e.g. [0] or [0,1,2,3]",
    )
    parser.add_argument("--num-kv-blocks", type=int, default=8, help="Number of KV blocks for blocked attention")
    parser.add_argument(
        "--kv-blocking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable KV-blocked attention. Use --no-kv-blocking for plain non-blocked baseline.",
    )
    parser.add_argument(
        "--skip-softmax",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable skip-softmax on top of KV blocking. Use --no-skip-softmax for KV-blocking baseline.",
    )
    parser.add_argument(
        "--skip-softmax-scale",
        type=float,
        default=None,
        help="Explicit BLASST scale. Effective lambda is scale / ctx_len and overrides phase scales.",
    )
    parser.add_argument(
        "--skip-softmax-prefill-scale",
        type=float,
        default=1.0,
        help="Prefill BLASST scale used when --skip-softmax-scale is not set",
    )
    parser.add_argument(
        "--skip-softmax-decode-scale",
        type=float,
        default=1.0,
        help="Decode BLASST scale used when --skip-softmax-scale is not set",
    )
    parser.add_argument(
        "--skip-softmax-min-keep-blocks",
        type=int,
        default=1,
        help="Minimum leading KV blocks to keep before applying skip-softmax",
    )
    args = parser.parse_args()

    qaic_config = None
    if args.kv_blocking:
        qaic_config = {
            "enable_blocking": True,
            "blocking_mode": "kv",
            "num_kv_blocks": args.num_kv_blocks,
            "skip_kv": True,
            "skip_softmax": args.skip_softmax,
            "skip_softmax_prefill_scale": args.skip_softmax_prefill_scale,
            "skip_softmax_decode_scale": args.skip_softmax_decode_scale,
            "skip_softmax_min_keep_blocks": args.skip_softmax_min_keep_blocks,
        }
        if args.skip_softmax_scale is not None:
            qaic_config["skip_softmax_scale"] = args.skip_softmax_scale

    print("qaic_config:")
    pprint(qaic_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = QEFFAutoModelForCausalLM.from_pretrained(args.model_name)

    qpc_path = model.compile(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        aic_hw_version=args.aic_hw_version,
        num_devices=(1 if args.device_group is None else len(args.device_group)),
        qaic_config=qaic_config,
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
    if not args.kv_blocking:
        perf_label = "plain non-blocked"
    elif args.skip_softmax:
        perf_label = "KV-blocked skip-softmax"
    else:
        perf_label = "KV-blocked no skip-softmax"
    print(f"Performance {perf_label}:")
    print(exec_info)


if __name__ == "__main__":
    main()
