# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Dynamo CausalLM example using KV head-parallel blocking with ONNX Loop export.

Requires PyTorch >= 2.13. Install dependencies before running:
    pip install -r examples/dynamo/causal_lm/requirements.txt
"""

import argparse

from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils import constants


def main():
    parser = argparse.ArgumentParser(
        description="Dynamo CausalLM export with kv_headpar blocking represented as ONNX Loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-1.5B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--num-hidden-layers", type=int, default=-1, help="Override number of hidden layers")
    parser.add_argument("--prompt", type=str, default="My name is", help="Input prompt for generation")
    parser.add_argument("--prefill-seq-len", type=int, default=32, help="Prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=128, help="Context (KV-cache) length")
    parser.add_argument("--generation-len", type=int, default=100, help="Number of new tokens to generate")
    parser.add_argument("--num-kv-blocks", type=int, default=2, help="Number of KV blocks")
    parser.add_argument("--headpar-split", type=int, default=None, help="Override split count for kv_headpar")
    parser.add_argument(
        "--dynamic-trip-count",
        action="store_true",
        help="Use runtime live-block count as the ONNX Loop bound instead of static max blocks",
    )
    parser.add_argument("--num-cores", type=int, default=constants.DEFAULT_AIC_NUM_CORES, help="Number of AI 100 cores")
    parser.add_argument(
        "--aic-hw-version", type=str, default=constants.DEFAULT_AIC_HW_VERSION, help="AIC hardware version"
    )
    parser.add_argument(
        "--enable-onnx-subfunctions",
        action="store_true",
        help="Enable ONNX subfunction extraction during dynamo export",
    )
    parser.add_argument(
        "--weight-free",
        action="store_true",
        help="Build the model on meta tensors and load weights at compile time",
    )
    parser.add_argument(
        "--device-group",
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        default=None,
        help="Device IDs (comma-separated), e.g. [0,1]",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    if args.num_hidden_layers > 0:
        config.num_hidden_layers = args.num_hidden_layers

    model = QEFFAutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        weight_free=args.weight_free,
    )

    qaic_config = {
        "blocking_mode": "kv_headpar",
        "num_kv_blocks": args.num_kv_blocks,
        "kv_loop_dynamic_trip_count": args.dynamic_trip_count,
    }
    if args.headpar_split is not None:
        qaic_config["headpar_split"] = args.headpar_split

    qpc_path = model.compile(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        aic_hw_version=args.aic_hw_version,
        num_devices=(1 if args.device_group is None else len(args.device_group)),
        qaic_config=qaic_config,
        dynamo=True,
        use_onnx_subfunctions=args.enable_onnx_subfunctions,
    )
    print(f"Model compiled to: {qpc_path}")

    exec_info = model.generate(
        tokenizer=tokenizer,
        prompts=[args.prompt],
        device_id=args.device_group,
        generation_len=args.generation_len,
    )

    print(f"\nPrompt   : {args.prompt}")
    print(f"Generated: {exec_info.generated_texts[0]}")


if __name__ == "__main__":
    main()
