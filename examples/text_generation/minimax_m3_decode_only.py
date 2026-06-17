# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse

import torch
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

MODEL_ID = "MiniMaxAI/MiniMax-M3"


def main():
    parser = argparse.ArgumentParser(description="Compile and generate with MiniMax-M3 decode-only PL=1.")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--num-cores", type=int, default=4)
    parser.add_argument("--generation-len", type=int, default=16)
    parser.add_argument("--prompt", default="Once upon a time,")
    parser.add_argument("--layerwise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--layerwise-window-size", type=int, default=1)
    parser.add_argument("--blocking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-kv-blocks", type=int, default=2)
    parser.add_argument("--mxfp6-matmul", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    qaic_config = None
    if args.blocking:
        qaic_config = {"enable_blocking": True, "blocking_mode": "kv", "num_kv_blocks": args.num_kv_blocks}

    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.float16,
        layerwise=args.layerwise,
        qaic_config=qaic_config,
    )

    qeff_model.compile(
        prefill_seq_len=1,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        mxfp6_matmul=args.mxfp6_matmul,
        num_devices=args.num_devices,
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
        retain_full_kv=True,
        layerwise=args.layerwise,
        layerwise_window_size=args.layerwise_window_size,
        qaic_config=qaic_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id or args.model_id)
    qeff_model.generate(prompts=[args.prompt], tokenizer=tokenizer, generation_len=args.generation_len)


if __name__ == "__main__":
    main()
