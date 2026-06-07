# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Compile and run GLM-MoE-DSA causal LM generation on AI 100.")
    parser.add_argument("--model-id", default="tiny-random/glm-5.1")
    parser.add_argument("--ctx-len", type=int, default=32)
    parser.add_argument("--prefill-seq-len", type=int, default=1)
    parser.add_argument("--generation-len", type=int, default=8)
    parser.add_argument("--num-cores", type=int, default=4)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--device-id", type=int, nargs="+", default=[0])
    parser.add_argument("--prompt", default="hello world")
    parser.add_argument("--dtype", default="float32", choices=["float16", "float32", "bfloat16"])
    args = parser.parse_args()

    if args.prefill_seq_len != 1:
        raise ValueError("This example is decode-only; compile it with --prefill-seq-len 1.")

    qaic_config = {
        "mla_absorption": {"cache_compressed": True, "absorption": True, "online": False},
        "dsa_impl": "dsa_par",
        "par_num_split": 4,
    }
    dtype = getattr(torch, args.dtype)

    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        dtype=dtype,
        qaic_config=qaic_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    qeff_model.compile(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        use_onnx_subfunctions=False,
        mxfp6_matmul=False,
    )
    qeff_model.generate(
        tokenizer=tokenizer,
        prompts=[args.prompt],
        device_id=args.device_id,
        generation_len=args.generation_len,
    )


if __name__ == "__main__":
    main()
