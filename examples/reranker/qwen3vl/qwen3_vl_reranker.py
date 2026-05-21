# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""CLI example for running Qwen3-VL reranker on AI100.

This script keeps only end-user orchestration:
- Parse compile/runtime args.
- Build one reference reranking payload.
- Print score list in document order.

All model/runtime logic lives in `reranker_model.py`.
"""

import argparse

from reranker_model import (
    DEFAULT_CTX_LEN,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_CORES,
    DEFAULT_NUM_DEVICES,
    QEffQwen3VLReranker,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for AI100 compile/inference knobs."""
    parser = argparse.ArgumentParser(description="Qwen3-VL reranker example.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--ctx-len", type=int, default=DEFAULT_CTX_LEN, help="Context length used at compile time.")
    parser.add_argument("--num-cores", type=int, default=DEFAULT_NUM_CORES, help="Number of AI100 cores.")
    parser.add_argument("--num-devices", type=int, default=DEFAULT_NUM_DEVICES, help="Number of AI100 devices.")
    parser.add_argument(
        "--mxfp6-matmul",
        action="store_true",
        help="Enable MXFP6 matmul during compile (default: disabled).",
    )
    parser.add_argument(
        "--compile-prefill-seq-len",
        type=int,
        default=None,
        help=(
            "Optional fixed prefill sequence length for compile/padding. "
            "Must be >= max prompt length of the current request."
        ),
    )
    return parser.parse_args()


def build_reference_inputs() -> dict:
    """Create the reference payload aligned with HF reranker-style usage."""
    return {
        "instruction": "Retrieve images or text relevant to the user's query.",
        "query": {"text": "A woman playing with her dog on a beach at sunset."},
        "documents": [
            {
                "text": (
                    "A woman shares a joyful moment with her golden retriever on a "
                    "sun-drenched beach at sunset, as the dog offers its paw in a "
                    "heartwarming display of companionship and trust."
                )
            },
            {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
            {
                "text": (
                    "A woman shares a joyful moment with her golden retriever on a "
                    "sun-drenched beach at sunset, as the dog offers its paw in a "
                    "heartwarming display of companionship and trust."
                ),
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
        ],
        "fps": 1.0,
    }


def main() -> None:
    """Run AI100 reranker inference and print scores."""
    args = parse_args()

    # Initialize the AI100-only reranker wrapper.
    model = QEffQwen3VLReranker(
        model_name_or_path=args.model_name,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        mxfp6_matmul=args.mxfp6_matmul,
        compile_prefill_seq_len=args.compile_prefill_seq_len,
    )

    # Score each document against the same instruction/query.
    inputs = build_reference_inputs()
    scores = model.process(inputs)

    # Final output: list of scores in input document order.
    print(scores)
    # [0.8624675869941711, 0.6706082820892334, 0.8116759657859802]


if __name__ == "__main__":
    main()
