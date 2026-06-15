# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""CLI example for running Qwen3-VL reranker on AI100.

This example intentionally exposes core QEff APIs to users:
- `QEFFAutoModelForImageTextToText.from_pretrained(...)`
- `model.compile(...)`
- AI100 runtime scoring using precompiled QPCs.

Qwen3-VL-specific reranker preprocessing/scoring remains in `reranker_model.py`.
"""

import argparse

from reranker_model import (
    QEffQwen3VLReranker,
    resolve_model_source,
)
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for AI100 compile/inference knobs."""
    parser = argparse.ArgumentParser(description="Qwen3-VL reranker example.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-Reranker-2B")
    parser.add_argument("--ctx-len", type=int, default=2048, help="Context length used at compile time.")
    parser.add_argument("--num-cores", type=int, default=16, help="Number of AI100 cores.")
    parser.add_argument("--num-devices", type=int, default=1, help="Number of AI100 devices.")
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

    # Resolve model source (HF repo id -> local snapshot path for stable loading).
    model_source = resolve_model_source(args.model_name)

    # 1) Load config + processor + QEff model through public QEff/HF APIs.
    config = AutoConfig.from_pretrained(model_source, trust_remote_code=True)
    if hasattr(config, "use_cache"):
        config.use_cache = True
    if hasattr(config, "text_config") and hasattr(config.text_config, "use_cache"):
        config.text_config.use_cache = True

    processor = AutoProcessor.from_pretrained(model_source, trust_remote_code=True)
    model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_source,
        kv_offload=True,
        trust_remote_code=True,
        config=config,
    )

    # 2) Build reranker helper and reference payload.
    reranker = QEffQwen3VLReranker(processor=processor, model=model)
    inputs = build_reference_inputs()

    # 3) Derive compile requirements from current payload.
    compile_specs = reranker.get_compile_specs(
        inputs=inputs,
        ctx_len=args.ctx_len,
        prefill_seq_len=args.compile_prefill_seq_len,
    )

    # 4) Compile using explicit QEff API and visible compile parameters.
    qpc_paths = model.compile(
        prefill_seq_len=compile_specs["prefill_seq_len"],
        ctx_len=compile_specs["ctx_len"],
        img_size=compile_specs["img_size"],
        height=compile_specs["height"],
        width=compile_specs["width"],
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        mxfp6_matmul=args.mxfp6_matmul,
    )

    # 5) Run AI100 scoring on precompiled QPCs.
    scores = reranker.process(
        inputs=inputs,
        qpc_paths=qpc_paths,
        prefill_seq_len=compile_specs["prefill_seq_len"],
    )

    print(scores)
    # [0.8624675869941711, 0.6706082820892334, 0.8116759657859802]


if __name__ == "__main__":
    main()
