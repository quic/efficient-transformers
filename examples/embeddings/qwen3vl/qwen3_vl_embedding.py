# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""CLI example for running Qwen3-VL embedding on AI100.

This example intentionally exposes core QEff APIs to users:
- ``QEFFAutoModelForImageTextToText.from_pretrained(...)``
- ``model.compile(...)``
- AI100 embedding generation using precompiled QPCs.

Qwen3-VL-specific embedding preprocessing/runtime remains in ``embedding_model.py``.
"""

import argparse

from embedding_model import QEffQwen3VLEmbedder, resolve_model_source
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.transformers.models.qwen3_vl._embedding_utils import configure_embedding_model_config

DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"
DEFAULT_CTX_LEN = 2048
DEFAULT_NUM_CORES = 16
DEFAULT_NUM_DEVICES = 1
DEFAULT_NUM_HIDDEN_LAYERS = 36
DEFAULT_VISION_DEPTH = 27
DEFAULT_DEEPSTACK_INDEX = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for AI100 compile/inference knobs."""
    parser = argparse.ArgumentParser(description="Qwen3-VL embedding example.")
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
    parser.add_argument("--num-hidden-layers", type=int, default=DEFAULT_NUM_HIDDEN_LAYERS)
    parser.add_argument("--vision-depth", type=int, default=DEFAULT_VISION_DEPTH)
    parser.add_argument("--deepstack-index", type=int, default=DEFAULT_DEEPSTACK_INDEX)
    return parser.parse_args()


def build_reference_inputs() -> dict:
    """Create the reference payload aligned with HF embedding-style usage."""
    return {
        "queries": [
            {"text": "A woman playing with her dog on a beach at sunset."},
            {"text": "Pet owner training dog outdoors near water."},
            {"text": "Woman surfing on waves during a sunny day."},
            {"text": "City skyline view from a high-rise building at night."},
        ],
        "documents": [
            {
                "text": (
                    "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, "
                    "as the dog offers its paw in a heartwarming display of companionship and trust."
                )
            },
            {"image": "https://picsum.photos/id/237/536/354"},
            {
                "text": (
                    "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, "
                    "as the dog offers its paw in a heartwarming display of companionship and trust."
                ),
                "image": "https://picsum.photos/id/237/536/354",
            },
        ],
    }


def main() -> None:
    """Run AI100 embedding inference and print query-document similarity matrix."""
    args = parse_args()

    # Resolve model source (HF repo id -> local snapshot path for stable loading).
    model_source = resolve_model_source(args.model_name)

    # 1) Load config + processor + QEff model through public QEff/HF APIs.
    config = AutoConfig.from_pretrained(model_source, trust_remote_code=True, padding=True)
    configure_embedding_model_config(
        config=config,
        num_hidden_layers=args.num_hidden_layers,
        vision_depth=args.vision_depth,
        deepstack_index=args.deepstack_index,
        export_embedding=True,
    )

    processor = AutoProcessor.from_pretrained(model_source, trust_remote_code=True, padding=True)
    model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_source,
        kv_offload=True,
        trust_remote_code=True,
        config=config,
        qaic_config={"export_embedding": True},
    )

    # 2) Build embedding helper and reference payload.
    embedder = QEffQwen3VLEmbedder(processor=processor, model=model)
    payload = build_reference_inputs()
    model_inputs = payload["queries"] + payload["documents"]

    # 3) Derive compile requirements from current payload.
    compile_specs = embedder.get_compile_specs(
        inputs=model_inputs,
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

    # 5) Run AI100 embedding generation on precompiled QPCs.
    embeddings = embedder.process(
        inputs=model_inputs,
        qpc_paths=qpc_paths,
        prefill_seq_len=compile_specs["prefill_seq_len"],
        normalize=True,
    )

    q_count = len(payload["queries"])
    similarity_scores = embeddings[:q_count] @ embeddings[q_count:].T
    print(similarity_scores.tolist())


if __name__ == "__main__":
    main()
