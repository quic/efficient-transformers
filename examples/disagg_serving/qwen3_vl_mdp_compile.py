# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Compile-only validation: Qwen3-VL-30B language-prefill QPC with MDP.

**Purpose — compile validation only.**
This script is intended *solely* to verify that the Qwen3-VL language-prefill
compile succeeds with Multi-Device Partitioning (MDP) options.  It does **not**
run inference, create a ``QAICInferenceSession``, or execute any generated QPCs.
Use it to confirm that the export + compile pipeline works end-to-end on your
hardware configuration before integrating the QPC into a serving stack.

The script exports and compiles the language-prefill partition of
Qwen/Qwen3-VL-30B-A3B-Instruct for disaggregated serving.  It uses:

  - ``mdp_ts_num_devices``  -- total number of AIC-100 devices across all
                               pipeline stages (default: 4).
  - ``mdp_num_partitions``  -- number of pipeline-parallel stages the model is
                               split into for prefill (default: 2).
  - ``mdp_strategy``        -- MDP partition-config generation strategy:
                               ``intersection`` (default) generates a compiler
                               dump first, then produces a compact MDP JSON
                               from the ONNX/compiler-node intersection;
                               ``onnx`` enumerates every ONNX graph node.
  - ``expert_parallel``     -- enables the MoE expert-parallel prefill path.

Running this script will **load / download the HF model weights** and invoke
the QEfficient export + compile pipeline.  Make sure:

  * A valid QEfficient installation and Qualcomm Cloud AI 100 toolchain are
    available in the active environment.
  * ``HF_HUB_CACHE`` points to a location with sufficient disk space if you
    want to redirect where the model is downloaded.
  * ``QEFF_HOME`` controls where QPC artifacts are written (optional; defaults
    to ``~/.cache/qeff``).

No secrets or absolute environment-specific paths are embedded here.
"""

import argparse
import sys

import torch
from transformers import AutoConfig

from QEfficient import QEFFAutoModelForImageTextToText

NUM_LAYERS = 2
VISION_DEPTH = 9


def parse_args() -> argparse.Namespace:
    """Return parsed command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compile Qwen3-VL-30B language-prefill QPC with MDP for disaggregated serving.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model_id",
        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
        help="HuggingFace model identifier.",
    )

    parser.add_argument("--batch_size", type=int, default=1, help="Prefill batch size.")
    parser.add_argument("--prefill_seq_len", type=int, default=128, help="Prefill sequence chunk length.")
    parser.add_argument("--ctx_len", type=int, default=4096, help="Maximum context length (KV-cache slots).")
    parser.add_argument("--height", type=int, default=354, help="Vision input height in pixels.")
    parser.add_argument("--width", type=int, default=536, help="Vision input width in pixels.")
    parser.add_argument("--num_cores", type=int, default=16, help="Number of AIC cores per device.")
    parser.add_argument("--mos", type=int, default=1, help="Memory-over-subscription factor.")

    parser.add_argument(
        "--mdp_ts_num_devices",
        type=int,
        default=4,
        help=(
            "Total AIC-100 devices used across all pipeline stages. "
            "Each stage receives mdp_ts_num_devices // mdp_num_partitions devices."
        ),
    )
    parser.add_argument(
        "--mdp_num_partitions",
        type=int,
        default=2,
        help="Number of pipeline-parallel partitions for disaggregated prefill.",
    )
    parser.add_argument(
        "--mdp_strategy",
        choices=["onnx", "intersection"],
        default="intersection",
        help=(
            "MDP partition-config generation strategy. "
            "'onnx' enumerates every ONNX graph node (~19 MB JSON). "
            "'intersection' first generates a compiler dump, then filters "
            "to exact Glow IR names."
        ),
    )
    parser.add_argument(
        "--expert_parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable MoE expert-parallel prefill compile options.",
    )
    parser.add_argument(
        "--expert_parallel_chunk_size",
        type=int,
        default=256,
        help="Packed chunk size used by the MoE expert-parallel prefill path.",
    )

    return parser.parse_args()


def main() -> None:
    """Load model and compile language-prefill QPC with MDP options.

    This function validates that the prefill compile step succeeds.  It
    intentionally stops after ``compile()`` returns and prints the QPC path.
    No ``QAICInferenceSession`` is created and no runtime inference is performed.
    """
    args = parse_args()

    config = AutoConfig.from_pretrained(args.model_id)
    config.dtype = "float16"
    config.vision_config.depth = VISION_DEPTH
    config.text_config.num_hidden_layers = NUM_LAYERS
    config.vision_config.deepstack_visual_indexes = [VISION_DEPTH - 1]

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        args.model_id,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        dtype=torch.float16,
        layerwise=False,
    )

    qaic_config = None
    if args.expert_parallel:
        qaic_config = {
            "moe_config": {
                "flavour": "expert_parallel",
                "expert_parallel_chunk_size": args.expert_parallel_chunk_size,
            }
        }

    prefill_qpc_path = qeff_model.compile(
        batch_size=args.batch_size,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        height=args.height,
        width=args.width,
        num_cores=args.num_cores,
        mos=args.mos,
        mdp_ts_num_devices=args.mdp_ts_num_devices,  # total devices spread across all pipeline stages
        mdp_num_partitions=args.mdp_num_partitions,  # number of pipeline-parallel stages (partitions)
        mdp_strategy=args.mdp_strategy,  # "intersection" auto-generates the compiler dump
        qaic_config=qaic_config,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        retain_full_kv=True,
        split_model_io=True,
        aic_enable_depth_first=True,
        prefill_only=True,
        enable_chunking=True,
        skip_vision=True,
        use_onnx_subfunctions=True,
        layerwise=False,
    )

    print(f"Prefill QPC path: {prefill_qpc_path}")


if __name__ == "__main__":
    sys.exit(main())
