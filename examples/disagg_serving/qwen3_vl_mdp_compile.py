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
                               ``onnx`` (default) enumerates every ONNX node;
                               ``intersection`` requires a prior
                               ``qaic-compile -mdp-dump-partition-config`` run
                               and produces a compact JSON.
  - ``mdp_compiler_dump_path`` -- path to the compiler-dump directory produced
                               by the prior ``intersection`` run.  Required when
                               ``--mdp_strategy intersection`` is chosen; the
                               argparse validator below enforces this.

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
        default="onnx",
        help=(
            "MDP partition-config generation strategy. "
            "'onnx' enumerates every ONNX graph node (~19 MB JSON). "
            "'intersection' filters to exact Glow IR names and requires "
            "--mdp_compiler_dump_path."
        ),
    )
    parser.add_argument(
        "--mdp_compiler_dump_path",
        default=None,
        help=(
            "Path to the directory produced by a prior "
            "'qaic-compile -mdp-dump-partition-config' run. "
            "Required when --mdp_strategy intersection is chosen."
        ),
    )

    args = parser.parse_args()

    if args.mdp_strategy == "intersection" and not args.mdp_compiler_dump_path:
        parser.error(
            "--mdp_compiler_dump_path is required when --mdp_strategy intersection is used. "
            "Run 'qaic-compile -mdp-dump-partition-config' first to produce the dump directory, "
            "then pass that path with --mdp_compiler_dump_path."
        )

    return args


def main() -> None:
    """Load model and compile language-prefill QPC with MDP options.

    This function validates that the prefill compile step succeeds.  It
    intentionally stops after ``compile()`` returns and prints the QPC path.
    No ``QAICInferenceSession`` is created and no runtime inference is performed.
    """
    args = parse_args()

    config = AutoConfig.from_pretrained(args.model_id)
    config.dtype = "float16"

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        args.model_id,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        dtype=torch.float16,
        layerwise=False,
    )

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
        mdp_strategy=args.mdp_strategy,  # "onnx" (default) or "intersection" (needs compiler dump)
        mdp_compiler_dump_path=args.mdp_compiler_dump_path,  # required only for intersection strategy
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
