# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Compile-only validation: Qwen3-VL-30B language-prefill QPC with MDP (intersection flow).

**Purpose — compile validation only.**
This script is intended *solely* to verify that the Qwen3-VL language-prefill
compile succeeds with Multi-Device Partitioning (MDP) using the two-pass
intersection strategy.  It does **not** run inference, create a
``QAICInferenceSession``, or execute any generated QPCs.  Use it to confirm
that the export + compile pipeline works end-to-end on your hardware
configuration before integrating the QPC into a serving stack.

**Two-pass intersection flow**

The script performs two sequential compile calls against the same loaded model:

Pass 1 — compiler dump
    Calls ``compile()`` *without* ``mdp_num_partitions`` (defaults to 1) but
    with ``mdp_dump_partition_config=<json_path>``.  This instructs the QAIC
    compiler to write the exact Glow IR node names it sees into the JSON file
    while still producing a (single-partition) QPC that is discarded.

Pass 2 — intersection compile
    Calls ``compile()`` with ``mdp_num_partitions``, ``mdp_strategy="intersection"``,
    and ``mdp_compiler_dump_path=<json_path>`` pointing at the file written in
    Pass 1.  The MDP generator intersects the ONNX-derived node superset with
    the compiler dump, producing a compact partition config (~1-2 MB vs ~19 MB
    for the ONNX-only strategy) that maps cleanly to hardware Glow IR names.

.. warning::
    MDP with ``use_onnx_subfunctions=False`` and **without** the intersection
    strategy enumerates every node in the ONNX graph (~19 MB JSON).  Many of
    those names do not match the Glow IR names the compiler actually uses,
    which can produce a very large node list that makes compilation extremely
    slow or causes it to stall.  This script always uses the two-pass
    intersection flow (Pass 1 dump + Pass 2 intersection compile) to avoid
    that problem.

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
import warnings

import torch
from transformers import AutoConfig

from QEfficient import QEFFAutoModelForImageTextToText

_DEFAULT_COMPILER_DUMP_JSON = "qwen3_vl_mdp_compiler_dump.json"


def parse_args() -> argparse.Namespace:
    """Return parsed command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Two-pass intersection compile for Qwen3-VL-30B language-prefill QPC with MDP. "
            "Pass 1 dumps the compiler MDP node list; Pass 2 compiles the final QPC using "
            "intersection to produce a compact partition config."
        ),
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
        "--num_devices",
        type=int,
        default=4,
        help=(
            "Total AIC-100 devices used across all pipeline stages. "
            "Each stage receives num_devices // mdp_num_partitions devices."
        ),
    )
    parser.add_argument(
        "--mdp_num_partitions",
        type=int,
        default=2,
        help="Number of pipeline-parallel partitions for disaggregated prefill (used in Pass 2).",
    )
    parser.add_argument(
        "--mdp_compiler_dump_path",
        default=_DEFAULT_COMPILER_DUMP_JSON,
        help=(
            "Path for the compiler MDP dump JSON file. "
            "Pass 1 writes this file; Pass 2 reads it. "
            "Relative paths are resolved from the current working directory."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run two-pass intersection compile for Qwen3-VL language-prefill.

    Pass 1: dump compiler MDP node list to a JSON file (no mdp_num_partitions).
    Pass 2: compile final QPC with mdp_num_partitions, mdp_strategy='intersection',
            and mdp_compiler_dump_path pointing at the Pass-1 dump.

    No ``QAICInferenceSession`` is created and no runtime inference is performed.
    """
    args = parse_args()

    warnings.warn(
        "MDP with use_onnx_subfunctions=False and without the intersection strategy can "
        "produce a very large node list (~19 MB JSON) of ONNX names that do not match "
        "Glow IR, making compilation extremely slow. This script uses the two-pass "
        "intersection flow (use_onnx_subfunctions=False + Pass 1 dump + Pass 2 intersection) "
        "to avoid that problem.",
        stacklevel=1,
    )

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

    _shared_compile_kwargs = dict(
        batch_size=args.batch_size,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        height=args.height,
        width=args.width,
        num_cores=args.num_cores,
        mos=args.mos,
        num_devices=args.num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        retain_full_kv=True,
        split_model_io=True,
        aic_enable_depth_first=True,
        prefill_only=True,
        enable_chunking=True,
        skip_vision=True,
        use_onnx_subfunctions=False,
        layerwise=False,
    )

    print(f"[Pass 1] Dumping compiler MDP partition config to: {args.mdp_compiler_dump_path}")
    qeff_model.compile(
        **_shared_compile_kwargs,
        mdp_dump_partition_config=args.mdp_compiler_dump_path,
    )
    print(f"[Pass 1] Compiler dump written to: {args.mdp_compiler_dump_path}")

    print(
        f"[Pass 2] Compiling final QPC with mdp_num_partitions={args.mdp_num_partitions}, "
        f"mdp_strategy='intersection', mdp_compiler_dump_path={args.mdp_compiler_dump_path}"
    )
    prefill_qpc_path = qeff_model.compile(
        **_shared_compile_kwargs,
        mdp_num_partitions=args.mdp_num_partitions,
        mdp_strategy="intersection",
        mdp_compiler_dump_path=args.mdp_compiler_dump_path,
    )
    print(f"[Pass 2] Prefill QPC path: {prefill_qpc_path}")


if __name__ == "__main__":
    sys.exit(main())
