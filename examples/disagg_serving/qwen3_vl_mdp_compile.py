# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Compile-only validation: Qwen3-VL language-prefill QPC with disaggregated MDP.

Runs export + compile for the language-prefill component only; no inference is performed.

MDP strategies:
  onnx         — single pass, ONNX-derived partition config (~19 MB JSON, may compile slowly).
  intersection — two-pass: Pass 1 dumps compiler node list, Pass 2 compiles with compact
                 intersected config (~1-2 MB, Glow-IR-aligned; recommended for large models).
  both         — run ONNX flow then intersection flow.

Requires a valid QEfficient installation and Qualcomm Cloud AI 100 toolchain.
Set HF_HUB_CACHE to control model download location; QEFF_HOME for QPC artifact output.
"""

import argparse
import sys

import torch
from transformers import AutoConfig

from QEfficient import QEFFAutoModelForImageTextToText

_DEFAULT_COMPILER_DUMP_JSON = "qwen3_vl_mdp_compiler_dump.json"


def parse_args() -> argparse.Namespace:
    """Return parsed command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compile Qwen3-VL-30B language-prefill QPC with MDP. "
            "Supports 'onnx' (direct compile), 'intersection' (two-pass compact compile), "
            "or 'both' flows.  No inference is performed."
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
        help="Number of pipeline-parallel partitions for disaggregated prefill.",
    )
    parser.add_argument(
        "--mdp_strategy",
        choices=["onnx", "intersection", "both"],
        default="intersection",
        help=(
            "MDP compile strategy to run. "
            "'onnx': single compile pass using ONNX-derived partition config. "
            "'intersection': two-pass compile — Pass 1 dumps compiler node list, "
            "Pass 2 compiles with the intersected compact config (recommended). "
            "'both': run ONNX strategy then intersection strategy."
        ),
    )
    parser.add_argument(
        "--mdp_compiler_dump_path",
        default=_DEFAULT_COMPILER_DUMP_JSON,
        help=(
            "Path for the compiler MDP dump JSON file used by the intersection strategy. "
            "Pass 1 writes this file; Pass 2 reads it. "
            "Relative paths are resolved from the current working directory. "
            "Ignored when --mdp_strategy onnx is selected."
        ),
    )

    return parser.parse_args()


def _run_onnx_flow(qeff_model, shared_kwargs: dict, mdp_num_partitions: int) -> str:
    """Run a single-pass ONNX-strategy MDP compile. Returns QPC path."""
    print(f"[ONNX] Compiling with mdp_num_partitions={mdp_num_partitions}, mdp_strategy='onnx'")
    qpc_path = qeff_model.compile(
        **shared_kwargs,
        mdp_num_partitions=mdp_num_partitions,
        mdp_strategy="onnx",
    )
    print(f"[ONNX] QPC path: {qpc_path}")
    return qpc_path


def _run_intersection_flow(
    qeff_model,
    shared_kwargs: dict,
    mdp_num_partitions: int,
    mdp_compiler_dump_path: str,
) -> str:
    """Run two-pass intersection MDP compile.

    Pass 1 dumps the compiler node list; Pass 2 compiles with the compact intersected config.
    Returns QPC path from Pass 2.
    """
    print(f"[Intersection Pass 1] Dumping compiler MDP partition config to: {mdp_compiler_dump_path}")
    qeff_model.compile(
        **shared_kwargs,
        mdp_dump_partition_config=mdp_compiler_dump_path,
    )
    print(f"[Intersection Pass 1] Compiler dump written to: {mdp_compiler_dump_path}")

    print(
        f"[Intersection Pass 2] Compiling final QPC with mdp_num_partitions={mdp_num_partitions}, "
        f"mdp_strategy='intersection', mdp_compiler_dump_path={mdp_compiler_dump_path}"
    )
    qpc_path = qeff_model.compile(
        **shared_kwargs,
        mdp_num_partitions=mdp_num_partitions,
        mdp_strategy="intersection",
        mdp_compiler_dump_path=mdp_compiler_dump_path,
    )
    print(f"[Intersection Pass 2] QPC path: {qpc_path}")
    return qpc_path


def main() -> None:
    """Run ONNX and/or intersection MDP compile for Qwen3-VL language-prefill.

    Strategy is governed by --mdp_strategy: 'onnx', 'intersection', or 'both'.
    No QAICInferenceSession is created; no runtime inference is performed.
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

    shared_compile_kwargs = dict(
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

    run_onnx = args.mdp_strategy in ("onnx", "both")
    run_intersection = args.mdp_strategy in ("intersection", "both")

    if run_onnx:
        _run_onnx_flow(qeff_model, shared_compile_kwargs, args.mdp_num_partitions)

    if run_intersection:
        _run_intersection_flow(
            qeff_model,
            shared_compile_kwargs,
            args.mdp_num_partitions,
            args.mdp_compiler_dump_path,
        )


if __name__ == "__main__":
    sys.exit(main())
