# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse

import QEfficient

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compilation script.")
    parser.add_argument("--onnx_path", "--onnx-path", required=True, help="Onnx Model Path")
    parser.add_argument(
        "--qpc-path",
        "--qpc_path",
        required=True,
        help="Compiled qpc binaries will be stored under this folder",
    )
    parser.add_argument("--batch_size", "--batch-size", type=int, default=1, help="Batch size for text generation")
    parser.add_argument(
        "--prompt_len",
        "--prompt-len",
        default=32,
        type=int,
        help="Sequence length for text generation.",
    )
    parser.add_argument("--ctx_len", "--ctx-len", default=128, type=int, help="Context length for text generation.")
    parser.add_argument(
        "--mxfp6",
        action="store_true",
        help="Compress constant MatMul weights to MXFP6 E2M3, default is no compression",
    )
    parser.add_argument(
        "--mxint8",
        action="store_true",
        help="Compress Present/Past KV to MXINT8 using CustomIO config, default is False",
    )
    parser.add_argument(
        "--num_cores",
        "--num-cores",
        required=True,
        type=int,
        help="num cores to compile the model on",
    )
    parser.add_argument(
        "--device_group",
        "--device-group",
        required=True,
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        help="Cloud AI 100 device ids (comma-separated) e.g. [0] ",
    )
    parser.add_argument(
        "--aic_enable_depth_first",
        "--aic-enable-depth-first",
        action="store_true",
        help="If passed, this option will be enabled during compilation, disabled by default",
    )
    parser.add_argument(
        "--mos",
        type=int,
        default=-1,
        help=" Effort level to reduce the on-chip memory",
    )
    parser.add_argument(
        "--full_batch_size",
        "--full_batch_size",
        type=int,
        default=1,
        help="Batch size for text generation"
    )

    # FIXME(ochougul): Allow extra compilation arguments
    args = parser.parse_args()
    QEfficient.compile(**vars(args))
