# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
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
        "--custom_io_file_path",
        "--custom-io-file-path",
        type=str,
        help="Path to custom IO file",
    )
    parser.add_argument(
        "--device_group",
        "--device-group",
        required=True,
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        help="Cloud AI 100 device ids (comma-separated) e.g. [0,1] ",
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
        "--full-batch-size",
        type=int,
        default=None,
        help="Set full batch size to enable continuous batching mode, default is None",
    )
    parser.add_argument(
        "--allow-mxint8-mdp-io",
        "--allow_mxint8_mdp_io",
        action="store_true",
        help="If passed, this option allows MXINT8 compression of MDP IO traffic",
    )
    parser.add_argument(
        "--enable_qnn",
        "--enable-qnn",
        nargs="?",
        const=True,
        type=str,
        default=False,
        help="Enables QNN. Optionally, a configuration file can be provided with [--enable_qnn CONFIG_FILE].\
             If not provided, the default configuration will be used.\
             Sample Config: QEfficient/compile/qnn_config.json",
    )

    args, compiler_options = parser.parse_known_args()

    if isinstance(args.enable_qnn, str):
        args.qnn_config = args.enable_qnn
        args.enable_qnn = True

    compiler_options_dict = {}
    for i in range(0, len(compiler_options)):
        if compiler_options[i].startswith("--"):
            key = compiler_options[i].lstrip("-").replace("-", "_")
            value = (
                compiler_options[i + 1]
                if i + 1 < len(compiler_options) and not compiler_options[i + 1].startswith("-")
                else True
            )
            compiler_options_dict[key] = value
    QEfficient.compile(**args.__dict__, **compiler_options_dict)
