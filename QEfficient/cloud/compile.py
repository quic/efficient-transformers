# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import json
import os
from typing import List

from QEfficient.exporter.export_utils import compile_kv_model_on_cloud_ai_100
from QEfficient.utils.logging_utils import logger


def create_and_dump_specializations(batch_size: int, prompt_len: int, ctx_len: int, path: str):
    # Create
    specializations = {
        "specializations": [
            {
                "batch_size": str(batch_size),
                "seq_len": str(prompt_len),
                "ctx_len": str(ctx_len),
            },
            {"batch_size": str(batch_size), "seq_len": "1", "ctx_len": str(ctx_len)},
        ]
    }
    # Dump
    with open(path, "w") as file:
        json.dump(specializations, file, indent=4)


def main(
    onnx_path: str,
    qpc_path: str,
    num_cores: int,
    device_group: List[int],
    aic_enable_depth_first: bool = False,
    mos: int = -1,
    batch_size: int = 1,
    prompt_len: int = 32,
    ctx_len: int = 128,
    mxfp6: bool = True,
    mxint8: bool = False,
) -> str:
    # Dynamically create the specializations JSON
    """
    Api() to compile the Onnx Model on Cloud AI 100 Platform with give config.
    ---------
    :param onnx_path: str. Generated Onnx Model Path.
    :base_path: str. Base path for the generated models.
    :batch_size: int. Batch size to compile the model for.
    :prompt_len: int. prompt len for the model to compile.
    :ctx_len: int. Maximum context length to compile the model.
    :mxfp6: bool. Enable compilation for MXFP6 precision
    :num_cores: int. Number of cores to compile model on. default: 16 available option: [1 to 16]
    """

    os.makedirs(qpc_path, exist_ok=True)
    specialization_json_path = os.path.join(qpc_path, "specializations.json")
    create_and_dump_specializations(
        batch_size=batch_size, prompt_len=prompt_len, ctx_len=ctx_len, path=specialization_json_path
    )

    # Select the customIO config based on the mx flag.
    if mxint8:
        custom_io_file_name = "custom_io_int8.yaml"
    else:
        custom_io_file_name = "custom_io_fp16.yaml"

    custom_io_file_path = os.path.join(os.path.dirname(onnx_path), custom_io_file_name)

    if not os.path.isfile(custom_io_file_path):
        raise FileNotFoundError(f"file {custom_io_file_path} needs to exist in the same directory as onnx model files.")

    _, qpc_path = compile_kv_model_on_cloud_ai_100(
        onnx_path=onnx_path,
        specializations_json=specialization_json_path,
        num_cores=num_cores,
        custom_io_path=custom_io_file_path,
        base_path=qpc_path,
        mxfp6=mxfp6,
        aic_enable_depth_first=aic_enable_depth_first,
        mos=mos,
        device_group=device_group,
    )

    logger.info(f"Compiled QPC files can be found here: {qpc_path}")
    return qpc_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compilation script.")
    parser.add_argument("--onnx_path", "--onnx-path", required=True, help="Onnx Model Path")
    parser.add_argument(
        "--qpc-path",
        "--qpc_path",
        required=False,
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
    args = parser.parse_args()
    main(**vars(args))
