# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
import shutil
import subprocess
from typing import List, Optional, Tuple

from QEfficient.utils.logging_utils import logger


def create_and_dump_specializations(
    batch_size: int, prompt_len: int, ctx_len: int, path: str, full_batch_size: Optional[int] = None
):
    # Create specialization file.
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
    # If continuous batching is enabled by proving full_batch_size we need to add FBS to the specialization file and update the batch size of decoder part to FBS
    if full_batch_size is not None:
        specializations["specializations"][0]["full_batch_size"] = str(full_batch_size)
        specializations["specializations"][1]["full_batch_size"] = str(full_batch_size)
        specializations["specializations"][1]["batch_size"] = str(full_batch_size)

    # Dump
    with open(path, "w") as file:
        json.dump(specializations, file, indent=4)


def compile_kv_model_on_cloud_ai_100(
    onnx_path: str,
    specializations_json: str,
    num_cores: int,
    base_path: str,
    mxfp6: bool,
    custom_io_path: str,
    aic_enable_depth_first: bool,
    mos: int = -1,
    device_group: Optional[List[int]] = None,
    **kwargs,
) -> Tuple[bool, str]:
    if kwargs:
        # FIXME
        raise NotImplementedError("Can't handle extra compilation args now!")
    aic_binary_dir = os.path.join(base_path, "qpcs")

    if os.path.isdir(aic_binary_dir):
        shutil.rmtree(aic_binary_dir)

    assert os.path.isfile(
        specializations_json
    ), f"Please use 'QEfficient.compile', as {specializations_json} file was not found"
    assert os.path.isfile(custom_io_path), f"{custom_io_path} file was not found!"
    command = [
        "/opt/qti-aic/exec/qaic-exec",
        f"-m={onnx_path}",
        "-aic-hw",
        "-aic-hw-version=2.0",
        f"-network-specialization-config={specializations_json}",
        "-convert-to-fp16",
        "-retained-state",
        f"-aic-num-cores={num_cores}",
        f"-custom-IO-list-file={custom_io_path}",
        "-compile-only",
        f"-aic-binary-dir={aic_binary_dir}",
    ]
    if mxfp6:
        command.append("-mxfp6-matmul")
    if mos > 0:
        command.append(f"-mos={mos}")
    if aic_enable_depth_first:
        command.append("-aic-enable-depth-first")
    if device_group is not None and len(device_group) > 1:
        mdp_ts_config = {
            "connections": [{"devices": list(range(len(device_group))), "type": "p2p"}],
            "partitions": [
                {
                    "name": "Partition0",
                    "devices": [{"deviceId": device, "numCores": num_cores} for device in range(len(device_group))],
                }
            ],
        }
        mdp_ts_config_path = os.path.join(base_path, "mdp_ts_config.json")
        with open(mdp_ts_config_path, "w") as file:
            json.dump(mdp_ts_config, file, indent=4)
        command.append(f"-mdp-load-partition-config={mdp_ts_config_path}")
    print("Running AI 100 compiler:", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Compilation Failed!!\n\nSTDOUT\n{result.stdout}\n\nSTDERR\n{result.stderr}")

    print("\n===================== Compilation Done! =====================\n")
    return result.returncode == 0, aic_binary_dir


def compile(
    onnx_path: str,
    qpc_path: str,
    num_cores: int,
    device_group: Optional[List[int]] = None,  #  FIXME: use num_devices instead
    aic_enable_depth_first: bool = False,
    mos: int = -1,
    batch_size: int = 1,
    prompt_len: int = 32,
    ctx_len: int = 128,
    mxfp6: bool = True,
    mxint8: bool = False,
    custom_io_file_path: Optional[str] = None,
    full_batch_size: Optional[int] = None,
    **kwargs,
) -> str:
    """
    Compiles the given ``ONNX`` model using Cloud AI 100 platform SDK compiler and saves the compiled ``qpc`` package at ``qpc_path``.
    Generates tensor-slicing configuration if multiple devices are passed in ``device_group``.

    This function will be deprecated soon and will be replaced by ``QEFFAutoModelForCausalLM.compile``.

    ``Mandatory`` Args:
        :onnx_path (str): Generated ``ONNX`` Model Path.
        :qpc_path (str): Path for saving compiled qpc binaries.
        :num_cores (int): Number of cores to compile the model on.
    ``Optional`` Args:
        :device_group (List[int]): Used for finding the number of devices to compile for. ``Defaults to None.``
        :aic_enable_depth_first (bool): Enables ``DFS`` with default memory size. ``Defaults to False.``
        :mos (int): Effort level to reduce the on-chip memory. ``Defaults to -1.``
        :batch_size (int): Batch size to compile the model for. ``Defaults to 1.``
        :prompt_len (int): Prompt length for the model to compile. ``Defaults to 32``
        :ctx_len (int): Maximum context length to compile the model. ``Defaults to 128``
        :mxfp6 (bool): Enable compilation for ``MXFP6`` precision.  ``Defaults to True.``
        :mxint8 (bool): Compress Present/Past KV to ``MXINT8`` using ``CustomIO`` config. ``Defaults to False.``
        :custom_io_file_path (str): Path to ``customIO`` file (formatted as a string). ``Defaults to None.``

    Returns:
        :str: Path to compiled ``qpc`` package.
    """
    os.makedirs(qpc_path, exist_ok=True)
    specialization_json_path = os.path.join(qpc_path, "specializations.json")
    # Dynamically create the specializations JSON
    create_and_dump_specializations(
        batch_size=batch_size,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        path=specialization_json_path,
        full_batch_size=full_batch_size,
    )

    # Select the customIO config based on the mx flag.
    custom_io_file_name = "custom_io_int8.yaml" if mxint8 else "custom_io_fp16.yaml"

    if custom_io_file_path is None:
        custom_io_file_path = os.path.join(os.path.dirname(onnx_path), custom_io_file_name)

    if not os.path.isfile(custom_io_file_path):
        raise FileNotFoundError(
            f"Custom IO file {custom_io_file_name} is not present at the expected path {custom_io_file_path}. Please pass the correct file path or rerun infer/export API"
        )

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
