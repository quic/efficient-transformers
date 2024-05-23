# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import os
from typing import List

from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

import QEfficient
from QEfficient.cloud.compile import main as compile
from QEfficient.exporter.export_hf_to_cloud_ai_100 import qualcomm_efficient_converter
from QEfficient.generation.text_generation_inference import cloud_ai_100_exec_kv
from QEfficient.utils import hf_download, onnx_exists, qpc_exists
from QEfficient.utils.constants import Constants
from QEfficient.utils.logging_utils import logger

"""
1. Check if compiled qpc for given config already exists, if it does jump to execute, else
2. Check if exported ONNX file already exists, if true, jump to compilation -> execution, else
3. Check if HF model exists in cache, if true, start transform -> export -> compilation -> execution, else,
4. Download HF model -> transform -> export -> compile -> execute
"""


def main(
    model_name: str,
    num_cores: int,
    prompt: str,
    aic_enable_depth_first: bool = False,
    mos: int = -1,
    cache_dir: str = Constants.CACHE_DIR,
    hf_token: str = None,
    batch_size: int = 1,
    prompt_len: int = 32,
    ctx_len: int = 128,
    mxfp6: bool = False,
    mxint8: bool = False,
    device_group: List[int] = [
        0,
    ],
) -> None:
    qpc_base_dir_name = (
        f"qpc_{num_cores}cores_{batch_size}BS_{prompt_len}PL_{ctx_len}CL_{mos}MOS_"
        + f"{len(device_group)}"
        + "devices"
        + ("_mxfp6_mxint8" if (mxfp6 and mxint8) else "_mxfp6" if mxfp6 else "_fp16_mxint8" if mxint8 else "_fp16")
    )

    # Get tokenizer
    if hf_token is not None:
        login(hf_token)
    model_hf_path = hf_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        ignore_patterns=["*.txt", "*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.msgpack", "*.h5"],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_hf_path, use_cache=True, padding_side="left", trust_remote_code=True
    )

    qpc_path_exists, qpc_dir_path = qpc_exists(model_name, qpc_base_dir_name)
    if qpc_path_exists:
        # execute
        logger.info("Pre-compiled qpc found! Trying to execute with given prompt")
        cloud_ai_100_exec_kv(tokenizer=tokenizer, qpc=qpc_dir_path, device_id=device_group, prompt=prompt)
        return

    onnx_path_exists, onnx_dir_path, onnx_model_path = onnx_exists(model_name)
    if onnx_path_exists:
        # Compile -> execute
        # We need to pass parent directory of qpc_dir_path, as the compile function handles the qpcs directory creation
        generated_qpc_path = compile(
            onnx_path=onnx_model_path,
            qpc_path=os.path.dirname(qpc_dir_path),
            num_cores=num_cores,
            batch_size=batch_size,
            prompt_len=prompt_len,
            ctx_len=ctx_len,
            mxfp6=mxfp6,
            mxint8=mxint8,
            aic_enable_depth_first=aic_enable_depth_first,
            mos=mos,
            device_group=device_group,
        )
        assert (
            generated_qpc_path == qpc_dir_path
        ), f"QPC files were generated at an unusual location, expected {qpc_dir_path}; got {generated_qpc_path}"
        cloud_ai_100_exec_kv(tokenizer=tokenizer, qpc=generated_qpc_path, device_id=device_group, prompt=prompt)
        return

    #############################################
    # hf model -> export -> compile -> execute
    #############################################
    model_hf = AutoModelForCausalLM.from_pretrained(model_hf_path, use_cache=True)
    # Easy and minimal api to update the model to QEff.
    model_transformed = QEfficient.transform(model_hf, type="Transformers", form_factor="cloud")
    logger.info(f"Model after Optimized transformations {model_transformed}")

    # Export to the Onnx
    logger.info(f"Exporting to Pytorch {model_name} to ONNX...")
    base_path, generated_onnx_path = qualcomm_efficient_converter(
        model_kv=model_transformed,
        onnx_dir_path=onnx_dir_path,
        model_name=model_name,
        kv=True,
        form_factor="cloud",
        return_path=True,
        tokenizer=tokenizer,
    )
    print(
        f"Generated Onnx_path {generated_onnx_path} and Onnx_model_path {onnx_model_path} and Onnx_dir_path is {onnx_dir_path}"
    )
    assert (
        generated_onnx_path == onnx_model_path
    ), f"ONNX files were generated at an unusual location, expected {onnx_model_path}, got {generated_onnx_path}"
    logger.info(f"Base Path is {base_path} and Onnx Model Path is : {generated_onnx_path}")

    # Compile
    # We need to pass parent directory of qpc_dir_path, as the compile function handles the qpcs directory creation
    generated_qpc_path = compile(
        onnx_path=onnx_model_path,
        qpc_path=os.path.dirname(qpc_dir_path),
        num_cores=num_cores,
        batch_size=batch_size,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        mxfp6=mxfp6,
        mxint8=mxint8,
        aic_enable_depth_first=aic_enable_depth_first,
        mos=mos,
        device_group=device_group,
    )
    assert (
        qpc_dir_path == generated_qpc_path
    ), f"QPC files were generated at an unusual location, expected {qpc_dir_path}; got {generated_qpc_path}"
    logger.info(f"Compiled qpc files can be found at : {generated_qpc_path}")

    # Execute
    cloud_ai_100_exec_kv(tokenizer=tokenizer, qpc=generated_qpc_path, device_id=device_group, prompt=prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference command, the model will be downloaded from HF, optmized, compiled, executed on AIC"
    )
    parser.add_argument("--model-name", "--model_name", required=True, help="HF Model card name/id")
    parser.add_argument(
        "--cache-dir", "--cache_dir", default=Constants.CACHE_DIR, required=False, help="Cache dir to store HF Downlods"
    )
    parser.add_argument(
        "--hf-token", "--hf_token", default=None, type=str, required=False, help="HF token id for private HF models"
    )
    parser.add_argument("--batch-size", "--batch_size", type=int, default=1, help="Batch size for text generation")
    parser.add_argument(
        "--prompt-len", "--prompt_len", default=32, type=int, help="Sequence length for text generation."
    )
    parser.add_argument("--ctx-len", "--ctx_len", default=128, type=int, help="Context length for text generation.")
    parser.add_argument(
        "--mxfp6", action="store_true", help="Compress constant MatMul weights to MXFP6 E2M3, default is no compression"
    )
    parser.add_argument(
        "--mxint8",
        action="store_true",
        help="Compress Present/Past KV to MXINT8 using CustomIO config, default is False",
    )
    parser.add_argument(
        "--num_cores", "--num-cores", type=int, required=True, help="Number of cores to compile on Cloud AI 100"
    )
    parser.add_argument(
        "--device_group",
        "--device-group",
        required=True,
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        help="Cloud AI 100 device ids (comma-separated) e.g. [0,1]  ",
    )
    parser.add_argument(
        "--prompt",
        type=lambda prompt: prompt.split("|"),
        default="My name is",
        help="Input prompt, if executing for batch size>1, pass input promprs in single string but seperate with pipe (|) symbol",
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
        help="Effort level to reduce the on-chip memory",
    )

    args = parser.parse_args()
    main(**args.__dict__)
