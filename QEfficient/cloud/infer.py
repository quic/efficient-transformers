# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import logging
import os
from typing import List, Optional

import QEfficient
from QEfficient.cloud.compile import main as compile
from QEfficient.exporter.export_hf_to_cloud_ai_100 import qualcomm_efficient_converter
from QEfficient.generation.text_generation_inference import (
    check_batch_size_and_num_prompts,
    cloud_ai_100_exec_kv,
)
from QEfficient.loader import QEFFAutoModel
from QEfficient.utils import load_hf_tokenizer, onnx_exists, qpc_exists
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
    prompt: Optional[str] = None, # type: ignore
    prompts_txt_file_path: Optional[str] = None,
    aic_enable_depth_first: bool = False,
    mos: int = -1,
    cache_dir: str = Constants.CACHE_DIR,
    hf_token: Optional[str] = None,
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

    prompt: List[str] = check_batch_size_and_num_prompts(prompt, prompts_txt_file_path, batch_size)

    # Get tokenizer
    tokenizer = load_hf_tokenizer(model_name=model_name, cache_dir=cache_dir, hf_token=hf_token)

    qpc_path_exists, qpc_dir_path = qpc_exists(model_name, qpc_base_dir_name)
    onnx_path_exists, onnx_dir_path, onnx_model_path = onnx_exists(model_name)

    if qpc_path_exists:
        # execute
        logger.info(f"Pre-compiled qpc found at {qpc_dir_path}! Executing with given prompt")
    elif onnx_path_exists:
        logger.info(f"Pre-exported ONNX files found at {onnx_dir_path}! Jumping to Compilation")
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
    else:
        #############################################
        # hf model -> export -> compile -> execute
        #############################################
        # Load hf model
        qeff_model = QEFFAutoModel.from_pretrained(pretrained_model_name_or_path=model_name, cache_dir=cache_dir, hf_token=hf_token)
        
        # Easy and minimal api to update the model to QEff.
        qeff_opt_model = QEfficient.transform(qeff_model, form_factor="cloud")
        logger.info(f"Model after Optimized transformations {qeff_opt_model}")

        # Export to the Onnx
        logger.info(f"Exporting Pytorch {model_name} model to ONNX...")
        # Need to split below function into two functions one which always takes QEFFAutoModel and other with same interface as below
        base_path, generated_onnx_path = qualcomm_efficient_converter(
            model_name=model_name,
            model_kv=qeff_opt_model, # type: ignore
            tokenizer=tokenizer,
            onnx_dir_path=onnx_dir_path,
            kv=True,
            form_factor="cloud",
            return_path=True,
        ) # type: ignore
        print(f"Generated Onnx_path {generated_onnx_path} and Onnx_model_path {onnx_model_path} and Onnx_dir_path is {onnx_dir_path}")
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
    cloud_ai_100_exec_kv(
        batch_size,
        tokenizer=tokenizer,
        qpc_path=qpc_dir_path,
        device_id=device_group,
        prompt=prompt,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference command, the model will be downloaded from HF, optmized, compiled, executed on Cloud AI 100"
    )
    parser.add_argument("--model-name", "--model_name", required=True, help="HF Model card name/id")
    parser.add_argument(
        "--cache-dir",
        "--cache_dir",
        default=Constants.CACHE_DIR,
        required=False,
        help="Cache dir to store HF Downloads",
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
        help="Input prompt, if executing for batch size>1, pass input prompts in single string but seperate with pipe (|) symbol",
    )
    parser.add_argument(
        "--prompts_txt_file_path",
        "--prompts-txt-file-path",
        type=str,
        help="File path for taking input prompts from txt file, sample prompts.txt file present in examples folder",
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
    #FIXME: Add verbose feature
    parser.add_argument(
        "--verbose","-v",
        action="store_true",
        help="pass to print info logs",
    )

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.INFO)
    main(**args.__dict__)
