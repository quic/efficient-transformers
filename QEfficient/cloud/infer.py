# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import os
from typing import List
from typing import Tuple

from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

import QEfficient
from QEfficient.cloud.compile import main as compile
from QEfficient.exporter.export_hf_to_cloud_ai_100 import qualcomm_efficient_converter
from QEfficient.generation.text_generation_inference import (
    check_batch_size_and_num_prompts,
    cloud_ai_100_exec_kv,
)
from QEfficient.utils import hf_download, onnx_exists, qpc_exists
from QEfficient.utils.constants import Constants
from QEfficient.utils.logging_utils import logger

"""
1. Check if compiled qpc for given config already exists, if it does jump to execute, else
2. Check if exported ONNX file already exists, if true, jump to compilation -> execution, else
3. Check if HF model exists in cache, if true, start transform -> export -> compilation -> execution, else,
4. Download HF model -> transform -> export -> compile -> execute
"""

def _construct_qpc_dir_name(num_cores: int, batch_size: int, prompt_len: int, ctx_len: int, mos: int, device_group: List[int], mxfp6: bool, mxint8: bool) -> str:
    """
    Construct the base directory name for QPC files
    Parameters:
    - num_cores (int): Number of cores to use in AI 100.
    - batch_size (int): Batch size for processing.
    - prompt_len (int): Length of the prompt.
    - ctx_len (int): Context length.
    - mos (int): Maximum out-channel split. 
    - device_group (List[int]): List of device IDs.
    - mxfp6 (bool): Flag for enabling MXFP6 precision.
    - mxint8 (bool): Flag for mixed INT8 precision.

    Returns:
    - str: The constructed directory name.
    """
    qpc_base_dir_name = (
        f"qpc_{num_cores}cores_{batch_size}BS_{prompt_len}PL_{ctx_len}CL_{mos}MOS_"
        + f"{len(device_group)}"
        + "devices"
        + ("_mxfp6_mxint8" if (mxfp6 and mxint8) else "_mxfp6" if mxfp6 else "_fp16_mxint8" if mxint8 else "_fp16")
    )
    return qpc_base_dir_name

def _get_tokenizer(model_name: str, cache_dir: str, hf_token: str = None) -> Tuple[AutoTokenizer, str]:
    """
    Login with the Hugging Face token if provided and download the model,
    Create and return the tokenizer.

    Parameters:
    - model_name (str): Model name to download.
    - cache_dir (str): The directory where the model should be cached.
    - hf_token (Optional[str]): The Hugging Face token for authentication.

    Returns:
    - AutoTokenizer: The tokenizer for the downloaded model.
    - str: Downloaded model path
    """
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
    return tokenizer, model_hf_path

def main(
    model_name: str,
    num_cores: int,
    prompt: str = None,
    prompts_txt_file_path: str = None,
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
    
    # construct the qpc_base dir name based on the parameter. 
    qpc_base_dir_name = _construct_qpc_dir_name(num_cores, batch_size, prompt_len, ctx_len, mos, device_group, mxfp6, mxint8)

    # check either a prompt or a file path is provided, standardizes prompts into a list, and validates that the batch size matches the number of prompts.
    prompt = check_batch_size_and_num_prompts(prompt, prompts_txt_file_path, batch_size)

    # get tokenizer
    tokenizer, model_hf_path = _get_tokenizer(model_name, cache_dir, hf_token)
    
    # check for qpc path or qpc directory 
    qpc_path_exists, qpc_dir_path = qpc_exists(model_name, qpc_base_dir_name)

    #############################################
    # Workflow
    # HF model -> export -> compile -> execute
    #############################################

    if qpc_path_exists:
        # Continue to execute with pre-compiled QPC
        logger.info("Pre-compiled QPC found! Executing with given prompt.")
    else:
        # Check for ONNX model existence or transform and export HF model
        onnx_path_exists, onnx_dir_path, onnx_model_path = onnx_exists(model_name)
        
        # Check if onnx file exists
        if not onnx_path_exists:
            # Transform hf model -> export -> compile
            model_hf = AutoModelForCausalLM.from_pretrained(model_hf_path, use_cache=True)
            
            # Easy and minimal api to update the model to QEff.
            model_transformed = QEfficient.transform(model_hf, type="Transformers", form_factor="cloud")
            logger.info(f"Model after Optimized transformations {model_transformed}")

            # Export to ONNX
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
            
            print(f"Generated Onnx_path {generated_onnx_path} and Onnx_model_path {onnx_model_path} and Onnx_dir_path is {onnx_dir_path}")

            assert (generated_onnx_path == onnx_model_path), f"ONNX files were generated at an unusual location, expected {onnx_model_path}, got {generated_onnx_path}"
            
            logger.info(f"Base Path is {base_path} and Onnx Model Path is : {generated_onnx_path}")
           
        # Compile the model and generate QPC
        # We need to pass parent directory of qpc_dir_path, as the compile function handles the qpc directory creation
        if not qpc_path_exists:
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

            assert (qpc_dir_path == generated_qpc_path), f"QPC files were generated at an unusual location, expected {qpc_dir_path}; got {generated_qpc_path}"
            logger.info(f"Compiled qpc files can be found at : {generated_qpc_path}")

    # Execute model on device
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

    args = parser.parse_args()
    main(**args.__dict__)
