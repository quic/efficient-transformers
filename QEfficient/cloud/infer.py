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
from QEfficient.generation.text_generation_inference import (
    check_batch_size_and_num_prompts,
    cloud_ai_100_exec_kv,
    get_compilation_batch_size,
    read_prompts_txt_file,
)
from QEfficient.utils import hf_download
from QEfficient.utils.constants import QEFF_MODELS_DIR, Constants
from QEfficient.utils.logging_utils import logger

"""
1. Check if compiled qpc for given config already exists, if it does jump to execute, else
2. Check if exported ONNX file already exists, if true, jump to compilation -> execution, else
3. Check if HF model exists in cache, if true, start transform -> export -> compilation -> execution, else,
4. Download HF model -> transform -> export -> compile -> execute
"""


def qpc_exists(qpc_dir_path: str) -> bool:
    """
    Checks if qpc files already exists, removes the directory if files have been manipulated.
    ---------
    :param dir_path: str. Path of qpc directory.
    :return: bool.
    """
    return os.path.isdir(qpc_dir_path) and os.path.isfile(os.path.join(qpc_dir_path, "programqpc.bin"))


def onnx_exists(onnx_file_path: str) -> bool:
    # todo(ochougul): add check for other files like raw input files, input_list.txt
    return os.path.isfile(onnx_file_path) and os.path.isfile(
        os.path.join(os.path.dirname(onnx_file_path), "custom_io.yaml")
    )


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
    device_group: List[int] = [
        0,
    ],
) -> None:
    # Make
    model_card_dir = os.path.join(QEFF_MODELS_DIR, str(model_name))
    os.makedirs(model_card_dir, exist_ok=True)

    qpc_base_dir_name = (
        f"qpc_{num_cores}cores_{batch_size}BS_{prompt_len}PL_{ctx_len}CL_"
        + f"{len(device_group)}"
        + "devices"
        + ("_mxfp6" if mxfp6 else "_fp16")
    )
    qpc_dir_path = os.path.join(model_card_dir, qpc_base_dir_name, "qpcs")

    onnx_dir_path = os.path.join(model_card_dir, "onnx")
    onnx_model_path = os.path.join(onnx_dir_path, model_name.replace("/", "_") + "_kv_clipped_fp16.onnx")

    assert (prompt is None and prompts_txt_file_path is not None) or (
        prompt is not None and prompts_txt_file_path is None
    ), "Please pass either single input string using --prompt or multiple inputs using --prompts_txt_file_path"

    if prompts_txt_file_path is not None:
        prompt = read_prompts_txt_file(prompts_txt_file_path)

    # Get tokenizer
    if hf_token is not None:
        login(hf_token)
    model_hf_path = hf_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        ignore_patterns=["*.txt", "*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf"],
    )
    tokenizer = AutoTokenizer.from_pretrained(model_hf_path, use_cache=True, padding_side="left")

    if qpc_exists(qpc_dir_path):
        # execute
        logger.info("Pre-compiled qpc found! Trying to execute with given prompt")
        compilation_batch_size = get_compilation_batch_size(qpc_dir_path)
        check_batch_size_and_num_prompts(prompt, compilation_batch_size)
        cloud_ai_100_exec_kv(
            compilation_batch_size=compilation_batch_size,
            tokenizer=tokenizer,
            qpc_path=qpc_dir_path,
            device_id=device_group,
            prompt=prompt,
        )
        return

    check_batch_size_and_num_prompts(prompt, batch_size)

    if onnx_exists(onnx_model_path):
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
            aic_enable_depth_first=aic_enable_depth_first,
            mos=mos,
            device_group=device_group,
        )
        assert (
            generated_qpc_path == qpc_dir_path
        ), f"QPC files were generated at an unusual location, expected {qpc_dir_path}; got {generated_qpc_path}"
        cloud_ai_100_exec_kv(
            compilation_batch_size=compilation_batch_size,
            tokenizer=tokenizer,
            qpc_path=qpc_dir_path,
            device_id=device_group,
            prompt=prompt,
        )
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
        compilation_batch_size=compilation_batch_size,
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
        type=str,
        help="Input prompt, if executing for batch size>1, use prompts_txt_file_path flag",
    )
    parser.add_argument(
        "--prompts_txt_file_path",
        "--prompts-txt-file-path",
        type=str,
        help="for batch size>1, pass input prompts in txt file, sample prompts.txt file present in examples folder",
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
