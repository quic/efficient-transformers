# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
1. Check if compiled qpc for given config already exists, if it does jump to execute, else
2. Check if exported ONNX file already exists, if true, jump to compilation -> execution, else
3. Check if HF model exists in cache, if true, start transform -> export -> compilation -> execution, else,
4. Download HF model -> transform -> export -> compile -> execute
"""

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
)
from QEfficient.utils import hf_download, onnx_exists, qpc_exists
from QEfficient.utils.constants import Constants


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
    device_group: List[int] = [0],
) -> None:
    """
    Main function to download, optimize, compile, and execute a model on Cloud AI 100.

    Parameters:
    - model_name: The name or ID of the Hugging Face model.
    - num_cores: Number of cores to compile on Cloud AI 100.
    - prompt: Input prompt for text generation.
    - prompts_txt_file_path: File path for input prompts from a text file.
    - aic_enable_depth_first: Enable depth-first search during compilation.
    - mos: Effort level to reduce on-chip memory.
    - cache_dir: Directory to store Hugging Face downloads.
    - hf_token: Hugging Face token for accessing private models.
    - batch_size: Batch size for text generation.
    - prompt_len: Sequence length for text generation.
    - ctx_len: Context length for text generation.
    - mxfp6: Compress constant MatMul weights to MXFP6 E2M3.
    - mxint8: Compress Present/Past KV to MXINT8 using CustomIO config.
    - device_group: Cloud AI 100 device IDs (comma-separated).
    """

    qpc_base_dir_name = create_qpc_base_dir_name(
        num_cores, batch_size, prompt_len, ctx_len, mos, len(device_group), mxfp6, mxint8
    )

    prompt = process_prompt(prompt, prompts_txt_file_path, batch_size)

    # Login to Hugging Face Hub if hf_token is provided
    if hf_token:
        login(hf_token)

    # Check if pre-compiled QPC/Onnx exists and execute if so
    qpc_path_exists, qpc_dir_path = qpc_exists(model_name, qpc_base_dir_name)
    onnx_path_exists, onnx_dir_path, onnx_model_path = onnx_exists(model_name)

    if qpc_path_exists:
        # Load the tokenizer
        tokenizer = load_tokenizer(model_name)
    else:
        # Download the model from Hugging Face Hub
        model_hf_path = hf_download(
            model_name,
            cache_dir,
            ignore_patterns=["*.txt", "*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.msgpack", "*.h5"],
        )
        tokenizer = load_tokenizer(model_hf_path)

    if qpc_path_exists:
        execute_with_qpc(tokenizer, batch_size, qpc_dir_path, device_group, prompt)
        return

    # Check if ONNX file exists and compile and execute if so
    if onnx_path_exists:
        compile_and_execute_onnx(
            onnx_model_path,
            tokenizer,
            batch_size,
            qpc_dir_path,
            num_cores,
            prompt_len,
            ctx_len,
            mxfp6,
            mxint8,
            aic_enable_depth_first,
            mos,
            device_group,
            prompt,
        )
        return

    # If none of the above conditions met, proceed with downloading, transforming, exporting, compiling, and executing
    model_hf = AutoModelForCausalLM.from_pretrained(model_hf_path, use_cache=True)
    model_transformed = QEfficient.transform(model_hf, type="Transformers", form_factor="cloud")
    onnx_model_path = export_to_onnx(model_transformed, model_name, tokenizer, onnx_dir_path, onnx_model_path)
    compile_and_execute_onnx(
        onnx_model_path,
        tokenizer,
        batch_size,
        qpc_dir_path,
        num_cores,
        prompt_len,
        ctx_len,
        mxfp6,
        mxint8,
        aic_enable_depth_first,
        mos,
        device_group,
        prompt,
    )


def create_qpc_base_dir_name(
    num_cores: int,
    batch_size: int,
    prompt_len: int,
    ctx_len: int,
    mos: int,
    device_count: int,
    mxfp6: bool,
    mxint8: bool,
) -> str:
    """
    Create the base directory name for the QPC based on the parameters.

    Returns:
    - A string representing the base directory name.
    """
    return f"qpc_{num_cores}cores_{batch_size}BS_{prompt_len}PL_{ctx_len}CL_{mos}MOS_{device_count}devices{'_mxfp6_mxint8' if mxfp6 and mxint8 else '_mxfp6' if mxfp6 else '_fp16_mxint8' if mxint8 else '_fp16'}"


def process_prompt(prompt: str, prompts_txt_file_path: str, batch_size: int) -> str:
    """
    Process the prompt for text generation.

    Returns:
    - The processed prompt.
    """
    return check_batch_size_and_num_prompts(prompt, prompts_txt_file_path, batch_size)


def load_tokenizer(model_hf_path: str) -> AutoTokenizer:
    """
    Load the tokenizer from the downloaded model.

    Returns:
    - An instance of the tokenizer.
    """
    return AutoTokenizer.from_pretrained(model_hf_path, use_cache=True, padding_side="left", trust_remote_code=True)


def execute_with_qpc(
    tokenizer: AutoTokenizer, batch_size: int, qpc_dir_path: str, device_group: List[int], prompt: str
) -> None:
    """
    Execute the model with the pre-compiled QPC.
    """
    cloud_ai_100_exec_kv(
        batch_size=batch_size, tokenizer=tokenizer, qpc_path=qpc_dir_path, device_id=device_group, prompt=prompt
    )


def compile_and_execute_onnx(
    onnx_model_path: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
    qpc_dir_path: str,
    num_cores: int,
    prompt_len: int,
    ctx_len: int,
    mxfp6: bool,
    mxint8: bool,
    aic_enable_depth_first: bool,
    mos: int,
    device_group: List[int],
    prompt,
) -> None:
    """
    Compile and execute the model using the ONNX file.
    """
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
    execute_with_qpc(tokenizer, batch_size, qpc_dir_path, device_group, prompt)


def export_to_onnx(
    transformed_model: AutoModelForCausalLM,
    model_name: str,
    tokenizer: AutoTokenizer,
    onnx_dir_path: str,
    onnx_model_path: str,
) -> str:
    """
    Export the transformed model to ONNX format.

    Returns:
    - The path to the generated ONNX file.
    """
    base_path, generated_onnx_path = qualcomm_efficient_converter(
        model_kv=transformed_model,
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
    return generated_onnx_path


def add_arguments(parser):
    """
    Add arguments to the argument parser.
    """
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
        "--num-cores", "--num_cores", type=int, required=True, help="Number of cores to compile on Cloud AI 100"
    )
    parser.add_argument(
        "--device-group",
        "--device_group",
        required=False,
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        help="Cloud AI 100 device ids (comma-separated) e.g. [0,1]",
    )
    parser.add_argument(
        "--prompt",
        type=lambda prompt: prompt.split("|"),
        help="Input prompt, if executing for batch size>1, pass input prompts in single string but separate with pipe (|) symbol",
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
    parser.add_argument("--mos", type=int, default=-1, help="Effort level to reduce the on-chip memory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference command, the model will be downloaded from HF, optimized, compiled, executed on Cloud AI 100"
    )
    add_arguments(parser)
    args = parser.parse_args()
    main(**args.__dict__)
