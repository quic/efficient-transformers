# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import os
from typing import Optional, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from QEfficient.exporter.export_hf_to_cloud_ai_100 import qualcomm_efficient_converter
from QEfficient.utils import onnx_exists, check_and_assign_cache_dir
from QEfficient.utils.constants import Constants
from QEfficient.utils.logging_utils import logger

# Specifically for Docker images.
ROOT_DIR = os.path.dirname(os.path.abspath(""))


def get_onnx_model_path(model_name: str, cache_dir: Optional[str] = None, tokenizer: Optional[Union[PreTrainedTokenizerFast, PreTrainedTokenizer]]=None, hf_token: Optional[str] = None, local_model_dir: Optional[str] = None):
    """
    exports the model to onnx if pre-exported file is not found and returns onnx_model_path
    """
    onnx_path_exists, onnx_dir_path, onnx_model_path = onnx_exists(model_name)
    if onnx_path_exists:
        logger.info(f"Pre-exported ONNX files found at {onnx_dir_path}! Jumping to Compilation")
    else:
        ###################
        # hf model -> export
        ####################
        # Export to the Onnx
        logger.info(f"Exporting Pytorch {model_name} model to ONNX...")
        _, generated_onnx_model_path = qualcomm_efficient_converter(
                model_name=model_name,
                local_model_dir=local_model_dir,
                tokenizer=tokenizer,
                onnx_dir_path=onnx_dir_path,
                kv=True,
                form_factor="cloud",
                return_path=True,
                hf_token=hf_token,
                cache_dir=cache_dir
            ) # type: ignore
        logger.info(f"Generated Onnx_path {generated_onnx_model_path} \nOnnx_model_path {onnx_model_path} \nand Onnx_dir_path is {onnx_dir_path}")
        assert (
                generated_onnx_model_path == onnx_model_path
            ), f"ONNX files were generated at an unusual location, expected {onnx_model_path}, got {generated_onnx_model_path}"
    return onnx_model_path


def main(
    model_name: str,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    local_model_dir: Optional[str] = None,
) -> None:
    """
    Api() for exporting to Onnx Model.
    ---------
    :param model_name: str. Hugging Face Model Card name, Example: gpt2
    :cache_dir: str. Cache dir to store the downloaded huggingface files.
    :hf_token: str. HuggingFace login token to access private repos.
    :local_model_dir: str. Path to custom model weights and config files.
    """
    check_and_assign_cache_dir(local_model_dir,cache_dir)
    get_onnx_model_path(model_name=model_name, cache_dir=cache_dir, hf_token=hf_token, local_model_dir=local_model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export script.")
    parser.add_argument("--model_name", "--model-name", required=True, help="HF Model card name/id")
    parser.add_argument("--local-model-dir", "--local_model_dir", required=False, help="Path to custom model weights and config files")
    parser.add_argument(
        "--cache_dir",
        "--cache-dir",
        required=False,
        default=None,
        help="Cache_dir to store the HF files",
    )
    parser.add_argument(
        "--hf-token", "--hf_token", default=None, type=str, required=False, help="HF token id for private HF models"
    )
    args = parser.parse_args()
    main(**args.__dict__)
