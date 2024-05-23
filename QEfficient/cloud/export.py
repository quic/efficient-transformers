# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import os

from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

import QEfficient
from QEfficient.exporter.export_hf_to_cloud_ai_100 import qualcomm_efficient_converter
from QEfficient.utils import hf_download, onnx_exists
from QEfficient.utils.constants import Constants
from QEfficient.utils.logging_utils import logger

# Specifically for Docker images.
ROOT_DIR = os.path.dirname(os.path.abspath(""))


def main(
    model_name: str,
    cache_dir: str,
    hf_token: str = None,
) -> None:
    """
    Api() for exporting to Onnx Model.
    ---------
    :param model_name: str. Hugging Face Model Card name, Example: gpt2
    :cache_dir: str. Cache dir to store the downloaded huggingface files.
    :hf_token: str. HuggingFace login token to access private repos.
    """
    onnx_path_exists, onnx_dir_path, onnx_model_path = onnx_exists(model_name)
    if onnx_path_exists:
        logger.warning(f"Generated Onnx files found {onnx_model_path}! Please use Infer/Compile Apis()")
        return

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
    model = AutoModelForCausalLM.from_pretrained(model_hf_path, use_cache=True)

    # Easy and minimal api to update the model to QEff.
    QEfficient.transform(model, type="Transformers", form_factor="cloud")
    print(f"Model after Optimized transformations {model}")

    # Export to the Onnx
    print(f"Exporting to Pytorch {model_name} to Onnx")
    base_path, onnx_path = qualcomm_efficient_converter(
        model_kv=model,
        model_name=model_name,
        tokenizer=tokenizer,
        kv=True,
        form_factor="cloud",
        return_path=True,
    )
    print(f"Base Path is {base_path} and Onnx Model Path is : {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export script.")
    parser.add_argument("--model_name", "--model-name", required=True, help="HF Model card name/id")
    parser.add_argument(
        "--cache_dir",
        "--cache-dir",
        required=False,
        default=Constants.CACHE_DIR,
        help="Cache_dir to store the HF files",
    )
    parser.add_argument(
        "--hf-token", "--hf_token", default=None, type=str, required=False, help="HF token id for private HF models"
    )
    args = parser.parse_args()
    main(**args.__dict__)
