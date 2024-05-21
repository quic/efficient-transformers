# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

import QEfficient
from QEfficient.cloud.infer import onnx_exists
from QEfficient.exporter.export_hf_to_cloud_ai_100 import qualcomm_efficient_converter
from QEfficient.utils import hf_download
from QEfficient.utils.constants import QEFF_MODELS_DIR, Constants
from QEfficient.utils.logging_utils import logger

# Specifically for Docker images.
ROOT_DIR = os.path.dirname(os.path.abspath(""))


def main(model_name: str, cache_dir: str) -> None:
    """
    Api() for exporting to Onnx Model.
    ---------
    :param model_name: str. Hugging Face Model Card name, Example: gpt2
    :cache_dir: str. Cache dir to store the downloaded huggingface files.
    """
    model_card_dir = os.path.join(QEFF_MODELS_DIR, str(model_name))
    os.makedirs(model_card_dir, exist_ok=True)

    onnx_dir_path = os.path.join(model_card_dir, "onnx")
    onnx_model_path = os.path.join(onnx_dir_path, model_name.replace("/", "_") + "_kv_clipped_fp16.onnx")

    if onnx_exists(onnx_model_path):
        logger.warning(f"Generated Onnx files found {onnx_model_path}! Please use Infer/Compile Apis()")
        return

    model_hf_path = hf_download(repo_id=model_name, hf_token=None, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_hf_path, use_cache=True, padding_side="left")
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
    args = parser.parse_args()
    main(args.model_name, args.cache_dir)
