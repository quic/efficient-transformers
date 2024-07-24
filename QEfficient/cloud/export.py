# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import os
from typing import Optional

import QEfficient
from QEfficient.utils import check_and_assign_cache_dir

# Specifically for Docker images.
ROOT_DIR = os.path.dirname(os.path.abspath(""))


def main(
    model_name: str,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    local_model_dir: Optional[str] = None,
) -> str:
    """
    Api() for exporting to Onnx Model.
    ---------
    :param model_name: str. Hugging Face Model Card name, Example: gpt2
    :cache_dir: str. Cache dir to store the downloaded huggingface files.
    :hf_token: str. HuggingFace login token to access private repos.
    :local_model_dir: str. Path to custom model weights and config files.
    """
    cache_dir = check_and_assign_cache_dir(local_model_dir, cache_dir)
    QEfficient.export(
        model_name=model_name,
        local_model_dir=local_model_dir,
        kv=True,
        form_factor="cloud",
        hf_token=hf_token,
        cache_dir=cache_dir,
    )  # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export script.")
    parser.add_argument("--model_name", "--model-name", required=True, help="HF Model card name/id")
    parser.add_argument(
        "--local-model-dir", "--local_model_dir", required=False, help="Path to custom model weights and config files"
    )
    parser.add_argument(
        "--cache_dir",
        "--cache-dir",
        required=False,
        help="Cache_dir to store the HF files",
    )
    parser.add_argument(
        "--hf-token", "--hf_token", default=None, type=str, required=False, help="HF token id for private HF models"
    )
    args = parser.parse_args()
    main(**args.__dict__)
