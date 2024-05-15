# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
from typing import List

from huggingface_hub import login
from transformers import AutoTokenizer

from QEfficient.generation.text_generation_inference import cloud_ai_100_exec_kv
from QEfficient.utils import hf_download
from QEfficient.utils.constants import Constants


def main(
    model_name: str,
    prompt: str,
    qpc_path: str,
    devices: List[int],
    cache_dir: str = Constants.CACHE_DIR,
    hf_token: str = None,
):
    """
    APi() to run the Model on Cloud AI 100 Platform.
    ---------
    :param model_name: str. Hugging Face Model Card name, Example: [gpt2]
    :prompt: str. Sample prompt for the model text generation
    :qpc_path: str.  Path to the generated binary after compilation.
    :devices: List[int]. Device Ids to be used for compilation. if devices > 1. Multiple Card setup is enabled.
    """
    if hf_token is not None:
        login(hf_token)
    # Download tokenizer along with model if it doesn't exist
    model_hf_path = hf_download(repo_id=model_name, cache_dir=cache_dir, allow_patterns=["*.json"])
    tokenizer = AutoTokenizer.from_pretrained(model_hf_path, use_cache=True, padding_side="left")

    cloud_ai_100_exec_kv(tokenizer=tokenizer, qpc=qpc_path, device_id=devices, prompt=prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execution script.")
    parser.add_argument(
        "--model_name", "--model-name", required=False, type=str, help="HF model card name for tokenizing the inputs"
    )
    parser.add_argument("--qpc_path", "--qpc-path", required=True, help="Path to generated QPC")
    parser.add_argument(
        "--prompt",
        type=lambda prompt: prompt.split("|"),
        default="My name is",
        help="Input prompt, if executing for batch size>1, pass input promprs in single string but seperate with pipe (|) symbol",
    )
    parser.add_argument(
        "--device_group",
        "--device-group",
        required=True,
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        help="cloud AI 100 device ids (comma-separated) e.g. [0]",
    )
    parser.add_argument(
        "--cache-dir", "--cache_dir", default=Constants.CACHE_DIR, required=False, help="Cache dir to store HF Downlods"
    )
    parser.add_argument(
        "--hf-token", "--hf_token", default=None, type=str, required=False, help="HF token id for private HF models"
    )
    args = parser.parse_args()
    main(args.model_name, args.prompt, args.qpc_path, args.device_group, args.cache_dir, args.hf_token)
