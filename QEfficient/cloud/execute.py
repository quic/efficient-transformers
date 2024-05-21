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
    qpc_path: str,
    device_group: List[int],
    prompt: str = None,
    prompts_txt_file_path: str = None,
    cache_dir: str = Constants.CACHE_DIR,
    hf_token: str = None,
):
    """
    APi() to run the Model on Cloud AI 100 Platform.
    ---------
    :param model_name: str. Hugging Face Model Card name, Example: "gpt2"
    :qpc_path: str.  Path to the generated binary after compilation.
    :device_group: List[int]. Device Ids to be used for compilation. if len(device_group) > 1. Multiple Card setup is enabled.
    :prompt: str. Sample prompt for the model text generation
    :prompts_txt_file_path: str. Path to txt file for multiple input prompts (in case of batch size > 1)
    """

    if hf_token is not None:
        login(hf_token)

    # Download tokenizer along with model if it doesn't exist
    model_hf_path = hf_download(repo_id=model_name, cache_dir=cache_dir, allow_patterns=["*.json", "*.py"])
    tokenizer = AutoTokenizer.from_pretrained(model_hf_path, use_cache=True, padding_side="left")

    # Execute
    cloud_ai_100_exec_kv(
        tokenizer=tokenizer,
        qpc_path=qpc_path,
        device_id=device_group,
        prompt=prompt,
        prompts_txt_file_path=prompts_txt_file_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execution script.")
    parser.add_argument(
        "--model_name", "--model-name", required=False, type=str, help="HF model card name for tokenizing the inputs"
    )
    parser.add_argument("--qpc_path", "--qpc-path", required=True, help="Path to generated QPC")
    parser.add_argument(
        "--device_group",
        "--device-group",
        required=True,
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        help="Cloud AI 100 device ids (comma-separated) e.g. [0]",
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
        "--cache-dir", "--cache_dir", default=Constants.CACHE_DIR, required=False, help="Cache dir to store HF Downlods"
    )
    parser.add_argument(
        "--hf-token", "--hf_token", default=None, type=str, required=False, help="HF token id for private HF models"
    )
    args = parser.parse_args()
    main(**args.__dict__)
