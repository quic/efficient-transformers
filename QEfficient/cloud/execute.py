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

from QEfficient.generation.text_generation_inference import (
    check_batch_size_and_num_prompts,
    cloud_ai_100_exec_kv,
    get_compilation_batch_size,
    read_prompts_txt_file,
)
from QEfficient.utils import hf_download
from QEfficient.utils.constants import Constants


def main(
    model_name: str,
    prompt: str,
    prompts_txt_file_path: str,
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

    assert (prompt is None and prompts_txt_file_path is not None) or (
        prompt is not None and prompts_txt_file_path is None
    ), "Please pass either single input string using --prompt or multiple inputs using --prompts_txt_file_path"

    if prompts_txt_file_path is not None:
        prompt = read_prompts_txt_file(prompts_txt_file_path)

    compilation_batch_size = get_compilation_batch_size(qpc_path)
    check_batch_size_and_num_prompts(prompt, compilation_batch_size)

    # Execute
    cloud_ai_100_exec_kv(
        compilation_batch_size=compilation_batch_size,
        tokenizer=tokenizer,
        qpc=qpc_path,
        device_id=devices,
        prompt=prompt,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execution script.")
    parser.add_argument(
        "--model_name", "--model-name", required=False, type=str, help="HF model card name for tokenizing the inputs"
    )
    parser.add_argument("--qpc_path", "--qpc-path", required=True, help="Path to generated QPC")
    parser.add_argument(
        "--prompt",
        type=str,
        help="Input prompt, if executing for batch size>1, use prompts_txt_file_path flag",
    )
    parser.add_argument(
        "--prompts_txt_file_path",
        "--prompts-txt-file-path-file-path",
        type=str,
        help="for batch size>1, pass input prompts in txt file, sample prompts.txt file present in examples folder",
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
    main(
        args.model_name,
        args.prompt,
        args.prompts_txt_file_path,
        args.qpc_path,
        args.device_group,
        args.cache_dir,
        args.hf_token,
    )
