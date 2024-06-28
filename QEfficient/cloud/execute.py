# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
from typing import List, Optional

from QEfficient.generation.text_generation_inference import (
    cloud_ai_100_exec_kv,
    get_input_prompts,
)
from QEfficient.utils import load_hf_tokenizer
from QEfficient.utils.constants import Constants


def main(
    model_name: str,
    qpc_path: str,
    device_group: List[int],
    prompt: Optional[str] = None,  # type: ignore
    prompts_txt_file_path: Optional[str] = None,
    generation_len: Optional[int] = None,
    cache_dir: Optional[str] = Constants.CACHE_DIR,
    hf_token: Optional[str] = None,
) -> None:
    """
    API() to run the Model on Cloud AI 100 Platform.
    ---------
    :param model_name: str. Hugging Face Model Card name, Example: "gpt2"
    :qpc_path: str.  Path to the generated binary after compilation.
    :device_group: List[int]. Device Ids to be used for compilation. if len(device_group) > 1. Multiple Card setup is enabled.
    :prompt: str. Sample prompt for the model text generation
    :prompts_txt_file_path: str. Path to txt file for multiple input prompts
    :generation_len: int. Number of tokens to be generated.
    :cache_dir: str. Cache dir where downloaded huggingface files are stored.
    :hf_token: str. HuggingFace login token to access private repos.
    """

    tokenizer = load_hf_tokenizer(model_name, cache_dir, hf_token)
    prompt: List[str] = get_input_prompts(prompt, prompts_txt_file_path)

    # Execute
    cloud_ai_100_exec_kv(
        tokenizer=tokenizer,
        qpc_path=qpc_path,
        device_id=device_group,
        prompt=prompt,
        generation_len=generation_len,
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
        type=lambda prompt: prompt.split("|"),
        help="Input prompt, if executing for batch size>1, pass input promprs in single string but seperate with pipe (|) symbol",
    )
    parser.add_argument(
        "--prompts_txt_file_path",
        "--prompts-txt-file-path",
        type=str,
        help="File path for taking input prompts from txt file, sample prompts.txt file present in examples folder",
    )
    parser.add_argument("--generation_len", "--generation-len", type=int, help="Number of tokens to generate")
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
    args = parser.parse_args()
    main(**args.__dict__)
