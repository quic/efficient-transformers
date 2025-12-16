# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
from typing import List, Optional
from QEfficient.generation.text_generation_inference import cloud_ai_100_exec_kv
from QEfficient.utils import load_hf_tokenizer


def main(
    model_name: str,
    qpc_path: str,
    device_group: List[int] = None,
    local_model_dir: Optional[str] = None,
    prompt: Optional[str] = None,  # type: ignore
    prompts_txt_file_path: Optional[str] = None,
    generation_len: Optional[int] = None,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    full_batch_size: Optional[int] = None,
):
    """
    Main function for the QEfficient execution CLI application.

    This function serves as the entry point for running a compiled model
    (QPC package) on the Cloud AI 100 Platform. It loads the necessary
    tokenizer and then orchestrates the text generation inference.

    Parameters
    ----------
    model_name : str
        Hugging Face Model Card name (e.g., ``gpt2``) for loading the tokenizer.
    qpc_path : str
        Path to the generated binary (QPC package) after compilation.

    Other Parameters
    ----------------
    device_group : List[int], optional
        List of device IDs to be used for inference. If `len(device_group) > 1`,
        a multi-card setup is enabled. Default is None.
    local_model_dir : str, optional
        Path to custom model weights and config files, used if not loading tokenizer
        from Hugging Face Hub. Default is None.
    prompt : str, optional
        Sample prompt(s) for the model text generation. For batch size > 1,
        pass multiple prompts separated by a pipe (``|``) symbol. Default is None.
    prompts_txt_file_path : str, optional
        Path to a text file containing multiple input prompts, one per line. Default is None.
    generation_len : int, optional
        Maximum number of tokens to be generated during inference. Default is None.
    cache_dir : str, optional
        Cache directory where downloaded HuggingFace files (like tokenizer) are stored.
        Default is None.
    hf_token : str, optional
        HuggingFace login token to access private repositories. Default is None.
    full_batch_size : int, optional
        Ignored in this context as continuous batching is managed by the compiled QPC.
        However, it might be passed through from CLI arguments. Default is None.

    Example
    -------
    To execute a compiled model from the command line:

    .. code-block:: bash

        python -m QEfficient.cloud.execute --model-name gpt2 --qpc-path /path/to/qpc/binaries --prompt "Hello world"

    For multi-device inference:

    .. code-block:: bash

        python -m QEfficient.cloud.execute --model-name gpt2 --qpc-path /path/to/qpc/binaries --device-group "[0,1]" --prompt "Hello | Hi"

    """
    tokenizer = load_hf_tokenizer(
        pretrained_model_name_or_path=(local_model_dir if local_model_dir else model_name),
        cache_dir=cache_dir,
        hf_token=hf_token,
    )

    # Execute
    cloud_ai_100_exec_kv(
        tokenizer=tokenizer,
        qpc_path=qpc_path,
        device_id=device_group,
        prompt=prompt,
        prompts_txt_file_path=prompts_txt_file_path,
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
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        help="Cloud AI 100 device ids (comma-separated) e.g. [0]",
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
        help="File path for taking input prompts from txt file, sample prompts.txt file present in examples/sample_prompts folder",
    )
    parser.add_argument("--generation_len", "--generation-len", type=int, help="Number of tokens to generate")
    parser.add_argument(
        "--local-model-dir", "--local_model_dir", required=False, help="Path to custom model weights and config files"
    )
    parser.add_argument(
        "--cache-dir",
        "--cache_dir",
        default=None,
        required=False,
        help="Cache dir to store HF Downloads",
    )
    parser.add_argument(
        "--full_batch_size",
        "--full-batch-size",
        type=int,
        default=None,
        help="Set full batch size to enable continuous batching mode, default is None",
    )
    parser.add_argument(
        "--hf-token", "--hf_token", default=None, type=str, required=False, help="HF token id for private HF models"
    )
    args = parser.parse_args()
    main(**args.__dict__)

