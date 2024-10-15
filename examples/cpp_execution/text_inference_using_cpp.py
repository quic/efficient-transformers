# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import argparse
import json
import logging
import os
import sys
from typing import List, Optional, Tuple, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM
from QEfficient.utils import check_and_assign_cache_dir, load_hf_tokenizer
from QEfficient.utils.logging_utils import logger

so_folder_path = os.path.abspath("examples/cpp_execution/build")
sys.path.append(so_folder_path)
import InferenceSetIOBufferExample  # noqa: E402


def main(
    model_name: str,
    num_cores: int,
    device_group: Optional[List[int]] = None,
    prompt: Optional[str] = None,  # type: ignore
    prompts_txt_file_path: Optional[str] = None,
    aic_enable_depth_first: bool = False,
    mos: int = -1,
    batch_size: int = 1,
    full_batch_size: Optional[int] = None,
    prompt_len: int = 32,
    ctx_len: int = 128,
    generation_len: Optional[int] = None,
    mxfp6: bool = False,
    mxint8: bool = False,
    local_model_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> None:
    """
    1. Check if compiled qpc for given config already exists, if it does jump to execute, else
    2. Check if exported ONNX file already exists, if true, jump to compilation -> execution, else
    3. Check if HF model exists in cache, if true, start transform -> export -> compilation -> execution, else,
    4. Download HF model -> transform -> export -> compile -> execute
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
        :num_cores (int): Number of cores to compile model on.
    ``Optional`` Args:
        :device_group (List[int]): Device Ids to be used for compilation. If ``len(device_group) > 1``, multiple Card setup is enabled. ``Defaults to None.``
        :prompt (str): Sample prompt for the model text generation. ``Defaults to None.``
        :prompts_txt_file_path (str): Path to txt file for multiple input prompts. ``Defaults to None.``
        :aic_enable_depth_first (bool): Enables ``DFS`` with default memory size. ``Defaults to False.``
        :mos (int): Effort level to reduce the on-chip memory. ``Defaults to -1.``
        :batch_size (int): Batch size to compile the model for. ``Defaults to 1.``
        :full_batch_size (int): Set full batch size to enable continuous batching mode. ``Default to None``
        :prompt_len (int): Prompt length for the model to compile. ``Defaults to 32.``
        :ctx_len (int): Maximum context length to compile the model. ``Defaults to 128.``
        :generation_len (int): Number of tokens to be generated. ``Defaults to False.``
        :mxfp6 (bool): Enable compilation for MXFP6 precision. ``Defaults to False.``
        :mxint8 (bool): Compress Present/Past KV to ``MXINT8`` using ``CustomIO`` config. ``Defaults to False.``
        :local_model_dir (str): Path to custom model weights and config files. ``Defaults to None.``
        :cache_dir (str): Cache dir where downloaded HuggingFace files are stored. ``Defaults to None.``
        :hf_token (str): HuggingFace login token to access private repos. ``Defaults to None.``

    .. code-block:: bash

        python -m examples.text_inference_from_cpp OPTIONS

    """
    cache_dir = check_and_assign_cache_dir(local_model_dir, cache_dir)
    tokenizer = load_hf_tokenizer(
        pretrained_model_name_or_path=(local_model_dir if local_model_dir else model_name),
        cache_dir=cache_dir,
        hf_token=hf_token,
    )

    qeff_model = AutoModelForCausalLM.from_pretrained(model_name)
    generated_qpc_path = qeff_model.compile(
        num_cores=14,
        mxfp6=True,
        device_group=[0],
    )

    #########
    # Execute
    #########
    cloud_ai_100_exec_kv(
        tokenizer=tokenizer,
        qpc_path=generated_qpc_path,
        prompt=prompt,
        device_id=device_group,
        prompts_txt_file_path=prompts_txt_file_path,
        generation_len=generation_len,
        full_batch_size=full_batch_size,
    )


def get_compilation_dims(qpc_path: str) -> Tuple[int, int]:
    qpc_base_path = os.path.dirname(os.path.normpath(qpc_path))
    specialization_file_path = os.path.join(qpc_base_path, "specializations.json")
    logger.info(f"specialization_file_path : {specialization_file_path}")

    if os.path.exists(specialization_file_path):
        with open(specialization_file_path, "r") as file:
            data = json.load(file)
    else:
        raise FileNotFoundError(f"expected specializations.json file at path, {qpc_base_path}")

    compilation_batch_size = int(data["specializations"][0]["batch_size"])
    compilation_ctx_len = int(data["specializations"][0]["ctx_len"])
    return compilation_batch_size, compilation_ctx_len


def read_prompts_txt_file(prompts_txt_file_path: str):
    prompt = []
    with open(prompts_txt_file_path, "r") as file:
        for line in file:
            prompt.append(line.strip())
    return prompt


def get_input_prompts(prompt: str, prompts_txt_file_path: str) -> List[str]:
    assert (
        prompt is not None or prompts_txt_file_path is not None
    ), "Please pass at least one argument either using --prompt or --prompts_txt_file_path"
    if prompts_txt_file_path is not None:
        if prompt is not None:
            logger.warning("Found inputs passed using txt file as well as CLI, taking inputs from given txt file")
        prompt = read_prompts_txt_file(prompts_txt_file_path)
    if isinstance(prompt, str):
        prompt = [prompt]
    return prompt


def fix_prompts(prompt: List[str], batch_size: int, full_batch_size: int = None):
    """
    Adjusts the list of prompts to match the required batch size.

    ``Mandatory`` Args:
        prompt (List[str]): List of input prompts.
        batch_size (int): The batch size to process at a time.

    ``Optional`` Args:
        full_batch_size (Optional[int]): The full batch size if different from batch_size.

    Returns:
        List[str]: Adjusted list of prompts.
    """
    exec_batch_size = full_batch_size if full_batch_size is not None else batch_size

    if len(prompt) < exec_batch_size:
        logger.warning("Number of prompts are less than batch size/full batch size, repeating to required batch size")
        prompt = (prompt * (exec_batch_size // len(prompt) + 1))[:exec_batch_size]
    elif full_batch_size is None and len(prompt) % batch_size != 0:
        logger.warning(
            "Number of prompts are not multiple of batch size, dropping last incomplete batch from given input prompts"
        )
        prompt = prompt[: batch_size * (len(prompt) // batch_size)]

    return prompt


def cloud_ai_100_exec_kv(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    qpc_path: str,
    prompt: Optional[List[str]] = None,
    prompts_txt_file_path: Optional[str] = None,
    device_id: Optional[List[int]] = None,
    generation_len: Optional[int] = None,
    enable_debug_logs: bool = False,
    stream: bool = True,
    full_batch_size: Optional[int] = None,
):
    batch_size, ctx_len = get_compilation_dims(qpc_path)
    prompt: List[str] = get_input_prompts(prompt, prompts_txt_file_path)
    prompt = fix_prompts(prompt, batch_size, full_batch_size)

    # ********* CPP Calling ********
    InferenceSetIOBufferExample.generatePrompt(
        tokenizer, qpc_path, batch_size, ctx_len, prompt, generation_len, device_id
    )


def tokenize_for_prefill(prompt, tokenizer, padded_len):
    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    return inputs


def tokenize_decode_output(tokenizer, generated_ids):
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print("Generated Text From CPP Inference: ")
    print(generated_texts)
    return generated_texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference command, the model will be downloaded from HF, optimized, compiled, executed on Cloud AI 100"
    )
    parser.add_argument("--model-name", "--model_name", required=True, help="HF Model card name/id")
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
        "--num_cores", "--num-cores", type=int, required=True, help="Number of cores to compile on Cloud AI 100"
    )
    parser.add_argument(
        "--device_group",
        "--device-group",
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        help="Cloud AI 100 device ids (comma-separated) e.g. [0,1]  ",
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
    parser.add_argument("--generation_len", "--generation-len", type=int, help="Number of tokens to generate")
    parser.add_argument(
        "--aic_enable_depth_first",
        "--aic-enable-depth-first",
        action="store_true",
        help="If passed, this option will be enabled during compilation, disabled by default",
    )
    parser.add_argument(
        "--mos",
        type=int,
        default=-1,
        help="Effort level to reduce the on-chip memory",
    )
    # FIXME: Add verbose feature
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="pass to print info logs",
    )
    parser.add_argument(
        "--full_batch_size",
        "--full_batch_size",
        type=int,
        default=None,
        help="Set full batch size to enable continuous batching mode, default is None",
    )

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.INFO)
    del args.verbose  # type: ignore
    main(**args.__dict__)
