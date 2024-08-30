# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import hashlib
import os
from typing import List, Optional, Tuple, Union

import requests
from huggingface_hub import login, snapshot_download
from requests.exceptions import HTTPError
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from QEfficient.utils.constants import QEFF_MODELS_DIR, Constants
from QEfficient.utils.logging_utils import logger


def login_and_download_hf_lm(model_name, *args, **kwargs):
    logger.info(f"loading HuggingFace model for {model_name}")
    hf_token = kwargs.pop("hf_token", None)
    cache_dir = kwargs.pop("cache_dir", None)
    if hf_token is not None:
        login(hf_token)
    model_name = hf_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        ignore_patterns=["*.txt", "*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.msgpack", "*.h5"],
    )
    return model_name


def hf_download(
    repo_id: Optional[str] = None,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
):
    # Setup cache and local dir
    local_dir = None
    if cache_dir is not None:
        cache_dir = f"{cache_dir}"
        local_dir = f"{cache_dir}/{repo_id}"

    os.makedirs(f"{cache_dir}/{repo_id}", exist_ok=True)
    max_retries = 5
    retry_count = 0
    while retry_count < max_retries:
        try:
            model_path = snapshot_download(
                repo_id,
                cache_dir=cache_dir,
                local_dir=local_dir,
                local_dir_use_symlinks=True,
                revision="main",
                resume_download=True,
                token=hf_token,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
            break
        except requests.ReadTimeout as e:
            print(f"Read timeout: {e}")
            retry_count += 1

        except HTTPError as e:
            retry_count = max_retries
            if e.response.status_code == 401:
                print("You need to pass a valid `--hf_token=...` to download private checkpoints.")
            else:
                raise e

    return model_path


def qpc_exists(model_name: str, qpc_base_dir_name: str) -> Tuple[bool, str]:
    """
    Checks if qpc dir exists.
    Returns
    1. Boolean variable indicating if qpc files exist
    2. Path of the qpc dir if found.
    ---------
    :param model_name: str. HF Model card name.
    :param dir_path: str. Path of qpc directory.
    :return: Union[Tuple[bool, str]]: qpc_exists and path to qpc directory
    """
    model_card_dir = os.path.join(QEFF_MODELS_DIR, str(model_name))
    os.makedirs(model_card_dir, exist_ok=True)

    qpc_dir_path = os.path.join(model_card_dir, qpc_base_dir_name, "qpcs")

    # Compute the boolean indicating if the QPC exists
    qpc_exists_bool = os.path.isdir(qpc_dir_path) and os.path.isfile(os.path.join(qpc_dir_path, "programqpc.bin"))

    return qpc_exists_bool, qpc_dir_path


def onnx_exists(model_name: str, base_dir_name: str) -> Tuple[bool, str, str]:
    """
    Checks if qpc files already exists, removes the directory if files have been manipulated.
    ---------
    :param model_name: str. HF Model card name.
    :return: Union[Tuple[bool, str, str]]: onnx_exists and path to onnx file and directory
    """
    model_card_dir = os.path.join(QEFF_MODELS_DIR, str(model_name))
    os.makedirs(model_card_dir, exist_ok=True)
    onnx_dir_path = os.path.join(model_card_dir, os.path.join(base_dir_name, "onnx"))
    onnx_model_path = os.path.join(onnx_dir_path, model_name.replace("/", "_") + "_kv_clipped_fp16.onnx")

    # Compute the boolean indicating if the ONNX model exists
    onnx_exists_bool = os.path.isfile(onnx_model_path) and os.path.isfile(
        os.path.join(os.path.dirname(onnx_model_path), "custom_io_fp16.yaml")
    )

    # Return the boolean, onnx_dir_path, and onnx_model_path
    return onnx_exists_bool, onnx_dir_path, onnx_model_path


def load_hf_tokenizer(
    pretrained_model_name_or_path: str,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    padding_side: str = "right",
    **kwargs,
) -> Union[PreTrainedTokenizerFast, PreTrainedTokenizer]:
    logger.info("Loading Tokenizer")
    if hf_token is not None:
        login(hf_token)
    # Download tokenizer along with model if it doesn't exist
    model_hf_path = (
        pretrained_model_name_or_path
        if os.path.isdir(pretrained_model_name_or_path)
        else hf_download(
            repo_id=pretrained_model_name_or_path, cache_dir=cache_dir, allow_patterns=["*.json", "*.py", "*token*"]
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_hf_path, padding_side=padding_side, trust_remote_code=True, **kwargs
    )
    padding_check_and_fix(tokenizer)  # Check and fix tokenizer viability

    return tokenizer


def get_qpc_dir_name_infer(
    num_cores, mos, batch_size, prompt_len, ctx_len, mxfp6, mxint8, device_group, full_batch_size
):
    qpc_base_dir_name = (
        f"model_files_{num_cores}cores_{batch_size}BS_{prompt_len}PL_{ctx_len}CL_{mos}MOS_"
        + f"{f'{full_batch_size}FBS_' if full_batch_size else ''}"
        + f"{len(device_group)}"
        + "devices"
        + ("_mxfp6_mxint8" if (mxfp6 and mxint8) else "_mxfp6" if mxfp6 else "_fp16_mxint8" if mxint8 else "_fp16")
    )

    return qpc_base_dir_name


def check_and_assign_cache_dir(local_model_dir, cache_dir):
    if local_model_dir is not None:
        if cache_dir is not None:
            logger.warning(
                f"Both local_model_dir ({local_model_dir}) and cache_dir ({cache_dir}) given. Using local_model_dir."
            )
        return None
    return cache_dir if cache_dir else Constants.CACHE_DIR


def padding_check_and_fix(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> None:
    """
    Checks and fixes tokenizer paddding side and pad_token_id viability.
    --------

    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]. Pass model tokenizer to check and fix.
    """
    if tokenizer.padding_side != "right":
        logger.warning(f"Setting tokenizer padding_side to 'right', got {tokenizer.padding_side}")
        tokenizer.padding_side = "right"

    if tokenizer.pad_token_id is None:
        assert tokenizer.eos_token_id is not None, "Found tokenizer.eos_token_id to be None, expected int"
        # If Pad token is out of range of vocab size
        if tokenizer.eos_token_id < tokenizer.vocab_size:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = tokenizer.vocab_size - 1


def generate_sha256_hash(data: str) -> str:
    """
    Generate a SHA-256 hash for the given data.

    :param data: The input data to hash.
    :return: The hexadecimal representation of the SHA-256 hash.
    """
    hash_object = hashlib.sha256()
    hash_object.update(data.encode("utf-8"))
    hash_hex = hash_object.hexdigest()

    return hash_hex
