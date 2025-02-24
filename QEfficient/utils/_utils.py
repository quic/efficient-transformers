# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import torch
from huggingface_hub import login, snapshot_download
from requests.exceptions import HTTPError
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from QEfficient.utils.constants import QEFF_MODELS_DIR, Constants
from QEfficient.utils.logging_utils import logger


class DownloadRetryLimitExceeded(Exception):
    """
    Used for raising error when hf_download fails to download the model after given max_retries.
    """


def login_and_download_hf_lm(model_name, *args, **kwargs):
    logger.info(f"loading HuggingFace model for {model_name}")
    hf_token = kwargs.pop("hf_token", None)
    cache_dir = kwargs.pop("cache_dir", None)
    if hf_token is not None:
        login(hf_token)
    model_path = hf_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        ignore_patterns=["*.txt", "*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.msgpack", "*.h5", "*.pth"],
    )
    return model_path


def hf_download(
    repo_id: Optional[str] = None,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
    max_retries: Optional[int] = Constants.MAX_RETRIES,
):
    # Setup cache_dir
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    retry_count = 0
    while retry_count < max_retries:
        try:
            model_path = snapshot_download(
                repo_id,
                cache_dir=cache_dir,
                revision="main",
                resume_download=True,
                token=hf_token,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
            break
        except requests.ReadTimeout as e:
            logger.info(f"Read timeout: {e}")
            retry_count += 1
        except HTTPError as e:
            if e.response.status_code == 401:
                logger.info("You need to pass a valid `--hf_token=...` to download private checkpoints.")
            raise e
        except OSError as e:
            if "Consistency check failed" in str(e):
                logger.info(
                    "Consistency check failed during model download. The file appears to be incomplete. Resuming the download..."
                )
                retry_count += 1
            else:
                raise e

    if retry_count >= max_retries:
        raise DownloadRetryLimitExceeded(
            f"Unable to download full model after {max_retries} tries. If the model fileS are huge in size, please try again."
        )
    return model_path


def qpc_exists(qpc_dir_path: str) -> bool:
    """
    Checks if qpc dir exists.
    Returns
    1. Boolean variable indicating if qpc files exist
    2. Path of the qpc dir if found.
    ---------

    :model_name: `str` - HF Model card name.
    :dir_path: `str` - Path of qpc directory.

    Return:
        qpc_exists and path to qpc directory
    """

    # Compute the boolean indicating if the QPC exists
    qpc_exists_bool = os.path.isdir(qpc_dir_path) and os.path.isfile(os.path.join(qpc_dir_path, "programqpc.bin"))

    return qpc_exists_bool


def get_onnx_dir_name(model_name, has_fbs):
    # Create a unique directory name for the ONNX model
    # Clearly indicate whether it's with or without FBS
    # Replace all hyphens with underscores
    model_name_safe = model_name.replace("/", "_").replace("-", "_")
    if has_fbs:
        return f"onnx_{model_name_safe}_with_fbs"
    else:
        return f"onnx_{model_name_safe}_without_fbs"


def onnx_exists(model_name: str, full_batch_size: int) -> Tuple[bool, str, str]:
    """
    Checks if qpc files already exists, removes the directory if files have been manipulated.
    ---------

    :model_name: `str`- HF Model card name.

    Return:
        onnx_exists and path to onnx file and directory
    """
    model_card_dir = os.path.join(QEFF_MODELS_DIR, str(model_name))
    os.makedirs(model_card_dir, exist_ok=True)

    # Determine if we're using full_batch_size
    has_fbs = full_batch_size is not None

    # ONNX handling
    onnx_dir_name = get_onnx_dir_name(model_name, has_fbs)
    onnx_dir_path = os.path.join(model_card_dir, onnx_dir_name)
    os.makedirs(onnx_dir_path, exist_ok=True)
    clipped_onnx_model_path = os.path.join(onnx_dir_path, model_name.replace("/", "_") + "_kv_clipped_fp16.onnx")
    unclipped_onnx_model_path = clipped_onnx_model_path.replace("_clipped_fp16.onnx", ".onnx")

    # Compute the boolean indicating if the ONNX model exists
    onnx_exists_bool = False
    onnx_model_path = None
    if os.path.isfile(os.path.join(onnx_dir_path, "custom_io_fp16.yaml")):
        if os.path.isfile(clipped_onnx_model_path):
            onnx_exists_bool = True
            onnx_model_path = clipped_onnx_model_path
        elif os.path.isfile(unclipped_onnx_model_path):
            onnx_exists_bool = True
            onnx_model_path = unclipped_onnx_model_path

    # Return the boolean, onnx_dir_path, and onnx_model_path
    return onnx_exists_bool, onnx_dir_path, onnx_model_path


def load_hf_tokenizer(
    pretrained_model_name_or_path: str,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    padding_side: str = "right",
    **kwargs,
) -> Union[PreTrainedTokenizerFast, PreTrainedTokenizer]:
    # FIXME: Fix kwargs to take token, cache_dir and pass via kwargs only on line 129
    logger.info("Loading Tokenizer")
    if hf_token is not None:
        login(hf_token)
    # Download tokenizer along with model if it doesn't exist
    model_hf_path = (
        pretrained_model_name_or_path
        if os.path.isdir(pretrained_model_name_or_path)
        else hf_download(
            repo_id=pretrained_model_name_or_path,
            cache_dir=cache_dir,
            allow_patterns=["*.json", "*.py", "*token*", "*.txt"],
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_hf_path, padding_side=padding_side, trust_remote_code=True, **kwargs
    )
    padding_check_and_fix(tokenizer)  # Check and fix tokenizer viability

    return tokenizer


def load_hf_processor(
    pretrained_model_name_or_path: str,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    **kwargs,
) -> Union[PreTrainedTokenizerFast, PreTrainedTokenizer]:
    logger.info("Loading Processor")
    if hf_token is not None:
        login(hf_token)
    # Download tokenizer along with model if it doesn't exist
    model_hf_path = (
        pretrained_model_name_or_path
        if os.path.isdir(pretrained_model_name_or_path)
        else hf_download(
            repo_id=pretrained_model_name_or_path,
            cache_dir=cache_dir,
            allow_patterns=["*.json", "*.py", "*token*", "*.txt"],
        )
    )
    processor = AutoProcessor.from_pretrained(model_hf_path, trust_remote_code=True, **kwargs)

    return processor


def get_qpc_dir_path(
    model_card_name,
    num_cores,
    mos,
    batch_size,
    prompt_len,
    ctx_len,
    mxfp6,
    mxint8,
    device_group,
    full_batch_size,
    num_speculative_tokens: Optional[int] = None,
    enable_qnn: Optional[bool] = False,
):
    # Create a unique directory name for the QPC model based on all parameters
    qpc_base_dir_name = (
        "qpc"
        + f"{'_qnn_' if enable_qnn else '_'}"
        + f"{num_cores}cores_{batch_size}bs_{prompt_len}pl_{ctx_len}cl_{mos}mos"
        + f"{f'_{full_batch_size}fbs_' if full_batch_size is not None else '_'}"
        + f"{f'_{num_speculative_tokens}nst_' if num_speculative_tokens is not None else ''}"
        + f"{len(device_group) if device_group is not None else 1}"
        + "devices"
        + ("_mxfp6_mxint8" if (mxfp6 and mxint8) else "_mxfp6" if mxfp6 else "_fp16_mxint8" if mxint8 else "_fp16")
    )
    model_card_dir = os.path.join(QEFF_MODELS_DIR, str(model_card_name))
    os.makedirs(model_card_dir, exist_ok=True)

    qpc_dir_path = os.path.join(model_card_dir, qpc_base_dir_name, "qpcs")
    return qpc_dir_path


def check_and_assign_cache_dir(local_model_dir, cache_dir):
    if local_model_dir is not None:
        if cache_dir is not None:
            logger.warning(
                f"Both local_model_dir ({local_model_dir}) and cache_dir ({cache_dir}) given. Using local_model_dir."
            )
        return None
    return cache_dir if cache_dir else None


def padding_check_and_fix(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> None:
    """
    Checks and fixes tokenizer padding side and pad_token_id viability.
    --------

    tokenizer: `Union[PreTrainedTokenizer, PreTrainedTokenizerFast]` - Pass model tokenizer to check and fix.
    """
    if tokenizer.padding_side != "right":
        logger.warning(f"Setting tokenizer padding_side to 'right', got {tokenizer.padding_side}")
        tokenizer.padding_side = "right"

    if tokenizer.pad_token_id is None:
        if not isinstance(tokenizer.eos_token_id, int):
            raise TypeError("Found tokenizer.eos_token_id to be None, expected int")
        # If Pad token is out of range of vocab size
        if tokenizer.eos_token_id < tokenizer.vocab_size:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = tokenizer.vocab_size - 1


def get_padding_shape_from_config(config, batch_size, seq_len):
    """
    Gets padding dims from model config - number of kv heads and d_head
    and returns padding shape - (batch_size, number of kv heads, seq_len, hidden size)
    required for initialization of past_key_values
    --------

    :config: AutoConfig from pretrained model.
    :batch_size: int. number of input prompts used to create inputs
    :seq_len: int. sequence length to run the model for.

    Return:
        List[int, int, int, int]
    """

    if hasattr(config, "n_head"):  # Assuming n_head is a key in the config (GPTs/CodeGen)
        n_heads = config.n_head
        d_head = config.n_embd // config.n_head
    elif hasattr(config, "num_key_value_heads") and hasattr(
        config, "num_attention_heads"
    ):  # Check for num_key_value_heads (Llama/Mistral)
        n_heads = config.num_key_value_heads

        if hasattr(config, "head_dim"):
            d_head = config.head_dim
        else:
            d_head = config.hidden_size // config.num_attention_heads

    elif hasattr(config, "n_heads"):  # Check for n_heads and d_model in the config (MPT Model)
        n_heads = config.n_heads
        d_head = config.d_model // config.n_heads
    elif hasattr(config, "new_decoder_architecture"):  # Check for Falcon
        new_decoder_architecture = getattr(config, "new_decoder_architecture")
        if new_decoder_architecture:  # multi_query is ignored when new_decoder_architecture is True
            n_heads = config.num_attention_heads
        else:
            if hasattr(config, "multi_query"):
                multi_query_value = getattr(config, "multi_query")
                if multi_query_value:
                    n_heads = 1  # MQA , multi query is true
                else:
                    n_heads = config.num_attention_heads
        d_head = config.hidden_size // config.num_attention_heads
    else:
        raise ValueError("Invalid model configuration: n_head/d_heads or num_key_value_heads not found.")
    padding_shape = [batch_size, n_heads, seq_len, d_head]
    if hasattr(config, "architectures") and config.architectures is not None:  # Check for Starcoder1 - 3D layout
        if "GPTBigCodeForCausalLM" in config.architectures:
            padding_shape = [batch_size, seq_len, d_head]
    return padding_shape


def get_num_layers_from_config(config):
    """
    Gets number of layers from model config
    --------

    :config: AutoConfig from pretrained model.

    Return:
        number of layers
    """

    if hasattr(config, "n_layer"):  # Assuming n_layer is a key in the config (GPTs/CodeGen)
        n_layer = config.n_layer
    elif hasattr(config, "num_hidden_layers"):  # llama/Mistral/Falcon
        n_layer = config.num_hidden_layers
    elif hasattr(config, "n_layers"):  # Check for n_layers in the config (MPT Model)
        n_layer = config.n_layers
    else:
        raise ValueError("Invalid model configuration: n_layer/n_layers or num_hidden_layers not found.")

    return n_layer


def execute_command(process: str, command: str, output_file_path: Optional[str] = None):
    """
    Executes the give command using subprocess.

    ``Mandatory`` Args:
        :process (str): Process name for which command is executed.
        :command (str): Command to be executed on shell.
    ``Optional`` Args:
        :output_file_path (str): If provided stdout & stderr for the executed command will be dumped to a file. ``Defaults to None.``

    """
    print(f"Running {process} command : \n {command}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
    except Exception as e:
        print("Execution failed: %s", e)

    if result.returncode != 0:
        raise RuntimeError(f"{process} failed Failed!!\n\nSTDOUT\n{result.stdout}\n\nSTDERR\n{result.stderr}")
    else:
        if output_file_path:
            stdout_path = os.path.join(output_file_path, f"{process}_stdout.txt")
            stderr_path = os.path.join(output_file_path, f"{process}_stderr.txt")
            # Write the output to a file
            try:
                with open(stdout_path, "w") as file:
                    file.write(result.stdout)
            except Exception as e:
                print(f"Failed to create {stdout_path}: {e}")
            try:
                with open(stderr_path, "w") as file:
                    file.write(result.stderr)
            except Exception as e:
                print(f"Failed to create {stderr_path}: {e}")


def load_json(file_path: str) -> Dict[Any, Any]:
    """
    Opens the given JSON file, load and return the JSON object.

    ``Mandatory`` Args:
        :file_path (str): JSON File to be opened.

    Return:
        JSON Object from the given file.

    """
    try:
        # Load the JSON config file
        with open(file_path, "r") as file:
            config_data = json.load(file)
    except Exception as e:
        raise ValueError(f"Failed to load json object from {file_path}: {e}")
    return config_data


def create_json(file_path: str, json_data: object):
    """
    Creates a JSON file with provided JSON data.

    ``Mandatory`` Args:
        :file_path (str): JSON File to be created.
        :json_data (object): JSON Data Object to be populated inside the created file.

    """
    try:
        with open(file_path, "w") as file:
            json.dump(json_data, file, indent=4)
    except Exception as e:
        print(f"Failed to create JSON File {file_path}: {e}")


def model_swap(func):
    def wrapper(*args, **kwargs):
        if "model" in kwargs and kwargs["model"] is not None:
            original_model = args[0].model
            args[0].model = kwargs["model"]
            onnx_path = func(*args, **kwargs)
            args[0].model = original_model
            return onnx_path

    return wrapper


@dataclass
class IOInfo:
    name: str
    datatype: torch.dtype
    shape: Tuple[Union[int, str], ...]

    def __repr__(self):
        return f"input_name:{self.name}\tdatatype:{self.datatype}\tshape:{self.shape}"
