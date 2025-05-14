# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import inspect
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import torch
import yaml
from huggingface_hub import login, snapshot_download
from requests.exceptions import HTTPError
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from QEfficient.utils.constants import QEFF_MODELS_DIR, Constants, QnnConstants
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


def get_num_layers_vlm(config):
    """
    Gets number of layers from model config of VLM
    --------

    :config: AutoConfig from pretrained model.

    Return:
        number of layers of text and vision part
    """

    if hasattr(config, "llm_config") and hasattr(config, "vision_config"):  # Intern
        n_layers_text = config.llm_config.num_hidden_layers
        n_layers_vision = config.vision_config.num_hidden_layers
    elif hasattr(config, "text_config") and hasattr(config, "vision_config"):  # Llava, Mllama
        n_layers_text = config.text_config.num_hidden_layers
        n_layers_vision = config.vision_config.num_hidden_layers

    return (n_layers_text, n_layers_vision)


def get_padding_shape_vlm(config, ctx_len, batch_size=1):
    """
    Gets padding dims for VLM models- number of kv heads and d_head
    and returns padding shape - (batch_size, number of kv heads, seq_len, hidden size)
    required for initialization of past_key_values
    --------

    :config: AutoConfig from pretrained model.
    :batch_size: int. number of input prompts used to create inputs
    :seq_len: int. sequence length to run the model for.

    Return:
        List[int, int, int, int]
    """
    if hasattr(config, "text_config"):
        n_heads = config.text_config.num_key_value_heads
        d_head = config.text_config.hidden_size // config.text_config.num_attention_heads
        padding_shape = [batch_size, n_heads, ctx_len, d_head]
    elif hasattr(config, "llm_config"):
        n_heads = config.llm_config.num_key_value_heads
        d_head = config.llm_config.hidden_size // config.llm_config.num_attention_heads
        padding_shape = [batch_size, n_heads, ctx_len, d_head]
    return padding_shape


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


def load_yaml(file_path: str) -> Dict[Any, Any]:
    """
    Opens the given YAML file, load and return the Dict.

    ``Mandatory`` Args:
        :file_path (str): YAML File to be opened.

    Return:
        Dict Object from the given file.

    """
    try:
        # Load the YAML config file
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)
    except Exception as e:
        raise ValueError(f"Failed to load YAML object from {file_path}: {e}")
    return config_data


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


def dump_qconfig(func):
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        create_and_dump_qconfigs(
            self.qpc_path,
            self.onnx_path,
            self.get_model_config,
            [cls.__name__ for cls in self._pytorch_transforms],
            [cls.__name__ for cls in self._onnx_transforms],
            kwargs.get("specializations"),
            kwargs.get("mdp_ts_num_devices", 1),
            kwargs.get("num_speculative_tokens"),
            **{
                k: v
                for k, v in kwargs.items()
                if k
                not in ["specializations", "mdp_ts_num_devices", "num_speculative_tokens", "custom_io", "onnx_path"]
            },
        )
        return result

    return wrapper


def create_and_dump_qconfigs(
    qpc_path,
    onnx_path,
    huggingface_config,
    pytorch_transforms,
    onnx_transforms,
    specializations,
    mdp_ts_num_devices,
    num_speculative_tokens,
    **compiler_options,
):
    """
    This Method creates a JSON file which contains all the configs for a model.
    Such as huggingface configs, QEff transforms, QAIC sdk version, QNN sdk, compilation dir, qpc dir and
    many other compilation options.
    """
    qnn_config = compiler_options["qnn_config"] if "qnn_config" in compiler_options else None
    enable_qnn = True if "qnn_config" in compiler_options else None

    qconfig_file_path = os.path.join(os.path.dirname(qpc_path), "qconfig.json")
    onnx_path = str(onnx_path)
    specializations_file_path = str(os.path.join(os.path.dirname(qpc_path), "specializations.json"))
    compile_dir = str(os.path.dirname(qpc_path))
    qnn_config_path = (
        (qnn_config if qnn_config is not None else "QEfficient/compile/qnn_config.json") if enable_qnn else None
    )

    # Extract QAIC SDK Apps Version from SDK XML file
    tree = ET.parse(Constants.SDK_APPS_XML)
    root = tree.getroot()
    qaic_version = root.find(".//base_version").text

    # Extract QNN SDK details from YAML file if the environment variable is set
    qnn_sdk_details = None
    qnn_sdk_path = os.getenv(QnnConstants.QNN_SDK_PATH_ENV_VAR_NAME)
    if enable_qnn and qnn_sdk_path:
        qnn_sdk_yaml_path = os.path.join(qnn_sdk_path, QnnConstants.QNN_SDK_YAML)
        with open(qnn_sdk_yaml_path, "r") as file:
            qnn_sdk_details = yaml.safe_load(file)

    # Ensure all objects in the configs dictionary are JSON serializable
    def make_serializable(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: make_serializable(value) for key, value in obj.items()}
        elif hasattr(obj, "__dict__"):
            return make_serializable(vars(obj))
        return str(obj)

    qconfigs = {
        "huggingface_config": make_serializable(huggingface_config),
        "qpc_config": {
            "QEff_config": {
                "pytorch_transforms": make_serializable(pytorch_transforms),
                "onnx_transforms": make_serializable(onnx_transforms),
                "onnx_path": onnx_path,
            },
        },
    }

    aic_compiler_config = {
        "apps_sdk_version": qaic_version,
        "compile_dir": compile_dir,
        "specializations_file_path": specializations_file_path,
        "specializations": make_serializable(specializations),
        "mdp_ts_num_devices": mdp_ts_num_devices,
        "num_speculative_tokens": num_speculative_tokens,
        **compiler_options,
    }
    qnn_config = {
        "enable_qnn": enable_qnn,
        "qnn_config_path": qnn_config_path,
    }
    # Put AIC or qnn details.
    if enable_qnn:
        qconfigs["qpc_config"]["qnn_config"] = qnn_config
        if qnn_sdk_details:
            qconfigs["qpc_config"]["qnn_config"].update(qnn_sdk_details)
    else:
        qconfigs["qpc_config"]["aic_compiler_config"] = aic_compiler_config

    create_json(qconfig_file_path, qconfigs)


def filter_kwargs(func, kwargs):
    """
    Filter a dictionary of keyword arguments to only include the valid arguments of a function.
    Args:
        func: The function to check the arguments for.
        kwargs: The dictionary of keyword arguments to filter.
    Returns:
        A new dictionary containing only the valid keyword arguments.
    """
    valid_args = inspect.signature(func).parameters
    return {key: value for key, value in kwargs.items() if key in valid_args}
