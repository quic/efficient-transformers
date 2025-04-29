# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import PretrainedConfig

from QEfficient.utils._utils import filter_kwargs


def get_speculative_config(pretrained_model_name_or_path, **kwargs) -> dict:
    if not isinstance(pretrained_model_name_or_path, (str, Path)):
        raise ValueError(
            f"`pretrained_config` must be a string or Path object but is of type {type(pretrained_model_name_or_path)}"
        )
    try:
        speculative_config, _ = PretrainedConfig.get_config_dict(
            pretrained_model_name_or_path, _configuration_file="speculator_config.json", **kwargs
        )
    except OSError as err:
        raise OSError(f"{err}.\nFile 'speculator_config.json' is expected to exist to apply turbo projections.")
    return speculative_config


def get_speculative_weights(pretrained_model_name_or_path, **kwargs) -> str:
    turbo_weights_file = "speculator.safetensors"
    hf_hub_kwargs = filter_kwargs(hf_hub_download, kwargs)
    if (local_path := Path(pretrained_model_name_or_path)).exists():
        if not local_path.is_dir():
            raise ValueError(f"local model path {local_path} must point to an existing dictionary")
        weights_path = local_path / turbo_weights_file
        if not weights_path.exists():
            raise FileNotFoundError(f"weights path {weights_path} does not exist.")
    else:
        weights_path = hf_hub_download(pretrained_model_name_or_path, filename=turbo_weights_file, **hf_hub_kwargs)
    return str(weights_path)
    