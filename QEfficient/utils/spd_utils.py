# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import PretrainedConfig

from QEfficient.utils.helper_utils import filter_kwargs


def _get_speculative_config(speculative_model, **kwargs) -> dict:
    assert isinstance(speculative_model, (str, Path))
    try:
        speculative_config, _ = PretrainedConfig.get_config_dict(
            speculative_model, _configuration_file="speculator_config.json", **kwargs
        )
    except OSError as err:
        raise OSError(f"{err}.\nFile 'speculator_config.json' is expected to exist to apply turbo projections.")
    speculative_config["model_type"] = "turbo"
    return speculative_config


def get_speculative_weights(speculative_model, **kwargs) -> str:
    turbo_weights_file = "speculator.safetensors"
    hf_hub_kwargs = filter_kwargs(hf_hub_download, kwargs)
    if (local_path := Path(speculative_model)).exists():
        assert local_path.is_dir()
        weights_path = local_path / turbo_weights_file
        assert weights_path.exists()
    else:
        weights_path = hf_hub_download(speculative_model, filename=turbo_weights_file, **hf_hub_kwargs)
    return str(weights_path)


def get_speculative_config(speculative_model, **kwargs):
    speculative_config: dict = _get_speculative_config(speculative_model, **kwargs)
    speculative_weights: str = get_speculative_weights(speculative_model, **kwargs)
    speculative_config["speculative_weights"] = speculative_weights
    return speculative_config
