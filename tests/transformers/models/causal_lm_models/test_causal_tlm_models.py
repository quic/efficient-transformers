# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os

import pytest

from QEfficient.utils.constants import Constants

from .check_causal_models import (
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100,
    get_custom_n_layers,
    get_hf_config_from_custom_config,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../configs/causal_model_configs.json")
with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    spd_models = config_data["spd_causal_lm_models"]
test_models_spd = [model["model_name"] for model in spd_models]
model_config_dict = {model["model_name"]: model for model in spd_models}


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_spd)
def test_full_causal_tlm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, manual_cleanup):

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        num_speculative_tokens=Constants.NUM_SPECULATIVE_TOKENS,
        manual_cleanup=manual_cleanup,
        num_devices=4,
    )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        num_speculative_tokens=Constants.NUM_SPECULATIVE_TOKENS,
        continuous_batching=True,
        manual_cleanup=manual_cleanup,
        num_devices=4,
    )


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_spd[:1])
def test_few_causal_tlm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, manual_cleanup):

    n_layer = get_custom_n_layers(model_name)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        num_speculative_tokens=Constants.NUM_SPECULATIVE_TOKENS,
        n_layer=n_layer,
        manual_cleanup=manual_cleanup,
    )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        num_speculative_tokens=Constants.NUM_SPECULATIVE_TOKENS,
        n_layer=n_layer,
        continuous_batching=True,
        manual_cleanup=manual_cleanup,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_spd[:1])
def test_dummy_causal_tlm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, manual_cleanup):

    hf_config = get_hf_config_from_custom_config(
        model_name, additional_params=model_config_dict[model_name].get("additional_params", {})
    )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        num_speculative_tokens=Constants.NUM_SPECULATIVE_TOKENS,
        config=hf_config,
        manual_cleanup=manual_cleanup,
    )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        num_speculative_tokens=Constants.NUM_SPECULATIVE_TOKENS,
        config=hf_config,
        continuous_batching=True,
        manual_cleanup=manual_cleanup,
    )
