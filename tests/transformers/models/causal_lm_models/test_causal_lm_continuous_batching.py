# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os

import pytest

from QEfficient.utils.test_utils import ModelConfig

from .check_causal_models import (
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100,
    get_custom_n_layers,
    get_hf_config_from_custom_config,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../configs/causal_model_configs.json")
with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    causal_lm_models = config_data["causal_lm_models"]
test_models_causal = [model["model_name"] for model in causal_lm_models]
model_config_dict = {model["model_name"]: model for model in causal_lm_models}


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal[1:2])
def test_full_causal_lm_pytorch_vs_ort_vs_ai100_cb(model_name):
    if model_name in ModelConfig.FULL_MODEL_TESTS_TO_SKIP:
        pytest.skip(f"Skipping full model test for {model_name} due to resource constraints.")
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
    )


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal[1:2])
def test_few_causal_lm_pytorch_vs_ort_vs_ai100_cb(model_name):

    n_layer = get_custom_n_layers(model_name)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        n_layer=n_layer,
        continuous_batching=True,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal[1:2])
def test_dummy_causal_lm_pytorch_vs_ort_vs_ai100_cb(model_name):

    hf_config = get_hf_config_from_custom_config(
        model_name, additional_params=model_config_dict[model_name].get("additional_params", {})
    )
    if model_name in ModelConfig.QUANTIZED_MODELS:
        n_layer = get_custom_n_layers(model_name)
        check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
            model_name,
            n_layer=n_layer,
            continuous_batching=True,
        )
    else:
        check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
            model_name,
            config=hf_config,
            continuous_batching=True,
        )
