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
    blockedKV_models = config_data["blockedKV_causal_lm_models"]
test_models_blockedKV = [model["model_name"] for model in blockedKV_models]
model_config_dict = {model["model_name"]: model for model in blockedKV_models}


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_blockedKV)
def test_full_causal_blockedKV_pytorch_vs_kv_vs_ort_vs_ai100(model_name):

    qaic_config = dict(num_kv_blocks=Constants.NUM_KV_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name=model_name, qaic_config=qaic_config)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, continuous_batching=True, qaic_config=qaic_config
    )


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_blockedKV)
def test_few_causal_blockedKV_pytorch_vs_kv_vs_ort_vs_ai100(model_name):

    n_layer = get_custom_n_layers(model_name)
    qaic_config = dict(num_kv_blocks=Constants.NUM_KV_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name=model_name, n_layer=n_layer, qaic_config=qaic_config)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, n_layer=n_layer, continuous_batching=True, qaic_config=qaic_config
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_blockedKV)
def test_dummy_causal_blockedKV_pytorch_vs_kv_vs_ort_vs_ai100(model_name):

    qaic_config = dict(num_kv_blocks=Constants.NUM_KV_BLOCKS)
    hf_config = get_hf_config_from_custom_config(
        model_name, additional_params=model_config_dict[model_name].get("additional_params", {})
    )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name=model_name, qaic_config=qaic_config, config=hf_config)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, continuous_batching=True, qaic_config=qaic_config, config=hf_config
    )
