# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import pytest
from transformers import AutoConfig

from QEfficient.utils.constants import Constants
from QEfficient.utils.test_utils import ModelConfig
from tests.test_matrix import entries_by_model_name, model_names, select_test_entries

from .check_causal_models import (
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100,
    get_custom_n_layers,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../configs/causal_model_configs.json")
spd_models = select_test_entries(CONFIG_PATH, "spd_causal_lm_models")
test_models_spd = model_names(spd_models)
model_config_dict = entries_by_model_name(spd_models)


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


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_spd)
def test_few_causal_tlm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, manual_cleanup):
    n_layer = get_custom_n_layers(model_name)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        num_speculative_tokens=Constants.NUM_SPECULATIVE_TOKENS,
        n_layer=n_layer,
        manual_cleanup=manual_cleanup,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_spd)
def test_dummy_causal_tlm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, manual_cleanup):
    custom_config = model_config_dict[model_name]
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        **custom_config.get("additional_params", {}),
    )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        num_speculative_tokens=Constants.NUM_SPECULATIVE_TOKENS,
        config=hf_config,
        manual_cleanup=manual_cleanup,
    )


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_spd)
def test_full_causal_tlm_pytorch_vs_kv_vs_ort_vs_ai100_CB(model_name, manual_cleanup):
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
@pytest.mark.parametrize("model_name", test_models_spd)
def test_few_causal_tlm_pytorch_vs_kv_vs_ort_vs_ai100_CB(model_name, manual_cleanup):
    n_layer = get_custom_n_layers(model_name)
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
@pytest.mark.parametrize("model_name", test_models_spd)
def test_dummy_causal_tlm_pytorch_vs_kv_vs_ort_vs_ai100_CB(model_name, manual_cleanup):
    custom_config = model_config_dict[model_name]
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        **custom_config.get("additional_params", {}),
    )

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        num_speculative_tokens=Constants.NUM_SPECULATIVE_TOKENS,
        config=hf_config,
        continuous_batching=True,
        manual_cleanup=manual_cleanup,
    )
