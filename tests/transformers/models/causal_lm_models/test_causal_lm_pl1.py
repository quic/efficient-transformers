# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import pytest
import torch
from transformers import AutoConfig

from QEfficient.utils.test_utils import ModelConfig
from tests.test_matrix import entries_by_model_name, model_names, select_test_entries

from .check_causal_models import (
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../configs/causal_model_configs.json")
causal_pl1_models = select_test_entries(CONFIG_PATH, "causal_lm_models_pl1")
test_models_pl1 = model_names(causal_pl1_models)
model_config_dict = entries_by_model_name(causal_pl1_models)


@pytest.mark.full_layers
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", test_models_pl1)
@pytest.mark.parametrize("retain_full_kv", [True, False])
def test_full_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100_pl1(model_name, retain_full_kv, manual_cleanup):
    if model_name == "gpt2" and retain_full_kv:
        pytest.skip("Skipping test for gpt2 with retain_full_kv=True as it is not supported.")

    torch.manual_seed(42)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        prompt_len=1,
        retain_full_kv=retain_full_kv,
        manual_cleanup=manual_cleanup,
        num_devices=4,
    )


@pytest.mark.few_layers
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", test_models_pl1)
@pytest.mark.parametrize("retain_full_kv", [True, False])
def test_few_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100_pl1(model_name, retain_full_kv, manual_cleanup):
    if model_name == "gpt2" and retain_full_kv:
        pytest.skip("Skipping test for gpt2 with retain_full_kv=True as it is not supported.")
    torch.manual_seed(42)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        n_layer=2,
        prompt_len=1,
        retain_full_kv=retain_full_kv,
        manual_cleanup=manual_cleanup,
    )


@pytest.mark.dummy_layers
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", test_models_pl1)
@pytest.mark.parametrize("retain_full_kv", [True, False])
def test_dummy_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100_pl1(model_name, retain_full_kv, manual_cleanup):
    if model_name == "gpt2" and retain_full_kv:
        pytest.skip("Skipping test for gpt2 with retain_full_kv=True as it is not supported.")

    torch.manual_seed(42)
    custom_config = model_config_dict[model_name]
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        **custom_config.get("additional_params", {}),
    )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        prompt_len=1,
        retain_full_kv=retain_full_kv,
        config=hf_config,
        manual_cleanup=manual_cleanup,
    )


@pytest.mark.full_layers
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", test_models_pl1)
@pytest.mark.parametrize("retain_full_kv", [True, False])
def test_full_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100_pl1_CB(model_name, retain_full_kv, manual_cleanup):
    if model_name == "gpt2" and retain_full_kv:
        pytest.skip("Skipping test for gpt2 with retain_full_kv=True as it is not supported.")
    torch.manual_seed(42)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        continuous_batching=True,
        prompt_len=1,
        retain_full_kv=retain_full_kv,
        manual_cleanup=manual_cleanup,
        num_devices=4,
    )


@pytest.mark.few_layers
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", test_models_pl1)
@pytest.mark.parametrize("retain_full_kv", [True, False])
def test_few_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100_pl1_CB(model_name, retain_full_kv, manual_cleanup):
    if model_name == "gpt2" and retain_full_kv:
        pytest.skip("Skipping test for gpt2 with retain_full_kv=True as it is not supported.")
    torch.manual_seed(42)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        n_layer=2,
        continuous_batching=True,
        prompt_len=1,
        retain_full_kv=retain_full_kv,
        manual_cleanup=manual_cleanup,
    )


@pytest.mark.dummy_layers
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", test_models_pl1)
@pytest.mark.parametrize("retain_full_kv", [True, False])
def test_dummy_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100_pl1_CB(model_name, retain_full_kv, manual_cleanup):
    if model_name == "gpt2" and retain_full_kv:
        pytest.skip("Skipping test for gpt2 with retain_full_kv=True as it is not supported.")

    torch.manual_seed(42)
    custom_config = model_config_dict[model_name]
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        **custom_config.get("additional_params", {}),
    )

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        continuous_batching=True,
        prompt_len=1,
        retain_full_kv=retain_full_kv,
        config=hf_config,
        manual_cleanup=manual_cleanup,
    )
