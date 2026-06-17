# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import pytest
import torch

from tests.utils.profile_test_config import load_test_config

from .check_causal_models import (
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100,
)

config_data = load_test_config("causal_model_configs")
causal_pl1_models = config_data["causal_lm_models_pl1"]
test_models_pl1 = [model["model_name"] for model in causal_pl1_models]
model_config_dict = {model["model_name"]: model for model in causal_pl1_models}


@pytest.mark.llm_model
@pytest.mark.qaic
@pytest.mark.parametrize("model_name", test_models_pl1)
@pytest.mark.parametrize("retain_full_kv", [True, False])
def test_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100_pl1(model_name, retain_full_kv):
    if model_name == "gpt2" and retain_full_kv:
        pytest.skip("Skipping test for gpt2 with retain_full_kv=True as it is not supported.")

    torch.manual_seed(42)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        prompt_len=1,
        retain_full_kv=retain_full_kv,
    )


@pytest.mark.llm_model
@pytest.mark.qaic
@pytest.mark.parametrize("model_name", test_models_pl1)
@pytest.mark.parametrize("retain_full_kv", [True, False])
def test_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100_pl1_CB(model_name, retain_full_kv):
    if model_name == "gpt2" and retain_full_kv:
        pytest.skip("Skipping test for gpt2 with retain_full_kv=True as it is not supported.")

    torch.manual_seed(42)

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        continuous_batching=True,
        prompt_len=1,
        retain_full_kv=retain_full_kv,
    )
