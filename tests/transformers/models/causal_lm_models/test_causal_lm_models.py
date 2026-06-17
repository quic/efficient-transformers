# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import pytest

from QEfficient.utils.test_utils import ModelConfig
from tests.utils.profile_test_config import load_test_config

from .check_causal_models import (
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100,
)

config_data = load_test_config("causal_model_configs")
causal_lm_models = config_data["causal_lm_models"]
test_models_causal = [model["model_name"] for model in causal_lm_models]


@pytest.mark.qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal)
def test_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name)


@pytest.mark.qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal)
def test_causal_lm_pytorch_vs_ort_vs_ai100_cb(model_name):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
    )
