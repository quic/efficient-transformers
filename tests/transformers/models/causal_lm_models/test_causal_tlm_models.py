# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import pytest

from QEfficient.utils.constants import Constants
from tests.utils.profile_test_config import load_test_config

from .check_causal_models import (
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100,
)

config_data = load_test_config("causal_model_configs")
spd_models = config_data["spd_causal_lm_models"]
test_models_spd = [model["model_name"] for model in spd_models]
model_config_dict = {model["model_name"]: model for model in spd_models}


@pytest.mark.qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_spd)
def test_causal_tlm_pytorch_vs_kv_vs_ort_vs_ai100(model_name):

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        num_speculative_tokens=Constants.NUM_SPECULATIVE_TOKENS,
    )


@pytest.mark.qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_spd)
def test_causal_tlm_pytorch_vs_kv_vs_ort_vs_ai100_CB(model_name):

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        num_speculative_tokens=Constants.NUM_SPECULATIVE_TOKENS,
        continuous_batching=True,
    )
