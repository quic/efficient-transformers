# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import pytest

from .check_causal_models import (
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100,
)

causal_lm_models_pl1_dict = {
    "gpt2": "hf-internal-testing/tiny-random-GPT2LMHeadModel",
    "openai/gpt-oss-20b": "tiny-random/gpt-oss-bf16",
}

if os.environ.get("QEFF_TEST_PROFILE", "").strip().lower() == "tiny_model":
    test_models_pl1 = list(causal_lm_models_pl1_dict.values())
else:
    test_models_pl1 = list(causal_lm_models_pl1_dict.keys())


@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", test_models_pl1)
@pytest.mark.parametrize("retain_full_kv", [True, False])
def test_export_compile_pl1(model_name, retain_full_kv):

    if model_name == "gpt2" and retain_full_kv:
        pytest.skip("Skipping test for gpt2 with retain_full_kv=True as it is not supported.")

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        prompt_len=1,
        retain_full_kv=retain_full_kv,
        export_compile_only=True,
    )


@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", test_models_pl1)
@pytest.mark.parametrize("retain_full_kv", [True, False])
def test_export_compile_pl1_cb(model_name, retain_full_kv):

    if model_name == "gpt2" and retain_full_kv:
        pytest.skip("Skipping test for gpt2 with retain_full_kv=True as it is not supported.")

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        continuous_batching=True,
        prompt_len=1,
        retain_full_kv=retain_full_kv,
        export_compile_only=True,
    )


@pytest.mark.qaic
@pytest.mark.llm
@pytest.mark.parametrize("model_name", test_models_pl1)
@pytest.mark.parametrize("retain_full_kv", [True, False])
def test_generate_pl1(model_name, retain_full_kv):

    if model_name == "gpt2" and retain_full_kv:
        pytest.skip("Skipping test for gpt2 with retain_full_kv=True as it is not supported.")

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        prompt_len=1,
        retain_full_kv=retain_full_kv,
    )


@pytest.mark.llm
@pytest.mark.qaic
@pytest.mark.parametrize("model_name", test_models_pl1)
@pytest.mark.parametrize("retain_full_kv", [True, False])
def test_generate_pl1_cb(model_name, retain_full_kv):

    if model_name == "gpt2" and retain_full_kv:
        pytest.skip("Skipping test for gpt2 with retain_full_kv=True as it is not supported.")

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        continuous_batching=True,
        prompt_len=1,
        retain_full_kv=retain_full_kv,
    )
