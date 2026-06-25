# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import pytest

from .check_sequence_models import check_seq_classification_pytorch_vs_ai100

seq_classification_models_dict = {
    "meta-llama/Llama-Prompt-Guard-2-22M": "hf-internal-testing/tiny-random-DebertaV2ForSequenceClassification"
}

if os.environ.get("QEFF_TEST_PROFILE", "").strip().lower() == "tiny_model":
    test_models = seq_classification_models_dict.values()
else:
    test_models = seq_classification_models_dict.keys()


@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models)
def test_export_compile(model_name):

    check_seq_classification_pytorch_vs_ai100(model_name=model_name, seq_len=32, export_compile_only=True)


@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models)
def test_export_compile_multiple_seq_len(model_name):

    check_seq_classification_pytorch_vs_ai100(model_name=model_name, seq_len=[32, 64, 128], export_compile_only=True)


@pytest.mark.llm_model
@pytest.mark.qaic
@pytest.mark.parametrize("model_name", test_models)
def test_generate(model_name):

    check_seq_classification_pytorch_vs_ai100(
        model_name=model_name,
        seq_len=32,
    )


@pytest.mark.llm_model
@pytest.mark.qaic
@pytest.mark.parametrize("model_name", test_models)
def test_generate_multiple_seq_len(model_name):

    check_seq_classification_pytorch_vs_ai100(
        model_name=model_name,
        seq_len=[32, 64, 128],
    )
