# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import pytest

from QEfficient.utils.test_utils import ModelConfig

from .check_embedding_models import check_embed_pytorch_vs_ort_vs_ai100

tiny_embedding_models = [
    pytest.param("hf-internal-testing/tiny-random-BertModel", "mean"),
    pytest.param("hf-internal-testing/tiny-random-BertModel", "cls"),
]

original_embedding_models = [
    pytest.param("jinaai/jina-embeddings-v2-base-code", "mean"),
    pytest.param("sentence-transformers/nli-bert-base-cls-pooling", "cls"),
]

if os.environ.get("QEFF_TEST_PROFILE", "").strip().lower() == "tiny_model":
    embed_test_models = tiny_embedding_models
else:
    embed_test_models = original_embedding_models


@pytest.mark.llm_model
@pytest.mark.parametrize("model_name, pooling", embed_test_models)
def test_export_compile(model_name, pooling):

    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    check_embed_pytorch_vs_ort_vs_ai100(
        model_name=model_name,
        seq_len=32,
        export_compile_only=True,
    )


@pytest.mark.llm_model
@pytest.mark.parametrize("model_name, pooling", embed_test_models)
def test_export_compile_with_pooling(model_name, pooling):

    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    check_embed_pytorch_vs_ort_vs_ai100(
        model_name=model_name,
        seq_len=32,
        pooling=pooling,
        export_compile_only=True,
    )


@pytest.mark.llm_model
@pytest.mark.parametrize("model_name, pooling", embed_test_models[:1])
def test_export_compile_with_multiple_seq_len(model_name, pooling):

    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    check_embed_pytorch_vs_ort_vs_ai100(model_name=model_name, seq_len=[32, 20], export_compile_only=True)


@pytest.mark.qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name, pooling", embed_test_models)
def test_generate(model_name, pooling):

    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    check_embed_pytorch_vs_ort_vs_ai100(
        model_name=model_name,
        seq_len=32,
    )


@pytest.mark.qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name, pooling", embed_test_models)
def test_generate_with_pooling(model_name, pooling):

    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    check_embed_pytorch_vs_ort_vs_ai100(
        model_name=model_name,
        seq_len=32,
        pooling=pooling,
    )


@pytest.mark.qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name, pooling", embed_test_models[:1])
def test_generate_with_multiple_seq_len(model_name, pooling):

    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    check_embed_pytorch_vs_ort_vs_ai100(model_name=model_name, seq_len=[32, 20])
