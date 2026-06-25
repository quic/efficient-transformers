# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import pytest

from .check_audio_embedding_models import check_ctc_pytorch_vs_kv_vs_ort_vs_ai100

audio_embedding_models = {"facebook/wav2vec2-base-960h": "hf-internal-testing/tiny-random-Wav2Vec2ForCTC"}

if os.environ.get("QEFF_TEST_PROFILE", "").strip().lower() == "tiny_model":
    test_models = audio_embedding_models.values()
else:
    test_models = audio_embedding_models.keys()


@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models)
def test_export_compile(model_name):

    check_ctc_pytorch_vs_kv_vs_ort_vs_ai100(model_name, export_compile_only=True)


@pytest.mark.qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models)
def test_generate(model_name):

    check_ctc_pytorch_vs_kv_vs_ort_vs_ai100(model_name)
