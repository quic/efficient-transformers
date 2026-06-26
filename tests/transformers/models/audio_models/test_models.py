# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import pytest

from .check_audio_models import check_seq2seq_pytorch_vs_kv_vs_ort_vs_ai100

audio_models = {"openai/whisper-tiny": "hf-internal-testing/tiny-random-WhisperForConditionalGeneration"}

if os.environ.get("QEFF_TEST_PROFILE", "").strip().lower() == "tiny_model":
    test_models = audio_models.values()
else:
    test_models = audio_models.keys()


@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", test_models)
def test_export_compile(model_name):

    check_seq2seq_pytorch_vs_kv_vs_ort_vs_ai100(model_name, export_compile_only=True)


@pytest.mark.qaic
@pytest.mark.llm
@pytest.mark.parametrize("model_name", test_models)
def test_generate(model_name):

    check_seq2seq_pytorch_vs_kv_vs_ort_vs_ai100(model_name)
