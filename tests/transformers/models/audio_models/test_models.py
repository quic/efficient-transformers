# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import pytest

from tests.utils.profile_test_config import load_test_config

from .check_audio_models import check_seq2seq_pytorch_vs_kv_vs_ort_vs_ai100

config_data = load_test_config("audio_model_configs")
test_models = config_data["speech_seq2seq_models"]
print(test_models)

@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models)
def test_export_compile(model_name):

    check_seq2seq_pytorch_vs_kv_vs_ort_vs_ai100(model_name, export_compile_only=True)


@pytest.mark.qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models)
def test_generate(model_name):

    check_seq2seq_pytorch_vs_kv_vs_ort_vs_ai100(model_name)
