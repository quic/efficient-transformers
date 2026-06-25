# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
import os

import pytest

from .check_embedding_models import check_qwen3_vl_embedding_cpu_vs_ai100_mad_parity

image_text_embedding_models = {
    "Qwen/Qwen3-VL-Embedding-8B": "optimum-intel-internal-testing/tiny-random-qwen3-vl-embedding"
}

if os.environ.get("QEFF_TEST_PROFILE", "").strip().lower() == "tiny_model":
    test_models = image_text_embedding_models.values()
else:
    test_models = image_text_embedding_models.keys()


@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_models)
def test_export_compile(model_name):

    check_qwen3_vl_embedding_cpu_vs_ai100_mad_parity(model_name, export_compile_only=True)


@pytest.mark.qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_models)
def test_generate(model_name):

    check_qwen3_vl_embedding_cpu_vs_ai100_mad_parity(model_name)
