# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------


import os

import pytest
import torch

from QEfficient.utils.test_utils import (
    ModelConfig,
)

from .check_image_text_to_text_models import check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100

image_text_custom_dtype_models_dict = {
    "OpenGVLab/InternVL2_5-1B": "optimum-intel-internal-testing/tiny-random-internvl2",
    "google/gemma-3-4b-it": "hf-internal-testing/tiny-random-Gemma3ForConditionalGeneration",
    "llava-hf/llava-1.5-7b-hf": "hf-internal-testing/tiny-random-LlavaForConditionalGeneration",
}

if os.environ.get("QEFF_TEST_PROFILE", "").strip().lower() == "tiny_model":
    test_custom_dtype_support_models = image_text_custom_dtype_models_dict.values()
else:
    test_custom_dtype_support_models = image_text_custom_dtype_models_dict.keys()


@pytest.mark.non_qaic
@pytest.mark.vlm
@pytest.mark.parametrize("model_name", test_custom_dtype_support_models)
def test_export_compile_custom_dtype(model_name):

    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")

    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        kv_offload=True,
        torch_dtype=torch.float16,
        export_compile_only=True,
    )


@pytest.mark.qaic
@pytest.mark.vlm
@pytest.mark.parametrize("model_name", test_custom_dtype_support_models)
def test_generate_custom_dtype(model_name):

    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")

    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        kv_offload=True,
        torch_dtype=torch.float16,
    )
