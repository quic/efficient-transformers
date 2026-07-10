# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------


import os

import pytest

from QEfficient.utils.test_utils import (
    ModelConfig,
)

from .check_image_text_to_text_models import (
    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100,
    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_CB,
)

image_text_models_dict = {
    "llava-hf/llava-1.5-7b-hf": "hf-internal-testing/tiny-random-LlavaForConditionalGeneration",
    # "meta-llama/Llama-4-Scout-17B-16E-Instruct": "optimum-intel-internal-testing/tiny-random-llama4",
    "google/gemma-3-4b-it": "hf-internal-testing/tiny-random-Gemma3ForConditionalGeneration",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": "tiny-random/mistral-3",
    "Qwen/Qwen2.5-VL-3B-Instruct": "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
    "Qwen/Qwen3-VL-2B-Instruct": "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration",
    "Qwen/Qwen3-VL-30B-A3B-Instruct": "hf-internal-testing/tiny-random-Qwen3VLMoeForConditionalGeneration",
    "Qwen/Qwen3.5-0.8B": "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration",
    "Qwen/Qwen3.6-35B-A3B": "trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6",
    # "allenai/Molmo-7B-D-0924": "allenai/Molmo-7B-D-0924",
    # "OpenGVLab/InternVL2_5-1B": "optimum-intel-internal-testing/tiny-random-internvl2",
    # "OpenGVLab/InternVL3_5-1B": "OpenGVLab/InternVL3_5-1B",
}

if os.environ.get("QEFF_TEST_PROFILE", "").strip().lower() == "tiny_model":
    test_models = image_text_models_dict.values()
else:
    test_models = image_text_models_dict.keys()


@pytest.mark.non_qaic
@pytest.mark.vlm
@pytest.mark.parametrize("model_name", test_models)
@pytest.mark.parametrize("kv_offload", [True, False])
def test_export_compile(model_name, kv_offload):

    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    if model_name in ModelConfig.DUAL_QPC_MODELS and not kv_offload:
        pytest.skip("These models require kv_offload=True for testing.")

    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        kv_offload=kv_offload,
        export_compile_only=True,
    )


@pytest.mark.non_qaic
@pytest.mark.vlm
@pytest.mark.parametrize("model_name", test_models)
@pytest.mark.parametrize("kv_offload", [True])  # TODO: Add support for kv_offload=False
def test_export_compile_cb(model_name, kv_offload):

    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    if model_name in ModelConfig.DUAL_QPC_MODELS and not kv_offload:
        pytest.skip("These models require kv_offload=True for testing.")

    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_CB(
        model_name,
        kv_offload=kv_offload,
        export_compile_only=True,
    )


@pytest.mark.qaic
@pytest.mark.vlm
@pytest.mark.parametrize("model_name", test_models)
@pytest.mark.parametrize("kv_offload", [True, False])
def test_generate(model_name, kv_offload):

    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    if model_name in ModelConfig.DUAL_QPC_MODELS and not kv_offload:
        pytest.skip("These models require kv_offload=True for testing.")

    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        kv_offload=kv_offload,
    )


@pytest.mark.qaic
@pytest.mark.vlm
@pytest.mark.parametrize("model_name", test_models)
@pytest.mark.parametrize("kv_offload", [True])  # TODO: Add support for kv_offload=False
def test_generate_cb(model_name, kv_offload):

    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    if model_name in ModelConfig.DUAL_QPC_MODELS and not kv_offload:
        pytest.skip("These models require kv_offload=True for testing.")

    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_CB(
        model_name,
        kv_offload=kv_offload,
    )
