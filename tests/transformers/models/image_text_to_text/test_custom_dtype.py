# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import json
import os

import pytest
import torch

from QEfficient.utils.test_utils import (
    ModelConfig,
)

from .test_image_text_to_text_models import (
    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../configs/image_text_model_configs.json")
with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    multimodal_models = config_data["image_text_custom_dtype_models"]
test_custom_dtype_support_models = [model_config["model_name"] for model_config in multimodal_models]
model_config_dict = {model["model_name"]: model for model in multimodal_models}

NEW_GENERATION_TOKENS = 10


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_custom_dtype_support_models)
@pytest.mark.parametrize("kv_offload", [True])
@pytest.mark.parametrize("torch_dtype", [torch.float16])
@pytest.mark.skip(
    reason="These tests are currently failing due to token mismatch. They need to be fixed and re-enabled."
)
def test_full_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_custom_dtype(
    model_name, kv_offload, torch_dtype, manual_cleanup
):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    if model_name in ModelConfig.DUAL_QPC_MODELS and not kv_offload:
        pytest.skip("These models require kv_offload=True for testing.")

    torch.manual_seed(42)
    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        kv_offload=kv_offload,
        manual_cleanup=manual_cleanup,
        num_devices=4,
        torch_dtype=torch_dtype,
    )


@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_custom_dtype_support_models)
@pytest.mark.parametrize("kv_offload", [True])
@pytest.mark.parametrize("torch_dtype", [torch.float16])
def test_few_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_custom_dtype(
    model_name, kv_offload, torch_dtype, manual_cleanup
):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    if model_name in ModelConfig.DUAL_QPC_MODELS and not kv_offload:
        pytest.skip("These models require kv_offload=True for testing.")

    torch.manual_seed(42)
    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        num_hidden_layers=model_config_dict[model_name]["num_layers"],
        kv_offload=kv_offload,
        manual_cleanup=manual_cleanup,
        torch_dtype=torch_dtype,
    )
