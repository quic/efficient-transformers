# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------


import json
import os
from typing import Optional

import onnx
import pytest
import torch
from transformers import AutoConfig

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.test_utils import ModelConfig, get_custom_n_layers, load_hf_causal_lm_model

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../configs/causal_model_configs.json")
with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    blockedKV_models = config_data["blockedKV_causal_lm_models"]
test_models_blockedKV = [model["model_name"] for model in blockedKV_models]
model_config_dict = {model["model_name"]: model for model in blockedKV_models}
torch.manual_seed(42)


def check_blockedKV_onnx_function_count_with_subfunction(
    model_name: str, manual_cleanup: callable, n_layer: int = -1, config: Optional[AutoConfig] = None
):
    """
    Export twice with `use_onnx_subfunctions=True`:
      1) without blocking
      2) with KV blocking (NUM_KV_BLOCKS=2)

    Verify that the number of ONNX functions is identical.
    """
    # Export with subfunctions, NO blocking
    model_no_block = load_hf_causal_lm_model(model_name, num_hidden_layers=n_layer, config=config)
    qeff_no_block = QEFFAutoModelForCausalLM(model_no_block, pretrained_model_name_or_path=model_name, qaic_config=None)
    qeff_no_block.export(use_onnx_subfunctions=True, offload_pt_weights=False)
    onnx_no_block = onnx.load(qeff_no_block.onnx_path, load_external_data=False)
    num_functions_no_block = len(onnx_no_block.functions)
    manual_cleanup(qeff_no_block.onnx_path)
    # Export with subfunctions, WITH KV blocking
    NUM_KV_BLOCKS = 2
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS)

    model_kv_block = load_hf_causal_lm_model(model_name, num_hidden_layers=n_layer, config=config)
    qeff_kv_block = QEFFAutoModelForCausalLM(
        model_kv_block, pretrained_model_name_or_path=model_name, qaic_config=qaic_config
    )
    qeff_kv_block.export(use_onnx_subfunctions=True, offload_pt_weights=False)
    onnx_kv_block = onnx.load(qeff_kv_block.onnx_path, load_external_data=False)
    num_functions_kv_block = len(onnx_kv_block.functions)

    assert num_functions_no_block == num_functions_kv_block
    manual_cleanup(qeff_kv_block.onnx_path)


@pytest.mark.full_layers
@pytest.mark.feature
@pytest.mark.parametrize("model_name", test_models_blockedKV)
def test_full_blockedKV_onnx_function_count_with_subfunction(model_name, manual_cleanup):
    # Keep model small for test runtime, and avoid CB path (not needed for function count).
    check_blockedKV_onnx_function_count_with_subfunction(model_name, manual_cleanup=manual_cleanup)


@pytest.mark.few_layers
@pytest.mark.feature
@pytest.mark.parametrize("model_name", test_models_blockedKV)
def test_few_blockedKV_onnx_function_count_with_subfunction(model_name, manual_cleanup):
    # Keep model small for test runtime, and avoid CB path (not needed for function count).
    n_layer = get_custom_n_layers(model_name)

    check_blockedKV_onnx_function_count_with_subfunction(model_name, n_layer=n_layer, manual_cleanup=manual_cleanup)


@pytest.mark.dummy_layers
@pytest.mark.feature
@pytest.mark.parametrize("model_name", test_models_blockedKV)
def test_dummy_blockedKV_onnx_function_count_with_subfunction(model_name, manual_cleanup):
    # Keep model small for test runtime, and avoid CB path (not needed for function count).
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        **model_config_dict[model_name].get("additional_params", {}),
    )
    n_layer = get_custom_n_layers(model_name)
    if model_name in ModelConfig.QUANTIZED_MODELS:
        n_layer = get_custom_n_layers(model_name)
        hf_config = None
    check_blockedKV_onnx_function_count_with_subfunction(
        model_name, n_layer=n_layer, config=hf_config, manual_cleanup=manual_cleanup
    )
