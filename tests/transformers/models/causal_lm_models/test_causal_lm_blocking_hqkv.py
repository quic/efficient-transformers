# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os

import pytest
from transformers import AutoConfig

from QEfficient.utils.test_utils import ModelConfig

from .check_causal_models import (
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100,
    get_custom_n_layers,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../configs/causal_model_configs.json")
with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    blockedKV_models = config_data["blockedKV_causal_lm_models"]
test_models_blockedKV = [model["model_name"] for model in blockedKV_models]
model_config_dict = {model["model_name"]: model for model in blockedKV_models}

ALL_BLOCKING_MODES = ["h", "q", "kv", "qkv", "hqkv", "kv_headpar"]

HEAD_BLOCK_SIZE = 8
NUM_KV_BLOCKS = 2
NUM_Q_BLOCKS = 2


def _build_qaic_config(blocking_mode):
    cfg = {"blocking_mode": blocking_mode}
    if blocking_mode in ("h", "hqkv"):
        cfg["head_block_size"] = HEAD_BLOCK_SIZE
    if blocking_mode in ("kv", "kv_headpar", "qkv", "hqkv"):
        cfg["num_kv_blocks"] = NUM_KV_BLOCKS
    if blocking_mode in ("q", "qkv", "hqkv"):
        cfg["num_q_blocks"] = NUM_Q_BLOCKS
    return cfg


@pytest.mark.full_layers
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.parametrize("blocking_mode", ALL_BLOCKING_MODES)
@pytest.mark.parametrize("model_name", test_models_blockedKV[:1])
def test_full_causal_all_blocking_pytorch_vs_kv_vs_ort_vs_ai100(model_name, blocking_mode, manual_cleanup):
    qaic_config = _build_qaic_config(blocking_mode)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        manual_cleanup=manual_cleanup,
        num_devices=4,
    )


@pytest.mark.few_layers
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.parametrize("blocking_mode", ALL_BLOCKING_MODES)
@pytest.mark.parametrize("model_name", test_models_blockedKV[:1])
def test_few_causal_all_blocking_pytorch_vs_kv_vs_ort_vs_ai100(model_name, blocking_mode, manual_cleanup):
    n_layer = get_custom_n_layers(model_name)
    qaic_config = _build_qaic_config(blocking_mode)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        n_layer=n_layer,
        manual_cleanup=manual_cleanup,
    )


@pytest.mark.dummy_layers
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.parametrize("blocking_mode", ALL_BLOCKING_MODES)
@pytest.mark.parametrize("model_name", test_models_blockedKV[:1])
def test_dummy_causal_all_blocking_pytorch_vs_kv_vs_ort_vs_ai100(model_name, blocking_mode, manual_cleanup):
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        **model_config_dict[model_name].get("additional_params", {}),
    )
    n_layer = -1
    if model_name in ModelConfig.QUANTIZED_MODELS:
        n_layer = get_custom_n_layers(model_name)
        hf_config = None
    qaic_config = _build_qaic_config(blocking_mode)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        n_layer=n_layer,
        config=hf_config,
        manual_cleanup=manual_cleanup,
    )


@pytest.mark.full_layers
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.parametrize("blocking_mode", ALL_BLOCKING_MODES)
@pytest.mark.parametrize("model_name", test_models_blockedKV[:1])
def test_full_causal_all_blocking_pytorch_vs_kv_vs_ort_vs_ai100_CB(model_name, blocking_mode, manual_cleanup):
    qaic_config = _build_qaic_config(blocking_mode)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
        num_devices=4,
    )


@pytest.mark.few_layers
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.parametrize("blocking_mode", ALL_BLOCKING_MODES)
@pytest.mark.parametrize("model_name", test_models_blockedKV[:1])
def test_few_causal_all_blocking_pytorch_vs_kv_vs_ort_vs_ai100_CB(model_name, blocking_mode, manual_cleanup):
    n_layer = get_custom_n_layers(model_name)
    qaic_config = _build_qaic_config(blocking_mode)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        n_layer=n_layer,
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
    )


@pytest.mark.dummy_layers
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.parametrize("blocking_mode", ALL_BLOCKING_MODES)
@pytest.mark.parametrize("model_name", test_models_blockedKV[:1])
def test_dummy_causal_all_blocking_pytorch_vs_kv_vs_ort_vs_ai100_CB(model_name, blocking_mode, manual_cleanup):
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        **model_config_dict[model_name].get("additional_params", {}),
    )
    n_layer = -1
    if model_name in ModelConfig.QUANTIZED_MODELS:
        n_layer = get_custom_n_layers(model_name)
        hf_config = None
    qaic_config = _build_qaic_config(blocking_mode)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        n_layer=n_layer,
        config=hf_config,
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
    )
