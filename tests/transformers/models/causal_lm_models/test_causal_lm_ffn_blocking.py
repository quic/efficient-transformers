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


@pytest.mark.full_layers
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", test_models_blockedKV[:1])
def test_full_causal_all_blocking_pytorch_vs_kv_vs_ort_vs_ai100(model_name, manual_cleanup):
    # Skip MoE models: FFN blocking covers dense FFNs, not MoE blocks.
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
    )
    if getattr(hf_config, "num_local_experts", 1) not in (None, 0, 1) or getattr(hf_config, "num_experts", 1) not in (
        None,
        0,
        1,
    ):
        pytest.skip(f"Skipping MoE model {model_name} for FFN blocking test")

    # FFN blocking sizes
    NUM_TOKEN_BLOCKS = 2
    NUM_WEIGHT_BLOCKS = 2

    # Attention blocking sizes (for the combined case).
    HEAD_BLOCK_SIZE = 8
    NUM_KV_BLOCKS = 2
    NUM_Q_BLOCKS = 2

    # token-only FFN blocking
    qaic_config = dict(enable_ffn_blocking=True, ffn_blocking_mode="t", num_token_blocks=NUM_TOKEN_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, qaic_config=qaic_config, manual_cleanup=manual_cleanup, num_devices=4
    )

    # weight-only FFN blocking
    qaic_config = dict(enable_ffn_blocking=True, ffn_blocking_mode="w", num_weight_blocks=NUM_WEIGHT_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, qaic_config=qaic_config, manual_cleanup=manual_cleanup, num_devices=4
    )

    # token+weight FFN blocking
    qaic_config = dict(
        enable_ffn_blocking=True,
        ffn_blocking_mode="tw",
        num_token_blocks=NUM_TOKEN_BLOCKS,
        num_weight_blocks=NUM_WEIGHT_BLOCKS,
    )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, qaic_config=qaic_config, manual_cleanup=manual_cleanup, num_devices=4
    )

    # token+weight FFN blocking + HQKV attention blocking
    qaic_config = dict(
        enable_blocking=True,
        enable_ffn_blocking=True,
        ffn_blocking_mode="tw",
        num_token_blocks=NUM_TOKEN_BLOCKS,
        num_weight_blocks=NUM_WEIGHT_BLOCKS,
        blocking_mode="hqkv",
        head_block_size=HEAD_BLOCK_SIZE,
        num_kv_blocks=NUM_KV_BLOCKS,
        num_q_blocks=NUM_Q_BLOCKS,
    )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, qaic_config=qaic_config, manual_cleanup=manual_cleanup, num_devices=4
    )


@pytest.mark.few_layers
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", test_models_blockedKV[:1])
def test_few_causal_all_blocking_pytorch_vs_kv_vs_ort_vs_ai100(model_name, manual_cleanup):
    # Skip MoE models: FFN blocking covers dense FFNs, not MoE blocks.
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
    )
    if getattr(hf_config, "num_local_experts", 1) not in (None, 0, 1) or getattr(hf_config, "num_experts", 1) not in (
        None,
        0,
        1,
    ):
        pytest.skip(f"Skipping MoE model {model_name} for FFN blocking test")

    # FFN blocking sizes
    NUM_TOKEN_BLOCKS = 2
    NUM_WEIGHT_BLOCKS = 2

    # Attention blocking sizes (for the combined case).
    HEAD_BLOCK_SIZE = 8
    NUM_KV_BLOCKS = 2
    NUM_Q_BLOCKS = 2

    n_layer = get_custom_n_layers(model_name)

    # token-only FFN blocking
    qaic_config = dict(enable_ffn_blocking=True, ffn_blocking_mode="t", num_token_blocks=NUM_TOKEN_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, n_layer=n_layer, qaic_config=qaic_config, manual_cleanup=manual_cleanup
    )

    # weight-only FFN blocking
    qaic_config = dict(enable_ffn_blocking=True, ffn_blocking_mode="w", num_weight_blocks=NUM_WEIGHT_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, n_layer=n_layer, qaic_config=qaic_config, manual_cleanup=manual_cleanup
    )

    # token+weight FFN blocking
    qaic_config = dict(
        enable_ffn_blocking=True,
        ffn_blocking_mode="tw",
        num_token_blocks=NUM_TOKEN_BLOCKS,
        num_weight_blocks=NUM_WEIGHT_BLOCKS,
    )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, n_layer=n_layer, qaic_config=qaic_config, manual_cleanup=manual_cleanup
    )

    # token+weight FFN blocking + HQKV attention blocking
    qaic_config = dict(
        enable_blocking=True,
        enable_ffn_blocking=True,
        ffn_blocking_mode="tw",
        num_token_blocks=NUM_TOKEN_BLOCKS,
        num_weight_blocks=NUM_WEIGHT_BLOCKS,
        blocking_mode="hqkv",
        head_block_size=HEAD_BLOCK_SIZE,
        num_kv_blocks=NUM_KV_BLOCKS,
        num_q_blocks=NUM_Q_BLOCKS,
    )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, n_layer=n_layer, qaic_config=qaic_config, manual_cleanup=manual_cleanup
    )


@pytest.mark.dummy_layers
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", test_models_blockedKV[:1])
def test_dummy_causal_all_blocking_pytorch_vs_kv_vs_ort_vs_ai100(model_name, manual_cleanup):
    # Skip MoE models: FFN blocking covers dense FFNs, not MoE blocks.
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
    )
    if getattr(hf_config, "num_local_experts", 1) not in (None, 0, 1) or getattr(hf_config, "num_experts", 1) not in (
        None,
        0,
        1,
    ):
        pytest.skip(f"Skipping MoE model {model_name} for FFN blocking test")

    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        **model_config_dict[model_name].get("additional_params", {}),
    )
    n_layer = -1
    if model_name in ModelConfig.QUANTIZED_MODELS:
        n_layer = get_custom_n_layers(model_name)
        hf_config = None

    # FFN blocking sizes
    NUM_TOKEN_BLOCKS = 2
    NUM_WEIGHT_BLOCKS = 2

    # Attention blocking sizes (for the combined case).
    HEAD_BLOCK_SIZE = 8
    NUM_KV_BLOCKS = 2
    NUM_Q_BLOCKS = 2

    n_layer = get_custom_n_layers(model_name)

    # token-only FFN blocking
    qaic_config = dict(enable_ffn_blocking=True, ffn_blocking_mode="t", num_token_blocks=NUM_TOKEN_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, n_layer=n_layer, qaic_config=qaic_config, config=hf_config, manual_cleanup=manual_cleanup
    )

    # weight-only FFN blocking
    qaic_config = dict(enable_ffn_blocking=True, ffn_blocking_mode="w", num_weight_blocks=NUM_WEIGHT_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, n_layer=n_layer, qaic_config=qaic_config, config=hf_config, manual_cleanup=manual_cleanup
    )

    # token+weight FFN blocking
    qaic_config = dict(
        enable_ffn_blocking=True,
        ffn_blocking_mode="tw",
        num_token_blocks=NUM_TOKEN_BLOCKS,
        num_weight_blocks=NUM_WEIGHT_BLOCKS,
    )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, n_layer=n_layer, qaic_config=qaic_config, config=hf_config, manual_cleanup=manual_cleanup
    )

    # token+weight FFN blocking + HQKV attention blocking
    qaic_config = dict(
        enable_blocking=True,
        enable_ffn_blocking=True,
        ffn_blocking_mode="tw",
        num_token_blocks=NUM_TOKEN_BLOCKS,
        num_weight_blocks=NUM_WEIGHT_BLOCKS,
        blocking_mode="hqkv",
        head_block_size=HEAD_BLOCK_SIZE,
        num_kv_blocks=NUM_KV_BLOCKS,
        num_q_blocks=NUM_Q_BLOCKS,
    )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, n_layer=n_layer, qaic_config=qaic_config, config=hf_config, manual_cleanup=manual_cleanup
    )


@pytest.mark.full_layers
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", test_models_blockedKV[:1])
def test_full_causal_all_blocking_pytorch_vs_kv_vs_ort_vs_ai100_CB(model_name, manual_cleanup):
    # Skip MoE models: FFN blocking covers dense FFNs, not MoE blocks.
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
    )
    if getattr(hf_config, "num_local_experts", 1) not in (None, 0, 1) or getattr(hf_config, "num_experts", 1) not in (
        None,
        0,
        1,
    ):
        pytest.skip(f"Skipping MoE model {model_name} for FFN blocking test")

    # FFN blocking sizes
    NUM_TOKEN_BLOCKS = 2
    NUM_WEIGHT_BLOCKS = 2

    # Attention blocking sizes (for the combined case).
    HEAD_BLOCK_SIZE = 8
    NUM_KV_BLOCKS = 2
    NUM_Q_BLOCKS = 2

    # token-only FFN blocking
    qaic_config = dict(enable_ffn_blocking=True, ffn_blocking_mode="t", num_token_blocks=NUM_TOKEN_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
        num_devices=4,
    )

    # weight-only FFN blocking
    qaic_config = dict(enable_ffn_blocking=True, ffn_blocking_mode="w", num_weight_blocks=NUM_WEIGHT_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
        num_devices=4,
    )

    # token+weight FFN blocking
    qaic_config = dict(
        enable_ffn_blocking=True,
        ffn_blocking_mode="tw",
        num_token_blocks=NUM_TOKEN_BLOCKS,
        num_weight_blocks=NUM_WEIGHT_BLOCKS,
    )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
        num_devices=4,
    )

    # token+weight FFN blocking + HQKV attention blocking
    qaic_config = dict(
        enable_blocking=True,
        enable_ffn_blocking=True,
        ffn_blocking_mode="tw",
        num_token_blocks=NUM_TOKEN_BLOCKS,
        num_weight_blocks=NUM_WEIGHT_BLOCKS,
        blocking_mode="hqkv",
        head_block_size=HEAD_BLOCK_SIZE,
        num_kv_blocks=NUM_KV_BLOCKS,
        num_q_blocks=NUM_Q_BLOCKS,
    )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, qaic_config=qaic_config, manual_cleanup=manual_cleanup, num_devices=4
    )


@pytest.mark.few_layers
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", test_models_blockedKV[:1])
def test_few_causal_all_blocking_pytorch_vs_kv_vs_ort_vs_ai100_CB(model_name, manual_cleanup):
    # Skip MoE models: FFN blocking covers dense FFNs, not MoE blocks.
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
    )
    if getattr(hf_config, "num_local_experts", 1) not in (None, 0, 1) or getattr(hf_config, "num_experts", 1) not in (
        None,
        0,
        1,
    ):
        pytest.skip(f"Skipping MoE model {model_name} for FFN blocking test")

    # FFN blocking sizes
    NUM_TOKEN_BLOCKS = 2
    NUM_WEIGHT_BLOCKS = 2

    # Attention blocking sizes (for the combined case).
    HEAD_BLOCK_SIZE = 8
    NUM_KV_BLOCKS = 2
    NUM_Q_BLOCKS = 2

    n_layer = get_custom_n_layers(model_name)

    # token-only FFN blocking
    qaic_config = dict(enable_ffn_blocking=True, ffn_blocking_mode="t", num_token_blocks=NUM_TOKEN_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        n_layer=n_layer,
        qaic_config=qaic_config,
        continuous_batching=True,
        manual_cleanup=manual_cleanup,
    )

    # weight-only FFN blocking
    qaic_config = dict(enable_ffn_blocking=True, ffn_blocking_mode="w", num_weight_blocks=NUM_WEIGHT_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        n_layer=n_layer,
        qaic_config=qaic_config,
        continuous_batching=True,
        manual_cleanup=manual_cleanup,
    )

    # token+weight FFN blocking
    qaic_config = dict(
        enable_ffn_blocking=True,
        ffn_blocking_mode="tw",
        num_token_blocks=NUM_TOKEN_BLOCKS,
        num_weight_blocks=NUM_WEIGHT_BLOCKS,
    )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        n_layer=n_layer,
        qaic_config=qaic_config,
        continuous_batching=True,
        manual_cleanup=manual_cleanup,
    )

    # token+weight FFN blocking + HQKV attention blocking
    qaic_config = dict(
        enable_blocking=True,
        enable_ffn_blocking=True,
        ffn_blocking_mode="tw",
        num_token_blocks=NUM_TOKEN_BLOCKS,
        num_weight_blocks=NUM_WEIGHT_BLOCKS,
        blocking_mode="hqkv",
        head_block_size=HEAD_BLOCK_SIZE,
        num_kv_blocks=NUM_KV_BLOCKS,
        num_q_blocks=NUM_Q_BLOCKS,
    )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        n_layer=n_layer,
        qaic_config=qaic_config,
        continuous_batching=True,
        manual_cleanup=manual_cleanup,
    )


@pytest.mark.dummy_layers
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", test_models_blockedKV[:1])
def test_dummy_causal_all_blocking_pytorch_vs_kv_vs_ort_vs_ai100_CB(model_name, manual_cleanup):
    # Skip MoE models: FFN blocking covers dense FFNs, not MoE blocks.
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
    )
    if getattr(hf_config, "num_local_experts", 1) not in (None, 0, 1) or getattr(hf_config, "num_experts", 1) not in (
        None,
        0,
        1,
    ):
        pytest.skip(f"Skipping MoE model {model_name} for FFN blocking test")

    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        **model_config_dict[model_name].get("additional_params", {}),
    )
    n_layer = -1
    if model_name in ModelConfig.QUANTIZED_MODELS:
        n_layer = get_custom_n_layers(model_name)
        hf_config = None

    # FFN blocking sizes
    NUM_TOKEN_BLOCKS = 2
    NUM_WEIGHT_BLOCKS = 2

    # Attention blocking sizes (for the combined case).
    HEAD_BLOCK_SIZE = 8
    NUM_KV_BLOCKS = 2
    NUM_Q_BLOCKS = 2

    n_layer = get_custom_n_layers(model_name)

    # token-only FFN blocking
    qaic_config = dict(enable_ffn_blocking=True, ffn_blocking_mode="t", num_token_blocks=NUM_TOKEN_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        n_layer=n_layer,
        qaic_config=qaic_config,
        config=hf_config,
        continuous_batching=True,
        manual_cleanup=manual_cleanup,
    )

    # weight-only FFN blocking
    qaic_config = dict(enable_ffn_blocking=True, ffn_blocking_mode="w", num_weight_blocks=NUM_WEIGHT_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        n_layer=n_layer,
        qaic_config=qaic_config,
        config=hf_config,
        continuous_batching=True,
        manual_cleanup=manual_cleanup,
    )

    # token+weight FFN blocking
    qaic_config = dict(
        enable_ffn_blocking=True,
        ffn_blocking_mode="tw",
        num_token_blocks=NUM_TOKEN_BLOCKS,
        num_weight_blocks=NUM_WEIGHT_BLOCKS,
    )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        n_layer=n_layer,
        qaic_config=qaic_config,
        config=hf_config,
        continuous_batching=True,
        manual_cleanup=manual_cleanup,
    )

    # token+weight FFN blocking + HQKV attention blocking
    qaic_config = dict(
        enable_blocking=True,
        enable_ffn_blocking=True,
        ffn_blocking_mode="tw",
        num_token_blocks=NUM_TOKEN_BLOCKS,
        num_weight_blocks=NUM_WEIGHT_BLOCKS,
        blocking_mode="hqkv",
        head_block_size=HEAD_BLOCK_SIZE,
        num_kv_blocks=NUM_KV_BLOCKS,
        num_q_blocks=NUM_Q_BLOCKS,
    )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        n_layer=n_layer,
        qaic_config=qaic_config,
        config=hf_config,
        continuous_batching=True,
        manual_cleanup=manual_cleanup,
    )
