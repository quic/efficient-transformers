# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import pytest

from tests.utils.profile_test_config import load_test_config

from .check_causal_models import (
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100,
)

config_data = load_test_config("causal_model_configs")
blockedKV_models = config_data["blockedKV_causal_lm_models"]
test_models_blockedKV = [model["model_name"] for model in blockedKV_models]
model_config_dict = {model["model_name"]: model for model in blockedKV_models}


@pytest.mark.llm_model
@pytest.mark.qaic
@pytest.mark.parametrize("model_name", test_models_blockedKV[:1])
def test_causal_all_blocking_pytorch_vs_kv_vs_ort_vs_ai100(model_name):
    HEAD_BLOCK_SIZE = 8
    NUM_KV_BLOCKS = 2
    NUM_Q_BLOCKS = 2

    # head blocking only
    qaic_config = dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
    )

    # kv blocking only
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
    )

    # q block only
    qaic_config = dict(enable_blocking=True, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
    )

    # qkv blocking
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
    )

    # head qkv blocking
    qaic_config = dict(
        enable_blocking=True,
        head_block_size=HEAD_BLOCK_SIZE,
        num_kv_blocks=NUM_KV_BLOCKS,
        num_q_blocks=NUM_Q_BLOCKS,
    )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name=model_name, qaic_config=qaic_config)


@pytest.mark.llm_model
@pytest.mark.qaic
@pytest.mark.parametrize("model_name", test_models_blockedKV[:1])
def test_causal_all_blocking_pytorch_vs_kv_vs_ort_vs_ai100_CB(model_name):
    HEAD_BLOCK_SIZE = 8
    NUM_KV_BLOCKS = 2
    NUM_Q_BLOCKS = 2

    # head blocking only
    qaic_config = dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        continuous_batching=True,
    )

    # kv blocking only
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        continuous_batching=True,
    )

    # q block only
    qaic_config = dict(enable_blocking=True, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        continuous_batching=True,
    )

    # qkv blocking
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        continuous_batching=True,
    )

    # head qkv blocking
    qaic_config = dict(
        enable_blocking=True,
        head_block_size=HEAD_BLOCK_SIZE,
        num_kv_blocks=NUM_KV_BLOCKS,
        num_q_blocks=NUM_Q_BLOCKS,
    )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        continuous_batching=True,
    )
