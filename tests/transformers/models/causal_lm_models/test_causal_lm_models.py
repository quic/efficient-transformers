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

from QEfficient.utils._utils import create_json
from QEfficient.utils.constants import QnnConstants
from QEfficient.utils.test_utils import ModelConfig

from .check_causal_models import (
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100,
    get_custom_n_layers,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../configs/causal_model_configs.json")
with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    causal_lm_models = config_data["causal_lm_models"]
test_models_causal = [model["model_name"] for model in causal_lm_models]
model_config_dict = {model["model_name"]: model for model in causal_lm_models}


@pytest.mark.full_layers# -----------------------------------------------------------------------------
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
    HEAD_BLOCK_SIZE = 8
    NUM_KV_BLOCKS = 2
    NUM_Q_BLOCKS = 2

    # head blocking only
    qaic_config = dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, qaic_config=qaic_config, manual_cleanup=manual_cleanup, num_devices=4
    )

    # kv blocking only
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, qaic_config=qaic_config, manual_cleanup=manual_cleanup, num_devices=4
    )

    # q block only
    qaic_config = dict(enable_blocking=True, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, qaic_config=qaic_config, manual_cleanup=manual_cleanup, num_devices=4
    )

    # qkv blocking
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, qaic_config=qaic_config, manual_cleanup=manual_cleanup, num_devices=4
    )

    # hq blocking (head + q, no kv)
    qaic_config = dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        manual_cleanup=manual_cleanup,
        num_devices=4,
    )

    # hkv blocking (head + kv, no q)
    qaic_config = dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE, num_kv_blocks=NUM_KV_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        manual_cleanup=manual_cleanup,
        num_devices=4,
    )


    # head qkv blocking
    qaic_config = dict(
        enable_blocking=True,
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
    HEAD_BLOCK_SIZE = 8
    NUM_KV_BLOCKS = 2
    NUM_Q_BLOCKS = 2
    n_layer = get_custom_n_layers(model_name)
    # head blocking only
    qaic_config = dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, qaic_config=qaic_config, n_layer=n_layer, manual_cleanup=manual_cleanup
    )

    # kv blocking only
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, qaic_config=qaic_config, n_layer=n_layer, manual_cleanup=manual_cleanup
    )

    # q block only
    qaic_config = dict(enable_blocking=True, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, qaic_config=qaic_config, n_layer=n_layer, manual_cleanup=manual_cleanup
    )

    # qkv blocking
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, qaic_config=qaic_config, n_layer=n_layer, manual_cleanup=manual_cleanup
    )

    # hq blocking (head + q, no kv)
    qaic_config = dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        n_layer=n_layer,
        manual_cleanup=manual_cleanup,
    )

    # hkv blocking (head + kv, no q)
    qaic_config = dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE, num_kv_blocks=NUM_KV_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        n_layer=n_layer,
        manual_cleanup=manual_cleanup,
    )

    # head qkv blocking
    qaic_config = dict(
        enable_blocking=True,
        head_block_size=HEAD_BLOCK_SIZE,
        num_kv_blocks=NUM_KV_BLOCKS,
        num_q_blocks=NUM_Q_BLOCKS,
    )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, qaic_config=qaic_config, n_layer=n_layer, manual_cleanup=manual_cleanup
    )


@pytest.mark.dummy_layers
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", test_models_blockedKV)
def test_dummy_causal_all_blocking_pytorch_vs_kv_vs_ort_vs_ai100(model_name, manual_cleanup):
    HEAD_BLOCK_SIZE = 8
    NUM_KV_BLOCKS = 2
    NUM_Q_BLOCKS = 2

    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        **model_config_dict[model_name].get("additional_params", {}),
    )
    n_layer = -1
    if model_name in ModelConfig.QUANTIZED_MODELS:
        n_layer = get_custom_n_layers(model_name)
        hf_config = None
    # head blocking only
    qaic_config = dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, qaic_config=qaic_config, n_layer=n_layer, config=hf_config, manual_cleanup=manual_cleanup
    )

    # kv blocking only
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, qaic_config=qaic_config, n_layer=n_layer, config=hf_config, manual_cleanup=manual_cleanup
    )

    # q block only
    qaic_config = dict(enable_blocking=True, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, qaic_config=qaic_config, n_layer=n_layer, config=hf_config, manual_cleanup=manual_cleanup
    )

    # qkv blocking
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, qaic_config=qaic_config, n_layer=n_layer, config=hf_config, manual_cleanup=manual_cleanup
    )

    # hq blocking (head + q, no kv)
    qaic_config = dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        n_layer=n_layer,
        config=hf_config,
        manual_cleanup=manual_cleanup,
    )

    # hkv blocking (head + kv, no q)
    qaic_config = dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE, num_kv_blocks=NUM_KV_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        n_layer=n_layer,
        config=hf_config,
        manual_cleanup=manual_cleanup,
    )

    # head qkv blocking
    qaic_config = dict(
        enable_blocking=True,
        head_block_size=HEAD_BLOCK_SIZE,
        num_kv_blocks=NUM_KV_BLOCKS,
        num_q_blocks=NUM_Q_BLOCKS,
    )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, qaic_config=qaic_config, n_layer=n_layer, config=hf_config, manual_cleanup=manual_cleanup
    )


@pytest.mark.full_layers
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", test_models_blockedKV[:1])
def test_full_causal_all_blocking_pytorch_vs_kv_vs_ort_vs_ai100_CB(model_name, manual_cleanup):
    HEAD_BLOCK_SIZE = 8
    NUM_KV_BLOCKS = 2
    NUM_Q_BLOCKS = 2

    # head blocking only
    qaic_config = dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
        num_devices=4,
    )

    # kv blocking only
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
        num_devices=4,
    )

    # q block only
    qaic_config = dict(enable_blocking=True, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
        num_devices=4,
    )

    # qkv blocking
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
        num_devices=4,
    )

    # hq blocking (head + q, no kv)
    qaic_config = dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
        num_devices=4,
    )

    # hkv blocking (head + kv, no q)
    qaic_config = dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE, num_kv_blocks=NUM_KV_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
        num_devices=4,
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
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
        num_devices=4,
    )


@pytest.mark.few_layers
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", test_models_blockedKV[:1])
def test_few_causal_all_blocking_pytorch_vs_kv_vs_ort_vs_ai100_CB(model_name, manual_cleanup):
    HEAD_BLOCK_SIZE = 8
    NUM_KV_BLOCKS = 2
    NUM_Q_BLOCKS = 2
    n_layer = get_custom_n_layers(model_name)
    # head blocking only
    qaic_config = dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        n_layer=n_layer,
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
    )

    # kv blocking only
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        n_layer=n_layer,
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
    )

    # q block only
    qaic_config = dict(enable_blocking=True, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        n_layer=n_layer,
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
    )

    # qkv blocking
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        n_layer=n_layer,
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
    )

    # hq blocking (head + q, no kv)
    qaic_config = dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        n_layer=n_layer,
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
    )

    # hkv blocking (head + kv, no q)
    qaic_config = dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE, num_kv_blocks=NUM_KV_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        n_layer=n_layer,
        manual_cleanup=manual_cleanup,
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
        n_layer=n_layer,
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
    )


@pytest.mark.dummy_layers
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", test_models_blockedKV)
def test_dummy_causal_all_blocking_pytorch_vs_kv_vs_ort_vs_ai100_CB(model_name, manual_cleanup):
    HEAD_BLOCK_SIZE = 8
    NUM_KV_BLOCKS = 2
    NUM_Q_BLOCKS = 2

    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        **model_config_dict[model_name].get("additional_params", {}),
    )
    n_layer = -1
    if model_name in ModelConfig.QUANTIZED_MODELS:
        n_layer = get_custom_n_layers(model_name)
        hf_config = None
    # head blocking only
    qaic_config = dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        n_layer=n_layer,
        config=hf_config,
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
    )

    # kv blocking only
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        n_layer=n_layer,
        config=hf_config,
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
    )

    # q block only
    qaic_config = dict(enable_blocking=True, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        n_layer=n_layer,
        config=hf_config,
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
    )

    # qkv blocking
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        n_layer=n_layer,
        config=hf_config,
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
    )

    # hq blocking (head + q, no kv)
    qaic_config = dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        n_layer=n_layer,
        config=hf_config,
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
    )

    # hkv blocking (head + kv, no q)
    qaic_config = dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE, num_kv_blocks=NUM_KV_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        qaic_config=qaic_config,
        n_layer=n_layer,
        config=hf_config,
        manual_cleanup=manual_cleanup,
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
        n_layer=n_layer,
        config=hf_config,
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
    )

@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal)
def test_full_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    if model_name in ModelConfig.FULL_MODEL_TESTS_TO_SKIP:
        pytest.skip(f"Skipping full model test for {model_name} due to resource constraints.")
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name, compare_results=True, manual_cleanup=manual_cleanup, num_devices=4
    )


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal)
def test_few_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    n_layer = get_custom_n_layers(model_name)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name=model_name, n_layer=n_layer, manual_cleanup=manual_cleanup)


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("use_onnx_subfunctions", [False, True])
@pytest.mark.parametrize("model_name", test_models_causal)
def test_few_causal_lm_onnx_mdp_compile_only(model_name, use_onnx_subfunctions, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    n_layer = get_custom_n_layers(model_name)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        n_layer=n_layer,
        manual_cleanup=manual_cleanup,
        compile_only=True,
        mdp_num_partitions=2,
        mdp_strategy="onnx",
        use_onnx_subfunctions=use_onnx_subfunctions,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal)
def test_dummy_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    custom_config = model_config_dict[model_name]
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        **custom_config.get("additional_params", {}),
    )
    if model_name in ModelConfig.QUANTIZED_MODELS:
        n_layer = get_custom_n_layers(model_name)
        check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, n_layer=n_layer, manual_cleanup=manual_cleanup)
    else:
        check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, config=hf_config, manual_cleanup=manual_cleanup)


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal)
def test_full_causal_lm_pytorch_vs_ort_vs_ai100_cb(model_name, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    if model_name in ModelConfig.FULL_MODEL_TESTS_TO_SKIP:
        pytest.skip(f"Skipping full model test for {model_name} due to resource constraints.")
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        manual_cleanup=manual_cleanup,
        num_devices=4,
    )


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal)
def test_few_causal_lm_pytorch_vs_ort_vs_ai100_cb(model_name, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    n_layer = get_custom_n_layers(model_name)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        n_layer=n_layer,
        continuous_batching=True,
        manual_cleanup=manual_cleanup,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal)
def test_dummy_causal_lm_pytorch_vs_ort_vs_ai100_cb(model_name, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    custom_config = model_config_dict[model_name]
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        **custom_config.get("additional_params", {}),
    )
    if model_name in ModelConfig.QUANTIZED_MODELS:
        n_layer = get_custom_n_layers(model_name)
        check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
            model_name,
            n_layer=n_layer,
            continuous_batching=True,
            manual_cleanup=manual_cleanup,
        )
    else:
        check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
            model_name,
            config=hf_config,
            continuous_batching=True,
            manual_cleanup=manual_cleanup,
        )


######################### QNN Tests #########################


@pytest.mark.on_qaic
@pytest.mark.qnn
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal)
def test_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100_qnn(model_name, manual_cleanup):
    """
    QNN Setup
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """
    qnn_config_json_path = os.path.join(os.getcwd(), "qnn_config.json")
    create_json(qnn_config_json_path, QnnConstants.QNN_SAMPLE_CONFIG)
    n_layer = get_custom_n_layers(model_name)

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        n_layer=n_layer,
        enable_qnn=True,
        qnn_config=qnn_config_json_path,
        manual_cleanup=manual_cleanup,
    )


@pytest.mark.on_qaic
@pytest.mark.qnn
@pytest.mark.llm_model
def test_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100_pl1_qnn(manual_cleanup):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model for a prompt length of 1, both with and without continuous batching.
    """
    model_name = "gpt2"
    prompt_len = 1

    qnn_config_json_path = os.path.join(os.getcwd(), "qnn_config.json")
    create_json(qnn_config_json_path, QnnConstants.QNN_SAMPLE_CONFIG)

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        prompt_len=prompt_len,
        enable_qnn=True,
        qnn_config=qnn_config_json_path,
        manual_cleanup=manual_cleanup,
        num_devices=4,
    )
