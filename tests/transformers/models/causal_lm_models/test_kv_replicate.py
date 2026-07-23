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

from .check_causal_models import check_kv_repeat_causal_lm_pytorch_vs_ai100, get_custom_n_layers
from .config import (
    COSINE_SIMILARITY_THRESHOLD,
    compile_params,
    export_params,
    generate_params,
    transform_params,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../configs/causal_model_configs.json")
with open(CONFIG_PATH) as f:
    _config_data = json.load(f)
    _kv_replicate_models = _config_data["kv_replicate_causal_lm_models"]

test_models_kv_replicate = [model["model_name"] for model in _kv_replicate_models]
model_config_dict = {model["model_name"]: model for model in _kv_replicate_models}


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_kv_replicate)
def test_full_kv_replicate_causal_lm_pytorch_vs_ai100(model_name):
    """Verify end-to-end KV-head replication at full model depth.

    Applies ``ReplicateKVHeadTransform`` to expand KV heads to match the query
    head count (GQA → MHA layout), exports to ONNX, compiles to a QPC, runs
    inference on the AIC device, and checks that output cosine similarity meets
    the configured threshold.  The replication factor is derived automatically
    from the model config.

    Args:
        model_name: HuggingFace model identifier (must be a GQA model where
            ``num_attention_heads > num_key_value_heads``).
    """
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    if model_name in ModelConfig.FULL_MODEL_TESTS_TO_SKIP:
        pytest.skip(f"Skipping full model test for {model_name} due to resource constraints.")

    check_kv_repeat_causal_lm_pytorch_vs_ai100(
        model_name=model_name,
        transform_params=transform_params,
        export_params=export_params,
        compile_params={**compile_params, "num_devices": 4},
        generate_params=generate_params,
        export_compile_only=False,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_kv_replicate)
def test_few_kv_replicate_causal_lm_pytorch_vs_ai100(model_name):
    """Verify KV-head replication with a reduced layer count.

    Loads the model with the minimum number of layers required for correctness
    (see ``get_custom_n_layers``), applies ``ReplicateKVHeadTransform``, and
    checks that output cosine similarity meets the configured threshold.

    Args:
        model_name: HuggingFace model identifier.
    """
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")

    n_layer = get_custom_n_layers(model_name)
    check_kv_repeat_causal_lm_pytorch_vs_ai100(
        model_name=model_name,
        n_layer=n_layer,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=False,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_kv_replicate)
def test_dummy_kv_replicate_causal_lm_pytorch_vs_ai100(model_name):
    """Verify KV-head replication with a dummy (tiny) config.

    Builds a minimal ``AutoConfig`` from the model card so that the model is
    instantiated with the real GQA architecture but tiny hidden dimensions.
    Applies ``ReplicateKVHeadTransform`` and checks that output cosine similarity
    meets the configured threshold.

    Args:
        model_name: HuggingFace model identifier.
    """
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")

    custom_config = model_config_dict[model_name]
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        **custom_config.get("additional_params", {}),
    )
    n_layer = get_custom_n_layers(model_name) if model_name in ModelConfig.QUANTIZED_MODELS else -1
    config_arg = None if model_name in ModelConfig.QUANTIZED_MODELS else hf_config

    check_kv_repeat_causal_lm_pytorch_vs_ai100(
        model_name=model_name,
        n_layer=n_layer,
        config=config_arg,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=False,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_kv_replicate)
def test_full_kv_replicate_causal_lm_pytorch_vs_ai100_cb(model_name):
    """Verify end-to-end KV-head replication in continuous-batching mode at full model depth.

    Applies ``ReplicateKVHeadTransform`` with continuous batching enabled,
    compiles with 4 devices, runs inference on the AIC device, and checks that
    output cosine similarity meets the configured threshold.

    Args:
        model_name: HuggingFace model identifier.
    """
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    if model_name in ModelConfig.FULL_MODEL_TESTS_TO_SKIP:
        pytest.skip(f"Skipping full model test for {model_name} due to resource constraints.")

    check_kv_repeat_causal_lm_pytorch_vs_ai100(
        model_name=model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params={**compile_params, "num_devices": 4},
        generate_params=generate_params,
        export_compile_only=False,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_kv_replicate)
def test_few_kv_replicate_causal_lm_pytorch_vs_ai100_cb(model_name):
    """Verify KV-head replication in continuous-batching mode with a reduced layer count.

    Loads the model with the minimum layer count, applies ``ReplicateKVHeadTransform``
    with continuous batching enabled, and checks that output cosine similarity
    meets the configured threshold.

    Args:
        model_name: HuggingFace model identifier.
    """
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")

    n_layer = get_custom_n_layers(model_name)
    check_kv_repeat_causal_lm_pytorch_vs_ai100(
        model_name=model_name,
        n_layer=n_layer,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=False,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_kv_replicate)
def test_dummy_kv_replicate_causal_lm_pytorch_vs_ai100_cb(model_name):
    """Verify KV-head replication in continuous-batching mode with a dummy config.

    Builds a minimal ``AutoConfig``, applies ``ReplicateKVHeadTransform`` with
    continuous batching enabled, and checks that output cosine similarity meets
    the configured threshold.

    Args:
        model_name: HuggingFace model identifier.
    """
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")

    custom_config = model_config_dict[model_name]
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        **custom_config.get("additional_params", {}),
    )
    n_layer = get_custom_n_layers(model_name) if model_name in ModelConfig.QUANTIZED_MODELS else -1
    config_arg = None if model_name in ModelConfig.QUANTIZED_MODELS else hf_config

    check_kv_repeat_causal_lm_pytorch_vs_ai100(
        model_name=model_name,
        n_layer=n_layer,
        config=config_arg,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=False,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )
