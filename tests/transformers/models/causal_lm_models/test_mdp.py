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

from .check_causal_models import check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100, get_custom_n_layers
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
    _mdp_models = _config_data["mdp_causal_lm_models"]

test_models_mdp = [model["model_name"] for model in _mdp_models]
model_config_dict = {model["model_name"]: model for model in _mdp_models}


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_mdp)
@pytest.mark.parametrize("mdp_num_partitions", [2, 4])
def test_full_mdp_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, mdp_num_partitions):
    """Verify end-to-end pipeline-parallel MDP compilation and inference at full model depth.

    Compiles the model with ``mdp_num_partitions`` pipeline-parallel partitions
    using the ONNX-topsort MDP strategy, runs inference on the AIC device, and
    checks that ``qconfig.json`` is present and that output cosine similarity
    meets the configured threshold.  Uses 4 devices to match the partition count.

    Args:
        model_name: HuggingFace model identifier.
        mdp_num_partitions: Number of pipeline-parallel MDP partitions (2 or 4).
    """
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    if model_name in ModelConfig.FULL_MODEL_TESTS_TO_SKIP:
        pytest.skip(f"Skipping full model test for {model_name} due to resource constraints.")

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        transform_params=transform_params,
        export_params=export_params,
        compile_params={**compile_params, "num_devices": 4, "mdp_num_partitions": mdp_num_partitions},
        generate_params=generate_params,
        export_compile_only=False,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_mdp)
@pytest.mark.parametrize("mdp_num_partitions", [2, 4])
def test_few_mdp_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, mdp_num_partitions):
    """Verify pipeline-parallel MDP compilation and inference with a reduced layer count.

    Loads the model with the minimum number of layers required for correctness
    (see ``get_custom_n_layers``), compiles with ``mdp_num_partitions`` partitions,
    and checks that ``qconfig.json`` is present and output similarity meets the
    configured threshold.

    Args:
        model_name: HuggingFace model identifier.
        mdp_num_partitions: Number of pipeline-parallel MDP partitions (2 or 4).
    """
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")

    n_layer = get_custom_n_layers(model_name)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        n_layer=n_layer,
        transform_params=transform_params,
        export_params=export_params,
        compile_params={**compile_params, "mdp_num_partitions": mdp_num_partitions},
        generate_params=generate_params,
        export_compile_only=False,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_mdp)
@pytest.mark.parametrize("mdp_num_partitions", [2, 4])
def test_dummy_mdp_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, mdp_num_partitions):
    """Verify pipeline-parallel MDP compilation and inference with a dummy (tiny) config.

    Builds a minimal ``AutoConfig`` from the model card so that the model is
    instantiated with the real architecture but tiny hidden dimensions, keeping
    the test fast.  Compiles with ``mdp_num_partitions`` partitions and checks
    that ``qconfig.json`` is present and output similarity meets the threshold.

    Args:
        model_name: HuggingFace model identifier.
        mdp_num_partitions: Number of pipeline-parallel MDP partitions (2 or 4).
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

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        n_layer=n_layer,
        config=config_arg,
        transform_params=transform_params,
        export_params=export_params,
        compile_params={**compile_params, "mdp_num_partitions": mdp_num_partitions},
        generate_params=generate_params,
        export_compile_only=False,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_mdp)
@pytest.mark.parametrize("mdp_num_partitions", [2, 4])
def test_full_mdp_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100_cb(model_name, mdp_num_partitions):
    """Verify pipeline-parallel MDP compilation and inference in continuous-batching mode at full depth.

    Enables continuous batching alongside ``mdp_num_partitions`` pipeline-parallel
    partitions, compiles with 4 devices, and checks that ``qconfig.json`` is
    present and output similarity meets the configured threshold.

    Args:
        model_name: HuggingFace model identifier.
        mdp_num_partitions: Number of pipeline-parallel MDP partitions (2 or 4).
    """
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    if model_name in ModelConfig.FULL_MODEL_TESTS_TO_SKIP:
        pytest.skip(f"Skipping full model test for {model_name} due to resource constraints.")

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params={**compile_params, "num_devices": 4, "mdp_num_partitions": mdp_num_partitions},
        generate_params=generate_params,
        export_compile_only=False,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_mdp)
@pytest.mark.parametrize("mdp_num_partitions", [2, 4])
def test_few_mdp_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100_cb(model_name, mdp_num_partitions):
    """Verify pipeline-parallel MDP compilation and inference in continuous-batching mode with reduced layers.

    Loads the model with the minimum layer count, enables continuous batching
    alongside ``mdp_num_partitions`` partitions, and checks that ``qconfig.json``
    is present and output similarity meets the configured threshold.

    Args:
        model_name: HuggingFace model identifier.
        mdp_num_partitions: Number of pipeline-parallel MDP partitions (2 or 4).
    """
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")

    n_layer = get_custom_n_layers(model_name)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        n_layer=n_layer,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params={**compile_params, "mdp_num_partitions": mdp_num_partitions},
        generate_params=generate_params,
        export_compile_only=False,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_mdp)
@pytest.mark.parametrize("mdp_num_partitions", [2, 4])
def test_dummy_mdp_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100_cb(model_name, mdp_num_partitions):
    """Verify pipeline-parallel MDP compilation and inference in continuous-batching mode with a dummy config.

    Builds a minimal ``AutoConfig``, enables continuous batching alongside
    ``mdp_num_partitions`` partitions, and checks that ``qconfig.json`` is
    present and output similarity meets the configured threshold.

    Args:
        model_name: HuggingFace model identifier.
        mdp_num_partitions: Number of pipeline-parallel MDP partitions (2 or 4).
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

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        n_layer=n_layer,
        config=config_arg,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params={**compile_params, "mdp_num_partitions": mdp_num_partitions},
        generate_params=generate_params,
        export_compile_only=False,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )
