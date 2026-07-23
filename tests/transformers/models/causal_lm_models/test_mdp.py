# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import pytest

from .check_causal_models import check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100
from .config import (
    COSINE_SIMILARITY_THRESHOLD,
    QEFF_TEST_PROFILE,
    causal_lm_models_dict,
    compile_params,
    export_params,
    generate_params,
    test_models_causal,
    transform_params,
)


@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", test_models_causal)
@pytest.mark.parametrize("mdp_num_partitions", [2, 4])
def test_fp16_export_compile_mdp(model_name, mdp_num_partitions):
    """Verify that FP16 export and MDP compilation succeed for the given partition count.

    Exports the model to ONNX in FP16 with ONNX subfunctions enabled, then
    compiles with ``mdp_num_partitions`` pipeline-parallel MDP partitions.
    Asserts that ``qconfig.json`` is present.  Inference is not run.

    Args:
        model_name: HuggingFace model identifier.
        mdp_num_partitions: Number of pipeline-parallel MDP partitions (2 or 4).
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        transform_params=transform_params,
        export_params=export_params,
        compile_params={**compile_params, "mdp_num_partitions": mdp_num_partitions},
        generate_params=generate_params,
        continuous_batching=False,
        export_compile_only=True,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )


@pytest.mark.llm
@pytest.mark.qaic
@pytest.mark.parametrize("model_name", test_models_causal)
@pytest.mark.parametrize("mdp_num_partitions", [2, 4])
def test_fp16_export_compile_generate_mdp(model_name, mdp_num_partitions):
    """Verify end-to-end FP16 inference quality with MDP pipeline-parallel partitioning.

    Exports in FP16 with ONNX subfunctions, compiles with ``mdp_num_partitions``
    pipeline-parallel MDP partitions, runs inference on the AIC device, and
    checks that the cosine similarity between the AIC output token sequences and
    the HF PyTorch reference sequences meets the configured threshold.

    Args:
        model_name: HuggingFace model identifier.
        mdp_num_partitions: Number of pipeline-parallel MDP partitions (2 or 4).
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        transform_params=transform_params,
        export_params=export_params,
        compile_params={**compile_params, "mdp_num_partitions": mdp_num_partitions},
        generate_params=generate_params,
        continuous_batching=False,
        export_compile_only=False,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )


@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", test_models_causal)
@pytest.mark.parametrize("mdp_num_partitions", [2, 4])
def test_fp16_export_compile_mdp_cb(model_name, mdp_num_partitions):
    """Verify that FP16 export and MDP compilation succeed in continuous-batching mode.

    Exports in FP16 with ONNX subfunctions and continuous batching enabled,
    compiles with ``mdp_num_partitions`` pipeline-parallel MDP partitions, and
    asserts that ``qconfig.json`` is present.  Inference is not run.

    Args:
        model_name: HuggingFace model identifier.
        mdp_num_partitions: Number of pipeline-parallel MDP partitions (2 or 4).
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params={**compile_params, "mdp_num_partitions": mdp_num_partitions},
        generate_params=generate_params,
        export_compile_only=True,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )


@pytest.mark.llm
@pytest.mark.qaic
@pytest.mark.parametrize("model_name", test_models_causal)
@pytest.mark.parametrize("mdp_num_partitions", [2, 4])
def test_fp16_export_compile_generate_mdp_cb(model_name, mdp_num_partitions):
    """Verify end-to-end FP16 inference quality with MDP partitioning in continuous-batching mode.

    Exports in FP16 with ONNX subfunctions and continuous batching enabled,
    compiles with ``mdp_num_partitions`` pipeline-parallel MDP partitions, runs
    inference on the AIC device, and checks that the cosine similarity between
    the AIC output sequences and the HF PyTorch reference sequences meets the
    configured threshold for every batch slot.

    Args:
        model_name: HuggingFace model identifier.
        mdp_num_partitions: Number of pipeline-parallel MDP partitions (2 or 4).
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params={**compile_params, "mdp_num_partitions": mdp_num_partitions},
        generate_params=generate_params,
        export_compile_only=False,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )
