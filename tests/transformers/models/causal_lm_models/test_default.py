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
def test_fp16_export_compile(model_name):
    """Verify that FP16 export and compilation succeed without errors.

    Exports the model to ONNX in FP16 with ONNX subfunctions enabled, compiles
    it to a QPC, and asserts that the resulting ``qconfig.json`` is present.
    Inference is not run; this test validates the export-to-compile pipeline only.
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        continuous_batching=False,
        export_compile_only=True,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )


@pytest.mark.llm
@pytest.mark.qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_fp16_export_compile_generate(model_name):
    """Verify end-to-end FP16 inference quality against the HF PyTorch reference.

    Exports in FP16 with ONNX subfunctions, compiles, runs inference on the AIC
    device, and checks that the cosine similarity between the AIC output token
    sequences and the HF PyTorch reference sequences meets the configured threshold.
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        continuous_batching=False,
        export_compile_only=False,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )


@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_fp16_export_compile_cb(model_name):
    """Verify that FP16 export and compilation succeed in continuous-batching mode.

    Enables continuous batching with ONNX subfunctions, exports to ONNX, compiles
    to a QPC, and asserts that ``qconfig.json`` is present.  Inference is not run.
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=True,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )


@pytest.mark.llm
@pytest.mark.qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_fp16_export_compile_generate_cb(model_name):
    """Verify end-to-end FP16 inference quality in continuous-batching mode.

    Enables continuous batching with ONNX subfunctions, compiles, runs inference
    on the AIC device, and checks that the cosine similarity between the AIC
    output token sequences and the HF PyTorch reference sequences meets the
    configured threshold for every batch slot.
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=False,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )
