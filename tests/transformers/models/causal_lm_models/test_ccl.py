# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import pytest
import torch

from .check_causal_models import check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100
from .config import (
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
def test_fp16_export_compile_ccl_cb(model_name):
    """
    Fp16 + Subfunction + CB + CCL  (end to end run+ output verification)
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    qaic_config = {
        "ccl_enabled": True,
    }
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params={**transform_params, "qaic_config": qaic_config},
        export_params=export_params,
        compile_params={
            **compile_params,
            "comp_ctx_lengths_prefill": [256, 500],
            "comp_ctx_lengths_decode": [512, 1024],
        },
        generate_params=generate_params,
        export_compile_only=True,
    )


@pytest.mark.llm
@pytest.mark.qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_fp16_export_compile_generate_ccl_cb(model_name):
    """
    Fp16 + Subfunction + CB + CCL  (end to end run+ output verification)
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    qaic_config = {
        "ccl_enabled": True,
    }
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params={**transform_params, "qaic_config": qaic_config},
        export_params=export_params,
        compile_params={
            **compile_params,
            "comp_ctx_lengths_prefill": [256, 500],
            "comp_ctx_lengths_decode": [512, 1024],
        },
        generate_params=generate_params,
        export_compile_only=False,
    )


@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_fp32_export_fp16_compile_ccl_cb(model_name):
    """
    FP32 export + FP16 compilation + Subfunction + CB + CCL  (end to end run+ output verification)
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    qaic_config = {
        "ccl_enabled": True,
    }
    temp_transform_params = {"torch_dtype": torch.float32, "qaic_config": qaic_config}
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=temp_transform_params,
        export_params=export_params,
        compile_params={
            **compile_params,
            "comp_ctx_lengths_prefill": [256, 500],
            "comp_ctx_lengths_decode": [512, 1024],
        },
        generate_params=generate_params,
        export_compile_only=True,
    )


@pytest.mark.llm
@pytest.mark.qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_fp32_export_fp16_compile_generate_ccl_cb(model_name):
    """
    FP32 export + FP16 compilation + Subfunction + CB + CCL  (end to end run+ output verification)
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    qaic_config = {
        "ccl_enabled": True,
    }
    temp_transform_params = {"torch_dtype": torch.float32, "qaic_config": qaic_config}
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=temp_transform_params,
        export_params=export_params,
        compile_params={
            **compile_params,
            "comp_ctx_lengths_prefill": [256, 500],
            "comp_ctx_lengths_decode": [512, 1024],
        },
        generate_params=generate_params,
        export_compile_only=False,
    )
