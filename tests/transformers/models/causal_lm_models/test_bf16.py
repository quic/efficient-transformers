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
)


@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_bf16_export_bf16_compile_ccl_cb(model_name):
    """
    BF16 export and BF16 compilation + Subfunction + CB + CCL (only till compilation)
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    qaic_config = {
        "ccl_enabled": True,
    }
    temp_transform_params = {"torch_dtype": torch.bfloat16, "qaic_config": qaic_config}
    temp_compile_params = {
        **compile_params,
        "num_cores": 4,
        "aic-hw-version": "ai200",
        "comp_ctx_lengths_prefill": [256, 500],
        "comp_ctx_lengths_decode": [512, 1024],
    }

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=temp_transform_params,
        export_params=export_params,
        compile_params=temp_compile_params,
        generate_params=generate_params,
        export_compile_only=True,
    )


@pytest.mark.skip(
    reason="BF16 export and BF16 compilation + Subfunction + CB + CCL (only till compilation) - not supported yet"
)
@pytest.mark.llm
@pytest.mark.qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_bf16_export_bf16_compile_generate_ccl_cb(model_name):
    """
    BF16 export and BF16 compilation + Subfunction + CB + CCL (only till compilation)
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    qaic_config = {
        "ccl_enabled": True,
    }
    temp_transform_params = {"torch_dtype": torch.bfloat16, "qaic_config": qaic_config}
    temp_compile_params = {
        **compile_params,
        "num_cores": 4,
        "aic-hw-version": "ai200",
        "comp_ctx_lengths_prefill": [256, 500],
        "comp_ctx_lengths_decode": [512, 1024],
    }

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=temp_transform_params,
        export_params=export_params,
        compile_params=temp_compile_params,
        generate_params=generate_params,
        export_compile_only=False,
    )
