# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import pytest

from .check_causal_models import check_prefix_caching_inference
from .config import (
    QEFF_TEST_PROFILE,
    causal_lm_models_dict,
    compile_params,
    export_params,
    generate_params,
    test_models_causal,
    transform_params,
)


@pytest.mark.non_qaic
@pytest.mark.llm
@pytest.mark.parametrize("model_name", test_models_causal)
def test_fp16_export_compile_prefix_caching_cb(model_name):
    """
    Fp16 + Subfunction + CB with prefix caching  (end to end run+ output verification)
    The test should first generate output with some prefix+suffix1 or batch_id and then confirm that we are still able to execute of prefix+suffix2 on same batch id and getting correct output.
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    temp_compile_params = {
        **compile_params,
        "full_batch_size": 2,
        "kv_cache_batch_size": 4,
    }

    check_prefix_caching_inference(
        model_name,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=temp_compile_params,
        generate_params=generate_params,
        continuous_batching=False,
        export_compile_only=True,
    )


@pytest.mark.qaic
@pytest.mark.llm
@pytest.mark.parametrize("model_name", test_models_causal)
def test_fp16_export_compile_generate_prefix_caching_cb(model_name):
    """
    Fp16 + Subfunction + CB with prefix caching  (end to end run+ output verification)
    The test should first generate output with some prefix+suffix1 or batch_id and then confirm that we are still able to execute of prefix+suffix2 on same batch id and getting correct output.
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    prefixes = ["Once upon a time ", "Once upon a time "]
    suffixes1 = ["in a land far away", "there was a small village"]
    suffixes2 = ["a little girl", "in a bustling city"]
    temp_generate_params = {**generate_params, "prefixes": prefixes, "suffixes1": suffixes1, "suffixes2": suffixes2}
    temp_compile_params = {
        **compile_params,
        "full_batch_size": 2,
        "kv_cache_batch_size": 4,
    }

    check_prefix_caching_inference(
        model_name,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=temp_compile_params,
        generate_params=temp_generate_params,
        continuous_batching=False,
        export_compile_only=False,
    )
