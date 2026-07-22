# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import pytest

from .check_causal_models import check_prefix_caching_inference
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


@pytest.mark.non_qaic
@pytest.mark.llm
@pytest.mark.parametrize("model_name", test_models_causal)
def test_fp16_export_compile_prefix_caching_cb(model_name):
    """Verify that FP16 export and compilation succeed for a prefix-caching continuous-batching model.

    Compiles the model with a KV-cache batch size larger than the active batch
    (``full_batch_size=2``, ``kv_cache_batch_size=4``) so that prefill KV state
    can be retained and reused across requests.  Asserts that ``qconfig.json``
    is present.  Inference is not run.
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
        continuous_batching=True,
        export_compile_only=True,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )


@pytest.mark.qaic
@pytest.mark.llm
@pytest.mark.parametrize("model_name", test_models_causal)
def test_fp16_export_compile_generate_prefix_caching_cb(model_name):
    """Verify prefix-caching KV-reuse correctness end-to-end on the AIC device.

    Compiles with ``full_batch_size=2`` and ``kv_cache_batch_size=4``, then runs
    a multi-stage inference scenario that exercises KV-cache reuse:

    1. Generates outputs for two prompts (prefix + suffix1) on batch slots 0 and 1.
    2. Manually prefills the same prompts on slots 2 and 3 and verifies that the
       step-by-step decode tokens match the session's accumulated ``generated_ids``
       via cosine similarity.
    3. Re-runs prefill on slot 0 with a modified input sharing the same prefix,
       and verifies that the cached prefix produces consistent logits via cosine
       similarity.
    4. Re-runs decode on slot 1 from the cached prefix state and verifies that
       the output sequence matches the original baseline via cosine similarity.
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
        continuous_batching=True,
        export_compile_only=False,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )
