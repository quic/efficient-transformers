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

# Blocking configuration constants used across both compile-only and generate tests.
HEAD_BLOCK_SIZE = 8
NUM_KV_BLOCKS = 2
NUM_Q_BLOCKS = 2


@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_fp16_export_compile_blocking_cb(model_name):
    """Verify that FP16 export and compilation succeed for all attention-blocking modes with continuous batching.

    Runs five sub-cases in sequence, each enabling a different combination of
    head, KV, and Q blocking via ``qaic_config``:

    1. Head blocking only (``head_block_size``).
    2. KV blocking only (``num_kv_blocks``).
    3. Q blocking only (``num_q_blocks``).
    4. Combined KV + Q blocking.
    5. Combined head + KV + Q blocking.

    Each sub-case asserts that ``qconfig.json`` is present after compilation.
    Inference is not run.
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    # head blocking only
    qaic_config = dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params={**transform_params, "qaic_config": qaic_config},
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=True,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )

    # kv blocking only
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params={**transform_params, "qaic_config": qaic_config},
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=True,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )

    # q blocking only
    qaic_config = dict(enable_blocking=True, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params={**transform_params, "qaic_config": qaic_config},
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=True,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )

    # combined kv + q blocking
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params={**transform_params, "qaic_config": qaic_config},
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=True,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )

    # combined head + kv + q blocking
    qaic_config = dict(
        enable_blocking=True,
        head_block_size=HEAD_BLOCK_SIZE,
        num_kv_blocks=NUM_KV_BLOCKS,
        num_q_blocks=NUM_Q_BLOCKS,
    )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params={**transform_params, "qaic_config": qaic_config},
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=True,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )


@pytest.mark.llm
@pytest.mark.qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_fp16_export_compile_generate_blocking_cb(model_name):
    """Verify end-to-end FP16 inference quality for all attention-blocking modes with continuous batching.

    Runs five sub-cases in sequence, each enabling a different combination of
    head, KV, and Q blocking via ``qaic_config``:

    1. Head blocking only (``head_block_size``).
    2. KV blocking only (``num_kv_blocks``).
    3. Q blocking only (``num_q_blocks``).
    4. Combined KV + Q blocking.
    5. Combined head + KV + Q blocking.

    Each sub-case runs inference on the AIC device and checks that the cosine
    similarity between the AIC output sequences and the HF PyTorch reference
    sequences meets the configured threshold for every batch slot.
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    # head blocking only
    qaic_config = dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params={**transform_params, "qaic_config": qaic_config},
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=False,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )

    # kv blocking only
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params={**transform_params, "qaic_config": qaic_config},
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=False,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )

    # q blocking only
    qaic_config = dict(enable_blocking=True, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params={**transform_params, "qaic_config": qaic_config},
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=False,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )

    # combined kv + q blocking
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS, num_q_blocks=NUM_Q_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params={**transform_params, "qaic_config": qaic_config},
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=False,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )

    # combined head + kv + q blocking
    qaic_config = dict(
        enable_blocking=True,
        head_block_size=HEAD_BLOCK_SIZE,
        num_kv_blocks=NUM_KV_BLOCKS,
        num_q_blocks=NUM_Q_BLOCKS,
    )
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params={**transform_params, "qaic_config": qaic_config},
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=False,
        cosine_similarity_threshold=COSINE_SIMILARITY_THRESHOLD,
    )
