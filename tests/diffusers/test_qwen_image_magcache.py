# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import numpy as np
import pytest
import torch

from QEfficient.diffusers.pipelines.qwen_image.magcache import QwenImageMagCacheRuntime, nearest_interp


@pytest.mark.diffusers
def test_nearest_interp_target_length_one_uses_last_value():
    src = np.asarray([0.1, 0.2, 0.9], dtype=np.float32)
    out = nearest_interp(src, 1)
    assert out.shape == (1,)
    assert np.isclose(out[0], src[-1])


@pytest.mark.diffusers
def test_prepare_ratios_cfg_and_non_cfg_lengths():
    ratios = [0.99, 0.98, 0.97, 0.96]

    cfg_runtime = QwenImageMagCacheRuntime(
        num_inference_steps=5,
        do_classifier_free_guidance=True,
        threshold=0.1,
        max_skip_steps=2,
        retention_ratio=0.0,
        ratios=ratios,
    )
    assert len(cfg_runtime._prepared_ratios) == 10

    non_cfg_runtime = QwenImageMagCacheRuntime(
        num_inference_steps=5,
        do_classifier_free_guidance=False,
        threshold=0.1,
        max_skip_steps=2,
        retention_ratio=0.0,
        ratios=ratios,
    )
    assert len(non_cfg_runtime._prepared_ratios) == 5


@pytest.mark.diffusers
def test_retention_window_behavior():
    runtime = QwenImageMagCacheRuntime(
        num_inference_steps=5,
        do_classifier_free_guidance=False,
        threshold=0.1,
        max_skip_steps=2,
        retention_ratio=0.4,
        ratios=[1.0] * 5,
    )

    allowed = [runtime._cache_allowed_for_call(i) for i in range(5)]
    assert allowed == [False, False, True, True, True]


@pytest.mark.diffusers
def test_skip_path_advances_call_index_and_respects_k_limit():
    runtime = QwenImageMagCacheRuntime(
        num_inference_steps=4,
        do_classifier_free_guidance=False,
        threshold=1.0,
        max_skip_steps=2,
        retention_ratio=0.0,
        ratios=[1.0] * 4,
    )

    assert runtime.should_skip("cond") is False
    runtime.complete_call("cond", torch.zeros(1))
    assert runtime.call_index == 1

    assert runtime.should_skip("cond") is True
    runtime.complete_skip("cond")
    assert runtime.call_index == 2

    assert runtime.should_skip("cond") is True
    runtime.complete_skip("cond")
    assert runtime.call_index == 3

    # Third consecutive skip exceeds K=2 and should force execution.
    assert runtime.should_skip("cond") is False
    runtime.complete_call("cond", torch.zeros(1))

    # End of denoise sequence resets runtime state for next image.
    assert runtime.call_index == 0
    assert runtime.stream_states["cond"].cached_residual is None
