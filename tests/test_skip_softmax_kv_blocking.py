# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import math

import torch

from QEfficient.blocking.blocked_attention_forwards import (
    _compute_skip_softmax_mask,
    _get_skip_softmax_log_lambda,
    update_running_softmax,
)


def test_skip_softmax_lambda_uses_explicit_scale_when_provided():
    query = torch.zeros(1, 2, 1, 4)

    log_lambda = _get_skip_softmax_log_lambda(
        skip_softmax=True,
        query=query,
        ctx_len=4096,
        skip_softmax_scale=16.0,
        skip_softmax_prefill_scale=999.0,
        skip_softmax_decode_scale=999.0,
    )

    assert torch.allclose(log_lambda.cpu(), torch.tensor(math.log(16.0 / 4096.0), dtype=torch.float32))


def test_skip_softmax_lambda_uses_decode_scale_for_q_len_one():
    query = torch.zeros(1, 2, 1, 4)

    log_lambda = _get_skip_softmax_log_lambda(
        skip_softmax=True,
        query=query,
        ctx_len=1024,
        skip_softmax_prefill_scale=8.0,
        skip_softmax_decode_scale=2.0,
    )

    assert torch.allclose(log_lambda.cpu(), torch.tensor(math.log(2.0 / 1024.0), dtype=torch.float32))


def test_skip_softmax_lambda_uses_prefill_scale_for_q_len_greater_than_one():
    query = torch.zeros(1, 2, 16, 4)

    log_lambda = _get_skip_softmax_log_lambda(
        skip_softmax=True,
        query=query,
        ctx_len=2048,
        skip_softmax_prefill_scale=4.0,
        skip_softmax_decode_scale=2.0,
    )

    assert torch.allclose(log_lambda.cpu(), torch.tensor(math.log(4.0 / 2048.0), dtype=torch.float32))


def test_skip_softmax_mask_does_not_skip_before_prior_contribution():
    block_max = torch.tensor([[[0.0]]])
    current_max = torch.tensor([[[10.0]]])
    current_denominator = torch.zeros(1, 1, 1)
    log_lambda = torch.tensor(math.log(1.0 / 1024.0), dtype=torch.float32)

    mask = _compute_skip_softmax_mask(
        block_max=block_max,
        current_max=current_max,
        current_denominator=current_denominator,
        log_lambda=log_lambda,
        block_idx=1,
        min_keep_blocks=1,
    )

    assert not mask.item()


def test_skip_softmax_mask_skips_after_min_keep_blocks():
    block_max = torch.tensor([[[0.0]]])
    current_max = torch.tensor([[[10.0]]])
    current_denominator = torch.ones(1, 1, 1)
    log_lambda = torch.tensor(math.log(1.0 / 1024.0), dtype=torch.float32)

    mask = _compute_skip_softmax_mask(
        block_max=block_max,
        current_max=current_max,
        current_denominator=current_denominator,
        log_lambda=log_lambda,
        block_idx=1,
        min_keep_blocks=1,
    )

    assert mask.item()


def test_update_running_softmax_preserves_state_for_skip_mask():
    current_max = torch.zeros(1, 1, 1)
    current_denominator = torch.ones(1, 1, 1)
    output = torch.ones(1, 1, 1, 2)

    scores = torch.full((1, 1, 1, 2), -10.0)
    value = torch.ones(1, 1, 2, 2) * 7.0
    skip_mask = torch.ones(1, 1, 1, dtype=torch.bool)

    next_max, next_den, next_output = update_running_softmax(
        current_max=current_max,
        attn_weights_block=scores,
        current_denominator=current_denominator,
        output=output,
        v_block=value,
        skip_mask=skip_mask,
    )

    assert torch.equal(next_max, current_max)
    assert torch.equal(next_den, current_denominator)
    assert torch.equal(next_output, output)


def test_update_running_softmax_updates_state_when_not_skipped():
    current_max = torch.zeros(1, 1, 1)
    current_denominator = torch.ones(1, 1, 1)
    output = torch.zeros(1, 1, 1, 2)

    scores = torch.zeros(1, 1, 1, 2)
    value = torch.ones(1, 1, 2, 2)
    skip_mask = torch.zeros(1, 1, 1, dtype=torch.bool)

    _, next_den, next_output = update_running_softmax(
        current_max, scores, current_denominator, output, value, skip_mask=skip_mask
    )

    assert torch.all(next_den > current_denominator)
    assert torch.all(next_output > output)
