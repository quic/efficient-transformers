# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Compute the maximum kv_block_size under an fp16 memory budget.

Constraints (bytes) per matmul:
1) [1, num_heads, q_len, 576] x [1, 1, 576, kv] -> [1, num_heads, q_len, kv]
2) [1, num_heads, q_len, kv] x [1, 1, kv, 512] -> [1, num_heads, q_len, 512]

For each matmul, sum(input_a + input_b + output) must be < budget.
The returned kv_block_size satisfies both constraints.
"""

from typing import List

from QEfficient.utils.constants import VTCM_SIZE_THRESHOLD

FP16_BYTES = 2
DEFAULT_NUM_HEADS = 64
VTCM_SIZE_THRESHOLD = int(VTCM_SIZE_THRESHOLD)


def matmul1_bytes(q_len: int, kv_block_size: int, num_heads: int = DEFAULT_NUM_HEADS) -> int:
    """Bytes for [1,num_heads,q,kv] x [1,1,kv,512] -> [1,num_heads,q,512] in fp16."""
    elems_a = num_heads * q_len * kv_block_size
    elems_b = kv_block_size * 512
    elems_out = num_heads * q_len * 512
    return FP16_BYTES * (elems_a + elems_b + elems_out)


def matmul2_bytes(q_len: int, kv_block_size: int, num_heads: int = DEFAULT_NUM_HEADS) -> int:
    """Bytes for [1,num_heads,q,576] x [1,1,576,kv] -> [1,num_heads,q,kv] in fp16."""
    elems_a = num_heads * q_len * 576
    elems_b = 576 * kv_block_size
    elems_out = num_heads * q_len * kv_block_size
    return FP16_BYTES * (elems_a + elems_b + elems_out)


def max_kv_block_size(
    q_len: int,
    budget_bytes: int = VTCM_SIZE_THRESHOLD,
    num_heads: int = DEFAULT_NUM_HEADS,
) -> int:
    """Return the largest integer kv_block_size that satisfies both matmul budgets.

    Returns 0 if no positive kv_block_size can satisfy the constraints.
    """
    if q_len < 0:
        raise ValueError("q_len must be non-negative")
    if budget_bytes <= 0:
        raise ValueError("budget_bytes must be positive")
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")

    # Enforce strict inequality in bytes:
    # FP16_BYTES * elems < budget_bytes  =>  elems <= floor((budget_bytes - 1)/FP16_BYTES)
    max_elems = (budget_bytes - 1) // FP16_BYTES

    # Matmul1 elements:
    #   A_elems = num_heads*q_len*kv
    #   B_elems = kv*512
    #   C_elems = num_heads*q_len*512
    # Enforce A_elems + B_elems + C_elems <= max_elems
    c1_elems = num_heads * q_len * 512
    rem1 = max_elems - c1_elems
    den1 = num_heads * q_len + 512  # kv coefficient from A_elems + B_elems
    k1 = rem1 // den1 if rem1 >= 0 else -1

    # Matmul2 elements:
    #   A_elems = num_heads*q_len*576
    #   B_elems = 576*kv
    #   C_elems = num_heads*q_len*kv
    # Enforce A_elems + B_elems + C_elems <= max_elems
    a2_elems = num_heads * q_len * 576
    rem2 = max_elems - a2_elems
    den2 = num_heads * q_len + 576  # kv coefficient from B_elems + C_elems
    k2 = rem2 // den2 if rem2 >= 0 else -1

    kv = min(k1, k2)
    return max(0, kv)


def block_candidates_generator(max_length: int) -> List[int]:
    block_list = []
    i = 1
    step = 1
    while i <= max_length:
        block_list.append(i)
        if i % (4 * step) == 0:
            step *= 2
        i += step
    return block_list


def get_num_kv_blocks_for_mla(q_len, num_heads, ctx_len):
    budget_bytes = VTCM_SIZE_THRESHOLD
    kv = max_kv_block_size(q_len, budget_bytes, num_heads)
    b1 = matmul1_bytes(q_len, kv, num_heads)
    b2 = matmul2_bytes(q_len, kv, num_heads)

    assert b1 < budget_bytes, "matmul1 is not under the budget"
    assert b2 < budget_bytes, "matmul2 is not under the budget"
    kv_block_size_list = block_candidates_generator(ctx_len)
    for i in range(len(kv_block_size_list) - 1):
        if kv_block_size_list[i] < kv < kv_block_size_list[i + 1]:
            kv_block_size = kv_block_size_list[i]
    return ctx_len // kv_block_size
