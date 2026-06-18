# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Shared low-level kernels for expert-blocked MoE prefill.

These helpers are model-agnostic. The per-expert MLP math (plain SiLU-GLU vs the
GPT-OSS clamped GLU with biases) is injected via the ``expert_mlp`` callable so a
single traced kernel serves every MoE model.
"""

from typing import Callable, Optional

import torch

from QEfficient.customop.ctx_scatter_gather import (
    CtxGatherFunc3DGeneralized,
    CtxScatterFunc3DGeneralized,
    CtxScatterFunc3DInt,
)


def build_matched_idx_from_cumsum(T2Ei: torch.Tensor) -> torch.Tensor:
    """Build packed->original token index from a per-token expert-match mask."""
    batch_size, seq_len = T2Ei.shape
    int32_max = torch.iinfo(torch.int32).max
    int32_max_scalar = torch.tensor(int32_max, dtype=torch.int32, device=T2Ei.device)
    token_idx = torch.arange(seq_len, dtype=torch.int32, device=T2Ei.device).unsqueeze(0).expand(batch_size, -1)
    valid_prefix = torch.cumsum(T2Ei.to(torch.int32), dim=1)
    valid_dest = valid_prefix - 1
    scatter_pos = torch.where(T2Ei, valid_dest, int32_max_scalar)
    # NOTE: expand_as(...) instead of torch.full_like(...) is the compiler-preferred
    # workaround for ConstantOfShape(INT32_MAX); both produce identical traced Ctx ops.
    matched_idx = int32_max_scalar.expand_as(token_idx)
    matched_idx = CtxScatterFunc3DInt.apply(
        matched_idx.unsqueeze(-1),
        scatter_pos,
        token_idx.unsqueeze(-1),
    ).squeeze(-1)
    return matched_idx


def cumsum_scatter_gather_update_expert_blocked(
    x: torch.Tensor,
    T2Ei: torch.Tensor,
    W_g: torch.Tensor,
    W_u: torch.Tensor,
    W_d: torch.Tensor,
    routing_weight: torch.Tensor,
    expert_out: torch.Tensor,
    expert_mlp: Callable,
    *,
    b_g: Optional[torch.Tensor] = None,
    b_u: Optional[torch.Tensor] = None,
    b_d: Optional[torch.Tensor] = None,
    packed_chunk_size: int,
    num_packed_chunks: int = 1,
) -> torch.Tensor:
    """Run one local expert slot over statically traced packed chunks.

    Accumulates one local expert's contribution in-place onto ``expert_out`` using a
    packed/cumsum layout so the MLP runs only over active rows, then scatters the
    weighted output back to original token positions.

    ``expert_mlp`` computes the per-expert projection
    ``(x_chunk, W_g, W_u, W_d, b_g, b_u, b_d) -> down_out`` (without the routing
    weight, which is applied here). ``num_packed_chunks`` controls the number of
    ONNX-traced chunk iterations; during export the sequence length can stay small
    while compile-time specialization maps those iterations to the caller's real
    prefill length.

    Shapes:
        x               : [T, H]
        T2Ei            : [num_nsp, T]            (bool)
        W_g, W_u        : [num_nsp, H, I]
        W_d             : [num_nsp, I, H]
        b_g, b_u        : [num_nsp, I]            (optional)
        b_d             : [num_nsp, H]            (optional)
        routing_weight  : [num_nsp, T, 1]
        expert_out      : [num_nsp, T, H]         (accumulator, in-out)
    """
    batch_size, seq_len = T2Ei.shape
    num_packed_chunks = max(1, int(num_packed_chunks))

    matched_idx = build_matched_idx_from_cumsum(T2Ei)
    valid_rows = torch.einsum("ij->i", T2Ei.to(torch.int32)).unsqueeze(1)
    x_expanded = x.unsqueeze(0).expand(batch_size, -1, -1)
    chunk_starts = [-(-chunk_idx * seq_len) // num_packed_chunks for chunk_idx in range(num_packed_chunks)]
    for chunk_idx, packed_start in enumerate(chunk_starts):
        packed_stop = seq_len if chunk_idx == num_packed_chunks - 1 else chunk_starts[chunk_idx + 1]
        chunk_rows = packed_stop - packed_start
        row_range = torch.arange(chunk_rows, dtype=torch.int32, device=x.device).unsqueeze(0)
        chunk_matched_idx = matched_idx[:, packed_start:packed_stop]

        x_chunk = CtxGatherFunc3DGeneralized.apply(x_expanded, chunk_matched_idx)

        down_chunk = expert_mlp(x_chunk, W_g, W_u, W_d, b_g, b_u, b_d)

        rw_chunk = CtxGatherFunc3DGeneralized.apply(routing_weight, chunk_matched_idx)
        down_chunk = down_chunk * rw_chunk
        expert_out_chunk = CtxGatherFunc3DGeneralized.apply(expert_out, chunk_matched_idx)
        updated_chunk = expert_out_chunk + down_chunk

        chunk_valid_rows = torch.clamp(
            valid_rows - packed_start,
            min=torch.zeros_like(valid_rows),
            max=torch.full_like(valid_rows, chunk_rows),
        )
        updated_chunk = torch.where(
            (row_range < chunk_valid_rows).unsqueeze(-1), updated_chunk, torch.zeros_like(updated_chunk)
        )
        expert_out = CtxScatterFunc3DGeneralized.apply(expert_out, chunk_matched_idx, updated_chunk)

    return expert_out
