# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""The three MoE forward flavours and the flavour selector.

All flavours consume canonical :class:`MoEWeights` plus a :class:`MoEProfile` and
return the routed expert output ``[T, H]``:

* ``moe_expert_parallel`` / ``moe_simple_loop`` consume a dense ``routing_weights [T, E]``.
* ``moe_decode_bmm`` consumes ``(topk_indices [T, K], topk_weights [T, K])``.
"""

from enum import Enum
from typing import Callable, Optional, Tuple

import torch

from QEfficient.customop.ctx_scatter_gather import (
    CtxGatherFunc3DGeneralized,
    CtxScatterFunc3DGeneralized,
    CtxScatterFunc3DInt,
)
from QEfficient.transformers.moe.profiles import MoEProfile
from QEfficient.transformers.moe.weights import MoEWeights


class MoEFlavour(str, Enum):
    EXPERT_PARALLEL = "expert_parallel"
    # Backward-compatible alias for one transition period.
    EXPERT_BLOCKED = "expert_parallel"
    SIMPLE_LOOP = "simple_loop"
    DECODE_BMM = "decode_bmm"

    @classmethod
    def _missing_(cls, value):
        if value == "expert_blocked":
            return cls.EXPERT_PARALLEL
        return None


def densify_topk(topk_indices: torch.Tensor, topk_weights: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Scatter ``(topk_indices, topk_weights)`` into a dense ``[T, E]`` routing matrix."""
    T = topk_indices.shape[0]
    routing_weights = topk_weights.new_zeros((T, num_experts))
    routing_weights.scatter_(1, topk_indices, topk_weights)
    return routing_weights


def _slot_bias(bias: Optional[torch.Tensor], local_experts: int, num_nsp: int) -> Optional[torch.Tensor]:
    if bias is None:
        return None
    return bias.view(local_experts, num_nsp, -1).transpose(0, 1).contiguous()


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
    """Run one local expert slot over statically traced packed chunks."""
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


def moe_expert_parallel(
    x: torch.Tensor,
    routing_weights: torch.Tensor,
    weights: MoEWeights,
    profile: MoEProfile,
    *,
    num_nsp: int,
    packed_chunk_size: int,
    num_packed_chunks: int = 1,
) -> torch.Tensor:
    """Prefill expert-parallel flavour: NUM_NSP experts per block, cumsum/scatter packed."""
    T, H = x.shape
    num_experts = weights.num_experts
    if num_experts % num_nsp != 0:
        raise ValueError(f"num_experts ({num_experts}) must be divisible by expert_parallel_num_nsp ({num_nsp})")
    local_experts = num_experts // num_nsp

    rw = routing_weights.transpose(0, 1).contiguous().view(local_experts, num_nsp, T).transpose(0, 1).contiguous()
    W_g = weights.gate.view(local_experts, num_nsp, H, -1).transpose(0, 1).contiguous()
    W_u = weights.up.view(local_experts, num_nsp, H, -1).transpose(0, 1).contiguous()
    W_d = weights.down.view(local_experts, num_nsp, -1, H).transpose(0, 1).contiguous()
    b_g = _slot_bias(weights.gate_bias, local_experts, num_nsp)
    b_u = _slot_bias(weights.up_bias, local_experts, num_nsp)
    b_d = _slot_bias(weights.down_bias, local_experts, num_nsp)

    expert_out = x.new_zeros((num_nsp, T, H))
    routing_weights_unsqueezed = rw.unsqueeze(-1)
    for slot in range(local_experts):
        T2Ei = rw[:, slot, :] > 0
        expert_out = cumsum_scatter_gather_update_expert_blocked(
            x=x,
            T2Ei=T2Ei,
            W_g=W_g[:, slot],
            W_u=W_u[:, slot],
            W_d=W_d[:, slot],
            routing_weight=routing_weights_unsqueezed[:, slot],
            expert_out=expert_out,
            expert_mlp=profile.expert_mlp,
            b_g=b_g[:, slot] if b_g is not None else None,
            b_u=b_u[:, slot] if b_u is not None else None,
            b_d=b_d[:, slot] if b_d is not None else None,
            packed_chunk_size=packed_chunk_size,
            num_packed_chunks=num_packed_chunks,
        )
    return torch.einsum("nth->th", expert_out)


# Backward-compatible helper name for one transition period.
moe_expert_blocked = moe_expert_parallel


def moe_simple_loop(
    x: torch.Tensor,
    routing_weights: torch.Tensor,
    weights: MoEWeights,
    profile: MoEProfile,
    *,
    prescale: bool = False,
) -> torch.Tensor:
    """Prefill simple-loop flavour: one masked pass per expert."""
    T, H = x.shape
    expert_out = x.new_zeros((T, H))
    for e in range(weights.num_experts):
        routing_weight = routing_weights[:, e].unsqueeze(-1)
        expert_input = x * routing_weight if prescale else x
        down = profile.expert_mlp(
            expert_input,
            weights.gate[e],
            weights.up[e],
            weights.down[e],
            weights.gate_bias[e] if weights.gate_bias is not None else None,
            weights.up_bias[e] if weights.up_bias is not None else None,
            weights.down_bias[e] if weights.down_bias is not None else None,
        )
        if prescale or profile.scale_mode == "pre":
            expert_out = expert_out + torch.where(routing_weight > 0, down, torch.zeros_like(down))
        else:
            expert_out = expert_out + down * routing_weight
    return expert_out


def moe_decode_bmm(
    x: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: MoEWeights,
    profile: MoEProfile,
    *,
    top_k: int,
) -> torch.Tensor:
    """Decode flavour: gather selected expert weights by index and batch-matmul."""
    T, H = x.shape
    idx = topk_indices.reshape(-1)
    gate_proj = weights.gate[idx]
    up_proj = weights.up[idx]
    down_proj = weights.down[idx]
    b_g = weights.gate_bias[idx] if weights.gate_bias is not None else None
    b_u = weights.up_bias[idx] if weights.up_bias is not None else None
    b_d = weights.down_bias[idx] if weights.down_bias is not None else None

    expert_in = x.unsqueeze(1).expand(-1, top_k, -1).contiguous().view(-1, 1, H)
    down = profile.expert_mlp(expert_in, gate_proj, up_proj, down_proj, b_g, b_u, b_d)
    experts_out = down.view(T, top_k, H)
    experts_out = experts_out * topk_weights.unsqueeze(-1)
    return torch.einsum("bnd->bd", experts_out)


# Models whose `auto` prefill flavour is expert-parallel even without chunking.
# GPT-OSS remains class-driven for now because its non-chunked prefill path has
# extra behavior beyond flavour selection.
_AUTO_EXPERT_PARALLEL_MODEL_TYPES = {
    "qwen3_moe",
    "qwen3_5_moe",
    "qwen3_vl_moe",
    "qwen3_vl_moe_text",
    "glm4_moe",
}


def select_moe_flavour(
    qaic_config: Optional[dict],
    model_type: str,
    *,
    is_prefill: bool,
    supports_blocking: bool,
    enable_chunking: bool,
    supports_decode_bmm: bool = True,
) -> MoEFlavour:
    """Resolve the MoE forward flavour for a module.

    Explicit overrides are read only from top-level ``qaic_config["moe_flavour"]``.
    ``auto`` uses gather+bmm for decode and the best supported prefill flavour for
    prefill.
    """
    override = (qaic_config or {}).get("moe_flavour", "auto")
    if isinstance(override, MoEFlavour):
        flavour = override
    elif override in (None, "auto"):
        if not is_prefill:
            flavour = MoEFlavour.DECODE_BMM if supports_decode_bmm else MoEFlavour.SIMPLE_LOOP
        elif supports_blocking and (enable_chunking or model_type in _AUTO_EXPERT_PARALLEL_MODEL_TYPES):
            flavour = MoEFlavour.EXPERT_PARALLEL
        else:
            flavour = MoEFlavour.SIMPLE_LOOP
    else:
        try:
            flavour = MoEFlavour(override)
        except ValueError as exc:
            raise ValueError(f"Unsupported qaic_config['moe_flavour']={override!r}") from exc

    if flavour is MoEFlavour.EXPERT_PARALLEL and not supports_blocking:
        raise AssertionError(
            f"moe flavour 'expert_parallel' requested for model_type {model_type!r} "
            "but the module does not set supports_moe_prefill_blocking=True"
        )
    if flavour is MoEFlavour.DECODE_BMM and not supports_decode_bmm:
        raise AssertionError(
            f"moe flavour 'decode_bmm' requested for model_type {model_type!r} "
            "but the module sets supports_moe_decode_bmm=False"
        )
    return flavour


def resolve_routing(
    routing,
    num_experts: int,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """Normalize a router output into (dense_routing_weights, optional topk pair).

    Accepts either a dense ``[T, E]`` tensor or a ``(topk_indices, topk_weights)``
    tuple and returns both representations (densifying the tuple when needed).
    """
    if torch.is_tensor(routing):
        return routing, None
    topk_indices, topk_weights = routing
    return densify_topk(topk_indices, topk_weights, num_experts), (topk_indices, topk_weights)
