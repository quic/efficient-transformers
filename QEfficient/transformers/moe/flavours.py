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
from typing import Optional, Tuple

import torch

from QEfficient.transformers.moe.kernels import cumsum_scatter_gather_update_expert_blocked
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
