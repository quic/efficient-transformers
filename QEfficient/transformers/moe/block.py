# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Shared MoE block orchestration.

``QEffMoEBlockMixin`` provides the ``forward`` that routes tokens, dispatches to the
configured flavour, and adds shared experts. Per-model blocks subclass it and
override only the variation points:

* ``route(x)``               -> (routing, router_logits); routing is dense ``[T,E]``
                                 or ``(topk_indices, topk_weights)``.
* ``get_moe_weights()``      -> canonical :class:`MoEWeights` (default: ``self.moe_weights``).
* ``moe_profile``            -> :class:`MoEProfile` (default: plain SiLU-GLU).
* ``apply_shared_experts``   -> add a shared-expert term (default: no-op).

The flavour is assigned at export time by ``OptimizedMoETransform`` via
``self._moe_flavour`` and the ``expert_blocking_*`` attributes.
"""

from typing import Optional

import torch

from QEfficient.transformers.moe.flavours import (
    MoEFlavour,
    moe_decode_bmm,
    moe_expert_blocked,
    moe_simple_loop,
    resolve_routing,
)
from QEfficient.transformers.moe.profiles import SILU_GLU_PROFILE, MoEProfile
from QEfficient.transformers.moe.weights import MoEWeights


class QEffMoEBlockMixin:
    # Default profile; models with non-standard activations override this.
    moe_profile: MoEProfile = SILU_GLU_PROFILE
    # Set by OptimizedMoETransform at export time; decode is the safe default.
    _moe_flavour: MoEFlavour = MoEFlavour.DECODE_BMM
    # Whether forward returns (out, router_logits) to match the HF MoE convention.
    _moe_return_router_logits: bool = False
    # Expert-blocking knobs, set by OptimizedMoETransform when flavour is EXPERT_BLOCKED.
    expert_blocking_num_nsp: Optional[int] = None
    expert_blocking_packed_chunk_size: Optional[int] = None
    expert_blocking_num_packed_chunks: int = 1

    # ---- variation points (override per model) --------------------------------
    def route(self, x: torch.Tensor):
        raise NotImplementedError

    def get_moe_weights(self) -> MoEWeights:
        return self.moe_weights

    def apply_shared_experts(self, out: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return out

    # ---- orchestration (shared) ----------------------------------------------
    def moe_dispatch(self, x: torch.Tensor, routing) -> torch.Tensor:
        weights = self.get_moe_weights()
        profile = self.moe_profile

        if self._moe_flavour is MoEFlavour.DECODE_BMM:
            dense, topk = resolve_routing(routing, weights.num_experts)
            if topk is None:
                raise ValueError("decode_bmm flavour requires route() to return (topk_indices, topk_weights)")
            topk_indices, topk_weights = topk
            return moe_decode_bmm(x, topk_indices, topk_weights, weights, profile, top_k=topk_indices.shape[1])

        dense, _ = resolve_routing(routing, weights.num_experts)
        if self._moe_flavour is MoEFlavour.EXPERT_BLOCKED:
            num_nsp = self.expert_blocking_num_nsp or weights.num_experts
            packed_chunk_size = self.expert_blocking_packed_chunk_size or x.shape[0]
            return moe_expert_blocked(
                x,
                dense,
                weights,
                profile,
                num_nsp=num_nsp,
                packed_chunk_size=packed_chunk_size,
                num_packed_chunks=getattr(self, "expert_blocking_num_packed_chunks", 1),
            )
        return moe_simple_loop(x, dense, weights, profile)

    def forward(self, hidden_states: torch.Tensor):
        B, S, H = hidden_states.shape
        x = hidden_states.view(B * S, H)
        routing, router_logits = self.route(x)
        out = self.moe_dispatch(x, routing)
        out = self.apply_shared_experts(out, x)
        out = out.view(B, S, H)
        if self._moe_return_router_logits:
            return out, router_logits
        return out
