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
``self._moe_flavour`` and the ``expert_parallel_*`` attributes.
"""

from typing import Optional, Tuple

import torch

from QEfficient.transformers.moe.flavours import (
    MoEFlavour,
    moe_decode_bmm,
    moe_expert_parallel,
    moe_simple_loop,
    resolve_routing,
)
from QEfficient.transformers.moe.profiles import SILU_GLU_PROFILE, MoEProfile
from QEfficient.transformers.moe.weights import MoEWeights


class QEffMoEBlockMixin:
    # Default profile; models with non-standard activations override this.
    moe_profile: MoEProfile = SILU_GLU_PROFILE
    supported_moe_flavours: Tuple[MoEFlavour, ...] = (MoEFlavour.SIMPLE_LOOP, MoEFlavour.DECODE_BMM)
    # Set by OptimizedMoETransform at export time; decode is the safe default.
    _moe_flavour: MoEFlavour = MoEFlavour.DECODE_BMM
    # Whether forward returns (out, router_logits) to match the HF MoE convention.
    _moe_return_router_logits: bool = False
    # Expert-parallel knobs, set by OptimizedMoETransform when flavour is EXPERT_PARALLEL.
    expert_parallel_num_nsp: Optional[int] = None
    expert_parallel_packed_chunk_size: Optional[int] = None
    expert_parallel_num_packed_chunks: int = 1

    @property
    def expert_blocking_num_nsp(self) -> Optional[int]:
        return self.expert_parallel_num_nsp

    @expert_blocking_num_nsp.setter
    def expert_blocking_num_nsp(self, value: Optional[int]) -> None:
        self.expert_parallel_num_nsp = value

    @property
    def expert_blocking_packed_chunk_size(self) -> Optional[int]:
        return self.expert_parallel_packed_chunk_size

    @expert_blocking_packed_chunk_size.setter
    def expert_blocking_packed_chunk_size(self, value: Optional[int]) -> None:
        self.expert_parallel_packed_chunk_size = value

    @property
    def expert_blocking_num_packed_chunks(self) -> int:
        return self.expert_parallel_num_packed_chunks

    @expert_blocking_num_packed_chunks.setter
    def expert_blocking_num_packed_chunks(self, value: int) -> None:
        self.expert_parallel_num_packed_chunks = value

    # ---- variation points (override per model) --------------------------------
    def route(self, x: torch.Tensor):
        raise NotImplementedError

    def get_moe_weights(self) -> MoEWeights:
        return self.moe_weights

    def apply_shared_experts(self, out: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return out

    def get_supported_moe_flavours(self) -> Tuple[MoEFlavour, ...]:
        return tuple(getattr(self, "supported_moe_flavours", QEffMoEBlockMixin.supported_moe_flavours))

    # ---- orchestration (shared) ----------------------------------------------
    def execute_moe_flavour(self, x: torch.Tensor, routing) -> torch.Tensor:
        weights = self.get_moe_weights()
        profile = self.moe_profile
        if callable(profile):
            profile = profile()

        flavour = getattr(self, "_moe_flavour", MoEFlavour.DECODE_BMM)
        if not isinstance(flavour, MoEFlavour):
            flavour = MoEFlavour(flavour)

        if flavour is MoEFlavour.DECODE_BMM:
            _, topk = resolve_routing(routing, weights.num_experts)
            if topk is None:
                raise ValueError("decode_bmm flavour requires route() to return (topk_indices, topk_weights)")
            topk_indices, topk_weights = topk
            return moe_decode_bmm(x, topk_indices, topk_weights, weights, profile, top_k=topk_indices.shape[1])

        dense, _ = resolve_routing(routing, weights.num_experts)
        if flavour is MoEFlavour.EXPERT_PARALLEL:
            num_nsp = getattr(self, "expert_parallel_num_nsp", None)
            if num_nsp is None:
                num_nsp = getattr(self, "expert_blocking_num_nsp", None) or weights.num_experts
            packed_chunk_size = getattr(self, "expert_parallel_packed_chunk_size", None)
            if packed_chunk_size is None:
                packed_chunk_size = getattr(self, "expert_blocking_packed_chunk_size", None) or x.shape[0]
            num_packed_chunks = getattr(
                self,
                "expert_parallel_num_packed_chunks",
                getattr(self, "expert_blocking_num_packed_chunks", 1),
            )
            return moe_expert_parallel(
                x,
                dense,
                weights,
                profile,
                num_nsp=num_nsp,
                packed_chunk_size=packed_chunk_size,
                num_packed_chunks=num_packed_chunks,
            )
        return moe_simple_loop(x, dense, weights, profile, prescale=profile.scale_mode == "pre")

    def moe_dispatch(self, x: torch.Tensor, routing) -> torch.Tensor:
        return self.execute_moe_flavour(x, routing)

    def forward(self, hidden_states: torch.Tensor):
        B, S, H = hidden_states.shape
        x = hidden_states.view(B * S, H)
        routing, router_logits = self.route(x)
        out = self.execute_moe_flavour(x, routing)
        out = self.apply_shared_experts(out, x)
        out = out.view(B, S, H)
        if self._moe_return_router_logits:
            return out, router_logits
        return out
