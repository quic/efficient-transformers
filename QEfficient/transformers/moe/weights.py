# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Canonical MoE expert-weight container and builders.

Every MoE model stores its expert weights differently (fused gate_up vs separate
gate/up, transposed or not, interleaved halves, module-list, compressed). These
helpers canonicalize all of them into a single orientation so the shared flavour
kernels can be model-agnostic:

    gate : [E, H, I]
    up   : [E, H, I]
    down : [E, I, H]

(optional per-expert biases: gate_bias/up_bias [E, I], down_bias [E, H]).
"""

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import torch
from torch import nn


@dataclass
class MoEWeights:
    """Expert weights in canonical orientation (gate/up ``[E,H,I]``, down ``[E,I,H]``)."""

    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    gate_bias: Optional[torch.Tensor] = None
    up_bias: Optional[torch.Tensor] = None
    down_bias: Optional[torch.Tensor] = None

    @property
    def num_experts(self) -> int:
        return self.gate.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.gate.shape[1]

    @property
    def intermediate_size(self) -> int:
        return self.gate.shape[2]

    @property
    def has_bias(self) -> bool:
        return self.gate_bias is not None


def _maybe_clone(t: Optional[torch.Tensor], clone: bool) -> Optional[torch.Tensor]:
    if t is None:
        return None
    return t.detach().clone() if clone else t


def build_canonical_expert_weights(
    *,
    down: torch.Tensor,
    gate_up: Optional[torch.Tensor] = None,
    gate: Optional[torch.Tensor] = None,
    up: Optional[torch.Tensor] = None,
    gate_up_bias: Optional[torch.Tensor] = None,
    down_bias: Optional[torch.Tensor] = None,
    fused: bool = True,
    fused_split_dim: int = 1,
    interleaved: bool = False,
    transpose_gate_up: bool = True,
    transpose_down: bool = True,
    clone: bool = True,
) -> MoEWeights:
    """Canonicalize tensor-level expert weights into :class:`MoEWeights`.

    Args:
        down: source down-projection weight (``[E,H,I]`` if ``transpose_down`` else ``[E,I,H]``).
        gate_up: fused gate+up source (used when ``fused=True``).
        gate, up: pre-split sources (used when ``fused=False``).
        gate_up_bias: fused bias source split the same way as ``gate_up``.
        down_bias: down-projection bias ``[E,H]``.
        fused: whether ``gate_up`` carries both halves and must be split.
        fused_split_dim: dim along which the fused halves are concatenated
            (1 for HF ``[E,2I,H]``; 2 for HF ``[E,H,2I]`` as in GPT-OSS).
        interleaved: GPT-OSS stores ``[gate0, up0, gate1, up1, ...]`` along the
            split dim; when True the halves are taken as ``[..., ::2]`` / ``[..., 1::2]``.
        transpose_gate_up: transpose the last two dims of gate/up to reach ``[E,H,I]``.
        transpose_down: transpose the last two dims of down to reach ``[E,I,H]``.
        clone: detach+clone the slices (set False to alias existing parameters).
    """
    if fused:
        if gate_up is None:
            raise ValueError("build_canonical_expert_weights(fused=True) requires gate_up")
        half = gate_up.shape[fused_split_dim] // 2
        if interleaved:
            idx_even = torch.arange(0, gate_up.shape[fused_split_dim], 2, device=gate_up.device)
            idx_odd = torch.arange(1, gate_up.shape[fused_split_dim], 2, device=gate_up.device)
            gate = gate_up.index_select(fused_split_dim, idx_even)
            up = gate_up.index_select(fused_split_dim, idx_odd)
        else:
            gate = gate_up.narrow(fused_split_dim, 0, half)
            up = gate_up.narrow(fused_split_dim, half, half)
    else:
        if gate is None or up is None:
            raise ValueError("build_canonical_expert_weights(fused=False) requires gate and up")

    gate_bias = up_bias = None
    if gate_up_bias is not None:
        # bias fused along the last dim ([E,2I]); same interleave rule as the weights.
        half_b = gate_up_bias.shape[-1] // 2
        if interleaved:
            gate_bias = gate_up_bias[..., ::2]
            up_bias = gate_up_bias[..., 1::2]
        else:
            gate_bias = gate_up_bias[..., :half_b]
            up_bias = gate_up_bias[..., half_b:]

    if transpose_gate_up:
        gate = gate.transpose(-1, -2)
        up = up.transpose(-1, -2)
    if transpose_down:
        down = down.transpose(-1, -2)

    return MoEWeights(
        gate=_maybe_clone(gate.contiguous() if clone else gate, clone),
        up=_maybe_clone(up.contiguous() if clone else up, clone),
        down=_maybe_clone(down.contiguous() if clone else down, clone),
        gate_bias=_maybe_clone(gate_bias, clone),
        up_bias=_maybe_clone(up_bias, clone),
        down_bias=_maybe_clone(down_bias, clone),
    )


def stack_expert_linears(
    experts: Iterable[Any],
    weight_getter: Callable[[Any], torch.Tensor],
    *,
    transpose: bool = True,
) -> torch.Tensor:
    """Stack per-expert weights from a module list into a single ``[E, *, *]`` tensor.

    ``weight_getter`` returns one expert's 2-D weight (e.g. ``lambda e: e.gate_proj.weight``);
    with ``transpose=True`` a ``nn.Linear`` weight ``[out, in]`` becomes ``[in, out]``.
    """
    slices = []
    for expert in experts:
        w = weight_getter(expert)
        if transpose:
            w = w.t()
        slices.append(w.unsqueeze(0))
    return torch.cat(slices, dim=0).contiguous()


def as_parameters(weights: MoEWeights) -> MoEWeights:
    """Wrap each present tensor of ``weights`` in an ``nn.Parameter`` (for module registration)."""
    return MoEWeights(
        gate=nn.Parameter(weights.gate),
        up=nn.Parameter(weights.up),
        down=nn.Parameter(weights.down),
        gate_bias=nn.Parameter(weights.gate_bias) if weights.gate_bias is not None else None,
        up_bias=nn.Parameter(weights.up_bias) if weights.up_bias is not None else None,
        down_bias=nn.Parameter(weights.down_bias) if weights.down_bias is not None else None,
    )
