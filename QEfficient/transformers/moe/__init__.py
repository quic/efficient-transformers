# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Common Mixture-of-Experts (MoE) infrastructure shared across model families.

This package owns the three MoE forward flavours (prefill expert-blocking, prefill
simple loop, decode gather+bmm), the canonical expert-weight container/builders, and
the per-expert activation profiles. Model files declare their variation points and
delegate the rest here so the math lives in one place.
"""

from QEfficient.transformers.moe.adapters import (
    MoEAdapterSpec,
    bind_moe_adapter_methods,
    build_moe_weights,
    get_moe_adapter_spec,
    register_moe_adapter,
)
from QEfficient.transformers.moe.block import QEffMoEBlockMixin
from QEfficient.transformers.moe.flavours import (
    MoEFlavour,
    densify_topk,
    moe_decode_bmm,
    moe_expert_blocked,
    moe_expert_parallel,
    moe_simple_loop,
    resolve_routing,
    select_moe_flavour,
)
from QEfficient.transformers.moe.kernels import (
    build_matched_idx_from_cumsum,
    cumsum_scatter_gather_update_expert_blocked,
)
from QEfficient.transformers.moe.profiles import (
    SILU_GLU_PROFILE,
    MoEProfile,
    gptoss_clamped_glu_mlp,
    silu_glu_mlp,
)
from QEfficient.transformers.moe.weights import (
    MoEWeights,
    as_parameters,
    build_canonical_expert_weights,
    stack_expert_linears,
)

__all__ = [
    "QEffMoEBlockMixin",
    "MoEAdapterSpec",
    "bind_moe_adapter_methods",
    "build_moe_weights",
    "get_moe_adapter_spec",
    "register_moe_adapter",
    "MoEFlavour",
    "densify_topk",
    "moe_decode_bmm",
    "moe_expert_blocked",
    "moe_expert_parallel",
    "moe_simple_loop",
    "resolve_routing",
    "select_moe_flavour",
    "build_matched_idx_from_cumsum",
    "cumsum_scatter_gather_update_expert_blocked",
    "SILU_GLU_PROFILE",
    "MoEProfile",
    "gptoss_clamped_glu_mlp",
    "silu_glu_mlp",
    "MoEWeights",
    "as_parameters",
    "build_canonical_expert_weights",
    "stack_expert_linears",
]
