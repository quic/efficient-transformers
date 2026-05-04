# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
FFN blocking utilities.

This mirrors the attention blocking structure:
- A small dataclass config + enum describing the blocking mode.
- A generic interface used by modeling code to dispatch to a blocked FFN forward.

FFN blocking modes:
- ""   : no blocking (default)
- "t"  : token blocking (split seq_len into blocks, full hidden intermediate each block)
- "w"  : weight blocking (split intermediate hidden dim into blocks, full seq_len)
- "tw" : both token + weight blocking
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional

import torch
from torch import nn

from QEfficient.blocking.blocked_ffn_forwards import (
    blocked_tokens_ffn_forward,
    blocked_tokens_weights_ffn_forward,
    blocked_weights_ffn_forward,
)


class FFNBlockingMode(str, Enum):
    NONE = ""
    T = "t"
    W = "w"
    TW = "tw"


@dataclass
class FFNBlockingConfig:
    mode: FFNBlockingMode = FFNBlockingMode.NONE
    num_token_blocks: Optional[int] = None
    num_weight_blocks: Optional[int] = None


_STRATEGIES: Dict[FFNBlockingMode, Callable[..., torch.Tensor]] = {
    FFNBlockingMode.T: blocked_tokens_ffn_forward,
    FFNBlockingMode.W: blocked_weights_ffn_forward,
    FFNBlockingMode.TW: blocked_tokens_weights_ffn_forward,
}


def generic_blocked_ffn_interface(
    *,
    w1: nn.Module,
    w2: nn.Module,
    x: torch.Tensor,
    blocking_config: FFNBlockingConfig,
    w3: Optional[nn.Module] = None,
    dropout: Optional[float] = None,
    activation_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Generic interface used by modeling files to route through a selected FFN blocking strategy.
    """
    if blocking_config is None or blocking_config.mode in (None, FFNBlockingMode.NONE, ""):
        act = activation_fn if activation_fn is not None else torch.nn.functional.silu
        # llama style mlp has up_proj, down_proj and gate_proj
        if w3 is not None:
            return w2(act(w1(x)) * w3(x))
        # gpt2 style mlp has gate_proj and down_proj
        else:
            return w2(act(w1(x)))

    strategy = _STRATEGIES.get(blocking_config.mode)
    if strategy is None:
        raise ValueError(f"Unsupported FFN blocking mode: {blocking_config.mode}")

    return strategy(
        w1=w1,
        w2=w2,
        w3=w3,
        x=x,
        num_token_blocks=blocking_config.num_token_blocks,
        num_weight_blocks=blocking_config.num_weight_blocks,
        activation_fn=activation_fn,
    )
