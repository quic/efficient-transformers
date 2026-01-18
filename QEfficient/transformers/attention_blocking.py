# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from transformers.cache_utils import Cache

from QEfficient.transformers.blocked_attention_utils import blocked_kv_attention_forward, q_blocked_attention_forward


@dataclass
class AttentionBlockingConfig:
    mode: str = "none"
    num_kv_blocks: Optional[int] = None
    num_q_blocks: Optional[int] = None
    head_block_size: Optional[int] = None


class BlockingStrategy:
    def apply(
        self,
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        cache_kwargs: Dict,
        layer_idx: int,
        past_key_value: Cache,
        config: AttentionBlockingConfig,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError


class KVBlockingStrategy(BlockingStrategy):
    def apply(
        self,
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        cache_kwargs: Dict,
        layer_idx: int,
        past_key_value: Cache,
        config: AttentionBlockingConfig,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return blocked_kv_attention_forward(
            module=module,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            scaling=scaling,
            num_kv_blocks=config.num_kv_blocks,
            cache_kwargs=cache_kwargs,
            layer_idx=layer_idx,
            past_key_value=past_key_value,
            **kwargs,
        )


class NoOpBlockingStrategy(BlockingStrategy):
    def apply(self, *args, **kwargs):
        raise NotImplementedError("No-op blocking strategy should not be used for attention execution.")


class QBlockingStrategy(BlockingStrategy):
    def apply(
        self,
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        cache_kwargs: Dict,
        layer_idx: int,
        past_key_value: Cache,
        config: AttentionBlockingConfig,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return q_blocked_attention_forward(
            module=module,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            scaling=scaling,
            num_q_blocks=config.num_q_blocks or 1,
        )


class HeadBlockingStrategy(BlockingStrategy):
    def apply(self, *args, **kwargs):
        raise NotImplementedError("Head-blocking strategy is not implemented yet.")


_STRATEGIES: Dict[str, BlockingStrategy] = {
    "none": NoOpBlockingStrategy(),
    "kv": KVBlockingStrategy(),
    "q": QBlockingStrategy(),
    "head": HeadBlockingStrategy(),
    "qkv": QBlockingStrategy(),
    "hqkv": HeadBlockingStrategy(),
}


def get_blocking_strategy(config: AttentionBlockingConfig) -> BlockingStrategy:
    return _STRATEGIES.get(config.mode, _STRATEGIES["none"])
