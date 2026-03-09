# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional

import torch
from transformers.cache_utils import Cache

from QEfficient.blocking.blocked_attention_forwards import (
    blocked_h_attention_forward,
    blocked_hqkv_attention_forward,
    blocked_kv_attention_forward,
    blocked_q_attention_forward,
    blocked_qkv_attention_forward,
    invalid_blocking_attention_forward,
)


class BlockingMode(str, Enum):
    NONE = ""
    KV = "kv"
    Q = "q"
    H = "h"
    QKV = "qkv"
    HQKV = "hqkv"


@dataclass
class AttentionBlockingConfig:
    mode: BlockingMode = BlockingMode.NONE
    num_kv_blocks: Optional[int] = None
    num_q_blocks: Optional[int] = None
    head_block_size: Optional[int] = None


def supports_blocked_kv(past_key_value: Optional[Cache]) -> bool:
    return past_key_value is not None and hasattr(past_key_value, "read_only_blockedKV")


_STRATEGIES: Dict[BlockingMode, Callable] = {
    BlockingMode.NONE: invalid_blocking_attention_forward,
    BlockingMode.KV: blocked_kv_attention_forward,
    BlockingMode.Q: blocked_q_attention_forward,
    BlockingMode.H: blocked_h_attention_forward,
    BlockingMode.QKV: blocked_qkv_attention_forward,
    BlockingMode.HQKV: blocked_hqkv_attention_forward,
}


def get_blocking_strategy(config: AttentionBlockingConfig) -> Callable:
    return _STRATEGIES.get(config.mode, _STRATEGIES[BlockingMode.NONE])


def generic_blocked_attention_interface(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    layer_idx: int,
    past_key_value: Cache,
    blocking_config: AttentionBlockingConfig,
    comp_ctx_lengths: Optional[torch.LongTensor] = None,
    batch_index: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_seen_tokens: Optional[int] = None,
    non_blocked_forward: Callable = None,
    **kwargs,
):
    use_kv_blocked = (
        blocking_config is not None and "kv" in blocking_config.mode and supports_blocked_kv(past_key_value)
    )
    use_blocking = blocking_config is not None and (blocking_config.mode != BlockingMode.NONE)

    if past_key_value is not None:
        if use_kv_blocked:
            cache_kwargs = {
                "batch_index": batch_index,
                "position_ids": position_ids,
                "past_seen_tokens": past_seen_tokens,
            }
            past_key_value.write_only(key, value, module.layer_idx, cache_kwargs)
        else:
            cache_kwargs = {"batch_index": batch_index, "position_ids": position_ids}
            if comp_ctx_lengths is not None:
                attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
                cache_kwargs["CCL"] = attention_mask.shape[-1]
            key, value = past_key_value.update(key, value, module.layer_idx, cache_kwargs)

    if use_blocking:
        strategy = get_blocking_strategy(blocking_config)
        attn_output, attn_weights = strategy(
            module=module,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            scaling=scaling,
            cache_kwargs=cache_kwargs,
            layer_idx=layer_idx,
            past_key_value=past_key_value,
            num_kv_blocks=blocking_config.num_kv_blocks,
            num_q_blocks=blocking_config.num_q_blocks,
            head_block_size=blocking_config.head_block_size,
        )
    else:
        attn_output, attn_weights = non_blocked_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            **kwargs,
        )

    return attn_output, attn_weights
