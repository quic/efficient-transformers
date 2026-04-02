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
    skip_kv: Optional[bool] = False


def supports_blocked_kv(past_key_value: Optional[Cache]) -> bool:
    return past_key_value is not None and hasattr(past_key_value, "read_only_blockedKV")


_STRATEGIES: Dict[BlockingMode, Callable] = {
    BlockingMode.KV: blocked_kv_attention_forward,
    BlockingMode.Q: blocked_q_attention_forward,
    BlockingMode.H: blocked_h_attention_forward,
    BlockingMode.QKV: blocked_qkv_attention_forward,
    BlockingMode.HQKV: blocked_hqkv_attention_forward,
}


def get_blocking_strategy(config: AttentionBlockingConfig) -> Callable:
    return _STRATEGIES.get(config.mode)


# helper function needed both in generic blocked approach and in other modeling files for non-blocked approach
def past_key_value_update(
    module,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    past_key_value: Cache,
    comp_ctx_lengths: Optional[torch.LongTensor] = None,
    batch_index: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    sliding_window: Optional[int] = None,
):
    if past_key_value is not None:
        cache_kwargs = {"batch_index": batch_index, "position_ids": position_ids}
        if sliding_window is not None:
            cache_kwargs.update(
                {
                    "is_sliding": sliding_window is not None,
                    "sliding_window": past_key_value.sliding_window_len,
                }
            )
        if comp_ctx_lengths is not None:
            attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
            cache_kwargs["CCL"] = attention_mask.shape[-1]
        key, value = past_key_value.update(key, value, module.layer_idx, cache_kwargs)
    return key, value, cache_kwargs


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
    score_mod: Optional[Callable] = None,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
    **kwargs,
):
    use_kv_blocked = (
        blocking_config is not None and "kv" in blocking_config.mode and supports_blocked_kv(past_key_value)
    )

    if past_key_value is not None:
        if use_kv_blocked and sliding_window is None:
            cache_kwargs = {
                "batch_index": batch_index,
                "position_ids": position_ids,
                "past_seen_tokens": past_seen_tokens,
            }
            if sliding_window is not None:
                cache_kwargs.update(
                    {
                        "is_sliding": sliding_window is not None,
                        "sliding_window": past_key_value.sliding_window_len,
                    }
                )
            past_key_value.write_only(key, value, module.layer_idx, cache_kwargs)
        else:
            key, value, cache_kwargs = past_key_value_update(
                module=module,
                key=key,
                value=value,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                position_ids=position_ids,
                sliding_window=sliding_window,
            )

    strategy = _STRATEGIES.get(blocking_config.mode)
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
        score_mod=score_mod,
        position_bias=position_bias,
        sinks=sinks,
    )

    return attn_output, attn_weights
