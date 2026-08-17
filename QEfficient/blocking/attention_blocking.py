# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

import torch
from transformers.cache_utils import Cache

from QEfficient.blocking.blocked_attention_forwards import (
    blocked_bhqkv_attention_forward,
    blocked_h_attention_forward,
    blocked_h_mla_attention_forward,
    blocked_hqkv_attention_forward,
    blocked_kv_attention_forward,
    blocked_kv_attention_forward_decode_headpar_batch,
    blocked_kv_attention_forward_headpar_offline,
    blocked_kv_attention_forward_prefill_headpar_offline,
    blocked_kv_mla_attention_forward,
    blocked_q_attention_forward,
    blocked_q_attention_forward_prefill,
    blocked_qkv_attention_forward,
    blocked_qkv_attention_forward_prefill_headpar_offline,
    blocked_qkv_attention_forward_prefill_online,
)


class BlockingMode(str, Enum):
    NONE = ""
    AUTO = "auto"
    # decode
    KV = "kv"
    KV_HEADPAR = "kv_headpar"
    KV_BATCH_FOLD = "kv_batch_fold"
    Q = "q"
    H = "h"
    QKV = "qkv"
    HQ = "hq"
    HKV = "hkv"
    HQKV = "hqkv"
    BHQKV = "bhqkv"
    # MLA
    KV_MLA = "kv_mla"
    H_MLA = "h_mla"
    # prefill
    PREFILL_Q = "prefill_q"
    PREFILL_KV = "prefill_kv"
    PREFILL_QKV = "prefill_qkv"
    PREFILL_ONLINE = "prefill_online"

    @classmethod
    def resolve(cls, mode: Optional[str | "BlockingMode"]) -> "BlockingMode":
        if mode is None:
            return cls.NONE
        resolved_mode = cls(mode)
        if resolved_mode == cls.AUTO:
            return cls.HQKV
        return resolved_mode

    @property
    def is_prefill(self) -> bool:
        return self.value.startswith("prefill_")

    @classmethod
    def get_final_mode(
        cls,
        blocking_config: "AttentionBlockingConfig",
        *,
        prefill_only: bool = False,
        is_mla: bool = False,
        mla_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "BlockingMode":
        if is_mla:
            _mla_map = {cls.KV: cls.KV_MLA, cls.H: cls.H_MLA}
            mode = _mla_map[blocking_config.mode]
        else:
            mode = blocking_config.mode
        cls._validate_config(mode, blocking_config, prefill_only=prefill_only, mla_kwargs=mla_kwargs)
        return mode

    @classmethod
    def _validate_config(
        cls,
        mode: "BlockingMode",
        blocking_config: "AttentionBlockingConfig",
        prefill_only: bool = False,
        mla_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if mode.is_prefill and not prefill_only:
            raise ValueError(
                f"BlockingMode.{mode.name} is a prefill-only mode; set prefill_only=True when calling the blocked attention interface"
            )
        missing = [f for f in BLOCKING_MODE_REQUIRED_PARAMS.get(mode, []) if getattr(blocking_config, f, None) is None]
        if missing:
            raise ValueError(f"BlockingMode.{mode.name} requires {missing} to be set in AttentionBlockingConfig")

        _REQUIRED_MLA_KWARGS: Dict["BlockingMode", list] = {
            cls.KV_MLA: ["per_head_k_up_normal", "per_head_v_up", "mla_absorption"],
            cls.H_MLA: [
                "q_a_proj_out",
                "fusedqk",
                "q_nope",
                "q_pe",
                "kva",
                "k_pe",
                "per_head_q_up",
                "per_head_k_up",
                "per_head_v_up",
                "per_head_k_up_normal",
                "mla_absorption",
            ],
        }
        if mode in _REQUIRED_MLA_KWARGS:
            mla = mla_kwargs or {}
            missing_mla = [k for k in _REQUIRED_MLA_KWARGS[mode] if mla.get(k) is None]
            if missing_mla:
                raise ValueError(f"BlockingMode.{mode.name} requires {missing_mla} to be set in mla_kwargs")


@dataclass
class AttentionBlockingConfig:
    mode: BlockingMode = BlockingMode.NONE
    num_kv_blocks: Optional[int] = None
    num_q_blocks: Optional[int] = None
    head_block_size: Optional[int] = None
    skip_kv: Optional[bool] = True
    num_batch_blocks: Optional[int] = None
    headpar_split: Optional[int] = None
    batch_fold: Optional[bool] = False
    n_rep_chunk: Optional[int] = None
    ctx_len: Optional[int] = None
    kv_block_unroll: Optional[int] = 1


# Required AttentionBlockingConfig fields per blocking mode.
BLOCKING_MODE_REQUIRED_PARAMS: Dict[BlockingMode, list] = {
    # decode
    BlockingMode.KV: ["num_kv_blocks"],
    BlockingMode.KV_BATCH_FOLD: ["num_kv_blocks"],
    BlockingMode.KV_HEADPAR: ["num_kv_blocks"],
    BlockingMode.Q: ["num_q_blocks"],
    BlockingMode.H: ["head_block_size"],
    BlockingMode.QKV: ["num_kv_blocks", "num_q_blocks"],
    BlockingMode.HQ: ["head_block_size", "num_q_blocks"],
    BlockingMode.HKV: ["head_block_size", "num_kv_blocks"],
    BlockingMode.HQKV: ["head_block_size", "num_kv_blocks", "num_q_blocks"],
    BlockingMode.BHQKV: ["head_block_size", "num_kv_blocks", "num_q_blocks", "num_batch_blocks"],
    # MLA
    BlockingMode.KV_MLA: ["num_kv_blocks"],
    BlockingMode.H_MLA: ["head_block_size"],
    # prefill
    BlockingMode.PREFILL_Q: ["num_q_blocks"],
    BlockingMode.PREFILL_KV: ["num_kv_blocks"],
    BlockingMode.PREFILL_QKV: ["num_kv_blocks", "num_q_blocks"],
    BlockingMode.PREFILL_ONLINE: ["num_kv_blocks", "num_q_blocks"],
}


def supports_blocked_kv(past_key_value: Optional[Cache]) -> bool:
    return past_key_value is not None and hasattr(past_key_value, "read_only_blockedKV")


_STRATEGIES: Dict[BlockingMode, Callable] = {
    # decode
    BlockingMode.KV: blocked_kv_attention_forward,
    BlockingMode.KV_HEADPAR: blocked_kv_attention_forward_headpar_offline,
    BlockingMode.KV_BATCH_FOLD: blocked_kv_attention_forward_decode_headpar_batch,
    BlockingMode.Q: blocked_q_attention_forward,
    BlockingMode.H: blocked_h_attention_forward,
    BlockingMode.QKV: blocked_qkv_attention_forward,
    BlockingMode.HQ: blocked_hqkv_attention_forward,
    BlockingMode.HKV: blocked_hqkv_attention_forward,
    BlockingMode.HQKV: blocked_hqkv_attention_forward,
    BlockingMode.BHQKV: blocked_bhqkv_attention_forward,
    # MLA
    BlockingMode.KV_MLA: blocked_kv_mla_attention_forward,
    BlockingMode.H_MLA: blocked_h_mla_attention_forward,
    # prefill
    BlockingMode.PREFILL_Q: blocked_q_attention_forward_prefill,
    BlockingMode.PREFILL_KV: blocked_kv_attention_forward_prefill_headpar_offline,
    BlockingMode.PREFILL_QKV: blocked_qkv_attention_forward_prefill_headpar_offline,
    BlockingMode.PREFILL_ONLINE: blocked_qkv_attention_forward_prefill_online,
}


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
    return key, value, attention_mask, cache_kwargs


def generic_blocked_attention_interface(
    module,
    query: Optional[torch.Tensor] = None,
    key: Optional[torch.Tensor] = None,
    value: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    scaling: Optional[float] = None,
    layer_idx: Optional[int] = None,
    past_key_value: Optional[Cache] = None,
    blocking_config: Optional[AttentionBlockingConfig] = None,
    comp_ctx_lengths: Optional[torch.LongTensor] = None,
    batch_index: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_seen_tokens: Optional[int] = None,
    non_blocked_forward: Optional[Callable] = None,
    score_mod: Optional[Callable] = None,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
    mla_kwargs: Optional[Dict[str, Any]] = None,
    is_mla: bool = False,
    prefill_only: bool = False,
    **kwargs,
):
    strategy = _STRATEGIES[
        BlockingMode.get_final_mode(blocking_config, prefill_only=prefill_only, is_mla=is_mla, mla_kwargs=mla_kwargs)
    ]

    cache_kwargs = {"position_ids": position_ids, "batch_index": batch_index}

    if not is_mla:
        cache_kwargs["past_seen_tokens"] = past_seen_tokens
        if prefill_only:
            if sliding_window is not None:
                cache_kwargs.update({"is_sliding": True, "sliding_window": past_key_value.sliding_window_len})
            past_key_value.write_only(key, value, module.layer_idx, cache_kwargs)
        elif past_key_value is not None:
            use_kv_blocked = "kv" in blocking_config.mode and supports_blocked_kv(past_key_value)
            if blocking_config.mode == BlockingMode.KV_BATCH_FOLD:
                past_key_value.write_only_batch(key, value, module.layer_idx, cache_kwargs)
            elif use_kv_blocked and sliding_window is None:
                past_key_value.write_only(key, value, module.layer_idx, cache_kwargs)
            else:
                key, value, attention_mask, cache_kwargs = past_key_value_update(
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

    attn_output, attn_weights = strategy(
        # common
        module=module,
        query=query,
        attention_mask=attention_mask,
        scaling=scaling,
        cache_kwargs=cache_kwargs,
        layer_idx=layer_idx,
        score_mod=score_mod,
        position_bias=position_bias,
        sinks=sinks,
        # standard (decode / prefill) inputs
        key=key,
        value=value,
        past_key_value=past_key_value,
        sliding_window=sliding_window,
        # blocking dimensions
        num_kv_blocks=blocking_config.num_kv_blocks,
        num_q_blocks=blocking_config.num_q_blocks,
        head_block_size=blocking_config.head_block_size,
        num_batch_blocks=blocking_config.num_batch_blocks,
        configured_split=blocking_config.headpar_split,
        ctx_len=blocking_config.ctx_len,
        kv_block_unroll=blocking_config.kv_block_unroll,
        skip_kv=blocking_config.skip_kv or False,
        # prefill-specific
        n_rep_chunk=blocking_config.n_rep_chunk,
        # MLA-specific
        **(mla_kwargs or {}),
        **kwargs,
    )

    return attn_output, attn_weights
