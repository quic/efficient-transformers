# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache

from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep) for GQA.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _get_kv_states(module: nn.Module, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    num_kv_groups = getattr(module, "num_key_value_groups", None)
    if num_kv_groups is None:
        return key, value
    return repeat_kv(key, num_kv_groups), repeat_kv(value, num_kv_groups)


def _normalize_int(value: Optional[torch.Tensor | int]) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value) if value is not None else 0


def update_running_softmax(
    current_max: torch.Tensor,
    attn_weights_block: torch.Tensor,
    current_denominator: torch.Tensor,
    output: torch.Tensor,
    v_block: torch.Tensor,
    skip_kv: bool = False,
    skip_future: Optional(torch.Tensor) = None,
):
    # Update Running row maximum
    prev_max = current_max
    current_max_updated = torch.max(prev_max, attn_weights_block.max(dim=3).values)
    delta_max = prev_max - current_max_updated

    current_exp = torch.exp(attn_weights_block - current_max_updated.unsqueeze(-1))

    # update running denominator
    prev_denominator = current_denominator
    curr_exp_sum = torch.einsum("bhqk->bhq", current_exp)
    current_denominator_updated = prev_denominator * torch.exp(delta_max) + curr_exp_sum

    prob = current_exp / current_denominator_updated.unsqueeze(-1)

    prev_output = output
    # if updating running softmax with attention sinks, we don't have v_block
    if v_block is not None:
        output_updated = ((prev_denominator / current_denominator_updated).unsqueeze(-1)) * prev_output * torch.exp(
            delta_max.unsqueeze(-1)
        ) + torch.matmul(prob, v_block)
    else:
        output_updated = (
            ((prev_denominator / current_denominator_updated).unsqueeze(-1))
            * prev_output
            * torch.exp(delta_max.unsqueeze(-1))
        )

    if skip_kv and (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
        current_max = torch.where(skip_future, prev_max, current_max_updated)
        current_denominator = torch.where(skip_future, prev_denominator, current_denominator_updated)
        output = torch.where(skip_future.unsqueeze(-1), prev_output, output_updated)
    else:
        # Eager mode
        current_max = current_max_updated
        current_denominator = current_denominator_updated
        output = output_updated
    return current_max, current_denominator, output


def blocked_kv_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_kv_blocks: int,
    cache_kwargs: Dict[str, Any],
    layer_idx: int,
    past_key_value: Cache,
    *,
    use_causal_mask: bool = False,
    sliding_window: Optional[int] = None,
    skip_kv: bool = False,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Initialize result tensor
    output = torch.zeros_like(query)

    # Initialize Running Maximum and Denominator
    batch_size, num_heads, seq_len, _ = query.shape
    current_max = torch.full(
        (batch_size, num_heads, seq_len),
        float(MIN_MASKED_ATTENTION_VALUE),
        device=query.device,
    )
    current_denominator = torch.zeros(batch_size, num_heads, seq_len, device=query.device)

    past_seen_tokens = cache_kwargs.get("past_seen_tokens")
    if torch.onnx.is_in_onnx_export():
        attention_mask = None
        use_causal_mask = True
    position_ids = cache_kwargs.get("position_ids")
    num_kv_blocks = max(1, num_kv_blocks)
    kv_block_size = -(-past_seen_tokens // num_kv_blocks)
    if hasattr(module, "config"):
        mask_dtype = module.config.torch_dtype
    else:
        mask_dtype = value.dtype
    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=mask_dtype, device=query.device)
    current_position = position_ids.max(dim=-1).values
    # needed for GPT-OSS
    if sinks is not None:
        sinks = sinks.reshape(1, -1, 1, 1).expand(batch_size, -1, seq_len, -1)

    for j in range(num_kv_blocks):
        start_index = j * kv_block_size
        if j == num_kv_blocks - 1:
            kv_len_block = past_seen_tokens - start_index
        else:
            kv_len_block = kv_block_size
        end_index = start_index + kv_len_block

        skip_future = None
        if skip_kv:
            skip_future = (torch.tensor(start_index, device=query.device) > current_position).all()
            # Eager mode Only
            if not torch.onnx.is_in_onnx_export() and not torch.jit.is_tracing():
                if skip_future.item():
                    break

        k_block, v_block = past_key_value.read_only_blockedKV(start_index, end_index, layer_idx, cache_kwargs)
        k_block_states, v_block_states = _get_kv_states(module, k_block, v_block)

        attn_weights_block = torch.matmul(query, k_block_states.transpose(2, 3)) * scaling
        # position bias needed for mpt model
        if position_bias is not None:
            attn_weights_block = attn_weights_block + position_bias[:, :, start_index:end_index]

        mask_block = None
        if attention_mask is not None:
            mask_block = attention_mask[..., start_index:end_index]
            if mask_block.shape[-1] != attn_weights_block.shape[-1]:
                mask_block = None

        if use_causal_mask or mask_block is None:
            target_length = torch.where(
                torch.tensor(past_seen_tokens, dtype=torch.int) < torch.tensor(end_index, dtype=torch.int),
                past_seen_tokens,
                end_index,
            )
            causal_mask_block = _create_causal_mask(
                position_ids=position_ids,
                target_length=target_length,
                sliding_window=sliding_window,
                start_index=start_index,
            )
            if mask_block is None:
                mask_block = causal_mask_block
            else:
                mask_block = mask_block.to(torch.bool) | causal_mask_block

        if mask_block is not None:
            attn_weights_block = torch.where(mask_block, masked_tensor, attn_weights_block)

        current_max, current_denominator, output = update_running_softmax(
            current_max, attn_weights_block, current_denominator, output, v_block_states, skip_kv, skip_future
        )

    # If present, apply Attention Sinks, needed for GPT-OSS
    if sinks is not None:
        _, _, output = update_running_softmax(current_max, sinks, current_denominator, output, None)

    attn_output = output.transpose(1, 2).contiguous()
    attn_weights = None

    return attn_output, attn_weights


def blocked_qkv_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_kv_blocks: int,
    num_q_blocks: int,
    cache_kwargs: Dict[str, Any],
    layer_idx: int,
    past_key_value: Cache,
    *,
    use_causal_mask: bool = False,
    sliding_window: Optional[int] = None,
    skip_kv: bool = False,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Initialize Running Maximum and Denominator
    batch_size, num_heads, seq_len, DH = query.shape

    past_seen_tokens = cache_kwargs.get("past_seen_tokens")
    if torch.onnx.is_in_onnx_export():
        attention_mask = None
        use_causal_mask = True
    position_ids = cache_kwargs.get("position_ids")

    num_q_blocks = max(1, num_q_blocks)
    q_block_positions = [-(-i * seq_len) // num_q_blocks for i in range(num_q_blocks)]
    num_kv_blocks = max(1, num_kv_blocks)
    kv_block_size = -(-past_seen_tokens // num_kv_blocks)

    q_output_blocks = []
    q_attn_blocks = []
    if hasattr(module, "config"):
        mask_dtype = module.config.torch_dtype
    else:
        mask_dtype = value.dtype
    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=mask_dtype, device=query.device)
    current_position = position_ids.max(dim=-1).values
    # needed for GPT-OSS
    if sinks is not None:
        sinks = sinks.reshape(1, -1, 1, 1).expand(batch_size, -1, seq_len, -1)

    for q_block_idx in range(num_q_blocks):
        q_start = q_block_positions[q_block_idx]
        if q_block_idx == num_q_blocks - 1:
            q_len_block = seq_len - q_start
        else:
            q_len_block = q_block_positions[q_block_idx + 1] - q_start

        q_block = query[:, :, q_start : q_start + q_len_block, :]

        current_max = torch.full(
            (batch_size, num_heads, q_len_block),
            float(MIN_MASKED_ATTENTION_VALUE),
            device=query.device,
        )
        current_denominator = torch.zeros(batch_size, num_heads, q_len_block, device=query.device)
        output_blocks = torch.zeros((batch_size, num_heads, q_len_block, DH), device=query.device, dtype=query.dtype)

        for j in range(num_kv_blocks):
            start_index = j * kv_block_size
            if j == num_kv_blocks - 1:
                kv_len_block = past_seen_tokens - start_index
            else:
                kv_len_block = kv_block_size
            end_index = start_index + kv_len_block

            skip_future = None
            if skip_kv:
                skip_future = (torch.tensor(start_index, device=query.device) > current_position).all()
                # Eager mode Only
                if not torch.onnx.is_in_onnx_export() and not torch.jit.is_tracing():
                    if skip_future.item():
                        break
            k_block, v_block = past_key_value.read_only_blockedKV(start_index, end_index, layer_idx, cache_kwargs)
            k_block_states, v_block_states = _get_kv_states(module, k_block, v_block)

            attn_weights_block = torch.matmul(q_block, k_block_states.transpose(2, 3)) * scaling
            # position bias needed for mpt model
            if position_bias is not None:
                attn_weights_block = attn_weights_block + position_bias[:, :, start_index:end_index]

            mask_block = None
            if attention_mask is not None:
                mask_block = attention_mask[..., start_index:end_index]
                if mask_block.shape[-1] != attn_weights_block.shape[-1]:
                    mask_block = None

            if use_causal_mask or mask_block is None:
                # target_length = min(total_seen_tokens, end_index)
                target_length = torch.where(
                    torch.tensor(past_seen_tokens, dtype=torch.int) < torch.tensor(end_index, dtype=torch.int),
                    past_seen_tokens,
                    end_index,
                )
                causal_mask_block = _create_causal_mask(
                    position_ids=position_ids,
                    target_length=target_length,
                    sliding_window=sliding_window,
                    start_index=start_index,
                )
                if mask_block is None:
                    mask_block = causal_mask_block
                else:
                    mask_block = mask_block.to(torch.bool) | causal_mask_block

            if mask_block is not None:
                attn_mask_block = mask_block[:, :, q_start : q_start + q_len_block, :]
                attn_weights_block = torch.where(attn_mask_block, masked_tensor, attn_weights_block)

            current_max, current_denominator, output_blocks = update_running_softmax(
                current_max,
                attn_weights_block,
                current_denominator,
                output_blocks,
                v_block_states,
                skip_kv,
                skip_future,
            )

        # If present, apply Attention Sinks, needed for GPT-OSS
        if sinks is not None:
            _, _, output_blocks = update_running_softmax(current_max, sinks, current_denominator, output_blocks, None)
        q_output_blocks.append(output_blocks)
        q_attn_blocks.append(attn_weights_block)

    attn_output = torch.cat(q_output_blocks, dim=2).transpose(1, 2).contiguous()
    attn_weights = torch.cat(q_attn_blocks, dim=2)

    return attn_output, attn_weights


def blocked_hqkv_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_kv_blocks: int,
    num_q_blocks: int,
    head_block_size: int,
    cache_kwargs: Dict[str, Any],
    layer_idx: int,
    past_key_value: Cache,
    *,
    use_causal_mask: bool = False,
    sliding_window: Optional[int] = None,
    skip_kv: bool = False,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Initialize Running Maximum and Denominator
    batch_size, num_heads, seq_len, DH = query.shape

    past_seen_tokens = cache_kwargs.get("past_seen_tokens")
    if torch.onnx.is_in_onnx_export():
        attention_mask = None
        use_causal_mask = True
    position_ids = cache_kwargs.get("position_ids")
    if head_block_size <= 0:
        head_block_size = num_heads
    num_head_blocks = math.ceil(num_heads / head_block_size)
    num_q_blocks = max(1, num_q_blocks)
    q_block_positions = [-(-i * seq_len) // num_q_blocks for i in range(num_q_blocks)]
    num_kv_blocks = max(1, num_kv_blocks)
    kv_block_size = -(-past_seen_tokens // num_kv_blocks)

    h_output_blocks = []
    h_attn_blocks = []
    if hasattr(module, "config"):
        mask_dtype = module.config.torch_dtype
    else:
        mask_dtype = value.dtype
    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=mask_dtype, device=query.device)
    current_position = position_ids.max(dim=-1).values
    # needed for GPT-OSS
    if sinks is not None:
        sinks = sinks.reshape(1, -1, 1, 1).expand(batch_size, -1, seq_len, -1)

    # Process each head block independently
    for head_block_idx in range(num_head_blocks):
        h_start = head_block_idx * head_block_size
        h_end = min(h_start + head_block_size, num_heads)

        # Extract head blocks
        q_g = query[:, h_start:h_end, :, :]

        q_output_blocks = []
        q_attn_blocks = []

        for q_block_idx in range(num_q_blocks):
            q_start = q_block_positions[q_block_idx]
            if q_block_idx == num_q_blocks - 1:
                q_len_block = seq_len - q_start
            else:
                q_len_block = q_block_positions[q_block_idx + 1] - q_start

            q_block = q_g[:, :, q_start : q_start + q_len_block, :]

            current_max = torch.full(
                (batch_size, h_end - h_start, q_len_block),
                float(MIN_MASKED_ATTENTION_VALUE),
                device=query.device,
            )
            current_denominator = torch.zeros(batch_size, h_end - h_start, q_len_block, device=query.device)
            output_blocks = torch.zeros(
                (batch_size, h_end - h_start, q_len_block, DH), device=query.device, dtype=query.dtype
            )

            for j in range(num_kv_blocks):
                start_index = j * kv_block_size
                if j == num_kv_blocks - 1:
                    kv_len_block = past_seen_tokens - start_index
                else:
                    kv_len_block = kv_block_size
                end_index = start_index + kv_len_block

                skip_future = None
                if skip_kv:
                    skip_future = (torch.tensor(start_index, device=query.device) > current_position).all()
                    # Eager mode Only
                    if not torch.onnx.is_in_onnx_export() and not torch.jit.is_tracing():
                        if skip_future.item():
                            break

                k_block, v_block = past_key_value.read_only_blockedKV(start_index, end_index, layer_idx, cache_kwargs)
                k_block_states, v_block_states = _get_kv_states(module, k_block, v_block)

                k_g = k_block_states[:, h_start:h_end, :, :]
                v_g = v_block_states[:, h_start:h_end, :, :]

                attn_weights_block = torch.matmul(q_block, k_g.transpose(2, 3)) * scaling
                # position bias needed for mpt model
                if position_bias is not None:
                    attn_weights_block = attn_weights_block + position_bias[h_start:h_end, :, start_index:end_index]

                mask_block = None
                if attention_mask is not None:
                    mask_block = attention_mask[..., start_index:end_index]
                    if mask_block.shape[-1] != attn_weights_block.shape[-1]:
                        mask_block = None

                if use_causal_mask or mask_block is None:
                    # target_length = min(total_seen_tokens, end_index)
                    target_length = torch.where(
                        torch.tensor(past_seen_tokens, dtype=torch.int) < torch.tensor(end_index, dtype=torch.int),
                        past_seen_tokens,
                        end_index,
                    )
                    causal_mask_block = _create_causal_mask(
                        position_ids=position_ids,
                        target_length=target_length,
                        sliding_window=sliding_window,
                        start_index=start_index,
                    )
                    if mask_block is None:
                        mask_block = causal_mask_block
                    else:
                        mask_block = mask_block.to(torch.bool) | causal_mask_block

                if mask_block is not None:
                    mask_block_g = mask_block[:, :, q_start : q_start + q_len_block, :]
                    attn_weights_block = torch.where(mask_block_g, masked_tensor, attn_weights_block)

                current_max, current_denominator, output_blocks = update_running_softmax(
                    current_max, attn_weights_block, current_denominator, output_blocks, v_g, skip_kv, skip_future
                )
            # If present, apply Attention Sinks, needed for GPT-OSS
            if sinks is not None:
                _, _, output_blocks = update_running_softmax(
                    current_max, sinks, current_denominator, output_blocks, None
                )
            q_output_blocks.append(output_blocks)
            q_attn_blocks.append(attn_weights_block)

        head_output = torch.cat(q_output_blocks, dim=2)
        head_attn_weights = torch.cat(q_attn_blocks, dim=2)
        h_output_blocks.append(head_output)
        h_attn_blocks.append(head_attn_weights)

    attn_output = torch.cat(h_output_blocks, dim=1).transpose(1, 2).contiguous()
    attn_weights = torch.cat(h_attn_blocks, dim=1)

    return attn_output, attn_weights


def blocked_bhqkv_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_kv_blocks: int,
    num_q_blocks: int,
    num_batch_blocks: int,
    head_block_size: int,
    cache_kwargs: Dict[str, Any],
    layer_idx: int,
    past_key_value: Cache,
    *,
    score_mod: Optional[Callable[[torch.Tensor, int, int], torch.Tensor]] = None,
    use_causal_mask: bool = False,
    sliding_window: Optional[int] = None,
    skip_kv: bool = False,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Initialize Running Maximum and Denominator
    batch_size, num_heads, seq_len, DH = query.shape

    past_seen_tokens = cache_kwargs.get("past_seen_tokens")
    if torch.onnx.is_in_onnx_export():
        attention_mask = None
        use_causal_mask = True
    position_ids = cache_kwargs.get("position_ids")
    if head_block_size <= 0:
        head_block_size = num_heads
    num_head_blocks = math.ceil(num_heads / head_block_size)
    num_q_blocks = max(1, _normalize_int(num_q_blocks))
    q_block_positions = [-(-i * seq_len) // num_q_blocks for i in range(num_q_blocks)]
    num_kv_blocks = max(1, num_kv_blocks)
    kv_block_size = -(-past_seen_tokens // num_kv_blocks)

    h_output_blocks = []
    h_attn_blocks = []

    num_batch_blocks = max(
        1, min(batch_size, _normalize_int(num_batch_blocks))
    )  # default to batch size for number of batch blocks
    batch_block_positions = [(i * batch_size) // num_batch_blocks for i in range(num_batch_blocks)]

    if hasattr(module, "config"):
        mask_dtype = module.config.torch_dtype
    else:
        mask_dtype = value.dtype
    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=mask_dtype, device=query.device)

    current_position = position_ids.max(dim=-1).values
    # needed for GPT-OSS
    if sinks is not None:
        sinks = sinks.reshape(1, -1, 1, 1).expand(batch_size, -1, seq_len, -1)

    # Process each head block independently
    for head_block_idx in range(num_head_blocks):
        h_start = head_block_idx * head_block_size
        h_end = min(h_start + head_block_size, num_heads)

        # Extract head blocks
        q_g = query[:, h_start:h_end, :, :]

        q_output_blocks = []
        q_attn_blocks = []

        for q_block_idx in range(num_q_blocks):
            q_start = q_block_positions[q_block_idx]
            if q_block_idx == num_q_blocks - 1:
                q_len_block = seq_len - q_start
            else:
                q_len_block = q_block_positions[q_block_idx + 1] - q_start

            q_block_head = q_g[:, :, q_start : q_start + q_len_block, :]

            batch_output_blocks = []
            batch_attn_blocks = []

            for b_block_idx in range(num_batch_blocks):
                batch_start = batch_block_positions[b_block_idx]
                if b_block_idx == num_batch_blocks - 1:
                    batch_len = batch_size - batch_start
                else:
                    batch_len = batch_block_positions[b_block_idx + 1] - batch_start

                q_block = q_block_head[batch_start : batch_start + batch_len, :, :, :]

                current_max = torch.full(
                    (batch_len, h_end - h_start, q_len_block),
                    float(MIN_MASKED_ATTENTION_VALUE),
                    device=query.device,
                )
                current_denominator = torch.zeros(batch_len, h_end - h_start, q_len_block, device=query.device)
                output_blocks = torch.zeros(
                    (batch_len, h_end - h_start, q_len_block, DH), device=query.device, dtype=query.dtype
                )

                for j in range(num_kv_blocks):
                    start_index = j * kv_block_size
                    if j == num_kv_blocks - 1:
                        kv_len_block = past_seen_tokens - start_index
                    else:
                        kv_len_block = kv_block_size
                    end_index = start_index + kv_len_block

                    skip_future = None
                    if skip_kv:
                        skip_future = (torch.tensor(start_index, device=query.device) > current_position).all()
                        # Eager mode Only
                        if not torch.onnx.is_in_onnx_export() and not torch.jit.is_tracing():
                            if skip_future.item():
                                break

                    k_block, v_block = past_key_value.read_only_blockedKV(
                        start_index, end_index, layer_idx, cache_kwargs
                    )
                    k_block_states, v_block_states = _get_kv_states(module, k_block, v_block)

                    k_g = k_block_states[batch_start : batch_start + batch_len, h_start:h_end, :, :]
                    v_g = v_block_states[batch_start : batch_start + batch_len, h_start:h_end, :, :]

                    attn_weights_block = torch.matmul(q_block, k_g.transpose(2, 3)) * scaling
                    # position bias needed for mpt model
                    if position_bias is not None:
                        attn_weights_block = attn_weights_block + position_bias[h_start:h_end, :, start_index:end_index]

                    mask_block = None
                    if attention_mask is not None:
                        mask_block = attention_mask[..., start_index:end_index]
                        if mask_block.shape[-1] != attn_weights_block.shape[-1]:
                            mask_block = None

                    if use_causal_mask or mask_block is None:
                        # target_length = min(total_seen_tokens, end_index)
                        target_length = torch.where(
                            torch.tensor(past_seen_tokens, dtype=torch.int) < torch.tensor(end_index, dtype=torch.int),
                            past_seen_tokens,
                            end_index,
                        )
                        causal_mask_block = _create_causal_mask(
                            position_ids=position_ids,
                            target_length=target_length,
                            sliding_window=sliding_window,
                            start_index=start_index,
                        )
                        if mask_block is None:
                            mask_block = causal_mask_block
                        else:
                            mask_block = mask_block.to(torch.bool) | causal_mask_block

                    if mask_block is not None:
                        mask_block_g = mask_block[
                            batch_start : batch_start + batch_len, :, q_start : q_start + q_len_block, :
                        ]
                        attn_weights_block = torch.where(mask_block_g, masked_tensor, attn_weights_block)

                    current_max, current_denominator, output_blocks = update_running_softmax(
                        current_max, attn_weights_block, current_denominator, output_blocks, v_g, skip_kv, skip_future
                    )
                batch_output_blocks.append(output_blocks)
                batch_attn_blocks.append(attn_weights_block)
            # If present, apply Attention Sinks, needed for GPT-OSS
            if sinks is not None:
                _, _, batch_output_blocks = update_running_softmax(
                    current_max, sinks, current_denominator, batch_output_blocks, None
                )
            q_output_blocks.append(torch.cat(batch_output_blocks, dim=0))
            q_attn_blocks.append(torch.cat(batch_attn_blocks, dim=0))

        head_output = torch.cat(q_output_blocks, dim=2)
        head_attn_weights = torch.cat(q_attn_blocks, dim=2)
        h_output_blocks.append(head_output)
        h_attn_blocks.append(head_attn_weights)

    attn_output = torch.cat(h_output_blocks, dim=1).transpose(1, 2).contiguous()
    attn_weights = torch.cat(h_attn_blocks, dim=1)

    return attn_output, attn_weights


def blocked_h_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    head_block_size: int,
    *,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    H-blocked attention that slices along head dimension to create blocks and processes each block.
    """
    batch_size, num_heads, q_len, _ = query.shape
    if head_block_size <= 0:
        head_block_size = num_heads
    num_head_blocks = math.ceil(num_heads / head_block_size)

    key_states, value_states = _get_kv_states(module, key, value)

    h_output_blocks = []
    h_attn_blocks = []

    if hasattr(module, "config"):
        mask_dtype = module.config.torch_dtype
    else:
        mask_dtype = value.dtype
    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=mask_dtype, device=query.device)

    # Process each head block independently
    for head_block_idx in range(num_head_blocks):
        h_start = head_block_idx * head_block_size
        h_end = min(h_start + head_block_size, num_heads)

        # Extract head blocks
        q_g = query[:, h_start:h_end, :, :]
        k_g = key_states[:, h_start:h_end, :, :]
        v_g = value_states[:, h_start:h_end, :, :]

        attn_weights = torch.matmul(q_g, k_g.transpose(2, 3)) * scaling

        # position bias needed for mpt
        if position_bias is not None:
            attn_weights = attn_weights + position_bias[h_start:h_end, :, :]
        if attention_mask is not None:
            attn_weights = torch.where(attention_mask, masked_tensor, attn_weights)
        # attention sinks needed for gpt-oss
        if sinks is not None:
            sinks_g = (
                module.sinks[h_start:h_end]
                .reshape(1, -1, 1, 1)
                .expand(attn_weights.shape[0], -1, attn_weights.shape[2], -1)
            )
            combined_logits = torch.cat([attn_weights, sinks_g], dim=-1)
            attn_weights = combined_logits - combined_logits.max(dim=-1, keepdim=True).values

        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        if sinks is not None:
            attn_weights = attn_weights[..., :-1]
        output_block = torch.matmul(attn_weights, v_g)

        h_output_blocks.append(output_block)
        h_attn_blocks.append(attn_weights)

    attn_output = torch.cat(h_output_blocks, dim=1).transpose(1, 2).contiguous()
    attn_weights = torch.cat(h_attn_blocks, dim=1)

    return attn_output, attn_weights


def blocked_q_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_q_blocks: int,
    *,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Q-blocked attention that slices the query sequence into blocks and processes each block.
    """
    batch_size, num_heads, q_len, _ = query.shape
    num_q_blocks = max(1, _normalize_int(num_q_blocks))
    key_states, value_states = _get_kv_states(module, key, value)

    q_block_positions = [-(-i * q_len) // num_q_blocks for i in range(num_q_blocks)]
    q_output_blocks = []
    q_attn_blocks = []

    if hasattr(module, "config"):
        mask_dtype = module.config.torch_dtype
    else:
        mask_dtype = value.dtype
    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=mask_dtype, device=query.device)

    for q_block_idx in range(num_q_blocks):
        q_start = q_block_positions[q_block_idx]
        if q_block_idx == num_q_blocks - 1:
            q_len_block = q_len - q_start
        else:
            q_len_block = q_block_positions[q_block_idx + 1] - q_start

        q_block = query[:, :, q_start : q_start + q_len_block, :]
        attn_mask_block = None
        if attention_mask is not None:
            attn_mask_block = attention_mask[:, :, q_start : q_start + q_len_block, :]

        attn_weights = torch.matmul(q_block, key_states.transpose(2, 3)) * scaling
        # position bias needed for mpt model
        if position_bias is not None:
            attn_weights = attn_weights + position_bias
        if attn_mask_block is not None:
            attn_weights = torch.where(attn_mask_block, masked_tensor, attn_weights)
        # attention sinks needed for gpt-oss
        if sinks is not None:
            sinks_g = sinks.reshape(1, -1, 1, 1).expand(batch_size, -1, q_len_block, -1)
            combined_logits = torch.cat([attn_weights, sinks_g], dim=3)
            attn_weights = combined_logits - combined_logits.max(dim=3, keepdim=True).values

        attn_weights = torch.softmax(attn_weights, dim=3, dtype=torch.float32).to(query.dtype)
        if sinks is not None:
            attn_weights = attn_weights[..., : key.shape[2]]
        output_block = torch.matmul(attn_weights, value_states)

        q_output_blocks.append(output_block)
        q_attn_blocks.append(attn_weights)

    attn_output = torch.cat(q_output_blocks, dim=2).transpose(1, 2).contiguous()
    attn_weights = torch.cat(q_attn_blocks, dim=2)

    return attn_output, attn_weights


def blocked_kv_mla_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    per_head_k_up_normal: torch.Tensor,
    per_head_v_up: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_kv_blocks: int,
    cache_kwargs: Dict[str, Any],
    layer_idx: int,
    compressed_kvs: Optional[torch.Tensor],
    mla_absorption: Dict[str, Any],
    *,
    use_causal_mask: bool = False,
    sliding_window: Optional[int] = None,
    skip_kv: bool = False,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Initialize result tensor
    batch_size, num_heads, seq_len, _ = query.shape
    output = torch.zeros(
        batch_size, num_heads, seq_len, module.config.kv_lora_rank, device=query.device, dtype=query.dtype
    )

    if hasattr(module, "config"):
        mask_dtype = module.config.torch_dtype
    else:
        mask_dtype = query.dtype
    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=mask_dtype, device=query.device)

    # Initialize Running Maximum and Denominator
    current_max = torch.full(
        (batch_size, num_heads, seq_len),
        float(MIN_MASKED_ATTENTION_VALUE),
        device=query.device,
        dtype=query.dtype,
    )
    skip_kv = True
    current_denominator = torch.zeros(batch_size, num_heads, seq_len, device=query.device, dtype=query.dtype)
    ctx_len = compressed_kvs.layers[layer_idx].ckv.shape[2]
    kv_block_size = -(-ctx_len // num_kv_blocks)

    position_ids = cache_kwargs.get("position_ids")
    current_position = position_ids.max(dim=-1).values

    for j in range(num_kv_blocks):
        start_index = j * kv_block_size
        if j == num_kv_blocks - 1:
            kv_len_block = ctx_len - start_index
        else:
            kv_len_block = kv_block_size
        end_index = start_index + kv_len_block

        skip_future = None
        if skip_kv:
            skip_future = (torch.tensor(start_index, device=query.device) > current_position).all()
            # Eager mode Only
            if not torch.onnx.is_in_onnx_export() and not torch.jit.is_tracing():
                if skip_future.item():
                    break

        compressed_kv_block = compressed_kvs.read_only_blocked_ckv(start_index, end_index, layer_idx, cache_kwargs)
        k_pe_block = compressed_kvs.read_only_blocked_k_pe(start_index, end_index, layer_idx, cache_kwargs)

        causal_mask_block = _create_causal_mask(
            position_ids=position_ids,
            target_length=end_index,
            start_index=start_index,
        )

        if mla_absorption is not None:
            absorption = mla_absorption.get("absorption", False)
        else:
            absorption = False

        k_heads, q_heads = compressed_kv_block.shape[1], query.shape[1]

        if k_heads > 1:
            num_heads_to_repeat = math.ceil(q_heads / k_heads)
            compressed_kv_block = (
                compressed_kv_block.unsqueeze(2)
                .expand(-1, -1, num_heads_to_repeat, -1, -1)
                .reshape(batch_size, num_heads_to_repeat * k_heads, -1, module.config.kv_lora_rank)
            )
            compressed_kv_block = compressed_kv_block[:, :q_heads, :, :]

            k_pe_block = (
                k_pe_block.unsqueeze(2)
                .expand(-1, -1, num_heads_to_repeat, -1, -1)
                .reshape(batch_size, num_heads_to_repeat * k_heads, -1, module.config.qk_rope_head_dim)
            )
            k_pe_block = k_pe_block[:, :q_heads, :, :]

        if absorption:
            krope_nope = torch.cat((compressed_kv_block, k_pe_block), dim=-1)
            attn_weights_block = torch.matmul(query, krope_nope.transpose(2, 3)) * scaling
            # [1, 64, q_len, 576] X [1, 1, 576, kv_block_size] -> [1, 64, q_len, kv_block_size]
            attn_weights_block = torch.where(causal_mask_block, masked_tensor, attn_weights_block)
            current_max, current_denominator, output = update_running_softmax(
                current_max,
                attn_weights_block,
                current_denominator,
                output,
                compressed_kv_block,
                skip_kv,
                skip_future,
            )  # [1, 64, q_len, kv_block_size] X [1, 1, kv_block_size, 512] -> [1, 64, q_len, 512]
        else:
            knope = torch.matmul(compressed_kv_block, per_head_k_up_normal)
            if k_heads == 1:
                k_pe_block = (
                    k_pe_block.unsqueeze(1)
                    .expand(-1, num_heads, -1, -1, -1)
                    .reshape(batch_size, num_heads, -1, module.config.qk_rope_head_dim)
                )
            krope_nope = torch.cat((knope, k_pe_block), dim=-1)
            attn_weights_block = torch.matmul(query, krope_nope.transpose(2, 3)) * scaling
            attn_weights_block = torch.where(causal_mask_block, masked_tensor, attn_weights_block)
            current_max, current_denominator, output = update_running_softmax(
                current_max,
                attn_weights_block,
                current_denominator,
                output,
                compressed_kv_block,
                skip_kv,
                skip_future,
            )

    attn_output = torch.matmul(output, per_head_v_up)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_weights = None

    return attn_output, attn_weights


def blocked_kv_par_prefill_only_online_mla_attention_forward(
    module: nn.Module,
    query: torch.Tensor,  # [B, NQH, QL, D_abs]  absorption-space Q
    per_head_v_up: torch.Tensor,  # [1, NQH, kv_lora_rank, v_head_dim]
    per_head_k_up_normal: torch.Tensor,  # [1, NQH, qk_nope_head_dim, kv_lora_rank] — for non-absorption K
    mla_absorption: bool,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_kv_blocks: int,
    par_num_split: int,  # T-dim split within each KV block (maps to NSP cores)
    cache_kwargs: Dict[str, Any],
    layer_idx: int,
    compressed_kvs,
    blocking_config,
    n_rep_chunk: int = 16,
    ql_chunk: int = 128,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    position_ids = cache_kwargs.get("position_ids")
    B, NQH, QL, D_abs = query.shape
    kv_lora_rank = module.config.kv_lora_rank
    split = par_num_split

    if mla_absorption.get("absorption", False):
        Hkv = getattr(module, "num_key_value_heads", 1)
        n_rep = NQH // Hkv
    else:
        Hkv = NQH
        n_rep = 1

    # q_fold without expand — expand only per-chunk inside loop
    q_fold = query.reshape(B, Hkv, n_rep, QL, D_abs)

    ctx_len = compressed_kvs.layers[layer_idx].ckv.shape[2]
    kv_block_size = -(-ctx_len // num_kv_blocks)
    T_h_nom = -(-kv_block_size // split)

    # kv_offsets 5D: [1, 1, split, 1, T_h_nom]
    kv_offsets = (
        torch.arange(split, device=query.device)[:, None] * T_h_nom
        + torch.arange(T_h_nom, device=query.device)[None, :]
    ).view(1, 1, split, 1, T_h_nom)

    current_position = position_ids.max(dim=-1).values
    # current_min_pos = position_ids.min()
    skip_kv = getattr(blocking_config, "skip_kv", True)

    # Running accumulators: [B, Hkv, split, n_rep, QL]
    m_acc = torch.full(
        (B, Hkv, split, n_rep, QL), float(MIN_MASKED_ATTENTION_VALUE), device=query.device, dtype=query.dtype
    )
    s_acc = torch.zeros(B, Hkv, split, n_rep, QL, device=query.device, dtype=query.dtype)
    o_acc = torch.zeros(B, Hkv, split, n_rep, QL, kv_lora_rank, device=query.device, dtype=query.dtype)

    for j in range(num_kv_blocks):
        start_index = j * kv_block_size
        kv_len_block = ctx_len - start_index if j == num_kv_blocks - 1 else kv_block_size
        end_index = start_index + kv_len_block
        T_orig = kv_len_block

        skip_future = None
        if skip_kv:
            skip_future = (torch.tensor(start_index, device=query.device) > current_position).all()
            if not torch.onnx.is_in_onnx_export() and not torch.jit.is_tracing():
                if skip_future.item():
                    break

        ckv_block = compressed_kvs.read_only_blocked_ckv(start_index, end_index, layer_idx, cache_kwargs)
        k_pe_block = compressed_kvs.read_only_blocked_k_pe(start_index, end_index, layer_idx, cache_kwargs)

        if mla_absorption.get("absorption", False):
            k_block = torch.cat((ckv_block, k_pe_block), dim=-1)
            ckv_for_v = ckv_block
        else:
            orig_Hkv = getattr(module, "num_key_value_heads", 1)
            n_rep_kv = NQH // orig_Hkv
            ckv_nqh = (
                ckv_block.unsqueeze(2).expand(-1, orig_Hkv, n_rep_kv, -1, -1).reshape(B, NQH, T_orig, kv_lora_rank)
            )
            k_pe_nqh = (
                k_pe_block.unsqueeze(2)
                .expand(-1, orig_Hkv, n_rep_kv, -1, -1)
                .reshape(B, NQH, T_orig, module.config.qk_rope_head_dim)
            )
            k_nope = torch.matmul(ckv_nqh, per_head_k_up_normal)
            k_block = torch.cat((k_nope, k_pe_nqh), dim=-1)
            ckv_for_v = ckv_nqh

        T_blk = T_orig
        pad = 0
        if T_blk % split != 0:
            pad = split - (T_blk % split)
            k_block = F.pad(k_block, (0, 0, 0, pad))
            ckv_for_v = F.pad(ckv_for_v, (0, 0, 0, pad))
            T_blk += pad
        T_h = T_blk // split

        K_5d = k_block.view(B, Hkv, split, T_h, D_abs)
        V_5d = ckv_for_v.view(B, Hkv, split, T_h, kv_lora_rank)

        off = kv_offsets if T_h == T_h_nom else kv_offsets[:, :, :, :, :T_h]

        ql_max: list = []
        ql_sum: list = []
        ql_out: list = []

        for t_start in range(0, QL, ql_chunk):
            t_end = min(t_start + ql_chunk, QL)
            tc = t_end - t_start

            rep_max: list = []
            rep_sum: list = []
            rep_out: list = []

            for r_start in range(0, n_rep, n_rep_chunk):
                r_end = min(r_start + n_rep_chunk, n_rep)
                rc = r_end - r_start

                # expand only (r_chunk × t_chunk) portion of Q
                Q_sub = q_fold[:, :, r_start:r_end, t_start:t_end, :].unsqueeze(2).expand(B, Hkv, split, rc, tc, D_abs)
                attn_c = torch.matmul(Q_sub, K_5d.unsqueeze(3).transpose(-1, -2)) * scaling

                if pad > 0:
                    chunk_start = torch.arange(split, device=attn_c.device) * T_h
                    valid_in_chunk = T_orig - chunk_start
                    k_idx = torch.arange(T_h, device=attn_c.device)
                    pad_mask = k_idx.unsqueeze(0) >= valid_in_chunk.unsqueeze(1)
                    attn_c = attn_c.masked_fill(pad_mask.view(1, 1, split, 1, 1, T_h), -3.0e4)

                # causal mask uses position_ids for this t_chunk
                pos_sub = position_ids[:, t_start:t_end]
                causal_mask_c = off.unsqueeze(3) > (pos_sub - start_index)[:, None, None, None, :, None]
                attn_c = attn_c.masked_fill(causal_mask_c, -3.0e4)

                m_c = attn_c.max(dim=-1).values
                exp_c = torch.exp(attn_c - m_c.unsqueeze(-1))

                if skip_kv and (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
                    m_c = torch.where(skip_future, torch.full_like(m_c, -3.0e4), m_c)
                    exp_c = torch.where(skip_future, torch.zeros_like(exp_c), exp_c)

                sum_c = exp_c.sum(dim=-1)
                out_c = torch.matmul(exp_c, V_5d.unsqueeze(3))

                if skip_kv and (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
                    sum_c = torch.where(skip_future, torch.zeros_like(sum_c), sum_c)
                    out_c = torch.where(skip_future, torch.zeros_like(out_c), out_c)

                rep_max.append(m_c)
                rep_sum.append(sum_c)
                rep_out.append(out_c)

            # cat along n_rep dim (dim=3) for this t_chunk
            ql_max.append(torch.cat(rep_max, dim=3))
            ql_sum.append(torch.cat(rep_sum, dim=3))
            ql_out.append(torch.cat(rep_out, dim=3))

        m_blk = torch.cat(ql_max, dim=4)
        sum_blk = torch.cat(ql_sum, dim=4)
        out_blk = torch.cat(ql_out, dim=4)

        # Online merge (o_acc unnormalized)
        new_m = torch.max(m_acc, m_blk)
        delta = m_acc - new_m
        new_s = s_acc * torch.exp(delta) + sum_blk
        new_o = torch.exp(delta).unsqueeze(-1) * o_acc + out_blk

        if skip_kv and (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
            m_acc = torch.where(skip_future, m_acc, new_m)
            s_acc = torch.where(skip_future, s_acc, new_s)
            o_acc = torch.where(skip_future.unsqueeze(-1), o_acc, new_o)
        else:
            m_acc, s_acc, o_acc = new_m, new_s, new_o

    # ── Merge across splits (Stage 2 only) ───────────────────────────────────
    m2 = m_acc.max(dim=2).values
    w2 = torch.exp(m_acc - m2.unsqueeze(2))
    s2 = (w2 * s_acc).sum(dim=2)
    o2 = (w2.unsqueeze(-1) * o_acc).sum(dim=2)  # o_acc unnormalized → no s_acc factor
    output = o2 / s2.unsqueeze(-1)  # single division at the end

    # [B, Hkv, n_rep, QL, kv_lora_rank] → [B, NQH, QL, kv_lora_rank]
    output = output.reshape(B, NQH, QL, kv_lora_rank)
    attn_output = torch.matmul(output, per_head_v_up)  # [B, NQH, QL, v_head_dim]
    attn_output = attn_output.transpose(1, 2).contiguous()  # [B, QL, NQH, v_head_dim]

    return attn_output, None


def blocked_kv_par_mla_attention_forward(
    module: nn.Module,
    query: torch.Tensor,  # [B, NQH, QL, D_abs]  absorption-space Q
    per_head_v_up: torch.Tensor,  # [1, NQH, kv_lora_rank, v_head_dim]
    per_head_k_up_normal: torch.Tensor,  # [1, NQH, qk_nope_head_dim, kv_lora_rank] — for non-absorption K
    mla_absorption: bool,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_kv_blocks: int,
    par_num_split: int,  # T-dim split within each KV block (maps to NSP cores)
    cache_kwargs: Dict[str, Any],
    layer_idx: int,
    compressed_kvs,
    blocking_config,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    GQA headpar-style MLA attention.
    Layout matches qwen3_gqa_kv_blocking_microbench.py:
      q_fold = query.reshape(B, Hkv, QL*n_rep, D)            # simple reshape, no permute
      Q_5d   = q_fold.unsqueeze(2).expand(..., split, ...)   # broadcast over split
      K_5d   = k_block.view(B, Hkv, split, T_h, D)          # consecutive T split
    Merge is two-stage offline (buffer all blocks):
      Stage 1: max/exp/sum across KV blocks
      Stage 2: max/exp/sum across splits
    """
    position_ids = cache_kwargs.get("position_ids")
    B, NQH, QL, D_abs = query.shape
    kv_lora_rank = module.config.kv_lora_rank
    split = par_num_split

    # absorption=True : all n_rep heads in a group share the same K (= ckv||k_pe)
    #                   → fold: Hkv = module.num_key_value_heads, n_rep = NQH // Hkv
    # absorption=False: each query head has its own K = ckv @ k_up_h
    #                   → cannot fold across heads; treat as Hkv=NQH, n_rep=1
    print("using super fast attn")
    if mla_absorption.get("absorption", False):
        Hkv = getattr(module, "num_key_value_heads", 1)
        n_rep = NQH // Hkv
    else:
        Hkv = NQH
        n_rep = 1
    # ── Q fold: reshape + unsqueeze + expand (GQA style) ─────────────────────
    rem_heads = NQH % Hkv
    old_NQH = None
    if rem_heads != 0:
        n_rep = n_rep + 1
        old_NQH = NQH
        ideal_heads = n_rep * Hkv - NQH
        pad_zeros = torch.zeros(B, ideal_heads, QL, D_abs, dtype=query.dtype)
        query = torch.cat([query, pad_zeros], dim=1)
        NQH = n_rep * Hkv
    q_fold = query.reshape(B, Hkv, QL * n_rep, D_abs)
    Q_5d = q_fold.unsqueeze(2).expand(B, Hkv, split, QL * n_rep, D_abs)

    ctx_len = compressed_kvs.layers[layer_idx].ckv.shape[2]
    kv_block_size = -(-ctx_len // num_kv_blocks)
    T_h_nom = -(-kv_block_size // split)  # ceiling — nominal T per split chunk

    # kv_offsets: consecutive layout, offset of position within block
    # offsets[s, t] = s*T_h_nom + t
    kv_offsets = (
        torch.arange(split, device=query.device)[:, None] * T_h_nom
        + torch.arange(T_h_nom, device=query.device)[None, :]
    ).view(1, 1, split, 1, T_h_nom)  # [1, 1, split, 1, T_h_nom]

    current_position = position_ids.max(dim=-1).values
    # current_min_pos = position_ids.min()
    skip_kv = getattr(blocking_config, "skip_kv", True)

    max_buf: list = []
    sum_buf: list = []
    out_buf: list = []

    for j in range(num_kv_blocks):
        start_index = j * kv_block_size
        kv_len_block = ctx_len - start_index if j == num_kv_blocks - 1 else kv_block_size
        end_index = start_index + kv_len_block
        T_orig = kv_len_block

        skip_future = None
        if skip_kv:
            skip_future = (torch.tensor(start_index, device=query.device) > current_position).all()
            if not torch.onnx.is_in_onnx_export() and not torch.jit.is_tracing():
                if skip_future.item():
                    break

        # Read KV block: [B, Hkv, T_orig, kv_lora_rank/qk_rope_head_dim]
        ckv_block = compressed_kvs.read_only_blocked_ckv(start_index, end_index, layer_idx, cache_kwargs)
        k_pe_block = compressed_kvs.read_only_blocked_k_pe(start_index, end_index, layer_idx, cache_kwargs)

        # K in absorption or non-absorption space: [B, Hkv, T_orig, D_abs]
        if mla_absorption.get("absorption", False):
            k_block = torch.cat((ckv_block, k_pe_block), dim=-1)  # [B, Hkv, T, 576]
            ckv_for_v = ckv_block  # [B, Hkv, T, 512]
        else:
            # Each query head needs its own K: expand ckv to NQH=Hkv, apply per-head k_up
            # ckv_block:  [B, orig_Hkv, T, kv_lora_rank]
            orig_Hkv = getattr(module, "num_key_value_heads", 1)
            n_rep_kv = NQH // orig_Hkv
            ckv_nqh = (
                ckv_block.unsqueeze(2).expand(-1, orig_Hkv, n_rep_kv, -1, -1).reshape(B, NQH, T_orig, kv_lora_rank)
            )  # [B, NQH, T, 512]
            k_pe_nqh = (
                k_pe_block.unsqueeze(2)
                .expand(-1, orig_Hkv, n_rep_kv, -1, -1)
                .reshape(B, NQH, T_orig, module.config.qk_rope_head_dim)
            )
            # per_head_k_up_normal: [1, NQH, kv_lora_rank, qk_nope_head_dim]
            k_nope = torch.matmul(ckv_nqh, per_head_k_up_normal)  # [B, NQH, T, 128]
            k_block = torch.cat((k_nope, k_pe_nqh), dim=-1)  # [B, NQH, T, 192]
            ckv_for_v = ckv_nqh  # [B, NQH, T, 512]

        # Pad T to multiple of split
        T_blk = T_orig
        pad = 0
        if T_blk % split != 0:
            pad = split - (T_blk % split)
            k_block = F.pad(k_block, (0, 0, 0, pad))
            ckv_for_v = F.pad(ckv_for_v, (0, 0, 0, pad))
            T_blk += pad
        T_h = T_blk // split

        # 5D K/V: [B, Hkv, split, T_h, D]
        K_5d = k_block.view(B, Hkv, split, T_h, D_abs)
        V_5d = ckv_for_v.view(B, Hkv, split, T_h, kv_lora_rank)

        # Attention scores: [B, Hkv, split, QL*n_rep, T_h]
        attn = torch.matmul(Q_5d, K_5d.transpose(-1, -2)) * scaling

        # Padding mask
        if pad > 0:
            chunk_start = torch.arange(split, device=attn.device) * T_h
            valid_in_chunk = T_orig - chunk_start
            k_idx = torch.arange(T_h, device=attn.device)
            pad_mask = k_idx.unsqueeze(0) >= valid_in_chunk.unsqueeze(1)  # [split, T_h]
            attn = attn.masked_fill(pad_mask.view(1, 1, split, 1, T_h), -3.0e4)

        # Causal mask: offsets within block vs query position
        off = kv_offsets if T_h == T_h_nom else kv_offsets[:, :, :, :, :T_h]
        causal_mask = off > (position_ids - start_index)[:, None, None, :, None]
        causal_mask = causal_mask.repeat(1, 1, 1, n_rep, 1)
        attn = attn.masked_fill(causal_mask, -3.0e4)

        m_blk = attn.max(dim=-1).values  # [B, Hkv, split, QL*n_rep]
        exp_blk = torch.exp(attn - m_blk.unsqueeze(-1))

        if skip_kv and (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
            m_blk = torch.where(skip_future, torch.full_like(m_blk, float(MIN_MASKED_ATTENTION_VALUE)), m_blk)
            exp_blk = torch.where(skip_future, torch.zeros_like(exp_blk), exp_blk)

        sum_blk = exp_blk.sum(dim=-1)  # [B, Hkv, split, QL*n_rep]
        out_blk = torch.matmul(exp_blk, V_5d)  # [B, Hkv, split, QL*n_rep, kv_lora_rank]

        if skip_kv and (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
            sum_blk = torch.where(skip_future, torch.zeros_like(sum_blk), sum_blk)
            out_blk = torch.where(skip_future, torch.zeros_like(out_blk), out_blk)

        max_buf.append(m_blk)
        sum_buf.append(sum_blk)
        out_buf.append(out_blk)

    # ── Stage 1: merge across KV blocks ──────────────────────────────────────
    max_stk = torch.stack(max_buf)  # [nkvb, B, Hkv, split, QL*n_rep]
    sum_stk = torch.stack(sum_buf)
    out_stk = torch.stack(out_buf)  # [nkvb, B, Hkv, split, QL*n_rep, kv_lora_rank]
    m1 = max_stk.max(dim=0).values
    w1 = torch.exp(max_stk - m1.unsqueeze(0))
    s1 = (w1 * sum_stk).sum(dim=0)  # [B, Hkv, split, QL*n_rep]
    o1 = (w1.unsqueeze(-1) * out_stk).sum(dim=0)  # [B, Hkv, split, QL*n_rep, kv_lora_rank]

    # ── Stage 2: merge across splits ─────────────────────────────────────────
    m2 = m1.max(dim=2).values  # [B, Hkv, QL*n_rep]
    w2 = torch.exp(m1 - m2.unsqueeze(2))
    s2 = (w2 * s1).sum(dim=2)
    o2 = (w2.unsqueeze(-1) * o1).sum(dim=2)  # [B, Hkv, QL*n_rep, kv_lora_rank]
    output = o2 / s2.unsqueeze(-1)

    # ── Unfold + v_up (GQA style) ─────────────────────────────────────────────
    # [B, Hkv, QL*n_rep, kv_lora_rank] → [B, NQH, QL, kv_lora_rank]
    output = output.view(B, Hkv, n_rep, QL, kv_lora_rank)
    output = output.view(B, Hkv, n_rep, QL, kv_lora_rank).reshape(B, NQH, QL, kv_lora_rank)
    if rem_heads != 0:
        output = output[:, :old_NQH, :, :]
    attn_output = torch.matmul(output, per_head_v_up)  # [B, NQH, QL, v_head_dim]
    attn_output = attn_output.transpose(1, 2).contiguous()  # [B, QL, NQH, v_head_dim]

    return attn_output, None


def blocked_h_mla_attention_forward(
    module: nn.Module,
    q_a_proj_out: torch.Tensor,
    fusedqk: torch.Tensor,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kva: torch.Tensor,
    k_pe: torch.Tensor,
    per_head_q_up: torch.Tensor,
    per_head_k_up: torch.Tensor,
    per_head_v_up: torch.Tensor,
    per_head_k_up_normal: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    mla_absorption: Dict[str, Any],
    head_block_size: int,
    *,
    position_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    H-blocked attention that slices along head dimension to create blocks and processes each block.
    """
    batch_size, num_heads, q_len, _ = q_pe.shape
    if head_block_size <= 0:
        head_block_size = num_heads
    num_head_blocks = math.ceil(num_heads / head_block_size)

    if hasattr(module, "config"):
        mask_dtype = module.config.torch_dtype
    else:
        mask_dtype = q_pe.dtype
    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=mask_dtype, device=q_pe.device)

    if mla_absorption is not None:
        absorption = mla_absorption.get("absorption", False)
        online = mla_absorption.get("online", False)
    else:
        absorption = False

    h_output_blocks = []
    h_attn_blocks = []
    # Process each head block independently
    for head_block_idx in range(num_head_blocks):
        h_start = head_block_idx * head_block_size
        h_end = min(h_start + head_block_size, num_heads)

        if absorption:
            if online:
                qup_kupT = torch.matmul(per_head_q_up[:, h_start:h_end, :, :], per_head_k_up[:, h_start:h_end, :, :])
                dq_qup_kupT = torch.matmul(q_a_proj_out, qup_kupT)
            else:
                dq_qup_kupT = torch.matmul(q_a_proj_out, fusedqk[:, h_start:h_end, :, :])
            qkupTrope_nope = torch.cat((dq_qup_kupT, q_pe[:, h_start:h_end, :, :]), dim=-1)
            krope_nope = torch.cat((kva, k_pe), dim=-1)
            attn_weights = torch.matmul(qkupTrope_nope, krope_nope.transpose(2, 3)) * scaling
        else:
            knope = torch.matmul(kva, per_head_k_up_normal[:, h_start:h_end, :, :])
            krope_nope = torch.cat((knope, k_pe), dim=-1)
            qrope_nope = torch.cat((q_nope[:, h_start:h_end, :, :], q_pe[:, h_start:h_end, :, :]), dim=-1)
            attn_weights = torch.matmul(qrope_nope, krope_nope.transpose(2, 3)) * scaling

        if attention_mask is not None:
            attn_weights = torch.where(attention_mask, masked_tensor, attn_weights)
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_pe.dtype)
        attn_output = torch.matmul(attn_weights, kva)
        attn_output = torch.matmul(attn_output, per_head_v_up[:, h_start:h_end, :, :])
        h_output_blocks.append(attn_output)
        h_attn_blocks.append(attn_weights)

    attn_output = torch.cat(h_output_blocks, dim=1).transpose(1, 2).contiguous()
    attn_weights = torch.cat(h_attn_blocks, dim=1)
    return attn_output, attn_weights
