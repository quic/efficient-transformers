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
    output = torch.zeros(batch_size, num_heads, seq_len, module.config.kv_lora_rank, device=query.device)

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
    )
    skip_kv = True
    current_denominator = torch.zeros(batch_size, num_heads, seq_len, device=query.device)

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
            krope_nope = torch.cat((knope, k_pe_block.expand(-1, num_heads, -1, -1)), dim=-1)
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
