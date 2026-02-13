# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import math
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


def supports_blocked_kv(past_key_value: Optional[Cache]) -> bool:
    return past_key_value is not None and hasattr(past_key_value, "read_only_blockedKV")


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
    score_mod: Optional[Callable[[torch.Tensor, int, int], torch.Tensor]] = None,
    use_causal_mask: bool = False,
    sliding_window: Optional[int] = None,
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

    past_seen_tokens = _normalize_int(cache_kwargs.get("past_seen_tokens"))
    total_seen_tokens = past_seen_tokens + query.shape[2]
    if torch.onnx.is_in_onnx_export():
        attention_mask = None
        use_causal_mask = True
    position_ids = cache_kwargs.get("position_ids")
    num_kv_blocks = _normalize_int(num_kv_blocks)
    block_size = -(-past_seen_tokens // num_kv_blocks) if num_kv_blocks > 0 else past_seen_tokens
    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32, device=query.device)

    current_position = position_ids.max(dim=-1).values

    for j in range(num_kv_blocks):
        start_index = j * block_size
        end_index = (j + 1) * block_size

        skip_future = (torch.tensor(start_index, device=query.device) > current_position).all()

        k_block, v_block = past_key_value.read_only_blockedKV(start_index, end_index, layer_idx, cache_kwargs)
        k_block_states, v_block_states = _get_kv_states(module, k_block, v_block)

        attn_weights_block = torch.matmul(query, k_block_states.transpose(2, 3)) * scaling
        if score_mod is not None:
            attn_weights_block = score_mod(attn_weights_block, start_index, end_index)

        mask_block = None
        if attention_mask is not None:
            mask_block = attention_mask[..., start_index:end_index]
            if mask_block.shape[-1] != attn_weights_block.shape[-1]:
                mask_block = None

        if use_causal_mask or mask_block is None:
            target_length = min(total_seen_tokens, end_index)
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

        # Update Running row maximum
        prev_max = current_max
        current_max_updated = torch.max(prev_max, attn_weights_block.max(dim=-1).values)
        delta_max = prev_max - current_max_updated

        current_exp = torch.exp(attn_weights_block - current_max_updated.unsqueeze(-1))

        # update running denominator
        prev_denominator = current_denominator
        curr_exp_sum = torch.einsum("bhqk->bhq", current_exp)
        current_denominator_updated = prev_denominator * torch.exp(delta_max) + curr_exp_sum

        prob = current_exp / current_denominator_updated.unsqueeze(-1)

        prev_output = output
        output_updated = ((prev_denominator / current_denominator_updated).unsqueeze(-1)) * prev_output * torch.exp(
            delta_max.unsqueeze(-1)
        ) + torch.matmul(prob, v_block_states)

        if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
            current_max = torch.where(skip_future, prev_max, current_max_updated)
            current_denominator = torch.where(skip_future, prev_denominator, current_denominator_updated)
            output = torch.where(skip_future.unsqueeze(-1), prev_output, output_updated)
        else:
            # Eager mode
            current_max = current_max_updated
            current_denominator = current_denominator_updated
            output = output_updated

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
    score_mod: Optional[Callable[[torch.Tensor, int, int], torch.Tensor]] = None,
    use_causal_mask: bool = False,
    sliding_window: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Initialize Running Maximum and Denominator
    batch_size, num_heads, seq_len, DH = query.shape

    past_seen_tokens = _normalize_int(cache_kwargs.get("past_seen_tokens"))
    total_seen_tokens = past_seen_tokens + query.shape[2]
    if torch.onnx.is_in_onnx_export():
        attention_mask = None
        use_causal_mask = True
    position_ids = cache_kwargs.get("position_ids")
    num_kv_blocks = _normalize_int(num_kv_blocks)
    num_q_blocks = max(1, _normalize_int(num_q_blocks))
    
    q_block_positions = [(i * seq_len) // num_q_blocks for i in range(num_q_blocks)]
    q_output_blocks = []
    q_attn_blocks = []

    block_size = -(-past_seen_tokens // num_kv_blocks) if num_kv_blocks > 0 else past_seen_tokens
    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32, device=query.device)

    current_position = position_ids.max(dim=-1).values

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
            start_index = j * block_size
            end_index = (j + 1) * block_size

            skip_future = (torch.tensor(start_index, device=query.device) > current_position).all()
            
            # Eager mode Only
            if not torch.onnx.is_in_onnx_export() and not torch.jit.is_tracing():
                if skip_future.item():
                    break

            k_block, v_block = past_key_value.read_only_blockedKV(start_index, end_index, layer_idx, cache_kwargs)
            k_block_states, v_block_states = _get_kv_states(module, k_block, v_block)

            attn_weights_block = torch.matmul(q_block, k_block_states.transpose(2, 3)) * scaling
            if score_mod is not None:
                attn_weights_block = score_mod(attn_weights_block, start_index, end_index)

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

            # Update Running row maximum
            prev_max = current_max
            current_max_updated = torch.max(prev_max, attn_weights_block.max(dim=-1).values)
            delta_max = prev_max - current_max_updated

            current_exp = torch.exp(attn_weights_block - current_max_updated.unsqueeze(-1))

            # update running denominator
            prev_denominator = current_denominator
            curr_exp_sum = torch.einsum("bhqk->bhq", current_exp)
            current_denominator_updated = prev_denominator * torch.exp(delta_max) + curr_exp_sum

            prob = current_exp / current_denominator_updated.unsqueeze(-1)

            prev_output = output_blocks
            output_updated = ((prev_denominator / current_denominator_updated).unsqueeze(-1)) * prev_output * torch.exp(
                delta_max.unsqueeze(-1)
            ) + torch.matmul(prob, v_block_states)

            if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
                current_max = torch.where(skip_future, prev_max, current_max_updated)
                current_denominator = torch.where(skip_future, prev_denominator, current_denominator_updated)
                output_blocks = torch.where(skip_future.unsqueeze(-1), prev_output, output_updated)
            else:
                # Eager mode
                current_max = current_max_updated
                current_denominator = current_denominator_updated
                output_blocks = output_updated
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
    score_mod: Optional[Callable[[torch.Tensor, int, int], torch.Tensor]] = None,
    use_causal_mask: bool = False,
    sliding_window: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Initialize result tensor
    output = torch.zeros_like(query)

    # Initialize Running Maximum and Denominator
    batch_size, num_heads, seq_len, DH = query.shape

    past_seen_tokens = _normalize_int(cache_kwargs.get("past_seen_tokens"))
    total_seen_tokens = past_seen_tokens + query.shape[2]
    if torch.onnx.is_in_onnx_export():
        attention_mask = None
        use_causal_mask = True
    position_ids = cache_kwargs.get("position_ids")
    num_kv_blocks = _normalize_int(num_kv_blocks)
    if head_block_size <= 0:
        head_block_size = num_heads
    num_head_blocks = math.ceil(num_heads / head_block_size)
    num_q_blocks = max(1, _normalize_int(num_q_blocks))
    
    q_block_positions = [(i * seq_len) // num_q_blocks for i in range(num_q_blocks)]
    
    h_output_blocks = []
    h_attn_blocks = []

    block_size = -(-past_seen_tokens // num_kv_blocks) if num_kv_blocks > 0 else past_seen_tokens
    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32, device=query.device)

    current_position = position_ids.max(dim=-1).values

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
            output_blocks = torch.zeros((batch_size, h_end - h_start, q_len_block, DH), device=query.device, dtype=query.dtype)
        
            for j in range(num_kv_blocks):
                start_index = j * block_size
                end_index = (j + 1) * block_size

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
                if score_mod is not None:
                    attn_weights_block = score_mod(attn_weights_block, start_index, end_index)

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

                # Update Running row maximum
                prev_max = current_max
                current_max_updated = torch.max(prev_max, attn_weights_block.max(dim=-1).values)
                delta_max = prev_max - current_max_updated

                current_exp = torch.exp(attn_weights_block - current_max_updated.unsqueeze(-1))

                # update running denominator
                prev_denominator = current_denominator
                curr_exp_sum = torch.einsum("bhqk->bhq", current_exp)
                current_denominator_updated = prev_denominator * torch.exp(delta_max) + curr_exp_sum

                prob = current_exp / current_denominator_updated.unsqueeze(-1)

                prev_output = output_blocks
                output_updated = ((prev_denominator / current_denominator_updated).unsqueeze(-1)) * prev_output * torch.exp(
                    delta_max.unsqueeze(-1)
                ) + torch.matmul(prob, v_g)

                if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
                    # skip_mask = skip_future.view(1, 1, 1).expand(batch_size, h_end - h_start, q_len_block)
                    current_max = torch.where(skip_future, prev_max, current_max_updated)
                    current_denominator = torch.where(skip_future, prev_denominator, current_denominator_updated)
                    output_blocks = torch.where(skip_future, prev_output, output_updated)
                else:
                    # Eager mode
                    current_max = current_max_updated
                    current_denominator = current_denominator_updated
                    output_blocks = output_updated
            q_output_blocks.append(output_blocks)
            q_attn_blocks.append(attn_weights_block)

        head_output = torch.cat(q_output_blocks, dim=2)
        head_attn_weights = torch.cat(q_attn_blocks, dim=2)
        h_output_blocks.append(head_output)
        h_attn_blocks.append(head_attn_weights)

    attn_output = torch.cat(h_output_blocks, dim=1).transpose(1, 2).contiguous()
    attn_weights = torch.cat(h_attn_blocks, dim=1)

    return attn_output, attn_weights

def h_blocked_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    head_block_size: int,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Q-blocked attention that slices the query sequence into blocks and processes each block.
    """
    batch_size, num_heads, q_len, _ = query.shape
    if head_block_size <= 0:
        head_block_size = num_heads
    num_head_blocks = math.ceil(num_heads / head_block_size)
    
    key_states, value_states = _get_kv_states(module, key, value)

    h_output_blocks = []
    h_attn_blocks = []

    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32, device=query.device)

    # Process each head block independently
    for head_block_idx in range(num_head_blocks):
        h_start = head_block_idx * head_block_size
        h_end = min(h_start + head_block_size, num_heads)

        # Extract head blocks
        q_g = query[:, h_start:h_end, :, :]
        k_g = key_states[:, h_start:h_end, :, :]
        v_g = value_states[:, h_start:h_end, :, :]

        attn_weights = torch.matmul(q_g, k_g.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = torch.where(attention_mask, masked_tensor, attn_weights)

        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        output_block = torch.matmul(attn_weights, v_g)

        h_output_blocks.append(output_block)
        h_attn_blocks.append(attn_weights)

    attn_output = torch.cat(h_output_blocks, dim=1).transpose(1, 2).contiguous()
    attn_weights = torch.cat(h_attn_blocks, dim=1)

    return attn_output, attn_weights


def q_blocked_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_q_blocks: int,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Q-blocked attention that slices the query sequence into blocks and processes each block.
    """
    batch_size, num_heads, q_len, _ = query.shape
    num_q_blocks = max(1, _normalize_int(num_q_blocks))
    key_states, value_states = _get_kv_states(module, key, value)

    q_block_positions = [(i * q_len) // num_q_blocks for i in range(num_q_blocks)]
    q_output_blocks = []
    q_attn_blocks = []

    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32, device=query.device)

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
        if attn_mask_block is not None:
            attn_weights = torch.where(attn_mask_block, masked_tensor, attn_weights)

        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        output_block = torch.matmul(attn_weights, value_states)

        q_output_blocks.append(output_block)
        q_attn_blocks.append(attn_weights)

    attn_output = torch.cat(q_output_blocks, dim=2).transpose(1, 2).contiguous()
    attn_weights = torch.cat(q_attn_blocks, dim=2)

    return attn_output, attn_weights