# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import math
import os
from typing import Optional

import torch


def get_attention_blocking_config():
    """
    Get attention blocking configuration from environment variables.

    Returns:
        tuple: (blocking_mode, head_block_size, num_kv_blocks, num_q_blocks)
            - blocking_mode (str): The blocking strategy ('kv', 'q', 'qkv', 'default')
            - head_block_size (int or None): Number of attention heads per block
            - num_kv_blocks (int or None): Number of key-value blocks
            - num_q_blocks (int or None): Number of query blocks
    """
    mode = os.environ.get("ATTENTION_BLOCKING_MODE", "default").lower()
    head_block_size = int(os.environ.get("head_block_size", 0)) or None
    num_kv_blocks = int(os.environ.get("num_kv_blocks", 0)) or None
    num_q_blocks = int(os.environ.get("num_q_blocks", 0)) or None

    # Validate blocking mode
    valid_modes = ["kv", "qkv", "q", "default"]
    if mode not in valid_modes:
        raise ValueError(f"Invalid ATTENTION_BLOCKING_MODE: {mode}. Must be one of {valid_modes}")

    return mode, head_block_size, num_kv_blocks, num_q_blocks


def apply_head_blocking(
    q: torch.FloatTensor,
    k: torch.FloatTensor,
    v: torch.FloatTensor,
    head_block_size: int,
    attention_mask: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Forward pass with head-only blocking (default mode).

    This method processes attention heads in blocks while computing full attention
    matrices for each head block. It's less memory-efficient than other blocking
    modes but simpler and faster for moderate sequence lengths.

    Args:
        q (torch.FloatTensor): Query tensor of shape (BS, NH, CL, DH)
        k (torch.FloatTensor): Key tensor of shape (BS, NH, CL, DH)
        v (torch.FloatTensor): Value tensor of shape (BS, NH, CL, DH)
        attention_mask (Optional[torch.FloatTensor]): Attention mask tensor

    Returns:
        torch.FloatTensor: Attention output of shape (BS, NH, CL, DH)
    """
    BS, NH, CL, DH = q.shape
    scale_factor = 1.0 / math.sqrt(DH)

    # Get head blocking configuration
    head_block_size = head_block_size or NH
    num_head_blocks = math.ceil(NH / head_block_size)

    # Optimization: Handle small sequences with standard attention
    BS, NH, K_CL, DH = k.shape
    if K_CL <= 512:
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
        if attention_mask is not None:
            scores = torch.where(attention_mask, scores, torch.tensor(-1e4, dtype=scores.dtype, device=scores.device))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)
        return out

    outputs = []

    # Process each head block independently
    for head_block_idx in range(num_head_blocks):
        h_start = head_block_idx * head_block_size
        h_end = min(h_start + head_block_size, NH)

        # Extract head blocks
        q_g = q[:, h_start:h_end, :, :]
        k_g = k[:, h_start:h_end, :, :]
        v_g = v[:, h_start:h_end, :, :]

        # Compute full attention matrix for this head block
        qkblock = torch.matmul(q_g, k_g.transpose(-2, -1)) * scale_factor

        # Standard softmax computation
        probs = torch.softmax(qkblock, dim=-1)

        # Compute attention output
        output_blocks = torch.matmul(probs, v_g)
        outputs.append(output_blocks)

    # Concatenate all head blocks along head dimension
    out = torch.cat(outputs, dim=1)  # (BS, NH, CL, DH)
    return out


def apply_kv_blocking(
    q: torch.FloatTensor,
    k: torch.FloatTensor,
    v: torch.FloatTensor,
    head_block_size: int,
    num_kv_blocks: int,
    attention_mask: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Forward pass with Key-Value blocking and head blocking.

    This method processes key-value pairs in blocks while keeping queries intact.
    It uses online softmax to maintain numerical stability and reduce memory usage
    compared to computing full attention matrices.

    Args:
        q (torch.FloatTensor): Query tensor of shape (BS, NH, CL, DH)
        k (torch.FloatTensor): Key tensor of shape (BS, NH, CL, DH)
        v (torch.FloatTensor): Value tensor of shape (BS, NH, CL, DH)
        attention_mask (Optional[torch.FloatTensor]): Attention mask tensor

    Returns:
        torch.FloatTensor: Attention output of shape (BS, NH, CL, DH)
    """
    BS, NH, CL, DH = q.shape
    scale_factor = 1.0 / math.sqrt(DH)

    # Get blocking configuration
    head_block_size = head_block_size or NH
    num_kv_blocks = num_kv_blocks or CL
    num_head_blocks = math.ceil(NH / head_block_size)
    block_positions = [(i * CL) // num_kv_blocks for i in range(num_kv_blocks)]

    # Handle small sequences with standard attention
    BS, NH, K_CL, DH = k.shape
    if K_CL <= 512:
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
        if attention_mask is not None:
            scores = torch.where(attention_mask, scores, torch.tensor(-1e4, dtype=scores.dtype, device=scores.device))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)
        return out

    head_outputs = []

    # Process each head block
    for head_block_idx in range(num_head_blocks):
        h_start = head_block_idx * head_block_size
        h_end = min(h_start + head_block_size, NH)
        num_h = h_end - h_start

        q_g = q[:, h_start:h_end, :, :]
        k_g = k[:, h_start:h_end, :, :]
        v_g = v[:, h_start:h_end, :, :]

        # Initialize online softmax statistics
        running_exp_sum = torch.zeros((BS, num_h, CL), device=q.device, dtype=q.dtype)
        running_max = torch.full((BS, num_h, CL), float("-inf"), device=q.device, dtype=q.dtype)
        output_blocks = torch.zeros_like(q_g)

        # Process K,V in blocks using online softmax
        for kv_block_idx in range(num_kv_blocks):
            ki = block_positions[kv_block_idx]

            # Calculate KV block size
            if kv_block_idx == num_kv_blocks - 1:
                real_kv_len = CL - ki
            else:
                real_kv_len = block_positions[kv_block_idx + 1] - ki

            k_block = k_g[:, :, ki : ki + real_kv_len, :]
            v_block = v_g[:, :, ki : ki + real_kv_len, :]

            # Compute attention scores for current KV block
            qkblock = torch.matmul(q_g, k_block.transpose(-2, -1)) * scale_factor

            # Online softmax: Update running maximum
            prev_max = running_max.clone()
            running_max = torch.maximum(prev_max, torch.max(qkblock, dim=-1)[0])

            # Calculate numerical stability adjustments
            delta_max = prev_max - running_max
            curr_exp = torch.exp(qkblock - running_max.unsqueeze(-1))

            # Update running sum of exponentials
            prev_exp_sum = running_exp_sum.clone()
            curr_exp_sum = torch.einsum("bhqk->bhq", curr_exp)
            running_exp_sum = prev_exp_sum * torch.exp(delta_max) + curr_exp_sum

            # Compute normalized attention weights
            inv_running_exp_sum = 1.0 / running_exp_sum
            softmax_qkblock = curr_exp * inv_running_exp_sum.unsqueeze(-1)

            # Update output with rescaling
            prev_out = output_blocks.clone()
            rescale_factor = (prev_exp_sum * inv_running_exp_sum) * torch.exp(delta_max)
            output_blocks = rescale_factor.unsqueeze(-1) * prev_out + torch.matmul(softmax_qkblock, v_block)

        head_outputs.append(output_blocks)

    out = torch.cat(head_outputs, dim=1)  # (BS, NH, CL, DH)
    return out


def apply_q_blocking(
    q: torch.FloatTensor,
    k: torch.FloatTensor,
    v: torch.FloatTensor,
    head_block_size: int,
    num_q_blocks: int,
    attention_mask: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Forward pass with Query blocking and head blocking.

    This method processes query tokens in blocks while keeping key-value pairs intact.
    It's useful when the sequence length is large but memory constraints are primarily
    due to the query dimension.

    Args:
        q (torch.FloatTensor): Query tensor of shape (BS, NH, CL, DH)
        k (torch.FloatTensor): Key tensor of shape (BS, NH, CL, DH)
        v (torch.FloatTensor): Value tensor of shape (BS, NH, CL, DH)
        attention_mask (Optional[torch.FloatTensor]): Attention mask tensor

    Returns:
        torch.FloatTensor: Attention output of shape (BS, NH, CL, DH)
    """
    BS, NH, CL, DH = q.shape
    scale_factor = 1.0 / math.sqrt(DH)

    # Get blocking configuration
    head_block_size = head_block_size or NH
    num_q_blocks = num_q_blocks or CL
    num_head_blocks = math.ceil(NH / head_block_size)
    q_block_positions = [(i * CL) // num_q_blocks for i in range(num_q_blocks)]

    # Handle small sequences with standard attention
    BS, NH, K_CL, DH = k.shape
    if K_CL <= 512:
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
        if attention_mask is not None:
            scores = torch.where(attention_mask, scores, torch.tensor(-1e4, dtype=scores.dtype, device=scores.device))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)
        return out

    head_outputs = []

    # Process each head block
    for head_block_idx in range(num_head_blocks):
        h_start = head_block_idx * head_block_size
        h_end = min(h_start + head_block_size, NH)

        q_g = q[:, h_start:h_end, :, :]
        k_g = k[:, h_start:h_end, :, :]
        v_g = v[:, h_start:h_end, :, :]

        q_output_list = []

        # Process queries in blocks
        for q_block_idx in range(num_q_blocks):
            qi = q_block_positions[q_block_idx]

            # Calculate Q block size
            if q_block_idx == num_q_blocks - 1:
                real_q_len = CL - qi
            else:
                real_q_len = q_block_positions[q_block_idx + 1] - qi

            q_block = q_g[:, :, qi : qi + real_q_len, :]

            # Compute attention for this query block against all keys
            scores = torch.matmul(q_block, k_g.transpose(-2, -1)) * scale_factor
            probs = torch.softmax(scores, dim=-1)
            out_block = torch.matmul(probs, v_g)

            q_output_list.append(out_block)

        # Concatenate query blocks
        head_output = torch.cat(q_output_list, dim=2)
        head_outputs.append(head_output)

    out = torch.cat(head_outputs, dim=1)  # (BS, NH, CL, DH)
    return out


def apply_qkv_blocking(
    q: torch.FloatTensor,
    k: torch.FloatTensor,
    v: torch.FloatTensor,
    head_block_size: int,
    num_kv_blocks: int,
    num_q_blocks: int,
    attention_mask: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Forward pass with combined Query, Key, Value blocking and head blocking.

    This method implements the most memory-efficient attention computation by blocking
    along all three dimensions: heads, queries, and key-values.

    Args:
        q (torch.FloatTensor): Query tensor of shape (BS, NH, CL, DH)
        k (torch.FloatTensor): Key tensor of shape (BS, NH, CL, DH)
        v (torch.FloatTensor): Value tensor of shape (BS, NH, CL, DH)
        attention_mask (Optional[torch.FloatTensor]): Attention mask tensor

    Returns:
        torch.FloatTensor: Attention output of shape (BS, NH, CL, DH)
    """
    BS, NH, CL, DH = q.shape
    scale_factor = 1.0 / math.sqrt(DH)

    # Get blocking configuration from environment variables
    head_block_size = head_block_size or NH
    num_kv_blocks = num_kv_blocks or CL
    num_q_blocks = num_q_blocks or CL
    num_head_blocks = math.ceil(NH / head_block_size)

    # Calculate block positions for even distribution
    kv_block_positions = [(i * CL) // num_kv_blocks for i in range(num_kv_blocks)]
    q_block_positions = [(i * CL) // num_q_blocks for i in range(num_q_blocks)]

    # Optimization: Use standard attention for small sequences
    BS, NH, K_CL, DH = k.shape
    if K_CL <= 512:
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
        if attention_mask is not None:
            scores = torch.where(attention_mask, scores, torch.tensor(-1e4, dtype=scores.dtype, device=scores.device))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)
        return out

    head_outputs = []

    # Process attention heads in blocks to reduce memory usage
    for head_block_idx in range(num_head_blocks):
        h_start = head_block_idx * head_block_size
        h_end = min(h_start + head_block_size, NH)
        num_h = h_end - h_start

        # Extract current head block
        q_g = q[:, h_start:h_end, :, :]
        k_g = k[:, h_start:h_end, :, :]
        v_g = v[:, h_start:h_end, :, :]
        q_output_list = []

        # Process queries in blocks within each head block
        for q_block_idx in range(num_q_blocks):
            qi = q_block_positions[q_block_idx]

            # Calculate actual Q block size (handle remainder for last block)
            if q_block_idx == num_q_blocks - 1:
                real_q_len = CL - qi
            else:
                real_q_len = q_block_positions[q_block_idx + 1] - qi

            q_block = q_g[:, :, qi : qi + real_q_len, :]

            # Initialize online softmax statistics for this Q block
            running_exp_sum = torch.zeros((BS, num_h, real_q_len), device=q.device, dtype=q.dtype)
            running_max = torch.full((BS, num_h, real_q_len), float("-inf"), device=q.device, dtype=q.dtype)
            output_blocks = torch.zeros((BS, num_h, real_q_len, DH), device=q.device, dtype=q.dtype)

            # Process K,V in blocks for this Q block (online softmax)
            for kv_block_idx in range(num_kv_blocks):
                ki = kv_block_positions[kv_block_idx]

                # Calculate actual KV block size
                if kv_block_idx == num_kv_blocks - 1:
                    real_kv_len = CL - ki
                else:
                    real_kv_len = kv_block_positions[kv_block_idx + 1] - ki

                k_block = k_g[:, :, ki : ki + real_kv_len, :]
                v_block = v_g[:, :, ki : ki + real_kv_len, :]

                # Compute attention scores for current Q-K block
                qkblock = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale_factor

                # Online softmax: Update running maximum
                prev_max = running_max.clone()
                if qkblock.shape[-1] == 0:
                    running_max = prev_max
                else:
                    running_max = torch.maximum(prev_max, torch.max(qkblock, dim=-1)[0])

                # Calculate adjustment factor for numerical stability
                delta_max = prev_max - running_max
                curr_exp = torch.exp(qkblock - running_max.unsqueeze(-1))

                # Online softmax: Update running sum of exponentials
                prev_exp_sum = running_exp_sum.clone()
                curr_exp_sum = torch.einsum("bhqk->bhq", curr_exp)
                running_exp_sum = prev_exp_sum * torch.exp(delta_max) + curr_exp_sum

                # Compute normalized attention weights for this block
                inv_running_exp_sum = 1.0 / running_exp_sum
                softmax_qkblock = curr_exp * inv_running_exp_sum.unsqueeze(-1)

                # Online softmax: Update output with rescaling of previous blocks
                prev_out = output_blocks.clone()
                rescale_factor = (prev_exp_sum * inv_running_exp_sum) * torch.exp(delta_max)
                output_blocks = rescale_factor.unsqueeze(-1) * prev_out + torch.matmul(softmax_qkblock, v_block)

            q_output_list.append(output_blocks)

        # Concatenate all Q blocks for this head block
        head_output = torch.cat(q_output_list, dim=2)
        head_outputs.append(head_output)

    # Concatenate all head blocks
    out = torch.cat(head_outputs, dim=1)
    return out


def compute_blocked_attention(
    q: torch.FloatTensor,
    k: torch.FloatTensor,
    v: torch.FloatTensor,
    head_block_size: int,
    num_kv_blocks: int,
    num_q_blocks: int,
    blocking_mode: str = "default",
    attention_mask: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Main dispatcher function for different attention blocking strategies.

    Args:
        q (torch.FloatTensor): Query tensor of shape (BS, NH, CL, DH)
        k (torch.FloatTensor): Key tensor of shape (BS, NH, CL, DH)
        v (torch.FloatTensor): Value tensor of shape (BS, NH, CL, DH)
        head_block_size (int) : Head blocking size
        num_kv_blocks (int) : Number of KV blocks
        num_q_blocks (int) : Number of Q blocks
        blocking_mode (str): Blocking strategy ('kv', 'q', 'qkv', 'default')
        attention_mask (Optional[torch.FloatTensor]): Attention mask tensor

    Returns:
        torch.FloatTensor: Attention output of shape (BS, NH, CL, DH)
    """
    if blocking_mode == "kv":
        return apply_kv_blocking(q, k, v, head_block_size, num_kv_blocks, attention_mask)
    elif blocking_mode == "q":
        return apply_q_blocking(q, k, v, head_block_size, num_q_blocks, attention_mask)
    elif blocking_mode == "qkv":
        return apply_qkv_blocking(q, k, v, head_block_size, num_kv_blocks, num_q_blocks, attention_mask)
    else:  # default
        return apply_head_blocking(q, k, v, head_block_size, attention_mask)
