# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn


def blocked_tokens_ffn_forward(
    w1: nn.Module,
    w2: nn.Module,
    x: torch.Tensor,
    num_token_blocks: int,
    w3: Optional[nn.Module] = None,
    activation_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Token blocking: Process tokens in blocks.
    Pattern: For each token_block:
        X = gate(token_block), Y = nonlinearity(X), Z = up(token_block), W = Y*Z
        outputs.append(down(W))
    Result: concatenate along token dimension
    """
    act = activation_fn if activation_fn is not None else F.silu

    _, seq_len, _ = x.shape

    token_block_size = math.ceil(seq_len / num_token_blocks)
    outputs = []

    for token_idx in range(num_token_blocks):
        start_idx = token_idx * token_block_size
        if token_idx == num_token_blocks - 1:
            token_len_block = seq_len - start_idx
        else:
            token_len_block = token_block_size
        end_idx = start_idx + token_len_block
        token_block = x[:, start_idx:end_idx, :]

        X = w1(token_block)
        Y = act(X)
        # gpt2 style mlp has w1 and w2 only while llama style mlp has w3 as well
        if w3 is not None:
            Z = w3(token_block)
            W = Y * Z
        else:
            W = Y

        output_block = w2(W)
        outputs.append(output_block)

    # Concatenate along token dimension
    return torch.cat(outputs, dim=1)  # [BS, seqlen, dim]


def blocked_weights_ffn_forward(
    w1: nn.Module,
    w2: nn.Module,
    x: torch.Tensor,
    num_weight_blocks: int,
    w3: Optional[nn.Module] = None,
    activation_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Weight blocking: Process intermediate hidden dimension in chunks to reduce peak intermediate size.

    For each hidden_block:
        X = linear(x, w1_block)
        Y = silu(X)
        Z = linear(x, w3_block)
        W = Y * Z
        result += linear(W, w2_block)
    """
    act = activation_fn if activation_fn is not None else F.silu

    bs, seq_len, dim = x.shape
    hidden_dim = w1.weight.shape[0]

    weight_block_size = math.ceil(hidden_dim / num_weight_blocks)

    for weight_idx in range(num_weight_blocks):
        start_idx = weight_idx * weight_block_size
        if weight_idx == num_weight_blocks - 1:
            weight_len_block = hidden_dim - start_idx
        else:
            weight_len_block = weight_block_size
        end_idx = start_idx + weight_len_block

        # Extract weight blocks.
        # w1/w3: [hidden_dim, dim], w2: [dim, hidden_dim]
        w1_block = w1.weight[start_idx:end_idx, :]
        if w3 is not None:
            w3_block = w3.weight[start_idx:end_idx, :]
        w2_block = w2.weight[:, start_idx:end_idx]

        w1_bias = w1.bias[start_idx:end_idx] if w1.bias is not None else None
        if w3 is not None:
            w3_bias = w3.bias[start_idx:end_idx] if w3.bias is not None else None
        w2_bias = w2.bias if (weight_idx == 0 and w2.bias is not None) else None

        X = F.linear(x, w1_block, w1_bias)
        Y = act(X)
        # gpt2 style mlp has w1 and w2 only while llama style mlp has w3 as well
        if w3 is not None:
            Z = F.linear(x, w3_block, w3_bias)
            W = Y * Z
        else:
            W = Y
        if weight_idx == 0:
            result_block = F.linear(W, w2_block, w2_bias)
        else:
            result_block += F.linear(W, w2_block, w2_bias)

    return result_block


def blocked_tokens_weights_ffn_forward(
    w1: nn.Module,
    w2: nn.Module,
    x: torch.Tensor,
    num_token_blocks: int,
    num_weight_blocks: int,
    w3: Optional[nn.Module] = None,
    activation_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Both token and weight blocking: process tokens in blocks, and within each token block,
    process intermediate hidden dimension in blocks.

    w1/w3: [hidden_dim, dim], w2: [dim, hidden_dim]
    """
    act = activation_fn if activation_fn is not None else F.silu

    bs, seq_len, dim = x.shape
    hidden_dim = w1.weight.shape[0]

    token_block_size = math.ceil(seq_len / num_token_blocks)
    weight_block_size = math.ceil(hidden_dim / num_weight_blocks)
    result = torch.zeros(bs, seq_len, dim, device=x.device, dtype=x.dtype)

    outputs = []

    for token_idx in range(num_token_blocks):
        t_start = token_idx * token_block_size
        if token_idx == num_token_blocks - 1:
            token_len_block = seq_len - t_start
        else:
            token_len_block = token_block_size
        t_end = t_start + token_len_block

        token_block = x[:, t_start:t_end, :]

        for weight_idx in range(num_weight_blocks):
            w_start = weight_idx * weight_block_size
            if weight_idx == num_weight_blocks - 1:
                weight_len_block = hidden_dim - w_start
            else:
                weight_len_block = weight_block_size
            w_end = w_start + weight_len_block

            w1_block = w1.weight[w_start:w_end, :]
            if w3 is not None:
                w3_block = w3.weight[w_start:w_end, :]
            w2_block = w2.weight[:, w_start:w_end]

            w1_bias = w1.bias[w_start:w_end] if w1.bias is not None else None
            if w3 is not None:
                w3_bias = w3.bias[w_start:w_end] if w3.bias is not None else None
            w2_bias = w2.bias if (weight_idx == 0 and w2.bias is not None) else None

            X = F.linear(token_block, w1_block, w1_bias)
            Y = act(X)
            # gpt2 style mlp has w1 and w2 only while llama style mlp has w3 as well
            if w3 is not None:
                Z = F.linear(token_block, w3_block, w3_bias)
                W = Y * Z
            else:
                W = Y

            if weight_idx == 0:
                result_block = F.linear(W, w2_block, w2_bias)
            else:
                result_block += F.linear(W, w2_block, w2_bias)

        outputs.append(result_block)

    result = torch.cat(outputs, dim=1)

    return result
