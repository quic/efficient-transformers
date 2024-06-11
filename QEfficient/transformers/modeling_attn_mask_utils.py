# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from transformers.modeling_attn_mask_utils import AttentionMaskConverter


def _create_causal_mask(
    position_ids,
    target_length,
    sliding_window: Optional[int] = None,
):
    """
    A utility attention mask class that allows one to:
        - Create a causal 4d mask
        - Create a causal 4d mask with slided window
    """
    if sliding_window is not None:
        query_indices = position_ids.unsqueeze(-1)
        kv_indices = torch.arange(target_length).view(1, -1)
        # --- Rolling buffer ---
        pos_max = position_ids.max(1, keepdim=True).values
        kv_start = (pos_max // target_length) * target_length
        kv_indices_high = kv_indices + kv_start
        kv_indices_low = torch.where(kv_indices_high < target_length, kv_indices, kv_indices_high - target_length)
        kv_indices = torch.where(kv_indices_high > pos_max, kv_indices_low, kv_indices_high)
        kv_indices = kv_indices.unsqueeze(1)
        # ------
        causal_mask = kv_indices > query_indices
        attention_mask = causal_mask

        window_indices = query_indices - sliding_window + 1
        window_mask = kv_indices < window_indices
        attention_mask = attention_mask | window_mask
        attention_mask = attention_mask.unsqueeze(1)
    else:
        query_indices = position_ids.unsqueeze(-1)
        kv_indices = torch.arange(target_length).view(1, 1, -1)
        attention_mask = kv_indices > query_indices
        attention_mask = attention_mask.unsqueeze(1)

    return attention_mask


@dataclass
class QEffAttentionMaskConverter(AttentionMaskConverter):
    """
    A utility attention mask class that allows one to:
        - Create a causal 4d mask
        - Create a causal 4d mask with slided window
        - Convert a 2d attention mask (batch_size, query_length) to a 4d attention mask (batch_size, 1, query_length,
          key_value_length)
    """

    is_causal: bool
    sliding_window: int
    cache_index: int

    def __init__(
        self,
        is_causal: bool,
        sliding_window: Optional[int] = None,
        cache_index: Optional[torch.LongTensor] = None,
    ):
        self.is_causal = is_causal
        self.sliding_window = sliding_window
        self.cache_index = cache_index

        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(
                f"Make sure that when passing `sliding_window` that its value is a strictly positive integer, not `{self.sliding_window}`"
            )

    def to_4d(
        self,
        attention_mask_2d: torch.Tensor,
        query_length: int,
        dtype: torch.dtype,
        key_value_length: Optional[int] = None,
        cache_index: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Converts 2D attention mask to 4D attention mask by expanding mask to (bsz, head_dim=1, query_length,
        key_value_length) shape and by adding a large negative bias to not-attended positions. If attention_mask is
        causal, a causal mask will be added.

        Also, For Cloud AI 100:
        Copied from AttentionMaskConverter: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_attn_mask_utils.py
        The only differences are:
        - add new args cache idx for the kv retention

        """
        input_shape = (attention_mask_2d.shape[0], query_length)

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        causal_4d_mask = None
        if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
            if key_value_length is None:
                raise ValueError(
                    "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask."
                )

            past_key_values_length = key_value_length - query_length if cache_index is None else key_value_length
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=attention_mask_2d.device,
                past_key_values_length=past_key_values_length,
                cache_index=cache_index,
                sliding_window=self.sliding_window,
            )
        elif self.sliding_window is not None:
            raise NotImplementedError("Sliding window is currently only implemented for causal masking")

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = self._expand_mask(attention_mask_2d, dtype, tgt_len=input_shape[-1]).to(
            attention_mask_2d.device
        )
        expanded_4d_mask = expanded_attn_mask if causal_4d_mask is None else expanded_attn_mask | causal_4d_mask
        return expanded_4d_mask

    @staticmethod
    def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
        cache_index: Optional[int] = None,
        sliding_window: Optional[int] = None,
    ):
        """
        Make causal mask used for bi-directional self-attention.

        Also, For Cloud AI 100:
        Copied from AttentionMaskConverter: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_attn_mask_utils.py
        The only differences are:
        - add new args cache idx for the kv retention
        """
        bsz, tgt_len = input_ids_shape
        src_len = past_key_values_length if past_key_values_length > 0 else tgt_len + past_key_values_length
        query_indices = torch.arange(tgt_len, device=device) + (cache_index if cache_index is not None else 0)
        key_indices = torch.arange(src_len, device=device)
        mask = query_indices.view(-1, 1) < key_indices.view(1, -1)

        # add lower triangular sliding window mask if necessary
        if sliding_window is not None:
            window_indices = query_indices - sliding_window + 1
            mask = mask | (key_indices.view(1, -1) < window_indices.view(-1, 1))

        return mask[None, None, :, :].expand(bsz, 1, tgt_len, src_len)

    @staticmethod
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(torch.bool)

        inverted_mask = ~expanded_mask

        return inverted_mask


def _qeff_prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
    cache_index: Optional[torch.LongTensor] = None,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        attention_mask (`torch.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        input_shape (`tuple(int)` or `list(int)` or `torch.Size`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        inputs_embeds (`torch.Tensor`):
            The embedded inputs as a torch Tensor.
        past_key_values_length (`int`):
            The length of the key value cache.
        sliding_window (`int`, *optional*):
            If the model uses windowed attention, a sliding window should be passed.
    """
    attn_mask_converter = QEffAttentionMaskConverter(
        is_causal=True, sliding_window=sliding_window, cache_index=cache_index
    )
    key_value_length = input_shape[-1] if past_key_values_length == 0 else past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask,
            input_shape[-1],
            key_value_length=key_value_length,
            dtype=inputs_embeds.dtype,
            cache_index=cache_index,
        )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0],
            input_shape[-1],
            key_value_length,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

    return attention_mask


def _qeff_prepare_4d_attention_mask(
    mask: torch.Tensor,
    dtype: torch.dtype,
    tgt_len: Optional[int] = None,
    cache_index: Optional[torch.LongTensor] = None,
):
    """
    Creates a non-causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        mask (`torch.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        dtype (`torch.dtype`):
            The torch dtype the created mask shall have.
        tgt_len (`int`):
            The target length or query length the created mask shall have.
    """
    return QEffAttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)
