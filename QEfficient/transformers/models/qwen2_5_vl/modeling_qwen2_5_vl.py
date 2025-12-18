# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLAttention,
    Qwen2_5_VLConfig,
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLModelOutputWithPast,
    Qwen2_5_VLRotaryEmbedding,
    Qwen2_5_VLTextModel,
    Qwen2_5_VLVisionAttention,
    apply_rotary_pos_emb_vision,
    repeat_kv,
    rotate_half,
)

from QEfficient.transformers.cache_utils import QEffDynamicCache

# from transformers import Qw
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo, get_padding_shape_from_config
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE
from QEfficient.utils.logging_utils import logger


def qeff_apply_rotary_pos_emb(q, k, cos, sin, position_ids, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension seperately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """

    mrope_section = mrope_section * 2
    cos = cos[position_ids]
    sin = sin[position_ids]

    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed.to(q.dtype), k_embed.to(k.dtype)


class QEffQwen2_5_VLVisionAttention(Qwen2_5_VLVisionAttention):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        attention_mask = torch.full(
            [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
        )

        # Create index grids
        seq_len = attention_mask.shape[-1]
        rows = torch.arange(seq_len).view(1, -1)
        cols = torch.arange(seq_len).view(-1, 1)

        # Prepare start and end indices
        start = cu_seqlens[:-1].view(-1, 1, 1)
        end = cu_seqlens[1:].view(-1, 1, 1)

        # Create block masks using broadcasting
        row_mask = (rows >= start) & (rows < end)
        col_mask = (cols >= start) & (cols < end)
        block_mask = row_mask & col_mask  # shape: (num_blocks, seq_len, seq_len)

        # Combine all blocks into one mask
        final_mask = torch.ones((seq_len, seq_len), dtype=torch.float32)
        final_mask[block_mask.any(dim=0)] = 0

        final_mask = torch.where(final_mask == 1.0, torch.finfo(q.dtype).min, final_mask)

        attention_mask[0] = final_mask

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class QEffQwen2_5_VisionTransformerPretrainedModel(Qwen2_5_VisionTransformerPretrainedModel):
    def rot_pos_emb(self, grid_thw):
        pos_ids = []

        bs, t, h, w = grid_thw.shape

        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // self.spatial_merge_size,
            self.spatial_merge_size,
            w // self.spatial_merge_size,
            self.spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // self.spatial_merge_size,
            self.spatial_merge_size,
            w // self.spatial_merge_size,
            self.spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)

        x_expanded = pos_ids.unsqueeze(0)
        x_expanded = x_expanded.expand(bs, -1, -1)
        pos_ids = x_expanded.reshape(-1, pos_ids.size(1))

        max_grid_size = max(grid_thw.shape)
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        bs, grid_t, grid_h, grid_w = grid_thw.shape

        llm_grid_h, llm_grid_w = (
            grid_h // self.spatial_merge_size,
            grid_w // self.spatial_merge_size,
        )
        index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)

        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size

        index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)

        index_padded = index_padded.reshape(
            grid_t,
            num_windows_h,
            vit_merger_window_size,
            num_windows_w,
            vit_merger_window_size,
        )
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t,
            num_windows_h * num_windows_w,
            vit_merger_window_size,
            vit_merger_window_size,
        )

        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)

        x_expanded = seqlens.unsqueeze(0)
        x_expanded = x_expanded.expand(bs, -1)
        seqlens = x_expanded.reshape(-1)

        index_padded = index_padded.reshape(-1)

        mask = (index_padded == -100).to(torch.int32)

        if torch.jit.is_tracing():
            order = torch.argsort(mask)
        else:
            order = torch.argsort(mask, stable=True)

        index_new = index_padded[order]
        index_new = index_new[: index.reshape(-1).size(0)]

        step = grid_t * llm_grid_h * llm_grid_w
        batch_indices = torch.arange(bs)
        batch_indices = batch_indices.view(-1, 1)
        offsets = batch_indices * step
        window_index_tmp = index_new.unsqueeze(0) + offsets
        window_index = window_index_tmp.reshape(-1)

        cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]

        cu_window_seqlens = torch.cat([torch.tensor([0], dtype=cu_seqlens_tmp.dtype), cu_seqlens_tmp])

        return window_index, cu_window_seqlens

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)

        cu_window_seqlens = cu_window_seqlens.to(
            device=hidden_states.device, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32
        )

        # cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)

        hidden_states = hidden_states[window_index, :, :]

        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        bs, t, h, w = grid_thw.shape

        t = torch.arange(t, t + 1).squeeze().expand(bs)
        h = torch.arange(h, h + 1).squeeze().expand(bs)
        w = torch.arange(w, w + 1).squeeze().expand(bs)

        cu_seqlens = (h * w).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )

        cu_seqlens = torch.cat([torch.tensor([0], dtype=cu_seqlens.dtype), cu_seqlens])

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states


class QEffQwen2_5_VLRotaryEmbedding(Qwen2_5_VLRotaryEmbedding):
    """
    Copied from LlamaForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    The only differences are:
    - Add static sin/cos computations.
    """

    def __init__(self, config: Qwen2_5_VLConfig, device=None):
        super().__init__(config=config)
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=self.original_max_seq_len, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype) * self.attention_scaling,
            self.sin_cached[:seq_len].to(dtype=x.dtype) * self.attention_scaling,
        )


def eager_attention_forward_blockedKV(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    num_kv_blocks: Optional[torch.Tensor] = None,
    cache_kwargs: Optional[Dict[str, Any]] = None,
    layer_idx: int = None,
    past_key_value: Optional[Cache] = None,
    **kwargs,
):
    # Initialize result tensor
    output = torch.zeros_like(query)

    # Initialize Running Maximum
    batch_size, num_heads, seq_len, _ = query.shape
    current_max = torch.full((batch_size, num_heads, seq_len), float(MIN_MASKED_ATTENTION_VALUE))

    # Initialize Denominator
    current_denominator = torch.zeros(batch_size, num_heads, seq_len)

    past_seen_tokens = cache_kwargs.get("past_seen_tokens")
    position_ids = cache_kwargs.get("position_ids")
    block_size = -(-past_seen_tokens // num_kv_blocks)
    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32)

    for j in range(num_kv_blocks):
        start_index = j * block_size
        end_index = (j + 1) * block_size
        K_block, V_block = past_key_value.read_only_blockedKV(start_index, end_index, layer_idx, cache_kwargs)
        K_block_states = repeat_kv(K_block, module.num_key_value_groups)
        V_block_states = repeat_kv(V_block, module.num_key_value_groups)
        past_seen_tokens_start = start_index
        past_seen_tokens_end = torch.where(
            torch.tensor(past_seen_tokens, dtype=torch.int) < torch.tensor(end_index, dtype=torch.int),
            past_seen_tokens,
            end_index,
        )
        causal_mask_block = _create_causal_mask(
            position_ids=position_ids, target_length=past_seen_tokens_end, start_index=past_seen_tokens_start
        )

        # Compute attention scores for the block
        attn_weights_block = torch.matmul(query, K_block_states.transpose(2, 3)) / math.sqrt(module.head_dim)
        if attention_mask is not None:
            attn_weights_block = torch.where(causal_mask_block, masked_tensor, attn_weights_block)

        # Update Running row maximum
        prev_max = current_max
        current_max = torch.max(prev_max, attn_weights_block.max(dim=-1).values)
        delta_max = prev_max - current_max

        current_exp = torch.exp(
            attn_weights_block - current_max.unsqueeze(-1)
        )  # Subract current_max from each column of attn_weights_block

        # update running denominator
        prev_denominator = current_denominator
        current_denominator = prev_denominator * torch.exp(delta_max) + current_exp.sum(axis=-1)

        prob = current_exp / current_denominator.unsqueeze(-1)

        prev_output = output
        output = ((prev_denominator / current_denominator).unsqueeze(-1)) * prev_output * torch.exp(
            delta_max.unsqueeze(-1)
        ) + torch.matmul(prob, V_block_states)
    attn_output = output.transpose(1, 2).contiguous()
    attn_weights = None

    return attn_output, attn_weights


def eager_attention_forward_q_blocked(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    **kwargs,
):
    """
    Q-blocked attention for Qwen2.5-VL.
    Blocks only the query SL dimension.

    Args:
        query: (BS, NH, Q_LEN, DH)
        key: (BS, NH_KV, KV_LEN, DH)
        value: (BS, NH_KV, KV_LEN, DH)
        attention_mask: (BS, NH, Q_LEN, KV_LEN) or broadcastable
    """
    BS, NH, Q_LEN, DH = query.shape
    _, _, KV_LEN, _ = key.shape

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    target_blocks_q = int(os.environ.get("num_q_blocks", Q_LEN))
    q_block_positions = [(i * Q_LEN) // target_blocks_q for i in range(target_blocks_q)]
    scaling = 1.0 / math.sqrt(module.head_dim)

    q_output_blocks = []
    q_attn_weights_blocks = []

    # Process each Q block
    for q_block_idx in range(target_blocks_q):
        qi = q_block_positions[q_block_idx]

        # Calculate Q block size
        if q_block_idx == target_blocks_q - 1:
            real_q_len = Q_LEN - qi
        else:
            real_q_len = q_block_positions[q_block_idx + 1] - qi

        # Extract Q block
        q_block = query[:, :, qi : qi + real_q_len, :]
        attn_mask_block = None
        if attention_mask is not None:
            attn_mask_block = attention_mask[:, :, qi : qi + real_q_len, :]

        # Compute attention scores for this Q block
        attn_weights = torch.matmul(q_block, key_states.transpose(2, 3)) * scaling
        if attn_mask_block is not None:
            attn_weights = torch.where(
                attn_mask_block,
                torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32, device=attn_weights.device),
                attn_weights,
            )

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

        # Compute output for this Q block
        output_block = torch.matmul(attn_weights, value_states)

        q_output_blocks.append(output_block)
        q_attn_weights_blocks.append(attn_weights)

    attn_output = torch.cat(q_output_blocks, dim=2)
    attn_output = attn_output.transpose(1, 2).contiguous()

    # Concatenate attention weights
    attn_weights = torch.cat(q_attn_weights_blocks, dim=2)

    return attn_output, attn_weights


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    num_kv_blocks: Optional[torch.Tensor] = None,
    cache_kwargs: Optional[Dict[str, Any]] = None,
    layer_idx: int = None,
    past_key_value: Optional[Cache] = None,
    **kwargs,
):
    """
    Wrapper that routes to blocked or default attention based on environment variable.
    """
    blocking_mode = os.environ.get("ATTENTION_BLOCKING_MODE", "default").lower()

    if blocking_mode == "q":
        return eager_attention_forward_q_blocked(module, query, key, value, attention_mask, **kwargs)
    elif blocking_mode != "q" and num_kv_blocks is not None:
        return eager_attention_forward_blockedKV(
            module,
            query,
            key,
            value,
            attention_mask,
            cache_kwargs=cache_kwargs,
            num_kv_blocks=num_kv_blocks,
            layer_idx=layer_idx,
            past_key_value=past_key_value,
            **kwargs,
        )
    elif blocking_mode == "default":
        # Original implementation
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) / math.sqrt(module.head_dim)

        if attention_mask is not None:
            attn_weights = torch.where(
                attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), attn_weights
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights
    else:
        raise ValueError(f"Invalid ATTENTION_BLOCKING_MODE: {blocking_mode}. Must be 'q' or 'default'")


class QEffQwen2_5_VLAttention(Qwen2_5_VLAttention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __qeff_init__(self):
        self.rotary_emb = QEffQwen2_5_VLRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        num_kv_blocks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        kv_seq_len = past_key_value.get_seq_length(self.layer_idx, cache_position)
        past_seen_tokens = past_key_value.get_seq_length() if past_key_value is not None else 0

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = qeff_apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids[1:], self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            if num_kv_blocks is not None:
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "batch_index": batch_index,
                    "position_ids": position_ids[0],
                    "past_seen_tokens": past_seen_tokens,
                }
                past_key_value.write_only(key_states, value_states, self.layer_idx, cache_kwargs)
            else:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "batch_index": batch_index,
                    "position_ids": position_ids[0],
                }
                if comp_ctx_lengths is not None:
                    attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
                    cache_kwargs["CCL"] = attention_mask.shape[-1]
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            num_kv_blocks=num_kv_blocks,
            cache_kwargs=cache_kwargs,
            layer_idx=self.layer_idx,
            past_key_value=past_key_value,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class QEffQwen2_5_VLDecoderLayer(Qwen2_5_VLDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        # position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class QEffQwen2_5_VLTextModel(Qwen2_5_VLTextModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens
        causal_mask = _create_causal_mask(
            position_ids=position_ids[0], target_length=target_length, sliding_window=self.config.sliding_window
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

        return (hidden_states, past_key_values)


class QEffQwen2_5_VLModel(Qwen2_5_VLModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        output = Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()


class QEffQwen_2_5_vl_EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.vision_model = self.model.visual

    def forward(self, pixel_values, image_grid_thw):
        image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        bs = image_grid_thw.shape[0]
        split_size = torch.floor_divide(torch.tensor(image_embeds.size(0)), bs)
        image_embeds = image_embeds.reshape(bs, split_size, image_embeds.size(1))

        return image_embeds


class QEffQwen_2_5_vl_DecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.language_model = self.model.model.language_model

    def forward(
        self,
        input_ids,
        vision_embeds,
        position_ids,
        image_idx,
        past_key_values,
        batch_index: Optional[torch.LongTensor] = None,
        comp_ctx_lengths: Optional[List[int]] = None,
    ):
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        B, N, C = inputs_embeds.shape
        selected = input_ids == self.model.config.image_token_id
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(selected.unsqueeze(0).shape[0]).view(-1, 1)
        image_features_expanded = vision_embeds.reshape(-1, C).unsqueeze(0)[indices0, indices1]
        image_input_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_input_embeds)
        outputs = self.model.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=True,
        )

        logit_index = position_ids[0].to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[torch.arange(position_ids[0].shape[0]).view(-1, 1), logit_index]
        logits = self.model.lm_head(hidden_states)
        image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)

        return logits, vision_embeds, image_idx, outputs.past_key_values


class QEffQwen_2_5_vl_ForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def get_qeff_vision_encoder(self):
        return QEffQwen_2_5_vl_EncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEffQwen_2_5_vl_DecoderWrapper(self)

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        **kwargs,
    ):
        inputs_shapes = {}
        inputs_shapes["input_ids"] = (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)

        vision_size = 3577
        inputs_shapes["vision_embeds"] = (
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            vision_size,
            self.model.config.hidden_size,
        )
        inputs_shapes["image_grid_thw"] = (1, 1, 98, 146)
        inputs_shapes["position_ids"] = (
            3,
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN,
        )
        inputs_shapes["pixel_values"] = (14308, 1176)
        inputs_shapes["image_idx"] = (1, 1)
        inputs_shapes["image_sizes"] = (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, 2)
        # Define inputs
        vision_inputs = {}
        lang_inputs = {}
        vision_inputs["pixel_values"] = torch.zeros((inputs_shapes["pixel_values"]), dtype=torch.float32)
        vision_inputs["image_grid_thw"] = torch.zeros((inputs_shapes["image_grid_thw"]), dtype=torch.int64)
        lang_inputs["input_ids"] = torch.zeros((inputs_shapes["input_ids"]), dtype=torch.int64)
        lang_inputs["vision_embeds"] = torch.zeros((inputs_shapes["vision_embeds"]), dtype=torch.float32)
        lang_inputs["position_ids"] = (
            (
                torch.arange(constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN, dtype=torch.int64)
                .view(1, constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)
                .repeat(constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, 1)
            )
            .unsqueeze(0)
            .repeat(4, 1, 1)
        )
        lang_inputs["image_idx"] = torch.zeros((inputs_shapes["image_idx"]), dtype=torch.int64)

        bs: int = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        fbs: int = constants.ONNX_EXPORT_EXAMPLE_FBS

        # Add data for KV
        kv_cache_shape = get_padding_shape_from_config(
            config=self.model.config.text_config,
            batch_size=fbs if continuous_batching else bs,
            seq_len=constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN,
        )

        lang_inputs["past_key_values"] = [[] for _ in range(self.model.config.text_config.num_hidden_layers)]
        for i in range(self.model.config.text_config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_inputs["past_key_values"][i].append(torch.zeros(kv_cache_shape, dtype=torch.float32))

        if continuous_batching:
            lang_inputs["batch_index"] = torch.arange(bs).view(bs, 1)

        if comp_ctx_lengths is not None:
            lang_inputs["comp_ctx_lengths"] = torch.randint(0, 100, (40,), dtype=torch.long)

        inputs = {}
        if kv_offload:
            inputs["vision"] = vision_inputs
            inputs["lang"] = lang_inputs
        else:
            lang_inputs.pop("vision_embeds")
            inputs = {**vision_inputs, **lang_inputs}

        return inputs

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        img_size: None,
        height: int = None,
        width: int = None,
        num_frames: int = 1,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
        **compiler_options,
    ):
        comp_ctx_lengths_prefill = compiler_options.pop("comp_ctx_lengths_prefill", None)
        comp_ctx_lengths_decode = compiler_options.pop("comp_ctx_lengths_decode", None)

        if height is None or width is None:
            height = constants.QWEN2_5_VL_HEIGHT
            width = constants.QWEN2_5_VL_WIDTH
            logger.warning(
                f"Setting height and width to be {height} and {width} respectively, as it was neither passed nor found in vision_config"
            )
        prefill_seq_len = prefill_seq_len if prefill_seq_len else 128
        ctx_len = ctx_len if ctx_len else constants.INTERN_CTX_LEN
        channel = 3
        patch_size = self.config.vision_config.patch_size
        temporal_patch_size = self.config.vision_config.temporal_patch_size

        IMAGE_FACTOR = 28
        MIN_PIXELS = 4 * 28 * 28
        MAX_PIXELS = 16384 * 28 * 28
        MAX_RATIO = 200

        def round_by_factor(number: int, factor: int) -> int:
            """Returns the closest integer to 'number' that is divisible by 'factor'."""
            return round(number / factor) * factor

        def ceil_by_factor(number: int, factor: int) -> int:
            """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
            return math.ceil(number / factor) * factor

        def floor_by_factor(number: int, factor: int) -> int:
            """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
            return math.floor(number / factor) * factor

        def smart_resize(
            height: int,
            width: int,
            factor: int = IMAGE_FACTOR,
            min_pixels: int = MIN_PIXELS,
            max_pixels: int = MAX_PIXELS,
        ) -> tuple[int, int]:
            """
            Rescales the image so that the following conditions are met:

            1. Both dimensions (height and width) are divisible by 'factor'.

            2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

            3. The aspect ratio of the image is maintained as closely as possible.
            """
            if max(height, width) / min(height, width) > MAX_RATIO:
                raise ValueError(
                    f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
                )
            h_bar = max(factor, round_by_factor(height, factor))
            w_bar = max(factor, round_by_factor(width, factor))
            if h_bar * w_bar > max_pixels:
                beta = math.sqrt((height * width) / max_pixels)
                h_bar = floor_by_factor(height / beta, factor)
                w_bar = floor_by_factor(width / beta, factor)
            elif h_bar * w_bar < min_pixels:
                beta = math.sqrt(min_pixels / (height * width))
                h_bar = ceil_by_factor(height * beta, factor)
                w_bar = ceil_by_factor(width * beta, factor)
            return h_bar, w_bar

        resized_height, resized_width = smart_resize(height=height, width=width)
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        grid_height = grid_h * grid_w
        grid_width = patch_size * patch_size * temporal_patch_size * channel
        vision_size = grid_height // 4
        vision_size = vision_size * num_frames
        grid_height = grid_height * batch_size

        vision = [
            {
                "batch_size": batch_size,
                "vision_size": vision_size,
                "grid_height": grid_height,
                "grid_width": grid_width,
                "grid_h": grid_h,
                "grid_w": grid_w,
            }
        ]

        if comp_ctx_lengths_prefill is not None:
            lang = []

            for i in range(0, len(comp_ctx_lengths_prefill)):
                lang_prefill = {
                    "batch_size": 1 if continuous_batching else batch_size,
                    "seq_len": prefill_seq_len,
                    "ctx_len": ctx_len,
                    "vision_size": vision_size,
                    "comp_ctx_lengths": comp_ctx_lengths_prefill[i],
                    "vision_batch_size": batch_size,
                }

                if continuous_batching:
                    lang_prefill["full_batch_size"] = kv_cache_batch_size
                else:
                    lang_prefill["batch_size"] = kv_cache_batch_size
                if full_batch_size:
                    lang_prefill["full_batch_exec_size"] = full_batch_size

                lang.append(lang_prefill)

            for i in range(0, len(comp_ctx_lengths_decode)):
                lang_decode = {
                    "batch_size": full_batch_size if continuous_batching else batch_size,
                    "seq_len": "1",
                    "ctx_len": ctx_len,
                    "vision_size": vision_size,
                    "comp_ctx_lengths": comp_ctx_lengths_decode[i],
                    "vision_batch_size": batch_size,
                }

                if continuous_batching:
                    lang_decode["full_batch_size"] = kv_cache_batch_size
                else:
                    lang_decode["batch_size"] = kv_cache_batch_size

                lang.append(lang_decode)
        else:
            lang_prefill = {
                "batch_size": 1 if continuous_batching else batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "vision_size": vision_size,
                "vision_batch_size": batch_size,
            }

            if continuous_batching:
                lang_prefill["full_batch_size"] = kv_cache_batch_size
            else:
                lang_prefill["batch_size"] = kv_cache_batch_size
            if full_batch_size:
                lang_prefill["full_batch_exec_size"] = full_batch_size

            lang_decode = {
                "batch_size": full_batch_size if continuous_batching else batch_size,
                "seq_len": 1,
                "ctx_len": ctx_len,
                "vision_size": vision_size,
                "vision_batch_size": batch_size,
            }

            if continuous_batching:
                lang_decode["full_batch_size"] = kv_cache_batch_size
            else:
                lang_decode["batch_size"] = kv_cache_batch_size

            lang = [lang_prefill, lang_decode]

        specializations = {}

        if kv_offload:
            specializations["vision"] = vision
            specializations["lang"] = lang
            return specializations, compiler_options
        else:
            lang[0].pop("vision_size")
            lang[1].pop("vision_size")
            return lang, compiler_options

    def get_onnx_dynamic_axes(
        self, comp_ctx_lengths: Optional[List[int]] = None, kv_offload: bool = False, continuous_batching: bool = False
    ):
        # Define dynamic axes
        num_layers = self.config.text_config.num_hidden_layers

        vision_dynamic_axes = {
            "pixel_values": {0: "grid_height", 1: "grid_width"},
            "image_grid_thw": {0: "batch_size", 2: "grid_h", 3: "grid_w"},
        }

        lang_dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {1: "batch_size", 2: "seq_len"},
            "vision_embeds": {0: "vision_batch_size", 1: "vision_size"},
        }

        for i in range(num_layers):
            lang_dynamic_axes[f"past_key.{i}"] = {
                0: "full_batch_size" if continuous_batching else "batch_size",
                2: "ctx_len",
            }
            lang_dynamic_axes[f"past_value.{i}"] = {
                0: "full_batch_size" if continuous_batching else "batch_size",
                2: "ctx_len",
            }

        if continuous_batching:
            lang_dynamic_axes["batch_index"] = {0: "batch_size"}

        if comp_ctx_lengths is not None:
            lang_dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}

        dynamic_axes = {}

        if kv_offload:
            dynamic_axes["vision"] = vision_dynamic_axes
            dynamic_axes["lang"] = lang_dynamic_axes
        else:
            lang_dynamic_axes.pop("vision_embeds")
            dynamic_axes = {**vision_dynamic_axes, **lang_dynamic_axes}
        return dynamic_axes

    def get_output_names(self, kv_offload: bool = False):
        vision_output_names = ["vision_embeds"]
        lang_output_names = ["logits"]
        for i in range(self.model.config.text_config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_output_names.append(f"past_{kv}.{i}_RetainedState")

        output_names = {}
        if kv_offload:
            lang_output_names.insert(1, "vision_embeds_RetainedState")
            lang_output_names.insert(2, "image_idx_output")
            output_names["vision"] = vision_output_names
            output_names["lang"] = lang_output_names
        else:
            lang_output_names.insert(1, "pixel_values_RetainedState")
            lang_output_names.insert(2, "image_idx_output")
            return lang_output_names
        return output_names

    def prepare_inputs_for_generation(self, inputs, prefill_seq_len=128, batch_size=1):
        input_ids_length = inputs["input_ids"].shape[1]

        inputs["position_ids"] = torch.arange(input_ids_length).view(1, 1, input_ids_length).expand(-1, batch_size, -1)

        pos_ids, rope_deltas = self.model.get_rope_index(
            inputs["input_ids"],
            None if "image_grid_thw" not in inputs else inputs["image_grid_thw"],
            video_grid_thw=None,
            second_per_grid_ts=None,
            attention_mask=inputs["attention_mask"],
        )

        inputs["position_ids"] = torch.cat((inputs["position_ids"], pos_ids), dim=0)

        num_chunks = -(input_ids_length // -prefill_seq_len)  # ceil divide without float
        padded_len = num_chunks * prefill_seq_len  # Convert to a multiple of prompt_len

        inputs["position_ids"] = F.pad(
            inputs["position_ids"], pad=(0, padded_len - input_ids_length), mode="constant", value=-1
        )

        inputs.pop("image_grid_thw", None)

        return inputs

    def get_inputs_info(self):
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="attention_mask", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="pixel_values", datatype=torch.float32, shape=("batch_size", 3, "image_size", "image_size")),
        ]
