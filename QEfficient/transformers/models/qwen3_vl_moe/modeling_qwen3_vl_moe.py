# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeForConditionalGeneration,
    Qwen3VLMoeModel,
    Qwen3VLMoeModelOutputWithPast,
    Qwen3VLMoeTextAttention,
    Qwen3VLMoeTextConfig,
    Qwen3VLMoeTextDecoderLayer,
    Qwen3VLMoeTextModel,
    Qwen3VLMoeTextRotaryEmbedding,
    Qwen3VLMoeTextSparseMoeBlock,
    Qwen3VLMoeVisionAttention,
    Qwen3VLMoeVisionModel,
    apply_rotary_pos_emb_vision,
    repeat_kv,
    rotate_half,
)

from QEfficient.transformers.cache_utils import QEffDynamicCache
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
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed.to(q.dtype), k_embed.to(k.dtype)


class QEffQwen3VLMoeTextRotaryEmbedding(Qwen3VLMoeTextRotaryEmbedding):
    def __init__(self, config: Qwen3VLMoeTextConfig, device=None):
        super().__init__(config, device)
        self.mrope_section = config.rope_scaling.get("mrope_section", [24, 20, 20])
        self._set_cos_sin_cache(
            seq_len=self.original_max_seq_len, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(3, 1, -1)  # (3, 1, seq_len)

        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, 1, -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # (3, 1, 1, seq_len)

        device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)

            freqs_interleaved = self._apply_interleaved_mrope_cached(freqs, self.mrope_section)
            emb = torch.cat((freqs_interleaved, freqs_interleaved), dim=-1)
            self.register_buffer(
                "cos_cached", (emb.cos() * self.attention_scaling).squeeze(0).to(dtype), persistent=False
            )
            self.register_buffer(
                "sin_cached", (emb.sin() * self.attention_scaling).squeeze(0).to(dtype), persistent=False
            )

    def _apply_interleaved_mrope_cached(self, freqs, mrope_section):
        freqs_t = freqs[0].clone()  # (bs, seq_len, head_dim // 2)
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(self, x, position_ids, seq_len=None):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        seq_len = position_ids.shape[-1]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        cos = self.cos_cached[:seq_len].to(dtype=x.dtype)
        sin = self.sin_cached[:seq_len].to(dtype=x.dtype)

        if position_ids.shape[1] > 1:
            cos = cos.unsqueeze(0).expand(position_ids.shape[1], -1, -1)
            sin = sin.unsqueeze(0).expand(position_ids.shape[1], -1, -1)
        else:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        return cos, sin


class QEffQwen3VLMoeVisionModel(Qwen3VLMoeVisionModel):
    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        max_hw = max(grid_thw.shape)
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device
        bs, num_frames, height, width = grid_thw.shape
        grid_thw = (torch.tensor(grid_thw.shape, dtype=torch.int64)).unsqueeze(0)

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        merged_h, merged_w = height // merge_size, width // merge_size

        block_rows = torch.arange(merged_h, device=device)  # block row indices
        block_cols = torch.arange(merged_w, device=device)  # block col indices
        intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
        intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

        # Compute full-resolution positions
        row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

        row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

        coords = torch.stack((row_idx, col_idx), dim=-1)

        if num_frames > 1:
            coords = coords.repeat(num_frames, 1)

        pos_ids = coords
        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        bs, t, h, w = grid_thw.shape
        h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        max_t = torch.tensor(self.num_grid_per_side - 1, device=h_idxs.device)

        h_idxs_ceil = torch.minimum(h_idxs_floor + 1, max_t)  # working
        w_idxs_ceil = torch.minimum(w_idxs_floor + 1, max_t)

        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor

        base_h = h_idxs_floor * self.num_grid_per_side
        base_h_ceil = h_idxs_ceil * self.num_grid_per_side

        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        ]

        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]

        idx_tensor = torch.stack(indices, dim=0).to(dtype=torch.long, device=self.pos_embed.weight.device)  # [4, h*w]

        weight_tensor = torch.stack(weights, dim=0).to(
            dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        pos_embed = patch_pos_embeds[0]
        pos_embed = pos_embed.repeat(t, 1)

        pos_embed = (
            pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        x_expanded = patch_pos_embeds.unsqueeze(0)
        x_expanded = x_expanded.expand(bs, -1, -1)
        patch_pos_embeds = x_expanded.reshape(-1, patch_pos_embeds.size(1))
        return patch_pos_embeds

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        bs, t, h, w = grid_thw.shape

        t = torch.arange(t, t + 1).squeeze().expand(bs)
        h = torch.arange(h, h + 1).squeeze().expand(bs)
        w = torch.arange(w, w + 1).squeeze().expand(bs)

        cu_seqlens = (h * w).cumsum(
            dim=0,
            dtype=torch.int32,
        )
        cu_seqlens = torch.cat([torch.tensor([0], dtype=cu_seqlens.dtype), cu_seqlens])

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)
        hidden_states = self.merger(hidden_states)
        return hidden_states, deepstack_feature_lists


class QEffQwen3VLMoeVisionAttention(Qwen3VLMoeVisionAttention):
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


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_kwargs: Optional[Dict[str, Any]] = None,
    layer_idx: int = None,
    past_key_value: Optional[Cache] = None,
    **kwargs,
):
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


class QEffQwen3VLMoeTextAttention(Qwen3VLMoeTextAttention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = qeff_apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids[1:], self.config.rope_scaling["mrope_section"]
        )
        if past_key_values is not None:
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
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            cache_kwargs=cache_kwargs,
            layer_idx=self.layer_idx,
            past_key_values=past_key_values,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_values


class QEffQwen3VLMoeTextDecoderLayer(Qwen3VLMoeTextDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        batch_index: Optional[torch.LongTensor] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
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
            past_key_values (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
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
            past_key_values=past_key_values,
            batch_index=batch_index,
            comp_ctx_lengths=comp_ctx_lengths,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states[0]

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        return outputs


class QEffQwen3VLMoeTextModel(Qwen3VLMoeTextModel):
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
        visual_pos_masks: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if self.config.use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
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
            position_ids=position_ids[0], target_length=target_length, sliding_window=None
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids[1:])
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
                past_key_values=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            layer_idx = 0
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class QEffQwen3VLMoeModel(Qwen3VLMoeModel):
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

        output = Qwen3VLMoeModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()


class QEffQwen3VLEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.vision_model = self.model.visual

    def forward(self, pixel_values, image_grid_thw):
        image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)[0]
        bs = image_grid_thw.shape[0]
        split_size = torch.floor_divide(torch.tensor(image_embeds.size(0)), bs)
        image_embeds = image_embeds.reshape(bs, split_size, image_embeds.size(1))
        return image_embeds


class QEffQwen3VLDecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.language_model = self.model.model

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


class QEffQwen3VLMoeTextSparseMoeBlock(Qwen3VLMoeTextSparseMoeBlock):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, H = hidden_states.shape
        T = B * S
        x = hidden_states.view(T, H)

        router_logits = self.gate(x)
        prob = F.softmax(router_logits, dim=-1, dtype=torch.float)
        top_w, top_i = torch.topk(prob, self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=1, keepdim=True)
        top_w = top_w.to(x.dtype)
        idx = top_i.reshape(-1)
        w_up = self.experts.gate_up_proj.index_select(0, idx)
        w_dn = self.experts.down_proj.index_select(0, idx)

        xk = x.unsqueeze(1).expand(-1, self.top_k, -1).contiguous()
        xk = xk.view(-1, 1, H)
        gate_up = torch.bmm(xk, w_up)
        I2 = gate_up.size(-1)
        half = I2 // 2
        gate, up = gate_up[..., :half], gate_up[..., half:]
        intermediate = up * self.experts.act_fn(gate)
        experts_out = torch.bmm(intermediate, w_dn)
        experts_out = experts_out.view(T, self.top_k, H) * top_w.unsqueeze(-1)
        experts_out = experts_out.sum(dim=1).view(B, S, H)

        return experts_out, router_logits


class QEffQwen3VLMoeForConditionalGeneration(Qwen3VLMoeForConditionalGeneration):
    def get_qeff_vision_encoder(self):
        return QEffQwen3VLEncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEffQwen3VLDecoderWrapper(self)

    # def forward(
    #     self,
    #     input_ids,
    #     position_ids,
    #     past_key_values,
    #     pixel_values:Optional[torch.FloatTensor] = None,
    #     image_idx:Optional[torch.LongTensor] = None,
    #     comp_ctx_lengths: Optional[List[int]] = None,
    #     batch_index: Optional[torch.LongTensor] = None,
    #     image_grid_thw=None,
    # ):
    #     image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)[0]
    #     bs = image_grid_thw.shape[0]
    #     split_size = torch.floor_divide(torch.tensor(image_embeds.size(0)), bs)

    #     inputs_embeds = self.model.get_input_embeddings()(input_ids)
    #     B, N, C = inputs_embeds.shape
    #     selected = input_ids == self.model.config.image_token_id
    #     indices1 = selected.to(torch.int64).cumsum(1) - 1
    #     indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
    #     indices0 = torch.arange(selected.unsqueeze(0).shape[0]).view(-1, 1)
    #     image_features_expanded = image_embeds.reshape(-1, C).unsqueeze(0)[indices0, indices1]
    #     image_input_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
    #     inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_input_embeds)
    #     outputs = self.language_model(
    #         inputs_embeds=inputs_embeds,
    #         position_ids=position_ids,
    #         past_key_values=past_key_values,
    #         comp_ctx_lengths=comp_ctx_lengths,
    #         batch_index=batch_index,
    #         use_cache=True,
    #     )
    #     logit_index = position_ids[0].to(torch.int32).argmax(1, keepdim=True)
    #     hidden_states = outputs.last_hidden_state[torch.arange(position_ids[0].shape[0]).view(-1, 1), logit_index]
    #     logits = self.lm_head(hidden_states)
    #     image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
    #     return logits, image_embeds, image_idx, outputs.past_key_values

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        **kwargs,
    ):
        inputs_shapes = {}
        inputs_shapes["input_ids"] = (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)
        vision_size = 187
        inputs_shapes["vision_embeds"] = (
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            vision_size,
            self.model.config.vision_config.out_hidden_size,
        )
        inputs_shapes["image_grid_thw"] = (1, 1, 22, 34)
        inputs_shapes["position_ids"] = (
            3,
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN,
        )
        inputs_shapes["pixel_values"] = (748, 1536)
        inputs_shapes["image_idx"] = (1, 1)
        inputs_shapes["image_sizes"] = (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, 2)

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
        # Add data for KV

        bs: int = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        fbs: int = constants.ONNX_EXPORT_EXAMPLE_FBS

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
            lang_inputs["comp_ctx_lengths"] = torch.randint(0, 100, (40,), dtype=torch.int8)
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
            height = 1365
            width = 2048
            logger.warning(
                "Setting height and width to be 1365 and 2048 respectively, as it was neither passed nor found in vision_config"
            )
        prefill_seq_len = prefill_seq_len if prefill_seq_len else 128
        ctx_len = ctx_len if ctx_len else constants.INTERN_CTX_LEN
        channel = 3
        patch_size = self.config.vision_config.patch_size
        temporal_patch_size = self.config.vision_config.temporal_patch_size

        IMAGE_FACTOR = 32
        MIN_PIXELS = 64 * 32 * 32
        MAX_PIXELS = 16384 * 32 * 32
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
            "vision_embeds": {0: "batch_size", 1: "vision_size"},
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
            attention_mask=inputs["attention_mask"],
        )

        inputs["position_ids"] = torch.cat((inputs["position_ids"], pos_ids), dim=0)

        num_chunks = -(input_ids_length // -prefill_seq_len)  # ceil divide without float
        padded_len = num_chunks * prefill_seq_len  # Convert to a multiple of prompt_len

        inputs["position_ids"] = F.pad(
            inputs["position_ids"], pad=(0, padded_len - input_ids_length), mode="constant", value=-1
        )
        return inputs

    def get_inputs_info(self):
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="attention_mask", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="pixel_values", datatype=torch.float32, shape=("batch_size", 3, "image_size", "image_size")),
        ]
