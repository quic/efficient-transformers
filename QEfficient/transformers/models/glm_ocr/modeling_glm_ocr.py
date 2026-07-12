# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import math
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.glm_ocr.modeling_glm_ocr import (
    GlmOcrForConditionalGeneration,
    GlmOcrModel,
    GlmOcrTextAttention,
    GlmOcrTextDecoderLayer,
    GlmOcrTextModel,
    GlmOcrTextRotaryEmbedding,
    GlmOcrVisionAttention,
    GlmOcrVisionModel,
    apply_rotary_pos_emb_vision,
    repeat_kv,
)

from QEfficient.blocking.attention_blocking import (
    AttentionBlockingConfig,
    BlockingMode,
    generic_blocked_attention_interface,
    past_key_value_update,
)
from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo, get_padding_shape_from_config
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


def qeff_apply_glm_mrope(freqs: torch.Tensor, mrope_section: List[int]) -> torch.Tensor:
    """Select each M-RoPE section (T/H/W) from its own frequency chunk.

    Args:
        freqs: (3, bs, seq_len, head_dim // 2)  — temporal/height/width stacked
        mrope_section: list of 3 ints [t_size, h_size, w_size] summing to head_dim//2
    Returns:
        (bs, seq_len, head_dim // 2)
    """
    chunks = freqs.split(mrope_section, dim=-1)
    return torch.cat([chunk[i % 3] for i, chunk in enumerate(chunks)], dim=-1)


def qeff_prepare_mrope_cos_sin(
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    mrope_section: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Index precomputed cos/sin by M-RoPE position_ids and apply GLM's interleaved layout.

    Args:
        cos, sin: (max_seq_len, head_dim)  — from QEffGlmOcrTextModel.cos_cached / sin_cached
        position_ids: (3, batch, seq_len)  — temporal/height/width positions
        mrope_section: [t, h, w] sizes (each in half-head_dim units)
    Returns:
        cos, sin: (batch, 1, seq_len, head_dim)  — ready for q/k multiplication
    """
    half = cos.shape[-1] // 2
    # Static gather for adjacent-pair duplication; repeat_interleave over this axis
    # mis-traces under this fork's ONNX export path.
    dup_idx = torch.arange(half * 2, device=cos.device) // 2
    cos = cos[..., :half][position_ids]
    sin = sin[..., :half][position_ids]
    cos = qeff_apply_glm_mrope(cos, mrope_section)
    sin = qeff_apply_glm_mrope(sin, mrope_section)
    cos = cos[..., dup_idx].unsqueeze(1)
    sin = sin[..., dup_idx].unsqueeze(1)
    return cos, sin


def qeff_apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply GLM's interleaved-pair M-RoPE rotation to query and key tensors.

    Implemented as a static gather + sign vector rather than stack/flatten;
    the latter mis-traces under this fork's ONNX export path.
    """
    device, dtype = q.device, q.dtype
    head_dim = q.shape[-1]
    swap_idx = torch.arange(head_dim, device=device).view(head_dim // 2, 2).flip(-1).flatten()
    signs = torch.tensor([-1.0, 1.0], device=device, dtype=dtype).repeat(head_dim // 2)
    q_rot = q[..., swap_idx] * signs
    k_rot = k[..., swap_idx] * signs
    q_embed = (q * cos) + (q_rot * sin)
    k_embed = (k * cos) + (k_rot * sin)
    return q_embed.to(dtype), k_embed.to(dtype)


class QEffGlmOcrVisionAttention(GlmOcrVisionAttention):
    """Replaces the dynamic cu_seqlens-split loop with a static block-diagonal mask."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]

        # QKV projection + reshape to (seq_len, n_heads, head_dim)
        q, k, v = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3).unbind(0)
        )

        # Q/K-norm (vision attention has per-head RMSNorm; text does not)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Vision RoPE
        if position_embeddings is None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Build static block-diagonal attention mask from cu_seqlens
        # Positions within the same image attend to each other; cross-image is masked.
        attention_mask = torch.full(
            [1, seq_length, seq_length],
            torch.finfo(q.dtype).min,
            device=q.device,
            dtype=q.dtype,
        )
        rows = torch.arange(seq_length, device=q.device).view(1, -1)  # (1, seq_len)
        cols = torch.arange(seq_length, device=q.device).view(-1, 1)  # (seq_len, 1)
        start = cu_seqlens[:-1].view(-1, 1, 1)  # (num_blocks, 1, 1)
        end = cu_seqlens[1:].view(-1, 1, 1)
        row_mask = (rows >= start) & (rows < end)  # (num_blocks, 1, seq_len)
        col_mask = (cols >= start) & (cols < end)  # (num_blocks, seq_len, 1)
        block_mask = row_mask & col_mask  # (num_blocks, seq_len, seq_len)

        # 0 = attend, 1 = mask; convert 1 → finfo.min
        final_mask = torch.ones((seq_length, seq_length), dtype=torch.float32, device=q.device)
        final_mask[block_mask.any(dim=0)] = 0.0
        final_mask = torch.where(final_mask == 1.0, torch.finfo(q.dtype).min, final_mask)
        attention_mask[0] = final_mask

        # Attention: (seq_len, n_heads, head_dim) → transpose for matmul
        q = q.transpose(0, 1)  # (n_heads, seq_len, head_dim)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).reshape(seq_length, self.dim)
        return self.proj(attn_output)


class QEffGlmOcrVisionModel(GlmOcrVisionModel):
    """Vectorises HF's Python-loop rot_pos_emb, which traces to ONNX sequence ops QAIC rejects."""

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        # grid_thw: (num_images, 3) — [temporal, height_patches, width_patches].
        # The grid is static at trace time, so .item() gives Python constants.
        h_val = grid_thw[0, 1].item()
        w_val = grid_thw[0, 2].item()
        t_val = grid_thw[0, 0].item()
        ms = self.spatial_merge_size

        device = hidden_states.device

        hidden_states = self.patch_embed(hidden_states)  # (h*w*t, hidden_size)

        # ---- Rotary position IDs (vectorised) — mirrors GlmOcrVisionModel.rot_pos_emb ----
        hpos_ids = (
            torch.arange(h_val, device=device)
            .unsqueeze(1)
            .expand(h_val, w_val)
            .reshape(h_val // ms, ms, w_val // ms, ms)
            .permute(0, 2, 1, 3)
            .flatten()
        )
        wpos_ids = (
            torch.arange(w_val, device=device)
            .unsqueeze(0)
            .expand(h_val, w_val)
            .reshape(h_val // ms, ms, w_val // ms, ms)
            .permute(0, 2, 1, 3)
            .flatten()
        )
        pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1)  # (h*w, 2)
        if t_val > 1:
            pos_ids = pos_ids.repeat(t_val, 1)

        max_grid_size = max(h_val, w_val)
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)  # (max_hw, head_dim/2)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)  # (h*w*t, head_dim)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        seq_len = h_val * w_val * t_val
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)

        hidden_states = self.post_layernorm(hidden_states)  # (h*w*t, hidden_size)

        # Reshape for Conv2d downsampler: (N_merged, ms, ms, C) → (N_merged, C, ms, ms)
        n_merged = seq_len // (ms * ms)
        hidden_states = hidden_states.view(n_merged, ms, ms, hidden_states.shape[-1]).permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).view(n_merged, self.config.out_hidden_size)

        image_embeds = self.merger(hidden_states)

        # FP16 overflow clamp
        image_embeds = image_embeds.clamp(-60000, 60000)
        return image_embeds


class QEffGlmOcrTextRotaryEmbedding(GlmOcrTextRotaryEmbedding):
    """Precomputes sin/cos at init so torch.jit.trace sees static buffers."""

    def __init__(self, config, device=None):
        super().__init__(config=config)
        self._set_cos_sin_cache(
            seq_len=self.original_max_seq_len,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


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
) -> Tuple[torch.Tensor, torch.Tensor]:
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) / math.sqrt(module.head_dim)
    if attention_mask is not None:
        # -1e9, not -inf: FP16 cannot represent -inf safely.
        attn_weights = torch.where(
            attention_mask,
            torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=query.dtype),
            attn_weights,
        )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class QEffGlmOcrTextAttention(GlmOcrTextAttention):
    """KV cache + precomputed M-RoPE. No q/k-norm (text attention only has o_proj + QKV)."""

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
        sin_cached: Optional[torch.Tensor] = None,
        cos_cached: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Q, K, V projections  (no q_norm/k_norm on text attention)
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = (
            self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        )

        # Apply precomputed M-RoPE (passed from QEffGlmOcrTextModel)
        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos_cached, sin_cached)

        blocking_config = getattr(self, "attn_blocking_config", AttentionBlockingConfig())
        use_blocking = blocking_config is not None and (blocking_config.mode != BlockingMode.NONE)

        if use_blocking:
            past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
            attn_output, attn_weights = generic_blocked_attention_interface(
                module=self,
                query=query_states,
                key=key_states,
                value=value_states,
                attention_mask=attention_mask,
                scaling=self.scaling,
                layer_idx=self.layer_idx,
                past_key_value=past_key_values,
                blocking_config=blocking_config,
                comp_ctx_length=comp_ctx_lengths,
                batch_index=batch_index,
                position_ids=position_ids[0],
                past_seen_tokens=past_seen_tokens,
            )
        else:
            key_states, value_states, attention_mask, _ = past_key_value_update(
                module=self,
                key=key_states,
                value=value_states,
                attention_mask=attention_mask,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                position_ids=position_ids[0],
            )
            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
            )

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_values


class QEffGlmOcrTextDecoderLayer(GlmOcrTextDecoderLayer):
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
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        sin_cached: Optional[torch.Tensor] = None,
        cos_cached: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

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
            sin_cached=sin_cached,
            cos_cached=cos_cached,
        )
        # GLM sandwich-norm: applied before the residual add, unlike the standard
        # input/post-attention-only layernorm pair.
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class QEffGlmOcrTextModel(GlmOcrTextModel):
    def __qeff_init__(self):
        """Called by ModuleMappingTransform after class replacement."""
        self.rotary_emb = QEffGlmOcrTextRotaryEmbedding(config=self.config)
        attention_scaling = getattr(self.rotary_emb, "attention_scaling", 1.0)
        self.cos_cached = nn.Parameter(self.rotary_emb.cos_cached * attention_scaling, requires_grad=False)
        self.sin_cached = nn.Parameter(self.rotary_emb.sin_cached * attention_scaling, requires_grad=False)

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
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        return_legacy_cache = False
        if self.config.use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        # GLM-OCR position_ids: (4, batch, seq_len)
        #   [0] = text 1D positions for causal mask
        #   [1:] = 3D M-RoPE (T, H, W) positions
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]  # (batch, seq_len)  — 1D text positions
            mrope_position_ids = position_ids[1:]  # (3, batch, seq_len) — T/H/W M-RoPE
        else:
            text_position_ids = position_ids if position_ids.ndim == 2 else position_ids[0]
            mrope_position_ids = (
                position_ids[1:] if position_ids.ndim == 3 else position_ids.unsqueeze(0).expand(3, -1, -1)
            )

        target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens
        causal_mask = _create_causal_mask(
            position_ids=text_position_ids,
            target_length=target_length,
            sliding_window=None,
        )

        hidden_states = inputs_embeds

        # Precomputed M-RoPE (faster than re-computing per layer)
        mrope_section = self.config.rope_parameters["mrope_section"]
        cos, sin = qeff_prepare_mrope_cos_sin(self.cos_cached, self.sin_cached, mrope_position_ids, mrope_section)

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
                sin_cached=sin,
                cos_cached=cos,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

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


class QEffGlmOcrModel(GlmOcrModel):
    """Thin wrapper; QEff logic lives in the text/vision sub-modules."""


class QEffGlmOcrEncoderWrapper(nn.Module):
    def __init__(self, model: "QEffGlmOcrForConditionalGeneration"):
        super().__init__()
        self.model = model
        self.model.vision_model = self.model.model.visual

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {self.model.model.visual.blocks[0].__class__}

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        image_embeds = self.model.model.visual(pixel_values, image_grid_thw)
        bs = image_grid_thw.shape[0]
        split_size = image_embeds.shape[0] // bs
        image_embeds = image_embeds.reshape(bs, split_size, image_embeds.shape[1])
        return image_embeds


class QEffGlmOcrDecoderWrapper(nn.Module):
    def __init__(self, model: "QEffGlmOcrForConditionalGeneration"):
        super().__init__()
        self.model = model
        self.language_model = model.model.language_model

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffGlmOcrTextDecoderLayer}

    def forward(
        self,
        input_ids: torch.LongTensor,
        vision_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        image_idx: torch.LongTensor,
        past_key_values,
        batch_index: Optional[torch.LongTensor] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
    ):
        inputs_embeds = self.model.model.get_input_embeddings()(input_ids)
        B, N, C = inputs_embeds.shape

        # masked_scatter is ONNX-unfriendly; replace with cumsum-indexed gather.
        selected = input_ids == self.model.config.image_token_id
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(B, device=input_ids.device).view(-1, 1)
        # vision_embeds: (vision_batch_size, vision_size, C) -> (total_tokens, C)
        total_tokens = vision_embeds.shape[0] * vision_embeds.shape[1]
        image_features_expanded = vision_embeds.reshape(total_tokens, C).unsqueeze(0)[indices0, indices1.clamp(min=0)]
        image_input_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
        # Decode step (seq_len == 1) has no image tokens; skip the merge.
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_input_embeds)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=True,
        )

        logit_index = position_ids[0].to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[torch.arange(B, device=input_ids.device).view(-1, 1), logit_index]
        logits = self.model.lm_head(hidden_states)
        image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        return logits, vision_embeds, image_idx, outputs.past_key_values


_SEQ_LEN = 592
_CTX_LEN = 1024


class QEffGlmOcrForConditionalGeneration(GlmOcrForConditionalGeneration):
    """QEff wrapper for GLM-OCR.

    Single-QPC (kv_offload=False): combined vision+language ONNX graph.
    Dual-QPC  (kv_offload=True):   separate vision/language sessions via
                                    get_qeff_vision_encoder / get_qeff_language_decoder.
    """

    def get_qeff_vision_encoder(self) -> QEffGlmOcrEncoderWrapper:
        return QEffGlmOcrEncoderWrapper(self)

    def get_qeff_language_decoder(self) -> QEffGlmOcrDecoderWrapper:
        return QEffGlmOcrDecoderWrapper(self)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        past_key_values,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_idx: Optional[torch.LongTensor] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ):
        image_embeds = self.model.visual(pixel_values, image_grid_thw)

        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        B, N, C = inputs_embeds.shape

        # masked_scatter is ONNX-unfriendly; replace with cumsum-indexed gather.
        selected = input_ids == self.config.image_token_id
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(B, device=input_ids.device).view(-1, 1)
        image_features_expanded = image_embeds.reshape(image_embeds.shape[0], C).unsqueeze(0)[
            indices0, indices1.clamp(min=0)
        ]
        image_input_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
        # Decode step (seq_len == 1) has no image tokens; skip the merge.
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_input_embeds)

        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=True,
        )

        logit_index = position_ids[0].to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[torch.arange(B, device=input_ids.device).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        next_image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        image_idx = torch.where(image_idx < next_image_idx, next_image_idx, image_idx)
        # pixel_values is not a retained state: the compiler DCEs the vision encoder in
        # the decode spec, so pixel_values is re-uploaded from host on every decode call.
        return logits, image_idx, outputs.past_key_values

    # ------------------------------------------------------------------
    # QEff interface: dummy inputs, specializations, dynamic axes, etc.
    # ------------------------------------------------------------------

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        **kwargs,
    ) -> Dict:
        bs: int = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        fbs: int = constants.ONNX_EXPORT_EXAMPLE_FBS

        # Vision config
        vision_cfg = self.config.vision_config
        patch_size = vision_cfg.patch_size
        temporal_patch_size = vision_cfg.temporal_patch_size
        spatial_merge_size = vision_cfg.spatial_merge_size
        image_size = vision_cfg.image_size
        grid_h = image_size // patch_size  # 24
        grid_w = image_size // patch_size  # 24
        n_patches = grid_h * grid_w  # 576
        vision_size = n_patches // (spatial_merge_size**2)  # 144
        grid_width = 3 * temporal_patch_size * patch_size * patch_size  # 1176

        # Text config
        text_cfg = self.config.text_config
        num_layers = text_cfg.num_hidden_layers

        vision_inputs = {
            "pixel_values": torch.zeros((n_patches, grid_width), dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, grid_h, grid_w]], dtype=torch.int64),  # (batch, 3): [t, h, w]
        }

        kv_shape = get_padding_shape_from_config(
            config=text_cfg,
            batch_size=fbs if continuous_batching else bs,
            seq_len=constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN,
        )
        past_kv = [[torch.zeros(kv_shape, dtype=torch.float32) for _ in range(2)] for _ in range(num_layers)]

        lang_inputs = {
            "input_ids": torch.zeros((bs, constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN), dtype=torch.int64),
            "vision_embeds": torch.zeros((bs, vision_size, vision_cfg.out_hidden_size), dtype=torch.float32),
            # position_ids: (4, batch, seq_len)  — first dim is text 1D, rest are M-RoPE
            "position_ids": (
                torch.arange(constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN, dtype=torch.int64)
                .view(1, constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)
                .repeat(bs, 1)
                .unsqueeze(0)
                .repeat(4, 1, 1)
            ),
            "image_idx": torch.zeros((1, 1), dtype=torch.int64),
            "past_key_values": past_kv,
        }

        if continuous_batching:
            lang_inputs["batch_index"] = torch.arange(bs).view(bs, 1)

        if comp_ctx_lengths is not None:
            lang_inputs["comp_ctx_lengths"] = torch.randint(0, 100, (40,), dtype=torch.int8)

        if kv_offload:
            return {"vision": vision_inputs, "lang": lang_inputs}
        else:
            lang_inputs.pop("vision_embeds")
            return {**vision_inputs, **lang_inputs}

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        img_size: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
        **compiler_options,
    ):
        comp_ctx_lengths_prefill = compiler_options.pop("comp_ctx_lengths_prefill", None)
        comp_ctx_lengths_decode = compiler_options.pop("comp_ctx_lengths_decode", None)

        vision_cfg = self.config.vision_config
        patch_size = vision_cfg.patch_size
        temporal_patch_size = vision_cfg.temporal_patch_size
        spatial_merge_size = vision_cfg.spatial_merge_size
        image_size = vision_cfg.image_size if img_size is None else img_size

        grid_h = image_size // patch_size
        grid_w = image_size // patch_size
        n_patches = grid_h * grid_w
        vision_size = n_patches // (spatial_merge_size**2)
        grid_height = n_patches * batch_size
        grid_width = 3 * temporal_patch_size * patch_size * patch_size

        prefill_seq_len = prefill_seq_len or _SEQ_LEN
        ctx_len = ctx_len or _CTX_LEN

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

        def _lang_prefill(extra=None):
            spec = {
                "batch_size": 1 if continuous_batching else batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "vision_size": vision_size,
                "vision_batch_size": batch_size,
                "grid_height": grid_height,
                "grid_width": grid_width,
            }
            if continuous_batching:
                spec["full_batch_size"] = kv_cache_batch_size
            else:
                spec["batch_size"] = kv_cache_batch_size
            if full_batch_size:
                spec["full_batch_exec_size"] = full_batch_size
            if extra:
                spec.update(extra)
            return spec

        def _lang_decode(extra=None):
            spec = {
                "batch_size": full_batch_size if continuous_batching else batch_size,
                "seq_len": 1,
                "ctx_len": ctx_len,
                "vision_size": vision_size,
                "vision_batch_size": batch_size,
                "grid_height": grid_height,
                "grid_width": grid_width,
            }
            if continuous_batching:
                spec["full_batch_size"] = kv_cache_batch_size
            else:
                spec["batch_size"] = kv_cache_batch_size
            if extra:
                spec.update(extra)
            return spec

        if comp_ctx_lengths_prefill is not None:
            lang = []
            for ccl in comp_ctx_lengths_prefill:
                lang.append(_lang_prefill({"comp_ctx_lengths": ccl}))
            for ccl in comp_ctx_lengths_decode:
                lang.append(_lang_decode({"comp_ctx_lengths": ccl}))
        else:
            lang = [_lang_prefill(), _lang_decode()]

        if kv_offload:
            return {"vision": vision, "lang": lang}, compiler_options
        else:
            # Vision dims must be included in every lang spec so the compiler
            # can resolve ONNX symbolic dims from the vision graph.
            return lang, compiler_options

    def get_onnx_dynamic_axes(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
    ) -> Dict:
        num_layers = self.config.text_config.num_hidden_layers

        vision_dynamic_axes = {
            "pixel_values": {0: "grid_height", 1: "grid_width"},
            "image_grid_thw": {0: "batch_size"},  # (batch, 3): [t, h, w] per image
        }

        lang_dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {1: "batch_size", 2: "seq_len"},
            "vision_embeds": {0: "vision_batch_size", 1: "vision_size"},
        }

        if continuous_batching:
            lang_dynamic_axes["batch_index"] = {0: "batch_size"}

        for i in range(num_layers):
            batch_dim = "full_batch_size" if continuous_batching else "batch_size"
            lang_dynamic_axes[f"past_key.{i}"] = {0: batch_dim, 2: "ctx_len"}
            lang_dynamic_axes[f"past_value.{i}"] = {0: batch_dim, 2: "ctx_len"}

        if comp_ctx_lengths is not None:
            lang_dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}

        if kv_offload:
            return {"vision": vision_dynamic_axes, "lang": lang_dynamic_axes}
        else:
            merged = {**vision_dynamic_axes, **lang_dynamic_axes}
            merged.pop("vision_embeds", None)
            # pixel_values and image_grid_thw shapes are baked as ONNX constants
            # (via .item() in QEffGlmOcrVisionModel.forward) — removing them from
            # dynamic axes prevents the "Inconsistent retained state" compiler error.
            merged.pop("pixel_values", None)
            merged.pop("image_grid_thw", None)
            return merged

    def get_output_names(self, kv_offload: bool = False) -> Union[List[str], Dict[str, List[str]]]:
        vision_outputs = ["vision_embeds"]
        lang_outputs = ["logits"]
        for i in range(self.config.text_config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_outputs.append(f"past_{kv}.{i}_RetainedState")

        if kv_offload:
            lang_outputs.insert(1, "vision_embeds_RetainedState")
            lang_outputs.insert(2, "image_idx_output")
            return {"vision": vision_outputs, "lang": lang_outputs}
        else:
            # Single-QPC: no pixel_values_RetainedState (compiler DCE causes inconsistency)
            lang_outputs.insert(1, "image_idx_output")
            return lang_outputs

    def prepare_inputs_for_generation(self, inputs, prefill_seq_len=128, batch_size=1):
        input_ids_length = inputs["input_ids"].shape[1]
        text_position_ids = torch.arange(input_ids_length).view(1, 1, input_ids_length).expand(-1, batch_size, -1)

        mm_token_type_ids = inputs.get("mm_token_type_ids")
        if mm_token_type_ids is None:
            mm_token_type_ids = torch.zeros_like(inputs["input_ids"], dtype=torch.int32)
            mm_token_type_ids = mm_token_type_ids.masked_fill(inputs["input_ids"] == self.config.image_token_id, 1)

        mrope_position_ids, rope_deltas = self.model.get_rope_index(
            input_ids=inputs["input_ids"],
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=None,
            attention_mask=inputs["attention_mask"],
        )
        self.model.rope_deltas = rope_deltas
        inputs["position_ids"] = torch.cat((text_position_ids, mrope_position_ids), dim=0)

        num_chunks = -(input_ids_length // -prefill_seq_len)
        padded_len = num_chunks * prefill_seq_len
        inputs["position_ids"] = F.pad(
            inputs["position_ids"], pad=(0, padded_len - input_ids_length), mode="constant", value=-1
        )
        return inputs

    def get_inputs_info(self) -> List[IOInfo]:
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="attention_mask", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(
                name="pixel_values",
                datatype=torch.float32,
                shape=("grid_height", "grid_width"),
            ),
        ]
