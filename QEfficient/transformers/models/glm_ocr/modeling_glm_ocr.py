# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
#
# GLM-OCR (zai-org/GLM-OCR) QEfficient modeling file.
#
# Architecture:
#   Text decoder : GlmOcrTextModel  (16L, h=1536, 16Q/8KV heads, head_dim=128, M-RoPE)
#   Vision enc   : GlmOcrVisionModel (ViT depth=24, h=1024, patch=14, spatial_merge=2)
#   Projector    : GlmOcrVisionPatchMerger  (1024*4 → 1536, out_hidden_size)
#   Image inject : masked_scatter in HF → replaced with torch.where (V4)
#
# Requires transformers 5.0.1dev0+. Import guarded in pytorch_transforms.py (U6).
#
# Reference: QEfficient/transformers/models/qwen3_vl/modeling_qwen3_vl.py
#
# Invariants applied (see onboarding-issues.md#modeling-file-invariants):
#   U1  MIN_MASKED_ATTENTION_VALUE = -1e9 in text eager_attention_forward
#   U2  INT32 gather index for last-token logit slice
#   U3  No boolean indexing in image injection (cumsum + torch.where)
#   U4  KVCacheTransform (native HF, no trust_remote_code)
#   U6  try/except ImportError in pytorch_transforms.py
#   V1  Vision projector FP16 clamp [-60000, 60000] after GlmOcrVisionPatchMerger
#   V2  Vision attention: einsum replaced by permute+matmul (block mask approach)
#   V4  masked_scatter → torch.where cumsum-based merge
#   V5  Vision attn_mask preserved via additive masking before softmax
#   V6  Registered to AutoModelForImageTextToText
#   V7  Negative-gather clamp: indices1.clamp(min=0)
#   V8  QEffDynamicCache.from_legacy_cache at QEffGlmOcrTextModel entry
#   V9  Single-QPC: vision dims included in lang specializations
#   V10 Decode no-op: torch.where(seq_len==1, text_embeds, merged)
#   V11 _create_causal_mask with position_ids[0] (text 1D ids)
#   V12 Runner calls merge_visual_inputs before generate (runner script, not here)
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
    apply_rotary_pos_emb,
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
from QEfficient.utils.logging_utils import logger


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# M-RoPE helpers. GLM-OCR's text decoder uses GLM-style INTERLEAVED rotary
# (contiguous-chunk M-RoPE section select on the half head_dim, then
# repeat_interleave(2) + rotate-adjacent-pairs) — NOT the Qwen-style strided
# M-RoPE + split-half rotation used by qwen3_vl. The two are numerically
# different; copying qwen3_vl's helpers here silently produced wrong logits
# (HF token 'A' vs QEff token ' twice' on real weights) while compiling and
# running without error. See transformers.models.glm_ocr.modeling_glm_ocr's
# apply_rotary_pos_emb / rotate_half_llm for the reference implementation.
# ---------------------------------------------------------------------------

def qeff_apply_glm_mrope(freqs: torch.Tensor, mrope_section: List[int]) -> torch.Tensor:
    """Select each M-RoPE section (T/H/W) from its own frequency chunk, contiguous layout.

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
    # Static gather for adjacent-pair duplication (equivalent to repeat_interleave(2, dim=-1)
    # but on a static axis only — repeat_interleave over a dynamic-length axis exports as
    # SplitToSequence/ConcatFromSequence which the compiler rejects, and repeat_interleave
    # over the static-only last axis nonetheless mis-traces here into an incorrectly-shaped
    # graph). idx = [0,0,1,1,2,2,...] length = head_dim.
    dup_idx = torch.arange(half * 2, device=cos.device) // 2
    cos = cos[..., :half][position_ids]           # (3, batch, seq_len, head_dim // 2)
    sin = sin[..., :half][position_ids]
    cos = qeff_apply_glm_mrope(cos, mrope_section)    # (batch, seq_len, head_dim // 2)
    sin = qeff_apply_glm_mrope(sin, mrope_section)
    cos = cos[..., dup_idx].unsqueeze(1)              # (batch, 1, seq_len, head_dim)
    sin = sin[..., dup_idx].unsqueeze(1)
    return cos, sin


def qeff_apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply GLM-style interleaved M-RoPE to query and key tensors.

    GLM's rotate is pair-swap-with-negate: (x0,x1,x2,x3,...) -> (-x1,x0,-x3,x2,...).
    Rather than express that with stack+flatten (which torch.jit.trace here mis-handles
    when downstream broadcasting depends on the collapsed shape), we implement the same
    pair-swap as a static gather + a static sign vector. Both index buffers are size
    (head_dim,) and shape-static — no dynamic seq_len math is involved.
    """
    device, dtype = q.device, q.dtype
    head_dim = q.shape[-1]
    # swap adjacent pairs: [1,0,3,2,5,4,...]
    swap_idx = torch.arange(head_dim, device=device).view(head_dim // 2, 2).flip(-1).flatten()
    # signs: [-1,+1,-1,+1,...] so pair (x_{2i}, x_{2i+1}) maps to (-x_{2i+1}, x_{2i})
    signs = torch.tensor([-1.0, 1.0], device=device, dtype=dtype).repeat(head_dim // 2)
    q_rot = q[..., swap_idx] * signs
    k_rot = k[..., swap_idx] * signs
    q_embed = (q * cos) + (q_rot * sin)
    k_embed = (k * cos) + (k_rot * sin)
    return q_embed.to(dtype), k_embed.to(dtype)


# ---------------------------------------------------------------------------
# Vision attention  —  ONNX-safe block-diagonal mask (V2, V5)
# ---------------------------------------------------------------------------

class QEffGlmOcrVisionAttention(GlmOcrVisionAttention):
    """Replace the dynamic cu_seqlens-split loop with a static block-diagonal mask.

    HF splits hidden_states into per-image chunks and processes them in a Python
    loop → ONNX NonZero / dynamic slice.  We build a block-diagonal boolean mask
    from cu_seqlens instead (same approach as QEffQwen3VLVisionAttention).
    """

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
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
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
        rows = torch.arange(seq_length, device=q.device).view(1, -1)   # (1, seq_len)
        cols = torch.arange(seq_length, device=q.device).view(-1, 1)   # (seq_len, 1)
        start = cu_seqlens[:-1].view(-1, 1, 1)   # (num_blocks, 1, 1)
        end = cu_seqlens[1:].view(-1, 1, 1)
        row_mask = (rows >= start) & (rows < end)  # (num_blocks, 1, seq_len)
        col_mask = (cols >= start) & (cols < end)  # (num_blocks, seq_len, 1)
        block_mask = row_mask & col_mask           # (num_blocks, seq_len, seq_len)

        # 0 = attend, 1 = mask; convert 1 → finfo.min
        final_mask = torch.ones((seq_length, seq_length), dtype=torch.float32, device=q.device)
        final_mask[block_mask.any(dim=0)] = 0.0
        final_mask = torch.where(final_mask == 1.0, torch.finfo(q.dtype).min, final_mask)
        attention_mask[0] = final_mask

        # Attention: (seq_len, n_heads, head_dim) → transpose for matmul
        q = q.transpose(0, 1)   # (n_heads, seq_len, head_dim)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask   # additive mask (V5)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).reshape(seq_length, self.dim)
        return self.proj(attn_output)


# ---------------------------------------------------------------------------
# Vision model  —  FP16 clamp (V1) and clean return type
# ---------------------------------------------------------------------------

class QEffGlmOcrVisionModel(GlmOcrVisionModel):
    """Static-only forward to avoid Python for-loops that produce ONNX sequence ops.

    HF GlmOcrVisionModel.rot_pos_emb() uses `for t, h, w in grid_thw:` with
    list.append / torch.cat → ONNX SplitToSequence/SequenceEmpty/ConcatFromSequence,
    rejected by QAIC.  We vectorise the same logic using .item() at trace time for
    static Python ints, then pure tensor ops from there.

    Also applies V1 (FP16 clamp) after the patch-merger output.
    """

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        # grid_thw: (num_images, 3) — each row is [temporal, height_patches, width_patches]
        # At ONNX trace time the grid is static, so .item() gives Python constants.
        h_val = grid_thw[0, 1].item()
        w_val = grid_thw[0, 2].item()
        t_val = grid_thw[0, 0].item()
        ms = self.spatial_merge_size   # int, already a Python constant

        device = hidden_states.device

        # ---- Patch embedding ----
        hidden_states = self.patch_embed(hidden_states)  # (h*w*t, hidden_size)

        # ---- Rotary position IDs (vectorised) — mirrors GlmOcrVisionModel.rot_pos_emb ----
        hpos_ids = (
            torch.arange(h_val, device=device)
            .unsqueeze(1).expand(h_val, w_val)
            .reshape(h_val // ms, ms, w_val // ms, ms)
            .permute(0, 2, 1, 3).flatten()
        )
        wpos_ids = (
            torch.arange(w_val, device=device)
            .unsqueeze(0).expand(h_val, w_val)
            .reshape(h_val // ms, ms, w_val // ms, ms)
            .permute(0, 2, 1, 3).flatten()
        )
        pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1)  # (h*w, 2)
        if t_val > 1:
            pos_ids = pos_ids.repeat(t_val, 1)

        # ---- Rotary embeddings (from GlmOcrVisionRotaryEmbedding) ----
        max_grid_size = max(h_val, w_val)
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)   # (max_hw, head_dim/2)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)    # (h*w*t, head_dim)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # ---- cu_seqlens (static for fixed grid) ----
        seq_len = h_val * w_val * t_val
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

        # ---- ViT blocks (uses QEffGlmOcrVisionAttention via KVCacheTransform) ----
        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)

        # ---- Post-norm + spatial merge (mirror HF GlmOcrVisionModel.forward) ----
        hidden_states = self.post_layernorm(hidden_states)  # (h*w*t, hidden_size)

        # Reshape for Conv2d downsampler: (N_merged, ms, ms, C) → (N_merged, C, ms, ms)
        n_merged = seq_len // (ms * ms)
        hidden_states = hidden_states.view(n_merged, ms, ms, hidden_states.shape[-1]).permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).view(n_merged, self.config.out_hidden_size)

        # Final merger projection
        image_embeds = self.merger(hidden_states)

        # V1: FP16 overflow clamp
        image_embeds = image_embeds.clamp(-60000, 60000)
        return image_embeds


# ---------------------------------------------------------------------------
# Text decoder: RotaryEmbedding — precomputed sin/cos cache
# ---------------------------------------------------------------------------

class QEffGlmOcrTextRotaryEmbedding(GlmOcrTextRotaryEmbedding):
    """Precompute sin/cos at init so torch.jit.trace sees static buffers."""

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


# ---------------------------------------------------------------------------
# Text eager attention forward (U1: -1e9 not -inf)
# ---------------------------------------------------------------------------

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
        # U1: use -1e9, not -inf (FP16 cannot represent -inf safely)
        attn_weights = torch.where(
            attention_mask,
            torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=query.dtype),
            attn_weights,
        )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Text attention
# ---------------------------------------------------------------------------

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
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

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


# ---------------------------------------------------------------------------
# Text decoder layer  —  pass-through for QEff parameters
# ---------------------------------------------------------------------------

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
        # GLM sandwich-norm: post_self_attn_layernorm before the residual add (missing here
        # dropped ~40 units of hidden-state magnitude vs HF and broke real-weight generation)
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


# ---------------------------------------------------------------------------
# Text model  —  QEffDynamicCache + precomputed M-RoPE (V8, V11)
# ---------------------------------------------------------------------------

class QEffGlmOcrTextModel(GlmOcrTextModel):
    def __qeff_init__(self):
        """Called by ModuleMappingTransform after class replacement."""
        self.rotary_emb = QEffGlmOcrTextRotaryEmbedding(config=self.config)
        attention_scaling = getattr(self.rotary_emb, "attention_scaling", 1.0)
        self.cos_cached = nn.Parameter(
            self.rotary_emb.cos_cached * attention_scaling, requires_grad=False
        )
        self.sin_cached = nn.Parameter(
            self.rotary_emb.sin_cached * attention_scaling, requires_grad=False
        )

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

        # V8: legacy cache conversion at model entry
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

        # Extract text position IDs for causal mask (V11)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]   # (batch, seq_len)  — 1D text positions
            mrope_position_ids = position_ids[1:]  # (3, batch, seq_len) — T/H/W M-RoPE
        else:
            text_position_ids = position_ids if position_ids.ndim == 2 else position_ids[0]
            mrope_position_ids = position_ids[1:] if position_ids.ndim == 3 else position_ids.unsqueeze(0).expand(3, -1, -1)

        target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens
        # V11: causal mask uses 1D text position IDs
        causal_mask = _create_causal_mask(
            position_ids=text_position_ids,
            target_length=target_length,
            sliding_window=None,
        )

        hidden_states = inputs_embeds

        # Precomputed M-RoPE (faster than re-computing per layer)
        mrope_section = self.config.rope_parameters["mrope_section"]
        cos, sin = qeff_prepare_mrope_cos_sin(
            self.cos_cached, self.sin_cached, mrope_position_ids, mrope_section
        )

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

        # V8: convert back to legacy cache for ONNX/JIT
        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# ---------------------------------------------------------------------------
# GlmOcrModel wrapper  —  thin subclass so KVCacheTransform has an entry
# ---------------------------------------------------------------------------

class QEffGlmOcrModel(GlmOcrModel):
    """Thin wrapper; actual QEff logic is in sub-modules (TextModel, VisionModel)."""

    pass


# ---------------------------------------------------------------------------
# Encoder wrapper  (for kv_offload=True dual-QPC)
# ---------------------------------------------------------------------------

class QEffGlmOcrEncoderWrapper(nn.Module):
    def __init__(self, model: "QEffGlmOcrForConditionalGeneration"):
        super().__init__()
        self.model = model
        # Expose as vision_model attribute for get_submodules_for_export compatibility
        self.model.vision_model = self.model.model.visual

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {self.model.model.visual.blocks[0].__class__}

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        image_embeds = self.model.model.visual(pixel_values, image_grid_thw)
        # image_embeds shape: (total_merged_tokens, out_hidden_size)
        bs = image_grid_thw.shape[0]
        split_size = image_embeds.shape[0] // bs
        image_embeds = image_embeds.reshape(bs, split_size, image_embeds.shape[1])
        return image_embeds


# ---------------------------------------------------------------------------
# Decoder wrapper  (for kv_offload=True dual-QPC)
# ---------------------------------------------------------------------------

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

        # V4: replace masked_scatter with static torch.where + cumsum indexing
        selected = input_ids == self.model.config.image_token_id
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(B, device=input_ids.device).view(-1, 1)
        # V7: clamp to avoid negative ONNX gather indices
        image_features_expanded = vision_embeds.reshape(vision_embeds.shape[0], C).unsqueeze(0)[
            indices0, indices1.clamp(min=0)
        ]
        image_input_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
        # V10: decode no-op
        inputs_embeds = torch.where(
            input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_input_embeds
        )

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=True,
        )

        # U2: INT32 logit index
        logit_index = position_ids[0].to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[
            torch.arange(B, device=input_ids.device).view(-1, 1), logit_index
        ]
        logits = self.model.lm_head(hidden_states)
        image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        return logits, vision_embeds, image_idx, outputs.past_key_values


# ---------------------------------------------------------------------------
# Top-level: QEffGlmOcrForConditionalGeneration
# ---------------------------------------------------------------------------

# Default dims for dummy input construction (336×336 image, patch_size=14)
_BS = 1
_FBS = 4
_SEQ_LEN = 592
_CTX_LEN = 1024
_VISION_SIZE = 144    # (336//14)^2 // (spatial_merge_size=2)^2 = 576 // 4
_GRID_HEIGHT = 576    # 24 * 24 patches per image
_GRID_WIDTH = 1176    # 3 * temporal_patch_size=2 * patch_size=14 * patch_size=14


class QEffGlmOcrForConditionalGeneration(GlmOcrForConditionalGeneration):
    """QEff wrapper for GLM-OCR.

    Single-QPC (kv_offload=False): combined vision+language ONNX graph.
    Dual-QPC  (kv_offload=True):   separate vision/language sessions via
                                    get_qeff_vision_encoder / get_qeff_language_decoder.
    """

    # V6: register to AutoModelForImageTextToText (done via pytorch_transforms)

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
        # --- Vision encoding ---
        # image_embeds shape: (total_merged_tokens, out_hidden_size=1536)
        image_embeds = self.model.visual(pixel_values, image_grid_thw)

        # --- Text embedding ---
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        B, N, C = inputs_embeds.shape

        # V4: replace masked_scatter with static torch.where + cumsum merge
        selected = input_ids == self.config.image_token_id
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(B, device=input_ids.device).view(-1, 1)
        # V7: clamp negative indices
        image_features_expanded = image_embeds.reshape(image_embeds.shape[0], C).unsqueeze(0)[
            indices0, indices1.clamp(min=0)
        ]
        image_input_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
        # V10: decode no-op — when seq_len==1 the vision encoder output is not needed
        inputs_embeds = torch.where(
            input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_input_embeds
        )

        # --- Language model ---
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=True,
        )

        # --- Logit extraction (U2: INT32 index) ---
        logit_index = position_ids[0].to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[
            torch.arange(B, device=input_ids.device).view(-1, 1), logit_index
        ]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        next_image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        image_idx = torch.where(image_idx < next_image_idx, next_image_idx, image_idx)
        # Note: pixel_values is NOT returned as a retained state — the compiler DCEs the
        # vision encoder in the decode spec (unused output), which would make
        # pixel_values_RetainedState inconsistent between specs.  The runtime re-uploads
        # pixel_values from host on every decode call (slightly less efficient, but correct).
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
        grid_h = image_size // patch_size   # 24
        grid_w = image_size // patch_size   # 24
        n_patches = grid_h * grid_w          # 576
        vision_size = n_patches // (spatial_merge_size ** 2)  # 144
        grid_width = 3 * temporal_patch_size * patch_size * patch_size  # 1176

        # Text config
        text_cfg = self.config.text_config
        num_layers = text_cfg.num_hidden_layers
        num_kv_heads = text_cfg.num_key_value_heads
        head_dim = text_cfg.head_dim

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
        vision_size = n_patches // (spatial_merge_size ** 2)
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
            # V9: vision dims must be included in every lang spec so the
            # compiler can resolve ONNX symbolic dims from the vision graph.
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
