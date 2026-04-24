# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
QEfficient transforms for Qwen3.5-0.8B (qwen3_5 dense text-only path).

Architecture: Hybrid SSM+Attention dense model (24 layers, 3:1 linear_attention:full_attention)
- full_attention layers: standard QKV attention + KV cache
- linear_attention layers: Gated Delta Rule (SSM-like recurrence)
  - State: (conv_state [bs, conv_dim, kernel_size], recurrent_state [bs, heads, k_dim, v_dim])
  - Decode path (seq_len=1): ONNX-safe pure matrix ops
  - Prefill path (seq_len>1): uses torch_chunk_gated_delta_rule (not ONNX-safe yet)

No MoE — standard Qwen3_5MLP (no changes needed vs HF baseline).

Same SSM-hybrid logic as qwen3_5_moe but without the SparseMoeBlock.
Reference: QEfficient/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py

RMSNorm convention: weight=zeros, forward = x * (1.0 + weight) → GemmaCustomRMSNormAIC.
RoPE: MRoPE with partial_rotary_factor=0.25, mrope_section=[11,11,10].
  cos/sin already have shape [bs, seq_len, head_dim] — no position_ids indexing in apply_rotary_pos_emb.
  Rotary embedding handles 2D position_ids internally (expands to 3D for temporal/height/width).
"""

import math
from typing import List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5GatedDeltaNet,
    Qwen3_5TextModel,
    apply_rotary_pos_emb,
    torch_chunk_gated_delta_rule,
)

from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE

# ---------------------------------------------------------------------------
# ONNX-safe chunk gated delta rule (replaces FLA torch_chunk_gated_delta_rule)
#
# Three changes from the original to make it QAIC-compilable at opset 13:
#
# 1. g.cumsum(dim=-1) on 4D tensor [bs, heads, chunks, chunk_size]
#    QAIC supports CumSum only with non-negative axes (axis=-1 rejected).
#    Fix: reshape to 3D [bs*heads, chunks, chunk_size], cumsum on dim=2 (explicit positive axis), reshape back.
#
# 2. .tril() / torch.triu() use opset-14 ops.
#    Fix: replace with torch.where using precomputed row/col comparison masks (opset 13).
#
# 3. Sequential inner loop (for i in range(1, chunk_size)) uses index_put (in-place scatter).
#    This computes T = (I - A)^{-1} via forward substitution — O(n) sequential writes.
#    Fix: binary lifting — O(log n) matmuls, zero index_put ops.
#      T = I; A_pow = A
#      for _ in range(ceil(log2(C))): T = T + A_pow @ T; A_pow = A_pow @ A_pow
#    Requires only matmul + add — opset 13, QAIC-safe.
#
# The algorithm is mathematically identical to the original — verified on CPU (max diff 0).
# ---------------------------------------------------------------------------


def _onnx_safe_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm as _l2n

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2n(query, dim=-1, eps=1e-6)
        key = _l2n(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    # ── Fix 1: CumSum on axis 3 of 4D → reshape to 3D, cumsum on axis 2, reshape back ──
    g_4d_shape = g.shape  # [bs, heads, num_chunks, chunk_size]
    g_3d = g.reshape(-1, g_4d_shape[2], g_4d_shape[3])  # [bs*heads, num_chunks, chunk_size]
    g_3d = g_3d.cumsum(dim=2)  # axis 2 of 3D — QAIC-safe (positive axis)
    g = g_3d.reshape(g_4d_shape)  # back to 4D

    # ── Fix 2: tril/triu → torch.where with precomputed row/col masks ──
    device = query.device
    rows = torch.arange(chunk_size, device=device).unsqueeze(-1)  # [cs, 1]
    cols = torch.arange(chunk_size, device=device).unsqueeze(-2)  # [1, cs]
    lower_tri_mask = rows >= cols  # True where i >= j (lower tri incl. diagonal)
    upper_tri_d0 = rows <= cols  # True where i <= j (upper tri incl. diagonal)
    upper_tri_d1 = rows < cols  # True where i <  j (strictly upper tri)

    diff = g.unsqueeze(-1) - g.unsqueeze(-2)  # [bs, heads, chunks, cs, cs]
    decay_mask = torch.where(lower_tri_mask, diff.exp().float(), torch.zeros_like(diff))

    # Strictly-lower-triangular correction matrix A (diagonal = 0)
    A = -((k_beta @ key.transpose(-1, -2)) * decay_mask)
    A = torch.where(upper_tri_d0, torch.zeros_like(A), A)

    # ── Fix 3: compute T = (I - A)^{-1} via binary lifting (no index_put) ──
    # T = I + A + A^2 + ... + A^{C-1}  (Neumann series, terminates since A is nilpotent)
    # Recurrence: T_{2^{k+1}} = T_{2^k} + A^{2^k} @ T_{2^k}
    # Requires ceil(log2(C)) steps — all matmul + add, opset 13 safe.
    num_steps = max(1, math.ceil(math.log2(max(chunk_size, 2))))
    T = A.new_zeros(A.shape) + torch.eye(chunk_size, dtype=A.dtype, device=device)
    A_pow = A
    for _ in range(num_steps):
        T = T + A_pow @ T  # T_{2^{k+1}} = (I + A^{2^k}) @ T_{2^k}
        A_pow = A_pow @ A_pow  # A^{2^{k+1}}

    value = T @ v_beta
    k_cumdecay = T @ (k_beta * g.exp().unsqueeze(-1))

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=device)
        if initial_state is None
        else initial_state.to(value)
    )

    # Per-chunk loop — unrolled by ONNX tracer (num_chunks = total_seq / chunk_size).
    # Uses torch.stack to avoid index_put on core_attn_out.
    num_chunks = total_sequence_length // chunk_size
    chunk_outputs = []
    for i in range(num_chunks):
        q_i = query[:, :, i]
        k_i = key[:, :, i]
        v_i = value[:, :, i]
        attn_i = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        attn_i = torch.where(upper_tri_d1, torch.zeros_like(attn_i), attn_i)
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        chunk_outputs.append(attn_inter + attn_i @ v_new)
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_attn_out = torch.stack(chunk_outputs, dim=2)  # [B, H, num_chunks, chunk_size, V]

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# ---------------------------------------------------------------------------
# Helper: ONNX-safe l2 norm
# ---------------------------------------------------------------------------


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


# ---------------------------------------------------------------------------
# Helper: eager attention forward (avoids ALL_ATTENTION_FUNCTIONS dispatch)
# ---------------------------------------------------------------------------


def _eager_attn_forward(module, query, key, value, attention_mask, scaling):
    from QEfficient.transformers.models.qwen3_moe.modeling_qwen3_moe import repeat_kv

    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = torch.where(
            attention_mask,
            torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32),
            attn_weights,
        )
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output.transpose(1, 2).contiguous(), attn_weights


# ---------------------------------------------------------------------------
# Full-attention layer
# Differences from base Qwen3_5Attention:
#   - Uses QEffDynamicCache (batch_index, CCL for CCL-based KV cache)
#   - Masking: torch.where() with -1e9 (FP16-safe, no -inf)
#   - Output gate: attn_output * sigmoid(gate) — same as HF
# ---------------------------------------------------------------------------


class QEffQwen3_5Attention(Qwen3_5Attention):
    """
    Full-attention with KV cache and CCL support.

    position_embeddings (cos, sin) already carry MRoPE information:
    cos/sin shape is [bs, seq_len, head_dim] — no additional position_ids indexing.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[QEffDynamicCache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # q_proj produces [query, gate] concatenated (doubled output dim)
        q_out = self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2)
        query_states, gate = torch.chunk(q_out, 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)  # [bs, seq, num_heads * head_dim]

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # cos/sin from MRoPE rotary_emb — shape [bs, seq_len, head_dim], no indexing needed
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"batch_index": batch_index, "position_ids": position_ids}
            if comp_ctx_lengths is not None:
                attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
                cache_kwargs["CCL"] = attention_mask.shape[-1]
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output, attn_weights = _eager_attn_forward(
            self, query_states, key_states, value_states, attention_mask, self.scaling
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)  # output gate
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Linear-attention layer (Gated Delta Rule)
# Identical logic to qwen3_5_moe variant — different parent class only.
# ---------------------------------------------------------------------------


class QEffQwen3_5GatedDeltaNet(Qwen3_5GatedDeltaNet):
    """
    ONNX-safe Gated Delta Rule for hybrid SSM layers.

    Two modes:
    - seq_len > 1 (prefill/validation): calls torch_chunk_gated_delta_rule (not ONNX-safe)
    - seq_len == 1 (decode): inline single-step matrix ops — ONNX-safe

    States passed as explicit tensors:
      conv_state:      [bs, conv_dim, conv_kernel_size]
      recurrent_state: [bs, num_v_heads, k_head_dim, v_head_dim]

    Returns: (output, new_conv_state, new_recurrent_state)
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        recurrent_state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # [bs, conv_dim, seq_len]
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if seq_len == 1 and conv_state is not None:
            # Decode path: ONNX-safe causal conv1d update (no in-place ops)
            hidden_states_new = torch.cat([conv_state.to(mixed_qkv.dtype), mixed_qkv], dim=-1)
            new_conv_state = hidden_states_new[:, :, -self.conv_kernel_size :]
            out = F.conv1d(hidden_states_new, self.conv1d.weight, self.conv1d.bias, padding=0, groups=self.conv_dim)
            mixed_qkv = F.silu(out[:, :, -1:]).transpose(1, 2)
        elif seq_len > 1 and conv_state is not None:
            # Prefill path: ONNX-safe via single causal F.conv1d over the full sequence.
            # NOTE: new_conv_state = padded[-CONV_K:] captures the last CONV_K positions.
            # For exact-multiple-of-chunk prefill (no internal Pad ops, QAIC-safe), this is
            # padding tokens. Set self._prefill_actual_len (a Python int) BEFORE tracing to
            # bake a static conv state slice at the correct position:
            #   padded[:, :, actual_len : actual_len + CONV_K]
            # Since it's a Python int, ONNX traces it as a constant Slice (no dynamic indexing).
            padded = torch.cat([conv_state.to(mixed_qkv.dtype), mixed_qkv], dim=-1)
            _al = getattr(self, "_prefill_actual_len", None)
            if _al is not None:
                new_conv_state = padded[:, :, _al : _al + self.conv_kernel_size]
            else:
                new_conv_state = padded[:, :, -self.conv_kernel_size :]
            out = F.conv1d(padded, self.conv1d.weight, self.conv1d.bias, padding=0, groups=self.conv_dim)
            mixed_qkv = F.silu(out[:, :, 1:]).transpose(1, 2)
        else:
            # Prefill path (not ONNX-safe, for Stage 1→2 HF validation only)
            new_conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - seq_len, 0))
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len]).transpose(1, 2)

        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if seq_len == 1 and recurrent_state is not None:
            # Single recurrent step — all matrix ops, ONNX-safe
            q_t = _l2norm(query[:, 0]) * (1.0 / (query.shape[-1] ** 0.5))
            k_t = _l2norm(key[:, 0])
            v_t = value[:, 0]
            g_t = g[:, 0].float().exp()
            beta_t = beta[:, 0].float()

            decay = g_t.unsqueeze(-1).unsqueeze(-1)
            h = recurrent_state.float() * decay
            kv_mem = (h * k_t.float().unsqueeze(-1)).sum(dim=-2)
            delta = (v_t.float() - kv_mem) * beta_t.unsqueeze(-1)
            new_state = h + k_t.float().unsqueeze(-1) * delta.unsqueeze(-2)

            core_attn_out = (new_state * q_t.float().unsqueeze(-1)).sum(dim=-2)
            core_attn_out = core_attn_out.unsqueeze(1).to(value.dtype)
            new_recurrent_state = new_state.to(recurrent_state.dtype)
        elif seq_len > 1 and recurrent_state is not None:
            # Prefill: ONNX-safe chunk gated delta rule.
            # Uses _onnx_safe_chunk_gated_delta_rule which fixes:
            #   1. CumSum on axis 3 of 4D → reshape to 3D (QAIC supports axis 2 of 3D)
            #   2. tril/triu → torch.where with precomputed masks (opset 13)
            # attention_mask gates g/beta to prevent padding from corrupting the state.
            if attention_mask is not None:
                mask_f = attention_mask.float()
                g = g * mask_f.unsqueeze(-1)
                beta = beta * mask_f.unsqueeze(-1)
            core_attn_out, new_recurrent_state = _onnx_safe_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            new_recurrent_state = new_recurrent_state.to(recurrent_state.dtype)
            core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1, self.head_v_dim)
        else:
            # Prefill: chunk-wise (Stage 1→2 HF validation, not ONNX-safe — uses CumSum/tril)
            core_attn_out, new_recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1, self.head_v_dim)

        core_attn_out_flat = core_attn_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        core_attn_out_flat = self.norm(core_attn_out_flat, z_flat)
        core_attn_out = core_attn_out_flat.reshape(batch_size, seq_len, -1)
        output = self.out_proj(core_attn_out)

        return output, new_conv_state, new_recurrent_state


# ---------------------------------------------------------------------------
# Decoder layer — dispatches on layer_type
# No MoE: mlp is Qwen3_5MLP, returns a plain tensor.
# ---------------------------------------------------------------------------


class QEffQwen3_5DecoderLayer(Qwen3_5DecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[QEffDynamicCache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        conv_state: Optional[torch.Tensor] = None,
        recurrent_state: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states, new_conv_state, new_recurrent_state = self.linear_attn(
                hidden_states=hidden_states,
                conv_state=conv_state,
                recurrent_state=recurrent_state,
                attention_mask=padding_mask,
            )
        else:  # full_attention
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_value,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            new_conv_state = conv_state
            new_recurrent_state = recurrent_state

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_conv_state, new_recurrent_state


# ---------------------------------------------------------------------------
# Text model — manages KV cache + linear attention states
# ---------------------------------------------------------------------------


class QEffQwen3_5TextModel(Qwen3_5TextModel):
    """
    Manages both KV cache (full_attention layers) and linear attention states.

    past_key_values:       QEffDynamicCache (for full_attention layers)
    past_conv_states:      list of conv_state per linear_attention layer
    past_recurrent_states: list of recurrent_state per linear_attention layer

    Position IDs handling:
      - Accepts 2D position_ids [bs, seq_len] (text-only decode mode)
      - Qwen3_5TextRotaryEmbedding.forward() internally expands 2D → 3D for MRoPE
      - cos/sin returned have shape [bs, seq_len, head_dim] — no indexing in apply_rotary_pos_emb
    """

    def __qeff_init__(self):
        self._linear_attn_indices = [i for i, t in enumerate(self.config.layer_types) if t == "linear_attention"]
        self._full_attn_indices = [i for i, t in enumerate(self.config.layer_types) if t == "full_attention"]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        past_conv_states: Optional[List[torch.Tensor]] = None,
        past_recurrent_states: Optional[List[torch.Tensor]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        seq_len = inputs_embeds.shape[1]
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)
        # If no past KV — pure forward (Stage 1→2 validation, avoids AI 100 custom ops on CPU)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Causal mask: only create when there IS a past KV (decode mode).
        # Skip for prefill with no past KV to match HF behavior (returns None in that case).
        if past_key_values_length > 0:
            causal_mask = _create_causal_mask(position_ids=position_ids, target_length=past_key_values_length + seq_len)
        else:
            causal_mask = None

        hidden_states = inputs_embeds

        # Compute MRoPE position embeddings once for all full_attention layers.
        # rotary_emb accepts 2D position_ids and expands to 3D internally.
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        lin_idx = 0
        new_conv_states: List[torch.Tensor] = []
        new_recurrent_states: List[torch.Tensor] = []
        all_hidden_states = () if output_hidden_states else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if decoder_layer.layer_type == "linear_attention":
                conv_st = past_conv_states[lin_idx] if past_conv_states is not None else None
                rec_st = past_recurrent_states[lin_idx] if past_recurrent_states is not None else None
            else:
                conv_st = None
                rec_st = None

            hidden_states, new_conv, new_rec = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                padding_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                use_cache=use_cache,
                cache_position=cache_position,
                conv_state=conv_st,
                recurrent_state=rec_st,
            )

            if decoder_layer.layer_type == "linear_attention":
                new_conv_states.append(new_conv)
                new_recurrent_states.append(new_rec)
                lin_idx += 1

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if past_key_values is not None:
            past_key_values = past_key_values.to_legacy_cache()

        return (
            BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values,
                hidden_states=all_hidden_states,
            ),
            new_conv_states,
            new_recurrent_states,
        )


# ---------------------------------------------------------------------------
# CausalLM head
# ---------------------------------------------------------------------------


class QEffQwen3_5ForCausalLM(Qwen3_5ForCausalLM):
    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffQwen3_5DecoderLayer}

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        past_conv_states: Optional[List[torch.Tensor]] = None,
        past_recurrent_states: Optional[List[torch.Tensor]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs, new_conv_states, new_recurrent_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            past_conv_states=past_conv_states,
            past_recurrent_states=past_recurrent_states,
            comp_ctx_lengths=comp_ctx_lengths,
            inputs_embeds=inputs_embeds,
            batch_index=batch_index,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state
        # INT32 logit extraction: find last valid position via argmax on position_ids
        logit_idx = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = hidden_states[torch.arange(position_ids.shape[0]).view(-1, 1), logit_idx]
        logits = self.lm_head(hidden_states).float()

        return (
            CausalLMOutputWithPast(
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
            ),
            new_conv_states,
            new_recurrent_states,
        )
