# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import math
import os
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from QEfficient.blocking.attention_blocking import (
    AttentionBlockingConfig,
    generic_blocked_attention_interface,
    generic_blocked_mla_attention_interface,
)
from QEfficient.customop.ctx_scatter_gather import (
    CtxGatherFunc3DGeneralized,
    CtxScatterFunc3DGeneralized,
    CtxScatterFunc3DInt,
)
from QEfficient.customop.rms_norm import CustomRMSNormFunc
from QEfficient.customop.matmulnbits import QMOE, QuantLinearTorchFunction
from QEfficient.customop.quantization_ops import CastToUInt4Func, DequantizeLinearFunc
from QEfficient.transformers.cache_utils import QEffDynamicCache, QEffDynamicCompressedKVRopeCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils.constants import MAX_POSITION_EMBEDDINGS, MIN_MASKED_ATTENTION_VALUE


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


# Find dim range bounds based on rotations
def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class QEffDeepseekV3CustomRMSNormAIC(nn.Module):
    """
    RMSNorm module that works by replacing the current module with compiler known custom-op.
    """

    def forward(self, hidden_states):
        """
        Forward pass of the RMSNorm module.

        Args:
            hidden_states (torch.Tensor): Input tensor to be normalized.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        return CustomRMSNormFunc.apply(
            hidden_states, self.weight, self.variance_epsilon if hasattr(self, "variance_epsilon") else self.eps
        )


class DeepseekV3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class DeepseekV3YarnRotaryEmbedding(DeepseekV3RotaryEmbedding):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        freq_inter = 1.0 / (
            self.scaling_factor * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(device=device, dtype=torch.float32)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def orig_apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
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
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class QEffDeepseekV3Attention(nn.Module):
    """Adapted DeepseekV3Attention with QEff logic, adding batch_index and proper position_ids handling."""

    def __qeff_init__(
        self,
    ):
        q_up, q_rope = self.q_b_proj.weight.T.view(
            -1, self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim
        ).split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        q_up = q_up.reshape(-1, self.num_heads * self.qk_nope_head_dim).unsqueeze(0)
        self.q_up = torch.nn.Parameter(q_up.detach().clone())

        q_rope = q_rope.reshape(-1, self.num_heads * self.qk_rope_head_dim).unsqueeze(0)
        self.q_rope = torch.nn.Parameter(q_rope.detach().clone())

        k_up, v_up = self.kv_b_proj.weight.T.view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        k_up = k_up.reshape(-1, self.num_heads * self.qk_nope_head_dim).unsqueeze(0)
        v_up = v_up.reshape(-1, self.num_heads * self.v_head_dim).unsqueeze(0)

        self.k_up = torch.nn.Parameter(k_up.detach())
        self.v_up = torch.nn.Parameter(v_up.detach())
        per_head_q_up = self.q_up.squeeze(0).view(-1, self.num_heads, self.qk_nope_head_dim).transpose(0, 1)
        per_head_k_up = (
            self.k_up.squeeze(0).view(-1, self.num_heads, self.qk_nope_head_dim).transpose(0, 1).transpose(1, 2)
        )
        per_head_v_up = self.v_up.squeeze(0).view(-1, self.num_heads, self.v_head_dim).transpose(0, 1)
        self.per_head_v_up = torch.nn.Parameter(per_head_v_up.unsqueeze(0).detach().clone())
        self.per_head_q_up = torch.nn.Parameter(per_head_q_up.unsqueeze(0).detach().clone())
        self.per_head_k_up = torch.nn.Parameter(per_head_k_up.unsqueeze(0).detach().clone())
        per_head_k_up_normal = self.per_head_k_up.transpose(2, 3)
        self.per_head_k_up_normal = torch.nn.Parameter(per_head_k_up_normal.detach().clone())

        fusedqk = torch.bmm(per_head_q_up, per_head_k_up).reshape(
            -1, self.num_heads, self.q_lora_rank, self.kv_lora_rank
        )
        self.fusedqk = torch.nn.Parameter(fusedqk.detach().clone())

    def fused_forward_h_blocking(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        compressed_kvs: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        mla_absorption: Optional[Dict[str, bool]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv = compressed_kv.view(bsz, q_len, -1, self.kv_lora_rank + self.qk_rope_head_dim).transpose(1, 2)

        kva = compressed_kv[:, :, :, : self.kv_lora_rank]
        k_pe = compressed_kv[:, :, :, self.kv_lora_rank :]

        q_a_proj_out = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_pe = torch.matmul(q_a_proj_out, self.q_rope)
        q_pe = q_pe.view(bsz, q_len, self.num_heads, self.qk_rope_head_dim).transpose(1, 2)
        q_nope = torch.bmm(q_a_proj_out, self.q_up)
        q_nope = q_nope.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim).transpose(1, 2)

        kva = self.kv_a_layernorm(kva)
        cache_kwargs = {"position_ids": position_ids, "batch_index": batch_index}
        if compressed_kvs is not None:
            kva = compressed_kvs.update_ckv(kva, self.layer_idx, cache_kwargs)

        cos, sin = self.rotary_emb(kva, seq_len=32 * 1024)
        q_pe, k_pe = orig_apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        if compressed_kvs is not None:
            k_pe = compressed_kvs.update_k_pe(k_pe, self.layer_idx, cache_kwargs)

        blocking_config = getattr(self, "attn_blocking_config", AttentionBlockingConfig())

        attn_output, attn_weights = generic_blocked_mla_attention_interface(
            module=self,
            q_a_proj_out=q_a_proj_out,
            fusedqk=self.fusedqk,
            q_nope=q_nope,
            q_pe=q_pe,
            kva=kva,
            k_pe=k_pe,
            per_head_q_up=self.per_head_q_up,
            per_head_k_up=self.per_head_k_up,
            per_head_v_up=self.per_head_v_up,
            per_head_k_up_normal=self.per_head_k_up_normal,
            attention_mask=attention_mask,
            scaling=self.softmax_scale,
            mla_absorption=mla_absorption,
            blocking_config=blocking_config,
            position_ids=position_ids,
            **kwargs,
        )

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, compressed_kvs

    def fused_forward_kv_blocking(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        compressed_kvs: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        mla_absorption: Optional[Dict[str, bool]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv = compressed_kv.view(bsz, q_len, -1, self.kv_lora_rank + self.qk_rope_head_dim).transpose(1, 2)

        kva = compressed_kv[:, :, :, : self.kv_lora_rank]
        k_pe = compressed_kv[:, :, :, self.kv_lora_rank :]

        q_a_proj_out = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_pe = torch.matmul(q_a_proj_out, self.q_rope)
        q_pe = q_pe.view(bsz, q_len, self.num_heads, self.qk_rope_head_dim).transpose(1, 2)

        kva = self.kv_a_layernorm(kva)
        cache_kwargs = {"position_ids": position_ids, "batch_index": batch_index}

        ## Write Only
        if compressed_kvs is not None:
            compressed_kvs.write_only_ckv(kva, self.layer_idx, cache_kwargs)

        cos, sin = self.rotary_emb(hidden_states, seq_len=32 * 1024)
        q_pe, k_pe = orig_apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        if compressed_kvs is not None:
            compressed_kvs.write_only_k_pe(k_pe, self.layer_idx, cache_kwargs)

        if mla_absorption is not None:
            absorption = mla_absorption.get("absorption", False)
            online = mla_absorption.get("online", False)
        else:
            absorption = False

        if absorption:
            if online:
                qup_kupT = torch.matmul(self.per_head_q_up, self.per_head_k_up)
                dq_qup_kupT = torch.matmul(q_a_proj_out, qup_kupT)
            else:
                dq_qup_kupT = torch.matmul(q_a_proj_out, self.fusedqk)
            qkupTrope_nope = torch.cat((dq_qup_kupT, q_pe), dim=-1)
            query = qkupTrope_nope
        else:
            q_nope = torch.bmm(q_a_proj_out, self.q_up)
            q_nope = q_nope.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim).transpose(1, 2)
            qnope_rope = torch.cat((q_nope, q_pe), dim=-1)
            query = qnope_rope

        blocking_config = getattr(self, "attn_blocking_config", AttentionBlockingConfig())

        attn_output, attn_weights = generic_blocked_mla_attention_interface(
            module=self,
            query=query,
            per_head_k_up_normal=self.per_head_k_up_normal,
            per_head_v_up=self.per_head_v_up,
            attention_mask=attention_mask,
            scaling=self.softmax_scale,
            layer_idx=self.layer_idx,
            compressed_kvs=compressed_kvs,
            mla_absorption=mla_absorption,
            blocking_config=blocking_config,
            position_ids=position_ids,
            **kwargs,
        )

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, compressed_kvs

    def fused_forward_orig(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        compressed_kvs: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        mla_absorption: Optional[Dict[str, bool]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # ---- KV compression ----
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv = compressed_kv.view(bsz, q_len, -1, self.kv_lora_rank + self.qk_rope_head_dim).transpose(1, 2)

        kva = compressed_kv[:, :, :, : self.kv_lora_rank]
        k_pe = compressed_kv[:, :, :, self.kv_lora_rank :]

        # ---- Q projections ----
        q_a_proj_out = self.q_a_layernorm(self.q_a_proj(hidden_states))

        q_pe = torch.bmm(q_a_proj_out, self.q_rope)
        q_pe = q_pe.view(bsz, q_len, self.num_heads, self.qk_rope_head_dim).transpose(1, 2)

        kva = self.kv_a_layernorm(kva)

        cache_kwargs = {"position_ids": position_ids, "batch_index": batch_index}
        if compressed_kvs is not None:
            kva = compressed_kvs.update_ckv(kva, self.layer_idx, cache_kwargs)

        # ---- MLA absorption flags ----
        if mla_absorption is not None:
            absorption = mla_absorption.get("absorption", False)
            online = mla_absorption.get("online", False)
        else:
            absorption = False

        # ---- Rotary ----
        cos, sin = self.rotary_emb(q_pe, seq_len=32 * 1024)  # Doesn't need q_pe as head_dim is initialized
        q_pe, k_pe = orig_apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        if compressed_kvs is not None:
            k_pe = compressed_kvs.update_k_pe(k_pe, self.layer_idx, cache_kwargs)

        k_heads, q_heads = kva.shape[1], q_pe.shape[1]

        if k_heads > 1:
            num_heads_to_repeat = math.ceil(q_heads / k_heads)

            kva_expanded = (
                kva.unsqueeze(2)
                .expand(-1, -1, num_heads_to_repeat, -1, -1)
                .reshape(bsz, num_heads_to_repeat * k_heads, -1, self.config.kv_lora_rank)
            )
            kva_expanded = kva_expanded[:, :q_heads, :, :]

            k_pe_expanded = (
                k_pe.unsqueeze(2)
                .expand(-1, -1, num_heads_to_repeat, -1, -1)
                .reshape(bsz, num_heads_to_repeat * k_heads, -1, self.config.qk_rope_head_dim)
            )
            k_pe_expanded = k_pe_expanded[:, :q_heads, :, :]
        else:
            kva_expanded = kva
            #k_pe_expanded = k_pe
            num_heads_to_repeat = math.ceil(q_heads / k_heads)
            k_pe_expanded = (
                k_pe.unsqueeze(2)
                .expand(-1, -1, num_heads_to_repeat, -1, -1)
                .reshape(bsz, num_heads_to_repeat * k_heads, -1, self.config.qk_rope_head_dim)
            )

        v_up_per_head = self.v_up.squeeze(0).view(self.kv_lora_rank, self.num_heads, self.v_head_dim).permute(1, 0, 2)
        value_states = torch.matmul(kva_expanded, v_up_per_head)

        if absorption:
            if online:
                out = torch.matmul(self.per_head_q_up, self.per_head_k_up)
                q_nope_compressed = torch.matmul(q_a_proj_out.unsqueeze(1), out)
            else:
                q_nope_compressed = torch.matmul(
                    q_a_proj_out.unsqueeze(1),
                    self.fusedqk,
                )
            query_states = torch.cat((q_nope_compressed, q_pe), dim=-1)
            key_states = torch.cat((kva_expanded, k_pe_expanded), dim=-1)
        else:
            q_nope = torch.bmm(q_a_proj_out, self.q_up)
            q_nope = q_nope.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim).transpose(1, 2)
            query_states = torch.cat((q_nope, q_pe), dim=-1)

            k_up_per_head = (
                self.k_up.squeeze(0).view(self.kv_lora_rank, self.num_heads, self.qk_nope_head_dim).permute(1, 0, 2)
            )
            k_nope = torch.matmul(kva_expanded, k_up_per_head)
            key_states = torch.cat((k_nope, k_pe_expanded), dim=-1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale

        if attention_mask is not None:
            attn_weights = torch.where(
                attention_mask,
                torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=attn_weights.dtype),
                attn_weights,
            )
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_pe.dtype)
        ## Do v_proj here
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, compressed_kvs

    def fused_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        compressed_kvs: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        mla_absorption: Optional[Dict[str, bool]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        blocking_config = getattr(self, "attn_blocking_config", AttentionBlockingConfig())
        if getattr(blocking_config, "mode", None) == "h":
            return self.fused_forward_h_blocking(
                hidden_states,
                position_embeddings,
                attention_mask,
                position_ids,
                past_key_value,
                compressed_kvs,
                batch_index,
                output_attentions,
                use_cache,
                cache_position,
                mla_absorption,
                **kwargs,
            )
        elif getattr(blocking_config, "mode", None) == "kv":
            return self.fused_forward_kv_blocking(
                hidden_states,
                position_embeddings,
                attention_mask,
                position_ids,
                past_key_value,
                compressed_kvs,
                batch_index,
                output_attentions,
                use_cache,
                cache_position,
                mla_absorption,
                **kwargs,
            )
        else:
            return self.fused_forward_orig(
                hidden_states,
                position_embeddings,
                attention_mask,
                position_ids,
                past_key_value,
                compressed_kvs,
                batch_index,
                output_attentions,
                use_cache,
                cache_position,
                mla_absorption,
                **kwargs,
            )

    def forward_full_kv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

        q_nope = q[:, :, :, : self.qk_nope_head_dim]
        q_pe = q[:, :, :, self.qk_nope_head_dim :]

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)

        kva = compressed_kv[:, :, : self.kv_lora_rank]
        k_pe = compressed_kv[:, :, self.kv_lora_rank :]

        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(kva))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope = kv[:, :, :, : self.qk_nope_head_dim]
        value_states = kv[:, :, :, self.qk_nope_head_dim :]

        cos, sin = self.rotary_emb(value_states, seq_len=32 * 1024)
        q_pe, k_pe = orig_apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = torch.cat((q_nope, q_pe), -1)
        k_pe_new = k_pe.expand(-1, self.num_heads, -1, -1)
        key_states = torch.cat((k_nope, k_pe_new), -1)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "batch_index": batch_index, "position_ids": position_ids}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale

        if attention_mask is not None:  # no matter the length, we just slice it
            attn_weights = torch.where(
                attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), attn_weights
            )

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value

    def forward_full_kv_h_blocking(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

        q_nope = q[:, :, :, : self.qk_nope_head_dim]
        q_pe = q[:, :, :, self.qk_nope_head_dim :]

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv = compressed_kv.view(bsz, q_len, -1, self.kv_lora_rank + self.qk_rope_head_dim).transpose(1, 2)

        kva = compressed_kv[:, :, :, : self.kv_lora_rank]
        k_pe = compressed_kv[:, :, :, self.kv_lora_rank :]

        kv = (
            self.kv_b_proj(self.kv_a_layernorm(kva))
            .view(
                bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )  # TODO : split this matmul #with k_up and v_up
            .transpose(1, 2)
        )

        k_nope = kv[:, :, :, : self.qk_nope_head_dim]
        value_states = kv[:, :, :, self.qk_nope_head_dim :]

        cos, sin = self.rotary_emb(value_states, seq_len=32 * 1024)
        q_pe, k_pe = orig_apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = torch.cat((q_nope, q_pe), -1)
        k_pe_new = k_pe.expand(-1, self.num_heads, -1, -1)
        key_states = torch.cat((k_nope, k_pe_new), -1)

        blocking_config = getattr(self, "attn_blocking_config", AttentionBlockingConfig())

        attn_output, attn_weights = generic_blocked_attention_interface(
            module=self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            scaling=self.softmax_scale,
            layer_idx=self.layer_idx,
            past_key_value=past_key_value,
            blocking_config=blocking_config,
            batch_index=batch_index,
            position_ids=position_ids,
        )
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        blocking_config = getattr(self, "attn_blocking_config", AttentionBlockingConfig())
        if getattr(blocking_config, "mode", None) == "h":
            return self.forward_full_kv_h_blocking(
                hidden_states,
                position_embeddings,
                attention_mask,
                position_ids,
                past_key_value,
                batch_index,
                output_attentions,
                use_cache,
                cache_position,
                **kwargs,
            )
        else:
            return self.forward_full_kv(
                hidden_states,
                position_embeddings,
                attention_mask,
                position_ids,
                past_key_value,
                batch_index,
                output_attentions,
                use_cache,
                cache_position,
                **kwargs,
            )


EXPERT_BLOCKING_NUM_NSP = int(os.environ.get("EXPERT_BLOCKING_NUM_NSP", "16"))
EXPERT_BLOCKING_PACKED_CHUNK_SIZE = int(os.environ.get("EXPERT_BLOCKING_PACKED_CHUNK_SIZE", "256"))


def _build_matched_idx_from_cumsum(T2Ei: torch.Tensor) -> torch.Tensor:
    """Build packed->original token index"""
    batch_size, seq_len = T2Ei.shape
    int32_max = torch.iinfo(torch.int32).max
    int32_max_scalar = torch.tensor(int32_max, dtype=torch.int32, device=T2Ei.device)
    token_idx = torch.arange(seq_len, dtype=torch.int32, device=T2Ei.device).unsqueeze(0).expand(batch_size, -1)
    valid_prefix = torch.cumsum(T2Ei.to(torch.int32), dim=1)
    valid_dest = valid_prefix - 1
    scatter_pos = torch.where(T2Ei, valid_dest, int32_max_scalar)
    # Once the compiler fix for ConstantOfShape(INT32_MAX) is available, this
    # can be switched back to ``torch.full_like(token_idx, int32_max)``.
    matched_idx = int32_max_scalar.expand_as(token_idx)
    matched_idx = CtxScatterFunc3DInt.apply(
        matched_idx.unsqueeze(-1),
        scatter_pos,
        token_idx.unsqueeze(-1),
    ).squeeze(-1)
    return matched_idx


class QEffDeepseekMoEGate(nn.Module):
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)
        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")

        ### select top-k experts
        if self.topk_method == "noaux_tc":
            assert not self.training
            scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)
            group_scores = (
                scores_for_choice.view(bsz * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
            )  # [n, n_group]
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group)
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
            topk_weight = scores.gather(1, topk_idx)
        else:
            raise NotImplementedError(f"insupportable TopK function for MoE gating: {self.topk_method}")

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor  # must multiply the scaling factor

        router_probs = tmp_scores
        router_weights = scores
        return topk_idx, topk_weight, router_probs, router_weights


class QEffDeepseekV3MoE(nn.Module):
    def __qeff_init__(
        self,
    ):
        # Get common parameters from first expert
        first_expert = self.experts[0]
        self.bits = first_expert.gate_proj.bits
        self.group_size = first_expert.gate_proj.group_size
        self.act_fn = first_expert.act_fn
        assert first_expert.gate_proj.act_order == first_expert.up_proj.act_order == first_expert.down_proj.act_order, (
            "act_order mismatch"
        )
        self.act_order = first_expert.gate_proj.act_order

        # Store dimensions for dequantization
        self.in_features_gate, self.out_features_gate = (
            first_expert.gate_proj.in_features,
            first_expert.gate_proj.out_features,
        )
        self.in_features_up, self.out_features_up = first_expert.up_proj.in_features, first_expert.up_proj.out_features
        self.in_features_down, self.out_features_down = (
            first_expert.down_proj.in_features,
            first_expert.down_proj.out_features,
        )

        # Stack all parameters along a new dimension (expert dimension)
        self.all_gate_qweight = torch.nn.Parameter(
            torch.stack([exp.gate_proj.qweight for exp in self.experts], dim=0).reshape(
                -1, self.out_features_gate, self.in_features_gate // 2
            ),
            requires_grad=False,
        )
        self.all_gate_scales = torch.nn.Parameter(
            torch.stack([exp.gate_proj.scales for exp in self.experts], dim=0).reshape(
                -1, self.out_features_gate, self.in_features_gate // self.group_size
            ),
            requires_grad=False,
        )
        # TODO: Since we know qzeros is always 8 -> Just embed this once into the operator as parameter -> explore this later
        self.all_gate_qzeros = torch.nn.Parameter(
            torch.stack([exp.gate_proj.qzeros for exp in self.experts], dim=0).reshape(
                -1, self.out_features_gate, self.in_features_gate // (self.group_size * 2)
            ),
            requires_grad=False,
        )
        self.all_gate_gidx = torch.nn.Parameter(
            torch.stack([exp.gate_proj.g_idx for exp in self.experts], dim=0), requires_grad=False
        )

        self.all_up_qweight = torch.nn.Parameter(
            torch.stack([exp.up_proj.qweight for exp in self.experts], dim=0).reshape(
                -1, self.out_features_up, self.in_features_up // 2
            ),
            requires_grad=False,
        )
        self.all_up_scales = torch.nn.Parameter(
            torch.stack([exp.up_proj.scales for exp in self.experts], dim=0).reshape(
                -1, self.out_features_up, self.in_features_up // self.group_size
            ),
            requires_grad=False,
        )
        self.all_up_qzeros = torch.nn.Parameter(
            torch.stack([exp.up_proj.qzeros for exp in self.experts], dim=0).reshape(
                -1, self.out_features_up, self.in_features_up // (self.group_size * 2)
            ),
            requires_grad=False,
        )
        self.all_up_gidx = torch.nn.Parameter(
            torch.stack([exp.up_proj.g_idx for exp in self.experts], dim=0), requires_grad=False
        )

        self.all_down_qweight = torch.nn.Parameter(
            torch.stack([exp.down_proj.qweight for exp in self.experts], dim=0).reshape(
                -1, self.out_features_down, self.in_features_down // 2
            ),
            requires_grad=False,
        )
        self.all_down_scales = torch.nn.Parameter(
            torch.stack([exp.down_proj.scales for exp in self.experts], dim=0).reshape(
                -1, self.out_features_down, self.in_features_down // self.group_size
            ),
            requires_grad=False,
        )
        self.all_down_qzeros = torch.nn.Parameter(
            torch.stack([exp.down_proj.qzeros for exp in self.experts], dim=0).reshape(
                -1, self.out_features_down, self.in_features_down // (self.group_size * 2)
            ),
            requires_grad=False,
        )
        self.all_down_gidx = torch.nn.Parameter(
            torch.stack([exp.down_proj.g_idx for exp in self.experts], dim=0), requires_grad=False
        )

    def moe_old(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)

        for i in range(self.gate.top_k):
            expert_idx = topk_indices[:, i]
            curr_weight = topk_weights[:, i]
            gate_qweight = self.all_gate_qweight[expert_idx].reshape(
                seq_len * self.out_features_gate,
                self.in_features_gate // self.group_size,
                (self.group_size * self.bits) // 8,
            )
            gate_scales = self.all_gate_scales[expert_idx].reshape(
                seq_len * self.out_features_gate * (self.in_features_gate // self.group_size)
            )
            gate_qzeros = self.all_gate_qzeros[expert_idx].reshape(
                seq_len * self.out_features_gate, self.in_features_gate // self.group_size
            )
            gate_gidx = self.all_gate_gidx[expert_idx].reshape(seq_len * self.in_features_gate)

            up_qweight = self.all_up_qweight[expert_idx].reshape(
                seq_len * self.out_features_up,
                self.in_features_up // self.group_size,
                (self.group_size * self.bits) // 8,
            )
            up_scales = self.all_up_scales[expert_idx].reshape(
                seq_len * self.out_features_up * (self.in_features_up // self.group_size)
            )
            up_qzeros = self.all_up_qzeros[expert_idx].reshape(
                seq_len * self.out_features_up, self.in_features_up // self.group_size
            )
            up_gidx = self.all_up_gidx[expert_idx].reshape(seq_len * self.in_features_up)

            down_qweight = self.all_down_qweight[expert_idx].reshape(
                seq_len * self.out_features_down,
                self.in_features_down // self.group_size,
                (self.group_size * self.bits) // 8,
            )
            down_scales = self.all_down_scales[expert_idx].reshape(
                seq_len * self.out_features_down * (self.in_features_down // self.group_size)
            )
            down_qzeros = self.all_down_qzeros[expert_idx].reshape(
                seq_len * self.out_features_down, self.in_features_down // self.group_size
            )
            down_gidx = self.all_down_gidx[expert_idx].reshape(seq_len * self.in_features_down)

            gate_out = QuantLinearTorchFunction.apply(
                hidden_states,
                gate_qweight,
                gate_scales,
                gate_qzeros,
                gate_gidx if self.act_order else None,
                self.bits,
                self.group_size,
                self.in_features_gate,
                self.out_features_gate * seq_len,
            )

            up_out = QuantLinearTorchFunction.apply(
                hidden_states,
                up_qweight,
                up_scales,
                up_qzeros,
                up_gidx if self.act_order else None,
                self.bits,
                self.group_size,
                self.in_features_up,
                self.out_features_up * seq_len,
            )

            hidden = self.act_fn(gate_out) * up_out
            down_out = QuantLinearTorchFunction.apply(
                hidden,
                down_qweight,
                down_scales,
                down_qzeros,
                down_gidx if self.act_order else None,
                self.bits,
                self.group_size,
                self.in_features_down,
                self.out_features_down,
            )
            down_out = down_out.reshape(seq_len, self.out_features_down)
            final_hidden_states += down_out * curr_weight.unsqueeze(1)

        return final_hidden_states

    def moe_weights_as_activations(self, hidden_states, router_probs, router_weights):
        return QMOE.apply(
            hidden_states,
            router_weights,
            self.fc1_experts_weights,
            self.fc1_scales,
            self.fc2_experts_weights,
            self.fc2_scales,
            self.fc3_experts_weights,
            self.fc3_scales,
            router_probs,
            self.config.hidden_act,
            self.group_size,
            self.bits,
            self.num_experts_per_tok,
        )

    @torch.no_grad()
    def original_moe(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        # sorted_tokens_shape = sorted_tokens.shape
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out

    def moe_waa_unpack(self, hidden_states, topk_indices, topk_weights):
        # GATHER - collect weights for selected experts
        gate_proj_qweight = self.all_gate_qweight[topk_indices.flatten()]
        gate_proj_scales = self.all_gate_scales[topk_indices.flatten()]
        gate_proj_qzeros = self.all_gate_qzeros[topk_indices.flatten()]

        up_proj_qweight = self.all_up_qweight[topk_indices.flatten()]
        up_proj_scales = self.all_up_scales[topk_indices.flatten()]
        up_proj_qzeros = self.all_up_qzeros[topk_indices.flatten()]

        down_proj_qweight = self.all_down_qweight[topk_indices.flatten()]
        down_proj_scales = self.all_down_scales[topk_indices.flatten()]
        down_proj_qzeros = self.all_down_qzeros[topk_indices.flatten()]

        gate_proj_unpacked = CastToUInt4Func.apply(gate_proj_qweight)
        gate_zeros_unpacked = CastToUInt4Func.apply(gate_proj_qzeros)
        gate_proj_dq = DequantizeLinearFunc.apply(
            gate_proj_unpacked, gate_proj_scales, gate_zeros_unpacked, self.group_size
        )

        up_proj_unpacked = CastToUInt4Func.apply(up_proj_qweight)
        up_zeros_unpacked = CastToUInt4Func.apply(up_proj_qzeros)
        up_proj_dq = DequantizeLinearFunc.apply(up_proj_unpacked, up_proj_scales, up_zeros_unpacked, self.group_size)

        down_proj_unpacked = CastToUInt4Func.apply(down_proj_qweight)
        down_zeros_unpacked = CastToUInt4Func.apply(down_proj_qzeros)
        down_proj_dq = DequantizeLinearFunc.apply(
            down_proj_unpacked, down_proj_scales, down_zeros_unpacked, self.group_size
        )

        # Reshape for bmm: (bs*seq_len*top_k, 1, hidden_size)
        expert_in = (
            hidden_states.unsqueeze(1).expand(-1, self.gate.top_k, -1).contiguous().view(-1, 1, self.in_features_gate)
        )

        gate_out = torch.bmm(expert_in, gate_proj_dq.transpose(1, 2))
        up_out = torch.bmm(expert_in, up_proj_dq.transpose(1, 2))
        hidden = self.act_fn(gate_out) * up_out
        down_out = torch.bmm(hidden, down_proj_dq.transpose(1, 2))

        down_out = down_out.view(-1, self.gate.top_k, self.out_features_down)

        down_out = down_out * topk_weights.unsqueeze(-1)

        return torch.einsum("abc-> ac", down_out)

    def forward(self, hidden_states):
        print("Using new MoE forward with weights as activations")
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights, router_probs, router_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        # hidden_states = self.moe_weights_as_activations(hidden_states, router_probs, router_weights).view(*orig_shape)
        hidden_states = self.moe_waa_unpack(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class QEffPrefillOnlyDeepseekV3MoE(nn.Module):

    def _cumsum_scatter_gather_update_expert_blocked(
        self,
        x: torch.Tensor,
        T2Ei: torch.Tensor,
        expert,
#        W_g: torch.Tensor,
#        W_u: torch.Tensor,
#        W_d: torch.Tensor,
        routing_weight: torch.Tensor,
        expert_out: torch.Tensor,
        act_fn,
        T: int,
        packed_chunk_size: int,
    ) -> torch.Tensor:
        """Cumsum-scatter-gather-update expert helper for NSP-blocked dispatch.

        Accumulates one local expert's contribution in-place onto ``expert_out``.
        Uses a packed/cumsum layout so the MLP runs only over active rows, then
        scatters the weighted output back to original token positions.

        Shapes:
            x               : [T, H]
            T2Ei            : [num_nsp, T]            (bool)
            W_g, W_u        : [num_nsp, H, I]
            W_d             : [num_nsp, I, H]
            routing_weight  : [num_nsp, T]
            expert_out      : [num_nsp, T, H]         (accumulator, in-out)
        """
        batch_size, seq_len = T2Ei.shape
        packed_chunk_size = max(1, min(packed_chunk_size, seq_len))

        matched_idx = _build_matched_idx_from_cumsum(T2Ei)
        valid_rows = T2Ei.to(torch.int32).sum(dim=1, keepdim=True)
        row_range = torch.arange(packed_chunk_size, dtype=torch.int32, device=x.device).unsqueeze(0)
        x_expanded = x.unsqueeze(0).expand(batch_size, -1, -1)
        rw_expanded = routing_weight.unsqueeze(-1)

        for packed_start in range(0, seq_len, packed_chunk_size):
            packed_stop = packed_start + packed_chunk_size
            chunk_matched_idx = matched_idx[:, packed_start:packed_stop]

            x_chunk = CtxGatherFunc3DGeneralized.apply(x_expanded, chunk_matched_idx)

            gate_prime = expert.gate_proj(x_chunk)
            up_prime = expert.up_proj(x_chunk)
            down_chunk = expert.down_proj((up_prime * act_fn(gate_prime)))

            #gate_prime = x_chunk @ W_g
            #up_prime = x_chunk @ W_u
            #down_chunk = (up_prime * act_fn(gate_prime)) @ W_d

            rw_chunk = CtxGatherFunc3DGeneralized.apply(rw_expanded, chunk_matched_idx)
            down_chunk = down_chunk * rw_chunk

            expert_out_chunk = CtxGatherFunc3DGeneralized.apply(expert_out, chunk_matched_idx)
            updated_chunk = expert_out_chunk + down_chunk

            chunk_valid_rows = torch.clamp(valid_rows - packed_start, min=0, max=packed_chunk_size)
            updated_chunk = torch.where(
                (row_range < chunk_valid_rows).unsqueeze(-1), updated_chunk, torch.zeros_like(updated_chunk)
            )
            expert_out = CtxScatterFunc3DGeneralized.apply(expert_out, chunk_matched_idx, updated_chunk)

        return expert_out


    def _forward_expert_blocked(self, x: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        T, H = x.shape
        num_nsp = EXPERT_BLOCKING_NUM_NSP
        if len(self.experts) % num_nsp != 0:
            raise ValueError(
                f"num_experts ({len(self.experts)}) must be divisible by EXPERT_BLOCKING_NUM_NSP ({num_nsp})"
            )
        local_experts = len(self.experts) // num_nsp
        rw = routing_weights.transpose(0, 1).contiguous().view(local_experts, num_nsp, T).transpose(0, 1).contiguous()
#        W_g = self.gate_proj_w.view(local_experts, num_nsp, H, -1).transpose(0, 1).contiguous()
#        W_u = self.up_proj_w.view(local_experts, num_nsp, H, -1).transpose(0, 1).contiguous()
#        W_d = self.down_proj_w.view(local_experts, num_nsp, -1, H).transpose(0, 1).contiguous()
        expert_out = x.new_zeros((num_nsp, T, H))
        for slot in range(local_experts):
            routing_weight = rw[:, slot, :]
            T2Ei = routing_weight > 0
            expert_out = self._cumsum_scatter_gather_update_expert_blocked(
                x=x,
                T2Ei=T2Ei,
                expert=self.experts[slot],
#                W_g=W_g[:, slot],
#                W_u=W_u[:, slot],
#                W_d=W_d[:, slot],
                routing_weight=routing_weight,
                expert_out=expert_out,
                act_fn=self.experts[0].act_fn,
                T=T,
                packed_chunk_size=EXPERT_BLOCKING_PACKED_CHUNK_SIZE,
            )
        return expert_out.sum(dim=0)


    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        topk_idx, topk_weight, router_probs, router_weights = self.gate(hidden_states)
        B, S, H = hidden_states.shape
        T = B * S
        x = hidden_states.view(T, H)

        routing_weights = torch.zeros(T, self.config.n_routed_experts)
        routing_weights.scatter_(1, topk_idx, topk_weight)

        if len(self.experts) % EXPERT_BLOCKING_NUM_NSP == 0:
            expert_out = self._forward_expert_blocked(x=x, routing_weights=routing_weights)
            return expert_out.view(B, S, H)

        final_hidden_states = x.new_zeros((T, H))
        for expert_idx in range(self.n_routed_experts):
            expert = self.experts[expert_idx]
            gate_out = expert.gate_proj(hidden_states)
            up_out = expert.up_proj(hidden_states)
            hidden = expert.act_fn(gate_out) * up_out
            expert_output = expert.down_proj(hidden)
            current_hidden_states = expert_output * routing_weights[:, expert_idx].unsqueeze(-1)
            final_hidden_states += current_hidden_states

        return final_hidden_states.view(B, S, H)


class QEffDeepseekV3DecoderLayer(nn.Module):
    """Adapted DeepseekV3DecoderLayer with batch_index and proper position_ids handling."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        compressed_kvs: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mla_absorption: Optional[Dict[str, bool]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        orig_hidden_states = self.input_layernorm(hidden_states)
        if mla_absorption is not None:
            cache_compressed = mla_absorption.get("cache_compressed", False)
        else:
            cache_compressed = False
        if cache_compressed:
            hidden_states, self_attn_weights, present_compressed_kvs = self.self_attn.fused_forward(
                hidden_states=orig_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
                compressed_kvs=compressed_kvs,
                batch_index=batch_index,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                mla_absorption=mla_absorption,
                **kwargs,
            )
        else:
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=orig_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
                batch_index=batch_index,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            if cache_compressed:
                outputs += (present_compressed_kvs,)
            else:
                outputs += (present_key_value,)

        return outputs


class QEffDeepseekV3Model(nn.Module):
    """Adapted DeepseekV3Model with batch_index and QEff rotary embedding."""

    def __qeff_init__(self):
        scaling_factor = self.config.rope_scaling["factor"]
        kwargs = {
            key: self.config.rope_scaling[key]
            for key in [
                "original_max_position_embeddings",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            ]
            if key in self.config.rope_scaling
        }
        self.rotary_emb = DeepseekV3YarnRotaryEmbedding(
            self.config.qk_rope_head_dim,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            scaling_factor=scaling_factor,
            base=self.config.rope_theta,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        compressed_kvs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        mla_absorption: Optional[Dict[str, bool]] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and not isinstance(past_key_values, Cache) and past_key_values is not None:
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        if mla_absorption is not None:
            cache_compressed = mla_absorption.get("cache_compressed", False)
        else:
            cache_compressed = False

        if cache_compressed:
            compressed_kvs = QEffDynamicCompressedKVRopeCache.from_legacy_cache(compressed_kvs)
            target_len = compressed_kvs.layers[0].ckv.shape[-2]
        else:
            target_len = past_key_values[0][0].shape[2]

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=target_len)
        hidden_states = inputs_embeds
        position_embeddings = None

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                compressed_kvs=compressed_kvs,
                past_key_value=past_key_values,
                batch_index=batch_index,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                mla_absorption=mla_absorption,
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        next_cache = next_cache.to_legacy_cache()
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class QEffDeepseekV3ForCausalLM(nn.Module):
    """Adapted DeepseekV3ForCausalLM with batch_index and QEff optimizations."""

    def get_submodules_for_export(self) -> Type[nn.Module]:
        """
        Return the set of class used as the repeated layer across the model for subfunction extraction.
        Notes:
            This method should return the *class object* (not an instance).
            Downstream code can use this to find/build subfunctions for repeated blocks.
        """
        return {self.model.layers[0].__class__}

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        compressed_kvs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        mla_absorption = getattr(self, "mla_absorption", None)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            compressed_kvs=compressed_kvs,
            past_key_values=past_key_values,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            mla_absorption=mla_absorption,
            **kwargs,
        )

        hidden_states = outputs[0]
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = hidden_states[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_dummy_pkv_cache(self, config, batch_size, seq_len):
        mla_absorption = getattr(self, "mla_absorption", None)
        if mla_absorption is not None:
            cache_compressed = mla_absorption.get("cache_compressed", False)
        else:
            cache_compressed = False

        dummy_cache = [[] for _ in range(config.num_hidden_layers)]
        if cache_compressed:
            for layer in self.model.layers:
                if layer is not None:
                    num_heads = layer.self_attn.kv_a_proj_with_mqa.weight.shape[0] // (
                        self.model.config.kv_lora_rank + config.qk_rope_head_dim
                    )
            cache_shape_1 = (batch_size, num_heads, seq_len, config.kv_lora_rank)
            cache_shape_2 = (batch_size, num_heads, seq_len, config.qk_rope_head_dim)
        else:
            cache_shape_1 = (
                batch_size,
                config.num_attention_heads,
                seq_len,
                config.qk_nope_head_dim + config.qk_rope_head_dim,
            )
            cache_shape_2 = (batch_size, config.num_attention_heads, seq_len, config.v_head_dim)

        for i in range(config.num_hidden_layers):
            dummy_cache[i].append(torch.zeros(cache_shape_1, dtype=config.torch_dtype))
            dummy_cache[i].append(torch.zeros(cache_shape_2, dtype=config.torch_dtype))

        return dummy_cache
