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

from QEfficient.customop.rms_norm import CustomRMSNormFunc
from QEfficient.transformers.cache_utils import QEffDynamicCache, QEffDynamicCompressedKVRopeCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


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

    # def forward(self, x, position_ids):
    #     seq_len = torch.max(position_ids) + 1
    #     if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
    #         self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

    #     # Use position_ids to slice the precomputed caches
    #     cos = self.cos_cached[position_ids]
    #     sin = self.sin_cached[position_ids]
    #     return cos.to(x.dtype), sin.to(x.dtype)


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


def update_running_softmax(
    current_max: torch.Tensor,
    attn_weights_block: torch.Tensor,
    current_denominator: torch.Tensor,
    output: torch.Tensor,
    v_block: torch.Tensor,
    skip_kv: bool = False,
    skip_future: Optional[torch.Tensor] = None,
):
    # Update Running row maximum
    prev_max = current_max
    current_max_updated = torch.max(prev_max, attn_weights_block.max(dim=3).values)
    delta_max = prev_max - current_max_updated

    current_exp = torch.exp(attn_weights_block - current_max_updated.unsqueeze(-1))

    # update running denominator
    prev_denominator = current_denominator
    curr_exp_sum = torch.einsum("bhqk->bhq", current_exp)
    current_denominator_updated = prev_denominator * torch.exp(delta_max) + curr_exp_sum

    prob = current_exp / current_denominator_updated.unsqueeze(-1)

    prev_output = output
    # if updating running softmax with attention sinks, we don't have v_block
    if v_block is not None:
        output_updated = ((prev_denominator / current_denominator_updated).unsqueeze(-1)) * prev_output * torch.exp(
            delta_max.unsqueeze(-1)
        ) + torch.matmul(prob, v_block)
    else:
        output_updated = (
            ((prev_denominator / current_denominator_updated).unsqueeze(-1))
            * prev_output
            * torch.exp(delta_max.unsqueeze(-1))
        )

    if skip_kv and (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
        current_max = torch.where(skip_future, prev_max, current_max_updated)
        current_denominator = torch.where(skip_future, prev_denominator, current_denominator_updated)
        output = torch.where(skip_future.unsqueeze(-1), prev_output, output_updated)
    else:
        # Eager mode
        current_max = current_max_updated
        current_denominator = current_denominator_updated
        output = output_updated
    return current_max, current_denominator, output


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
        per_head_v_up = self.v_up.squeeze(0).view(-1, self.num_heads, self.qk_nope_head_dim).transpose(0, 1)
        self.per_head_v_up = torch.nn.Parameter(per_head_v_up.unsqueeze(0).detach().clone())
        self.per_head_q_up = torch.nn.Parameter(per_head_q_up.unsqueeze(0).detach().clone())
        self.per_head_k_up = torch.nn.Parameter(per_head_k_up.unsqueeze(0).detach().clone())
        per_head_k_up_normal = self.per_head_k_up.transpose(2, 3)
        self.per_head_k_up_normal = torch.nn.Parameter(per_head_k_up_normal.detach().clone())

        fusedqk_list = []
        for i in range(self.num_heads):
            fusedqk_list.append(torch.matmul(per_head_q_up[i, :, :], per_head_k_up[i, :, :]))
        fusedqk = torch.cat(fusedqk_list, dim=0)
        fusedqk = fusedqk.reshape(1, self.num_heads, -1, self.kv_lora_rank)

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

        if mla_absorption is not None:
            enable_absorption = mla_absorption.get("enable", False)
            absorb_online = mla_absorption.get("online", False)
        else:
            enable_absorption = False

        n_head_ckv = kva.shape[1]

        cos, sin = self.rotary_emb(kva, seq_len=32 * 1024)
        q_pe, k_pe = orig_apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        if compressed_kvs is not None:
            k_pe = compressed_kvs.update_k_pe(k_pe, self.layer_idx, cache_kwargs)

        attn_output_list = []
        attn_weights_list = []
        for head_block_idx in range(self.num_heads // n_head_ckv):
            h_start = head_block_idx * n_head_ckv
            h_end = min(h_start + n_head_ckv, self.num_heads)

            if enable_absorption:
                if absorb_online:
                    qup_kupT = torch.matmul(
                        self.per_head_q_up[:, h_start:h_end, :, :], self.per_head_k_up[:, h_start:h_end, :, :]
                    )
                    dq_qup_kupT = torch.matmul(q_a_proj_out, qup_kupT)
                else:
                    dq_qup_kupT = torch.matmul(q_a_proj_out, self.fusedqk[:, h_start:h_end, :, :])
                qkupTrope_nope = torch.cat((dq_qup_kupT, q_pe[:, h_start:h_end, :, :]), dim=-1)
                krope_nope = torch.cat((kva, k_pe), dim=-1)
                attn_weights = torch.matmul(qkupTrope_nope, krope_nope.transpose(2, 3)) * self.softmax_scale
                if attention_mask is not None:
                    attn_weights = torch.where(
                        attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), attn_weights
                    )
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(qkupTrope_nope.dtype)
                attn_output = torch.matmul(attn_weights, kva)
                attn_output = torch.matmul(attn_output, self.per_head_v_up[:, h_start:h_end, :, :])
                attn_output_list.append(attn_output)
                attn_weights_list.append(attn_weights)

            else:
                knope = torch.matmul(kva, self.per_head_k_up_normal[:, h_start:h_end, :, :])
                krope_nope = torch.cat((knope, k_pe), dim=-1)
                qrope_nope = torch.cat((q_nope[:, h_start:h_end, :, :], q_pe[:, h_start:h_end, :, :]), dim=-1)
                attn_weights = torch.matmul(qrope_nope, krope_nope.transpose(2, 3)) * self.softmax_scale
                if attention_mask is not None:
                    attn_weights = torch.where(
                        attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), attn_weights
                    )
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(qrope_nope.dtype)
                attn_output = torch.matmul(attn_weights, kva)
                attn_output = torch.matmul(
                    attn_output, self.per_head_v_up[:, h_start:h_end, :, :]
                )  # TODO: merge this matmul with o_proj
                attn_output_list.append(attn_output)
                attn_weights_list.append(attn_weights)

        attn_output = torch.cat(attn_output_list, dim=1)
        attn_weights = torch.cat(attn_weights_list, dim=1)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)  # 7168, 8192

        return attn_output, attn_weights, compressed_kvs

    def fused_forward_blocked_kv(
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
        num_kv_blocks = 4

        ctx_len = compressed_kvs.layers[0].ckv.shape[2]
        kv_block_size = -(-ctx_len // num_kv_blocks)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv = compressed_kv.view(bsz, q_len, -1, self.kv_lora_rank + self.qk_rope_head_dim).transpose(1, 2)

        kva = compressed_kv[:, :, :, : self.kv_lora_rank]
        k_pe = compressed_kv[:, :, :, self.kv_lora_rank :]

        q_a_proj_out = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_pe = torch.matmul(q_a_proj_out, self.q_rope)
        q_pe = q_pe.view(bsz, q_len, self.num_heads, self.qk_rope_head_dim).transpose(1, 2)
        q_nope = torch.matmul(q_a_proj_out, self.q_up)
        q_nope = q_nope.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim).transpose(1, 2)

        kva = self.kv_a_layernorm(kva)
        cache_kwargs = {"position_ids": position_ids, "batch_index": batch_index}

        if mla_absorption is not None:
            enable_absorption = mla_absorption.get("enable", False)
            absorb_online = mla_absorption.get("online", False)
        else:
            enable_absorption = False

        ## Write Only
        if compressed_kvs is not None:
            compressed_kvs.write_only_ckv(kva, self.layer_idx, cache_kwargs)

        cos, sin = self.rotary_emb(hidden_states, seq_len=32 * 1024)
        q_pe, k_pe = orig_apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        if compressed_kvs is not None:
            compressed_kvs.write_only_k_pe(k_pe, self.layer_idx, cache_kwargs)

        if enable_absorption:
            if absorb_online:
                qup_kupT = torch.matmul(self.per_head_q_up, self.per_head_k_up)
                dq_qup_kupT = torch.matmul(q_a_proj_out, qup_kupT)
            else:
                dq_qup_kupT = torch.matmul(q_a_proj_out, self.fusedqk)
            qkupTrope_nope = torch.cat((dq_qup_kupT, q_pe), dim=-1)
            batch_size, num_heads, seq_len, _ = qkupTrope_nope.shape
        else:
            batch_size, num_heads, seq_len, _ = q_nope.shape

        current_position = position_ids.max(dim=-1).values
        skip_kv = True
        output = torch.zeros(batch_size, self.num_heads, seq_len, self.kv_lora_rank, device=hidden_states.device)

        current_max = torch.full(
            (batch_size, num_heads, seq_len),
            float(MIN_MASKED_ATTENTION_VALUE),
            device=hidden_states.device,
        )
        current_denominator = torch.zeros(batch_size, num_heads, seq_len, device=hidden_states.device)

        for j in range(num_kv_blocks):
            start_index = j * kv_block_size
            if j == num_kv_blocks - 1:
                kv_len_block = ctx_len - start_index
            else:
                kv_len_block = kv_block_size
            end_index = start_index + kv_len_block

            skip_future = None
            if skip_kv:
                skip_future = (torch.tensor(start_index, device=hidden_states.device) > current_position).all()
                # Eager mode Only
                if not torch.onnx.is_in_onnx_export() and not torch.jit.is_tracing():
                    if skip_future.item():
                        break

            compressed_kv_block = compressed_kvs.read_only_blocked_ckv(
                start_index, end_index, self.layer_idx, cache_kwargs
            )
            k_pe_block = compressed_kvs.read_only_blocked_k_pe(start_index, end_index, self.layer_idx, cache_kwargs)

            causal_mask_block = _create_causal_mask(
                position_ids=position_ids,
                target_length=end_index,
                start_index=start_index,
            )

            if enable_absorption:
                krope_nope = torch.cat((compressed_kv_block, k_pe_block), dim=-1)
                attn_weights_block = (
                    torch.matmul(qkupTrope_nope, krope_nope.transpose(2, 3)) * self.softmax_scale
                )  # [1, 64, q_len, 576] X [1, 1, 576, kv_block_size] -> [1, 64, q_len, kv_block_size]
                attn_weights_block = torch.where(
                    causal_mask_block, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), attn_weights_block
                )
                current_max, current_denominator, output = update_running_softmax(
                    current_max,
                    attn_weights_block,
                    current_denominator,
                    output,
                    compressed_kv_block,
                    skip_kv,
                    skip_future,
                )
                # [1, 64, q_len, kv_block_size] X [1, 1, kv_block_size, 512] -> [1, 64, q_len, 512]
            else:
                knope = torch.matmul(compressed_kv_block, self.per_head_k_up_normal)
                krope_nope = torch.cat((knope, k_pe_block.expand(-1, self.num_heads, -1, -1)), dim=-1)
                qrope_nope = torch.cat((q_nope, q_pe), dim=-1)
                attn_weights_block = torch.matmul(qrope_nope, krope_nope.transpose(2, 3)) * self.softmax_scale
                attn_weights_block = torch.where(
                    causal_mask_block, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), attn_weights_block
                )
                current_max, current_denominator, output = update_running_softmax(
                    current_max,
                    attn_weights_block,
                    current_denominator,
                    output,
                    compressed_kv_block,
                    skip_kv,
                    skip_future,
                )

        attn_output = torch.matmul(output, self.per_head_v_up)  # TODO: merge this matmul with o_proj
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, compressed_kvs

    def fused_forward_basic(
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
        q_pe = torch.bmm(q_a_proj_out, self.q_rope)
        q_pe = q_pe.view(bsz, q_len, self.num_heads, self.qk_rope_head_dim).transpose(1, 2)
        q_nope = torch.bmm(q_a_proj_out, self.q_up)
        q_nope = q_nope.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim).transpose(1, 2)

        kva = self.kv_a_layernorm(kva)
        cache_kwargs = {"position_ids": position_ids, "batch_index": batch_index}
        if compressed_kvs is not None:
            kva = compressed_kvs.update_ckv(kva, self.layer_idx, cache_kwargs)

        if mla_absorption is not None:
            enable_absorption = mla_absorption.get("enable", False)
            absorb_online = mla_absorption.get("online", False)
        else:
            enable_absorption = False

        cos, sin = self.rotary_emb(hidden_states, seq_len=32 * 1024)
        q_pe, k_pe = orig_apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        if compressed_kvs is not None:
            k_pe = compressed_kvs.update_k_pe(k_pe, self.layer_idx, cache_kwargs)

        if enable_absorption:
            if absorb_online:
                out = torch.matmul(self.per_head_q_up, self.per_head_k_up)
                q_nope_compressed = torch.matmul(q_a_proj_out, out)
            else:
                q_nope_compressed = torch.matmul(q_a_proj_out, self.fusedqk)

            query_states = torch.cat((q_nope_compressed, q_pe), dim=-1)
            key_states = torch.cat((kva, k_pe), dim=-1)
        else:
            q_nope = torch.bmm(q_a_proj_out, self.q_up)
            q_nope = q_nope.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim).transpose(1, 2)
            query_states = torch.cat((q_nope, q_pe), dim=-1)

            k_up_per_head = (
                self.k_up.squeeze(0).view(self.kv_lora_rank, self.num_heads, self.qk_nope_head_dim).permute(1, 0, 2)
            )
            k_nope = torch.matmul(kva, k_up_per_head)
            key_states = torch.cat((k_nope, k_pe.expand(-1, self.num_heads, -1, -1)), dim=-1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        if attention_mask is not None:  # no matter the length, we just slice it
            attn_weights = torch.where(
                attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), attn_weights
            )

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_pe.dtype)
        attn_output = torch.matmul(attn_weights, kva)
        attn_output = torch.matmul(attn_output, self.per_head_v_up)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, compressed_kvs

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
        print("using orig forward")

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

        # kva = compressed_kv

        # ---- MLA absorption flags ----
        if mla_absorption is not None:
            enable_absorption = mla_absorption.get("enable", False)
            absorb_online = mla_absorption.get("online", False)
        else:
            enable_absorption = False

        n_head_ckv = kva.shape[1]
        p = self.num_heads // n_head_ckv
        seq_kv = kva.shape[2]

        # ---- Rotary ----
        cos, sin = self.rotary_emb(q_pe, seq_len=32 * 1024)  # Doesn't need q_pe as head_dim is initialized
        q_pe, k_pe = orig_apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        if compressed_kvs is not None:
            k_pe = compressed_kvs.update_k_pe(k_pe, self.layer_idx, cache_kwargs)

        kva_expanded = (
            kva.unsqueeze(2).expand(-1, -1, p, -1, -1).reshape(bsz, self.num_heads, seq_kv, self.kv_lora_rank)
        )

        k_pe_expanded = (
            k_pe.unsqueeze(2).expand(-1, -1, p, -1, -1).reshape(bsz, self.num_heads, seq_kv, self.qk_rope_head_dim)
        )

        v_up_per_head = self.v_up.squeeze(0).view(self.kv_lora_rank, self.num_heads, self.v_head_dim).permute(1, 0, 2)

        value_states = torch.matmul(kva_expanded, v_up_per_head)

        if enable_absorption:
            if absorb_online:
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
        if os.environ.get("KIMI_BLOCKING", "0") == "h":
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
        elif os.environ.get("KIMI_BLOCKING", "0") == "kv":
            return self.fused_forward_blocked_kv(
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
        elif os.environ.get("KIMI_BLOCKING", "0") == "basic":
            return self.fused_forward_basic(
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

        return attn_output, attn_weights, past_key_value, value_states

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

        kva = compressed_kv[:, :, : self.kv_lora_rank]
        k_pe = compressed_kv[:, :, self.kv_lora_rank :]

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

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "batch_index": batch_index, "position_ids": position_ids}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        n_head_ckv = 4  # kva.shape[1]

        attn_output_list = []
        attn_weights_list = []
        for head_block_idx in range(self.num_heads // n_head_ckv):
            h_start = head_block_idx * n_head_ckv
            h_end = min(h_start + n_head_ckv, self.num_heads)

            attn_weights = (
                torch.matmul(query_states[:, h_start:h_end, :, :], key_states[:, h_start:h_end, :, :].transpose(2, 3))
                * self.softmax_scale
            )

            if attention_mask is not None:
                attn_weights = torch.where(
                    attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), attn_weights
                )
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states[:, h_start:h_end, :, :])
            attn_output_list.append(attn_output)
            attn_weights_list.append(attn_weights)

        attn_output = torch.cat(attn_output_list, dim=1)
        attn_weights = torch.cat(attn_weights_list, dim=1)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value, value_states

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
        if os.environ.get("KIMI_BLOCKING", "0") == "h":
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


class QEffDeepseekV3MoE(nn.Module):
    def __qeff_init__(
        self,
    ):
        self.all_gate_proj = torch.nn.Parameter(
            torch.cat(
                [exp.gate_proj.compressor.decompress_module(exp.gate_proj).T.unsqueeze(0) for exp in self.experts],
                dim=0,
            )
        )
        self.all_up_proj = torch.nn.Parameter(
            torch.cat(
                [exp.up_proj.compressor.decompress_module(exp.up_proj).T.unsqueeze(0) for exp in self.experts], dim=0
            )
        )
        self.all_down_proj = torch.nn.Parameter(
            torch.cat(
                [exp.down_proj.compressor.decompress_module(exp.down_proj).T.unsqueeze(0) for exp in self.experts],
                dim=0,
            )
        )
        self.act_fn = self.experts[0].act_fn

    def moe(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)

        gate_proj = self.all_gate_proj[topk_indices.flatten()]
        up_proj = self.all_up_proj[topk_indices.flatten()]
        down_proj = self.all_down_proj[topk_indices.flatten()]
        expert_in = (
            hidden_states.unsqueeze(1).expand(-1, self.gate.top_k, -1).contiguous().view(-1, 1, self.config.hidden_size)
        )
        gate_out = torch.bmm(expert_in, gate_proj)
        up_out = torch.bmm(expert_in, up_proj)
        hidden = self.act_fn(gate_out) * up_out
        expert_output = torch.bmm(hidden, down_proj)
        experts_out = expert_output.view(seq_len, self.gate.top_k, self.config.hidden_size)
        experts_out = experts_out * topk_weights.unsqueeze(-1)

        final_hidden_states = torch.einsum("abc->ac", experts_out)

        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class QEffPrefillOnlyDeepseekV3MoE(nn.Module):
    def __qeff_init__(
        self,
    ):
        for exp in self.experts:
            gate_proj = torch.nn.Linear(self.config.hidden_size, self.config.moe_intermediate_size, bias=False)
            up_proj = torch.nn.Linear(self.config.hidden_size, self.config.moe_intermediate_size, bias=False)
            down_proj = torch.nn.Linear(self.config.moe_intermediate_size, self.config.hidden_size, bias=False)

            gate_proj.weight = torch.nn.Parameter(exp.gate_proj.compressor.decompress_module(exp.gate_proj))
            up_proj.weight = torch.nn.Parameter(exp.up_proj.compressor.decompress_module(exp.up_proj))
            down_proj.weight = torch.nn.Parameter(exp.down_proj.compressor.decompress_module(exp.down_proj))

            setattr(exp, "gate_proj", gate_proj)
            setattr(exp, "up_proj", up_proj)
            setattr(exp, "down_proj", down_proj)

    def moe(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, expert_mask: torch.Tensor, num_experts: int):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        for expert_idx in range(num_experts):
            expert = self.experts[expert_idx]
            gate_out = expert.gate_proj(hidden_states)
            up_out = expert.up_proj(hidden_states)
            hidden = expert.act_fn(gate_out) * up_out
            expert_output = expert.down_proj(hidden)
            current_hidden_states = expert_output * expert_mask[:, expert_idx].unsqueeze(-1)
            final_hidden_states += current_hidden_states

        print("\n\ninside prefill only moe\n")
        return final_hidden_states.type(hidden_states.dtype)

    def orig_moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        r"""
        CALL FOR CONTRIBUTION! I don't have time to optimise this right now, but expert weights need to be fused
        to not have to do a loop here (deepseek has 256 experts soooo yeah).
        """
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = expert_mask.permute(2, 0, 1)
        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output)

        # in original deepseek, the output of the experts are gathered once we leave this module
        # thus the moe module is itelsf an IsolatedParallel module
        # and all expert are "local" meaning we shard but we don't gather
        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states):
        """
        Forward pass of MoE block.
        """
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        # orig_out = self.orig_moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        mask = torch.zeros(hidden_states.shape[0], self.config.n_routed_experts)
        mask.scatter_(1, topk_indices, topk_weights)
        if os.environ.get("NUM_FFN_BLOCKS", None) is not None and os.environ.get("FFN_W_BLOCK_SIZE", None) is not None:
            hidden_states = self.moe_blocked_weights_forward(
                hidden_states, topk_weights, mask, self.config.n_routed_experts
            ).view(*orig_shape)
        elif os.environ.get("NUM_FFN_BLOCKS", None) is not None:
            hidden_states = self.moe_blocked_forward(
                hidden_states, topk_weights, mask, self.config.n_routed_experts
            ).view(*orig_shape)
        else:
            hidden_states = self.moe(hidden_states, topk_weights, mask, self.config.n_routed_experts).view(*orig_shape)

        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


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
        enable_mla: Optional[bool] = False,
        mla_absorption: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        orig_hidden_states = self.input_layernorm(hidden_states)
        if enable_mla:
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
            hidden_states, self_attn_weights, present_key_value, vs = self.self_attn(
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
            if enable_mla:
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
            max_position_embeddings=32 * 1024,
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
        enable_mla = getattr(self, "enable_mla", False)

        if enable_mla:
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
                enable_mla=getattr(self, "enable_mla", False),
                mla_absorption=getattr(self, "mla_absorption_config", None),
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
