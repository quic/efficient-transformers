# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import math
import os
from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    BaseModelOutputWithPooling,
    Qwen3_5MoeAttention,
    Qwen3_5MoeCausalLMOutputWithPast,
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeModel,
    Qwen3_5MoeModelOutputWithPast,
    Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeTextModel,
    Qwen3_5MoeTextRotaryEmbedding,
    Qwen3_5MoeVisionAttention,
    Qwen3_5MoeVisionModel,
    apply_rotary_pos_emb_vision,
    l2norm,
    repeat_kv,
    rotate_half,
)

from QEfficient.blocking.attention_blocking import (
    AttentionBlockingConfig,
    BlockingMode,
    generic_blocked_attention_interface,
)
from QEfficient.customop.ctx_scatter_gather import (
    CtxGatherFunc3DGeneralized,
    CtxScatterFunc3DGeneralized,
    CtxScatterFunc3DInt,
)
from QEfficient.customop.rms_norm import CustomRMSNormFunc
from QEfficient.transformers.cache_utils import QEffDynamicLayer
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo, get_padding_shape_from_config
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE
from QEfficient.utils.logging_utils import logger

# EXPERT_BLOCKING_NUM_NSP = 16
# EXPERT_BLOCKING_PACKED_CHUNK_SIZE = 32


class QEffQwen3_5MoeGatedDeltaNetCustomRMSNormAIC(nn.Module):
    """
    RMSNorm module that works by replacing the current module with compiler known custom-op.
    """

    def forward(self, hidden_states, gate):
        return (
            CustomRMSNormFunc.apply(
                hidden_states, self.weight, self.variance_epsilon if hasattr(self, "variance_epsilon") else self.eps
            )
        ) * F.silu(gate.to(torch.float32))


class QEffQwen3_5MoeDynamicCache(Cache):
    """
    Hybrid cache for Qwen3.5 models.

    Full-attention layers retain KV cache, while linear-attention layers retain
    convolution and recurrent states.
    """

    def __init__(self, config):
        super().__init__(layers=[])
        self.config = config
        self.layer_types = list(config.layer_types)
        self.transformer_layers = [i for i, layer_type in enumerate(self.layer_types) if layer_type == "full_attention"]
        self.last_linear_layer = next(
            (i for i in range(len(self.layer_types) - 1, -1, -1) if self.layer_types[i] == "linear_attention"),
            None,
        )
        self.kv_layers = [
            QEffDynamicLayer() if layer_type == "full_attention" else None for layer_type in self.layer_types
        ]
        self.conv_states = [None for _ in self.layer_types]
        self.recurrent_states = [None for _ in self.layer_types]

    @classmethod
    def from_legacy_cache(
        cls,
        config,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor, ...], ...]] = None,
    ) -> "QEffQwen3_5MoeDynamicCache":
        cache = cls(config)
        if past_key_values is None:
            return cache

        #
        for layer_idx, layer_state in enumerate(past_key_values):
            if cache.layer_types[layer_idx] == "full_attention":
                #
                key_states, value_states = layer_state
                layer = QEffDynamicLayer()
                layer.keys = key_states
                layer.values = value_states
                cache.kv_layers[layer_idx] = layer
            else:
                conv_state, recurrent_state = layer_state
                cache.conv_states[layer_idx] = conv_state
                cache.recurrent_states[layer_idx] = recurrent_state
        return cache

    def __len__(self):
        return len(self.layer_types)

    @property
    def key_cache(self):
        return [None if layer is None else layer.keys for layer in self.kv_layers]

    @property
    def value_cache(self):
        return [None if layer is None else layer.values for layer in self.kv_layers]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        layer = self.kv_layers[layer_idx]
        if layer is None:
            raise ValueError(f"Layer {layer_idx} is not a full_attention layer")
        return layer.update(key_states, value_states, cache_kwargs)

    def get_seq_length(self, layer_idx: Optional[int] = 0, cache_position: Optional[torch.LongTensor] = None) -> int:
        del cache_position
        if not self.transformer_layers:
            return 0
        if layer_idx not in self.transformer_layers:
            layer_idx = self.transformer_layers[0]
        layer = self.kv_layers[layer_idx]
        return 0 if layer is None or layer.keys is None else layer.keys.shape[-2]

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> Tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length(layer_idx)
        return query_length + past_seen_tokens, kv_offset

    def read_only_blockedKV(self, start_index: int, end_index: int, layer_idx: int, cache_kwargs: dict):
        layer = self.kv_layers[layer_idx]
        if layer is None:
            raise ValueError(f"Layer {layer_idx} is not a full_attention layer")
        return layer.read_only_blockedKV(start_index, end_index, cache_kwargs)

    def write_only(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs: dict):
        layer = self.kv_layers[layer_idx]
        if layer is None:
            raise ValueError(f"Layer {layer_idx} is not a full_attention layer")
        return layer.write_only(key_states, value_states, cache_kwargs)

    @property
    def has_previous_state(self) -> bool:
        if self.last_linear_layer is None:
            return False
        return self.conv_states[self.last_linear_layer] is not None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx, layer_type in enumerate(self.layer_types):
            if layer_type == "full_attention":
                layer = self.kv_layers[layer_idx]
                if layer is not None and layer.keys is not None:
                    device = layer.keys.device
                    beam_idx_device = beam_idx.to(device)
                    layer.keys = layer.keys.index_select(0, beam_idx_device)
                    layer.values = layer.values.index_select(0, beam_idx_device)
            elif self.conv_states[layer_idx] is not None:
                device = self.conv_states[layer_idx].device
                beam_idx_device = beam_idx.to(device)
                self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx_device)
                self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(0, beam_idx_device)

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, ...], ...]:
        legacy_cache = ()
        for layer_idx, layer_type in enumerate(self.layer_types):
            if layer_type == "full_attention":
                layer = self.kv_layers[layer_idx]
                if layer is None or layer.keys is None:
                    legacy_cache += ((torch.empty(0), torch.empty(0)),)
                else:
                    legacy_cache += ((layer.keys, layer.values),)
            else:
                conv_state = self.conv_states[layer_idx]
                recurrent_state = self.recurrent_states[layer_idx]
                legacy_cache += (
                    (
                        torch.empty(0) if conv_state is None else conv_state,
                        torch.empty(0) if recurrent_state is None else recurrent_state,
                    ),
                )
        return legacy_cache


class QEffQwen3_5MoeTextRotaryEmbedding(Qwen3_5MoeTextRotaryEmbedding):
    """
    QEff wrapper for Qwen3.5 text RoPE.

    Similar to Qwen3, this precomputes a reusable base cache and then indexes it
    with the current 3D RoPE position ids before applying the Qwen3.5 MRoPE
    interleaving pattern.
    """

    def __init__(self, config, device=None):
        super().__init__(config=config, device=device)
        self._set_cos_sin_cache(
            seq_len=self.original_max_seq_len,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])

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


def qeff_apply_interleaved_mrope(freqs, mrope_section):
    """Apply interleaved MRoPE to 3D rotary embeddings.
    Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    interleaved [THWTHWTHW...TT], preserving frequency continuity.
    args:
        x: (3, bs, seq_len, head_dim // 2)
        mrope_section: (3,)
    returns:
        x_t: (bs, seq_len, head_dim // 2)
    """

    half_shape = freqs[0].shape[-1] // 2
    freqs_t = freqs[0]
    for dim, offset in enumerate((1, 2), start=1):  # H, W
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
        offset += half_shape
        length += half_shape
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t


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

    cos = cos[position_ids]
    sin = sin[position_ids]

    cos = qeff_apply_interleaved_mrope(cos, mrope_section)
    sin = qeff_apply_interleaved_mrope(sin, mrope_section)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # import ipdb; ipdb.set_trace()
    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[:, :, :, :rotary_dim], q[:, :, :, rotary_dim:]
    k_rot, k_pass = k[:, :, :, :rotary_dim], k[:, :, :, rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)

    return q_embed, k_embed


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    #
    # MIN_MASKED_ATTENTION_VALUE = -10000
    if attention_mask is not None:
        attn_weights = torch.where(
            attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), attn_weights
        )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def qeff_torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    position_ids: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    idx = position_ids[0].flatten()
    zeros = torch.zeros(state_len, dtype=idx.dtype, device=idx.device)
    out = torch.cat([zeros, idx], dim=0)
    order = torch.argsort(out)  # sorted positions
    last4_positions = order[-state_len:]  # (4,)

    # ad_on = torch.where(hidden_states.shape[2] == torch.tensor(1), torch.tensor(1), cache_position.argmax(0))
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)

    updated_conv_state = hidden_states_new.index_select(2, last4_positions.long())
    # updated_conv_state = hidden_states_new[:, :, -state_len:].to(hidden_states_new.dtype)
    # updated_conv_state = hidden_states_new[:, :, position_ids[0].argmax(1) + 1: position_ids[0].argmax(1) + state_len].to(hidden_states_new.dtype)
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:]).to(hidden_states.dtype)
    return out, updated_conv_state


class QEffQwen3_5MoeAttention(Qwen3_5MoeAttention):
    """
    Full-attention path with QEff cache updates for retained-state export.
    """

    def __qeff_init__(self):
        # pass
        self.rotary_emb = QEffQwen3_5MoeTextRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[QEffQwen3_5MoeDynamicCache] = None,
        position_ids: Optional[torch.LongTensor] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        kv_seq_len = past_key_values.get_seq_length(self.layer_idx, cache_position)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = qeff_apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids[1:], self.rotary_emb.mrope_section
        )

        past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
        blocking_config = getattr(self, "attn_blocking_config", AttentionBlockingConfig())
        use_blocking = (
            past_key_values is not None and blocking_config is not None and (blocking_config.mode != BlockingMode.NONE)
        )

        if use_blocking:
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
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                scaling=self.scaling,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QEffQwen3_5MoeGatedDeltaNet(Qwen3_5MoeGatedDeltaNet):
    """
    Linear-attention path with explicit conv/recurrent retained-state updates.
    """

    def __qeff_init__(self):
        self.chunk_gated_delta_rule = self.torch_chunk_gated_delta_rule_qeff
        chunk_size = 64  # must match what's used in the function

        # Precompute all constant masks — no triu/tril with diagonal args at runtime
        # mask_causal: upper triangular including diagonal (diagonal=0)
        # = triu(ones, diagonal=0)
        mask_causal = torch.ones(chunk_size, chunk_size, dtype=torch.bool)
        for i in range(chunk_size):
            for j in range(i + 1):
                mask_causal[i, j] = False
        self.register_buffer("_mask_causal", mask_causal, persistent=False)
        # shape: (C, C), True above diagonal inclusive

        # mask_strict: strict upper triangular (diagonal=1)
        # = triu(ones, diagonal=1)
        mask_strict = torch.zeros(chunk_size, chunk_size, dtype=torch.bool)
        for i in range(chunk_size):
            for j in range(i + 1, chunk_size):
                mask_strict[i, j] = True
        self.register_buffer("_mask_strict", mask_strict, persistent=False)
        # shape: (C, C), True strictly above diagonal

        # ones_lower: lower triangular all-ones for cumsum replacement
        # = tril(ones, diagonal=0)
        ones_lower = torch.zeros(chunk_size, chunk_size)
        for i in range(chunk_size):
            for j in range(i + 1):
                ones_lower[i, j] = 1.0
        self.register_buffer("_ones_lower", ones_lower, persistent=False)
        # shape: (C, C)

        # eye: identity matrix
        self.register_buffer("_eye", torch.eye(chunk_size), persistent=False)

    def torch_chunk_gated_delta_rule_qeff(
        self,
        query,
        key,
        value,
        g,
        beta,
        position_ids,
        chunk_size=64,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        mask_causal=None,
        mask_strict=None,
        ones_lower=None,
        eye=None,
    ):
        initial_dtype = query.dtype
        if use_qk_l2norm_in_kernel:
            query = l2norm(query, dim=-1, eps=1e-6)
            key = l2norm(key, dim=-1, eps=1e-6)
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
        ]

        mask = (position_ids[0] != -1).unsqueeze(1)

        zeros = torch.zeros(g.shape, dtype=g.dtype, device=g.device)

        g = torch.where(mask, g, zeros)
        # beta = torch.where(mask, beta, zeros)

        qkv_zeros = torch.zeros(key.shape, dtype=key.dtype, device=key.device)
        key = torch.where(mask.unsqueeze(-1), key, qkv_zeros)
        query = torch.where(mask.unsqueeze(-1), query, qkv_zeros)
        value = torch.where(mask.unsqueeze(-1), value, qkv_zeros)

        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))

        # ck = g.clone()
        g = F.pad(g, (0, pad_size))
        total_sequence_length = sequence_length + pad_size
        scale = 1 / (query.shape[-1] ** 0.5)
        query = query * scale

        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)
        # reshape to chunks
        query, key, value, k_beta, v_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
        ]
        g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

        #
        # chunk decay
        # g = g.cumsum(dim=-1)

        L = g.size(-1)
        idx = torch.arange(L, device=g.device)
        mask_g = (idx.unsqueeze(1) >= idx.unsqueeze(0)).to(g.dtype)

        g = g @ mask_g.T

        #
        # decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril() # original decay_mask

        diff = g.unsqueeze(-1) - g.unsqueeze(-2)  # (B, H, num_chunks, C, C)
        diff = diff * (~mask_strict).float()  # zero upper triangle (strict)
        decay_mask = diff.exp().float()
        decay_mask = decay_mask * (~mask_strict).float()  # ensure upper is zero

        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
        # for i in range(1, chunk_size):
        #     row = attn[..., i, :i].clone()
        #     sub = attn[..., :i, :i].clone()
        #     attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        # attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

        ## Approximation code ##
        # A = attn
        # L = torch.eye(chunk_size, device=attn.device, dtype=attn.dtype)
        # Ak = A

        # K = 16
        # for _ in range(K):
        #     L = L + Ak
        #     Ak = Ak @ A

        # attn = L

        ## Factorized Approximation code ##
        eye = torch.eye(chunk_size, device=attn.device, dtype=attn.dtype)  #
        L = eye.clone()
        Apow = attn

        K = 32
        for _ in range(int(math.log2(K))):
            L = L @ (eye + Apow)
            Apow = Apow @ Apow  # square for next power

        attn = L

        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

        last_recurrent_state = (
            torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
            if initial_state is None
            else initial_state.to(value)
        )
        core_attn_out = torch.zeros_like(value)
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

        # for each chunk
        for i in range(0, total_sequence_length // chunk_size):
            q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
            attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
            v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
            v_new = v_i - v_prime
            attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
            core_attn_out[:, :, i] = attn_inter + attn @ v_new
            last_recurrent_state = (
                last_recurrent_state * g[:, :, i, -1, None, None].exp()
                + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
            )

        if not output_final_state:
            last_recurrent_state = None
        core_attn_out = core_attn_out.reshape(
            core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
        )
        core_attn_out = core_attn_out[:, :, :sequence_length]
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
        return core_attn_out, last_recurrent_state

    def _recurrent_step_batched(self, query, key, value, g, beta, recurrent_state):
        """
        Pure tensor ops, no loop, no padding.
        Works for any T but intended for T=1 decode.
        Shapes: query/key/value (B, T, H, d_k/d_v)
        """
        dtype = query.dtype

        # L2 norm (matching chunk kernel behavior)
        q = query.float()
        k = key.float()
        q = q * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)
        v = value.float()

        scale = 1.0 / (q.shape[-1] ** 0.5)
        q = q * scale  # (B, T, H, d_k)

        # For T=1 decode, this is a single step
        # Transpose to (B, H, T, d_k/d_v) to match recurrent state layout
        q = q.transpose(1, 2)  # (B, H, T, d_k)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        b = beta.transpose(1, 2).float().unsqueeze(-1)  # (B, H, T, 1)
        decay = g.transpose(1, 2).float().exp()  # (B, H, T)
        decay = decay.unsqueeze(-1).unsqueeze(-1)  # (B, H, T, 1, 1)

        S = recurrent_state.float()  # (B, H, d_k, d_v)

        # Single step — no loop because T=1
        # S update
        S_decayed = S * decay[:, :, 0]  # (B, H, d_k, d_v)
        kv_mem = (S_decayed * k[:, :, 0].unsqueeze(-1)).sum(dim=-2)  # (B, H, d_v)
        delta = (v[:, :, 0] - kv_mem) * b[:, :, 0]  # (B, H, d_v)
        S_new = S_decayed + k[:, :, 0].unsqueeze(-1) * delta.unsqueeze(-2)  # (B, H, d_k, d_v)
        out = (S_new * q[:, :, 0].unsqueeze(-1)).sum(dim=-2)  # (B, H, d_v)

        out = out.unsqueeze(2).transpose(1, 2).to(dtype)  # (B, 1, H, d_v) → (B, T, H, d_v)
        return out, S_new.to(recurrent_state.dtype)

    def forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None, position_ids=None):
        batch_size, seq_len, _ = hidden_states.shape

        # ── Projections ──────────────────────────────────────
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        beta = self.in_proj_b(hidden_states).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias)

        # ── Conv (unified, handles T=1 and T=N) ──────────────
        if cache_params is not None:
            #
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]
            mixed_qkv, new_conv_state = qeff_torch_causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                position_ids,
                self.conv1d.bias,
            )
            cache_params.conv_states[self.layer_idx] = new_conv_state
        else:
            recurrent_state = None
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        # ── Split Q/K/V ──────────────────────────────────────
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        # ── Recurrent State ───────────────────────────────────
        if cache_params is not None:
            # Decode branch — pure tensor ops, no loop, no padding
            # Shape: (B, 1, H, d_v), (B, H, d_k, d_v)
            recurrent_out, recurrent_S = self._recurrent_step_batched(query, key, value, g, beta, recurrent_state)

            # Prefill branch — chunked parallel scan
            # Shape: (B, T, H, d_v), (B, H, d_k, d_v)
            chunk_out, chunk_S = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                position_ids=position_ids,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                mask_causal=self._mask_causal,
                mask_strict=self._mask_strict,
                ones_lower=self._ones_lower,
                eye=self._eye,
            )

            # Select based on seq_len
            # is_decode is SCALAR — torch.where broadcasts efficiently
            # HW predicates entire branch at runtime
            is_decode = hidden_states.shape[1] == torch.tensor(1)

            core_attn_out = torch.where(is_decode, recurrent_out, chunk_out)
            last_recurrent_state = torch.where(is_decode, recurrent_S, chunk_S)

            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        else:
            # No cache — prefill only, no state needed
            core_attn_out, _ = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                mask_causal=self._mask_causal,
                mask_strict=self._mask_strict,
                ones_lower=self._ones_lower,
                eye=self._eye,
            )

        #
        # ── Output ────────────────────────────────────────────
        core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        # core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        return self.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))

    @staticmethod
    def apply_mask_to_padding_states(hidden_states, attention_mask):
        if attention_mask is not None and attention_mask.shape[1] > 1:
            dtype = hidden_states.dtype
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
        return hidden_states


class QEffQwen3_5MoeDecoderLayer(Qwen3_5MoeDecoderLayer):
    def __qeff_init__(self):
        #
        if self.layer_type == "linear_attention":
            self.linear_attn.__class__ = QEffQwen3_5MoeGatedDeltaNet
            self.linear_attn.__qeff_init__()
        elif self.layer_type == "full_attention":
            self.self_attn.__class__ = QEffQwen3_5MoeAttention
            self.self_attn.__qeff_init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[QEffQwen3_5MoeDynamicCache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        del use_cache
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # For the MoE layers, we need to unpack
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states
        return hidden_states


class QEffQwen3_5MoeTextModel(Qwen3_5MoeTextModel):
    # def __qeff_init__(self):
    #     self.rotary_emb = QEffQwen3_5MoeTextRotaryEmbedding(config=self.config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[QEffQwen3_5MoeDynamicCache, Tuple[Tuple[torch.FloatTensor, ...], ...]]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_legacy_cache = False

        if past_key_values is not None and not isinstance(past_key_values, QEffQwen3_5MoeDynamicCache):
            return_legacy_cache = True
            past_key_values = QEffQwen3_5MoeDynamicCache.from_legacy_cache(self.config, past_key_values)
        elif use_cache and past_key_values is None:
            past_key_values = QEffQwen3_5MoeDynamicCache(self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens
        causal_mask = _create_causal_mask(
            position_ids=position_ids[0], target_length=target_length, sliding_window=None
        )
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids[1:])
        # position_embeddings = None
        all_hidden_states = () if output_hidden_states else None
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_mask = linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            # break

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class QEffQwen3_5MoeForCausalLM(Qwen3_5MoeForCausalLM):
    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffQwen3_5MoeDecoderLayer}

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if hasattr(past_key_values, "reorder_cache"):
            past_key_values.reorder_cache(beam_idx)
        return past_key_values

    def _iter_retained_state_names(self) -> List[str]:
        names = []
        for layer_idx, layer_type in enumerate(self.config.layer_types):
            if layer_type == "full_attention":
                names.extend([f"past_key.{layer_idx}", f"past_value.{layer_idx}"])
            else:
                names.extend([f"conv_state.{layer_idx}", f"recurrent_state.{layer_idx}"])
        return names

    def get_retained_state_names(self) -> List[str]:
        return self._iter_retained_state_names()

    def get_onnx_retained_state_specs(
        self,
        batch_size: int,
        seq_len: int,
        kv_cache_shape: List[int],
        continuous_batching: bool = False,
        retain_full_kv: bool = False,
    ) -> dict:
        del seq_len, retain_full_kv
        batch_axis_name = "full_batch_size" if continuous_batching else "batch_size"
        specs = {
            "past_key_values": [],
            "input_names": [],
            "output_names": [],
            "dynamic_axes": {},
        }

        for layer_idx, layer_type in enumerate(self.config.layer_types):
            if layer_type == "full_attention":
                layer_names = [f"past_key.{layer_idx}", f"past_value.{layer_idx}"]
                layer_tensors = [
                    torch.zeros(tuple(kv_cache_shape), dtype=torch.float32),
                    torch.zeros(tuple(kv_cache_shape), dtype=torch.float32),
                ]
                layer_axes = [
                    {0: batch_axis_name, 2: "ctx_len"},
                    {0: batch_axis_name, 2: "ctx_len"},
                ]
            else:
                layer = self.model.layers[layer_idx].linear_attn
                conv_shape = (batch_size, layer.conv_dim, layer.conv_kernel_size)
                recurrent_shape = (batch_size, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim)
                layer_names = [f"conv_state.{layer_idx}", f"recurrent_state.{layer_idx}"]
                layer_tensors = [
                    torch.zeros(conv_shape, dtype=torch.float32),
                    torch.zeros(recurrent_shape, dtype=torch.float32),
                ]
                layer_axes = [{0: batch_axis_name}, {0: batch_axis_name}]

            specs["past_key_values"].append(layer_tensors)
            for name, axes in zip(layer_names, layer_axes):
                specs["input_names"].append(name)
                specs["output_names"].append(f"{name}_RetainedState")
                specs["dynamic_axes"][name] = axes

        return specs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[QEffQwen3_5MoeDynamicCache, Tuple[Tuple[torch.FloatTensor, ...], ...]]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        del logits_to_keep
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        if position_ids is None:
            hidden_states = outputs.last_hidden_state[:, -1:, :]
        else:
            text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
            logit_index = text_position_ids.to(torch.int32).argmax(1, keepdim=True)
            hidden_states = outputs.last_hidden_state[torch.arange(text_position_ids.shape[0]).view(-1, 1), logit_index]

        logits = self.lm_head(hidden_states).float()
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class QEffQwen3_5MoeModel(Qwen3_5MoeModel):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | Qwen3_5MoeModelOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_outputs: BaseModelOutputWithPooling = self.get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            )
            image_embeds = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # if pixel_values_videos is not None:
        #     video_outputs: BaseModelOutputWithPooling = self.get_video_features(
        #         pixel_values_videos, video_grid_thw, return_dict=True
        #     )
        #     video_embeds = video_outputs.pooler_output
        #     video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        #     _, video_mask = self.get_placeholder_mask(
        #         input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        #     )
        #     inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        return Qwen3_5MoeModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )


class QEffQwen3_5MoeVisionModel(Qwen3_5MoeVisionModel):
    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        max_hw = max(grid_thw.shape)
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device
        bs, num_frames, height, width = grid_thw.shape
        grid_thw = (torch.tensor(grid_thw.shape, dtype=torch.int64)).unsqueeze(0)

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        merged_h, merged_w = height // merge_size, width // merge_size

        block_rows = torch.arange(merged_h, device=device)
        block_cols = torch.arange(merged_w, device=device)
        intra_row = torch.arange(merge_size, device=device)
        intra_col = torch.arange(merge_size, device=device)

        row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

        row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

        coords = torch.stack((row_idx, col_idx), dim=-1)

        if num_frames > 1:
            coords = coords.repeat(num_frames, 1)

        pos_ids = coords
        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        bs, t, h, w = grid_thw.shape
        h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        max_t = torch.tensor(self.num_grid_per_side - 1, device=h_idxs.device)

        h_idxs_ceil = torch.minimum(h_idxs_floor + 1, max_t)
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

        idx_tensor = torch.stack(indices, dim=0).to(dtype=torch.long, device=self.pos_embed.weight.device)

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

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
        hidden_states = self.merger(hidden_states)
        return hidden_states


class QEffQwen3_5MoeVisionAttention(Qwen3_5MoeVisionAttention):
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
        seq_len = attention_mask.shape[-1]
        rows = torch.arange(seq_len).view(1, -1)
        cols = torch.arange(seq_len).view(-1, 1)

        start = cu_seqlens[:-1].view(-1, 1, 1)
        end = cu_seqlens[1:].view(-1, 1, 1)
        row_mask = (rows >= start) & (rows < end)
        col_mask = (cols >= start) & (cols < end)
        block_mask = row_mask & col_mask

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


class QEffQwen3_5MoeEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_submodules_for_export(self) -> Type[nn.Module]:
        if hasattr(self.model.model, "visual") and hasattr(self.model.model.visual, "blocks"):
            return {self.model.model.visual.blocks[0].__class__}
        if hasattr(self.model.model, "vision_model") and hasattr(self.model.model.vision_model, "blocks"):
            return {self.model.model.vision_model.blocks[0].__class__}
        return set()

    def forward(self, pixel_values, image_grid_thw):
        if hasattr(self.model.model, "visual"):
            image_outputs = self.model.model.visual(pixel_values, grid_thw=image_grid_thw)
            image_embeds = image_outputs[0] if isinstance(image_outputs, tuple) else image_outputs
        else:
            image_outputs: BaseModelOutputWithPooling = self.model.model.get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            )
            image_embeds = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(pixel_values.device, pixel_values.dtype)
        bs = image_grid_thw.shape[0]
        split_size = torch.floor_divide(torch.tensor(image_embeds.size(0)), bs)
        image_embeds = image_embeds.reshape(bs, split_size, image_embeds.size(1))
        return image_embeds


class QEffQwen3_5MoeDecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.language_model = self.model.model.language_model

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffQwen3_5MoeDecoderLayer}

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
        inputs_embeds = self.model.model.get_input_embeddings()(input_ids)
        _, _, channel_size = inputs_embeds.shape
        selected = input_ids == self.model.config.image_token_id
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(selected.unsqueeze(0).shape[0]).view(-1, 1)
        image_features_expanded = vision_embeds.reshape(-1, channel_size).unsqueeze(0)[indices0, indices1]
        image_input_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
        inputs_embeds = image_input_embeds

        outputs = self.language_model(
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
        return logits, vision_embeds, image_idx, outputs.past_key_values[: len(past_key_values)]


class QEffQwen3_5MoeForConditionalGeneration(Qwen3_5MoeForConditionalGeneration):
    def get_qeff_vision_encoder(self):
        return QEffQwen3_5MoeEncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEffQwen3_5MoeDecoderWrapper(self)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> tuple | Qwen3_5MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:

        ```python
        >>> from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

        >>> model = Qwen3_5MoeForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]

        >>> inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=1024)
        >>> generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        >>> output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        >>> print(output_text)
        ```
        """

        #

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )

        hidden_states = outputs[0]

        logit_index = position_ids[0].to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[torch.arange(position_ids[0].shape[0]).view(-1, 1), logit_index]
        #
        logits = self.lm_head(hidden_states)

        # loss = None
        # if labels is not None:
        #     loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return logits, outputs.past_key_values[: len(past_key_values)]

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        img_size: None,
        height: int = None,
        width: int = None,
        time: int = 1,
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
            height = constants.QWEN3_VL_HEIGHT
            width = constants.QWEN3_VL_WIDTH
            logger.warning(
                f"Setting height and width to be {height} and {width} respectively, as it was neither passed nor found in vision_config"
            )

        prefill_seq_len = prefill_seq_len if prefill_seq_len else 128
        ctx_len = ctx_len if ctx_len else constants.INTERN_CTX_LEN
        kv_cache_batch_size = kv_cache_batch_size or full_batch_size or batch_size
        channel = 3
        patch_size = self.config.vision_config.patch_size
        temporal_patch_size = getattr(self.config.vision_config, "temporal_patch_size", 1)

        IMAGE_FACTOR = 32
        MIN_PIXELS = 64 * 32 * 32
        MAX_PIXELS = 16384 * 32 * 32
        MAX_RATIO = 200

        def round_by_factor(number: int, factor: int) -> int:
            return round(number / factor) * factor

        def ceil_by_factor(number: int, factor: int) -> int:
            return math.ceil(number / factor) * factor

        def floor_by_factor(number: int, factor: int) -> int:
            return math.floor(number / factor) * factor

        def smart_resize(
            height: int,
            width: int,
            factor: int = IMAGE_FACTOR,
            min_pixels: int = MIN_PIXELS,
            max_pixels: int = MAX_PIXELS,
        ) -> tuple[int, int]:
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
        vision_size = (grid_height // 4) * num_frames * time
        grid_height = grid_height * time * batch_size

        vision = [
            {
                "batch_size": batch_size,
                "vision_size": vision_size,
                "grid_height": grid_height,
                "grid_width": grid_width,
                "time": time,
                "grid_h": grid_h,
                "grid_w": grid_w,
            }
        ]

        def _build_lang_spec(seq_len_val, comp_ctx_len=None):
            spec = {
                "batch_size": full_batch_size
                if (continuous_batching and seq_len_val == 1)
                else (1 if continuous_batching else batch_size),
                "seq_len": seq_len_val,
                "ctx_len": ctx_len,
            }
            if kv_offload:
                spec["vision_size"] = vision_size
                spec["vision_batch_size"] = batch_size
            if comp_ctx_len is not None:
                spec["comp_ctx_lengths"] = comp_ctx_len
            if continuous_batching:
                spec["full_batch_size"] = kv_cache_batch_size
            else:
                spec["batch_size"] = kv_cache_batch_size
            if full_batch_size and seq_len_val != 1:
                spec["full_batch_exec_size"] = full_batch_size
            return spec

        lang = []
        if comp_ctx_lengths_prefill is not None:
            for comp_ctx in comp_ctx_lengths_prefill:
                lang.append(_build_lang_spec(prefill_seq_len, comp_ctx_len=comp_ctx))
            for comp_ctx in comp_ctx_lengths_decode or []:
                lang.append(_build_lang_spec(1, comp_ctx_len=comp_ctx))
        else:
            lang.append(_build_lang_spec(prefill_seq_len))
            lang.append(_build_lang_spec(1))

        if kv_offload:
            return {"vision": vision, "lang": lang}, compiler_options

        for spec in lang:
            spec.pop("vision_size", None)
            spec.pop("vision_batch_size", None)
        return lang, compiler_options

    def get_onnx_dynamic_axes(
        self, comp_ctx_lengths: Optional[List[int]] = None, kv_offload: bool = False, continuous_batching: bool = False
    ):
        num_layers = self.config.text_config.num_hidden_layers
        batch_axis_name = "full_batch_size" if continuous_batching else "batch_size"

        vision_dynamic_axes = {
            "pixel_values": {0: "grid_height", 1: "grid_width"},
            "image_grid_thw": {0: "batch_size", 1: "time", 2: "grid_h", 3: "grid_w"},
        }

        lang_dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {1: "batch_size", 2: "seq_len"},
            "vision_embeds": {0: "vision_batch_size", 1: "vision_size"},
        }

        for i in range(num_layers):
            if self.config.text_config.layer_types[i] == "full_attention":
                lang_dynamic_axes[f"past_key.{i}"] = {0: batch_axis_name, 2: "ctx_len"}
                lang_dynamic_axes[f"past_value.{i}"] = {0: batch_axis_name, 2: "ctx_len"}
            else:
                lang_dynamic_axes[f"past_key.{i}"] = {0: batch_axis_name}
                lang_dynamic_axes[f"past_value.{i}"] = {0: batch_axis_name}

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
            dynamic_axes = lang_dynamic_axes

        return dynamic_axes

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        **kwargs,
    ):
        inputs_shapes = {}

        dummy_seq_len = 128
        inputs_shapes["input_ids"] = (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, dummy_seq_len)

        inputs_shapes["position_ids"] = (
            3,
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            dummy_seq_len,
        )
        inputs_shapes["pixel_values"] = (11008, 1536)
        inputs_shapes["image_grid_thw"] = (
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            1,
            86,
            128,
        )
        inputs_shapes["vision_embeds"] = (
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            2752,
            2048,
        )
        inputs_shapes["image_idx"] = (1, 1)

        vision_inputs = {}
        lang_inputs = {}
        vision_inputs["pixel_values"] = torch.zeros((inputs_shapes["pixel_values"]), dtype=torch.float32)
        vision_inputs["image_grid_thw"] = torch.zeros((inputs_shapes["image_grid_thw"]), dtype=torch.int64)
        lang_inputs["input_ids"] = torch.zeros((inputs_shapes["input_ids"]), dtype=torch.int64)
        lang_inputs["vision_embeds"] = torch.zeros((inputs_shapes["vision_embeds"]), dtype=torch.float32)
        lang_inputs["position_ids"] = (
            (
                torch.arange(dummy_seq_len, dtype=torch.int64)
                .view(1, dummy_seq_len)
                .repeat(constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, 1)
            )
            .unsqueeze(0)
            .repeat(4, 1, 1)
        )
        lang_inputs["image_idx"] = torch.zeros((inputs_shapes["image_idx"]), dtype=torch.int64)

        bs: int = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        fbs: int = constants.ONNX_EXPORT_EXAMPLE_FBS

        # Add data for KV
        # kv_cache_shape = get_padding_shape_from_config(
        #     config=self.model.config.text_config,
        #     batch_size=fbs if continuous_batching else bs,
        #     seq_len=dummy_seq_len,
        # )

        kv_cache_shape = get_padding_shape_from_config(
            config=self.model.config.text_config,
            batch_size=fbs if continuous_batching else bs,
            seq_len=dummy_seq_len,
        )

        linear_batch_size = fbs if continuous_batching else bs

        lang_inputs["past_key_values"] = [[] for _ in range(self.model.config.text_config.num_hidden_layers)]
        for i in range(self.model.config.text_config.num_hidden_layers):
            if self.model.config.text_config.layer_types[i] == "full_attention":
                for kv in ["key", "value"]:
                    lang_inputs["past_key_values"][i].append(torch.zeros(kv_cache_shape, dtype=torch.float32))
            else:
                layer = self.model.language_model.layers[i].linear_attn
                conv_shape = (linear_batch_size, layer.conv_dim, layer.conv_kernel_size)
                recurrent_shape = (linear_batch_size, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim)
                lang_inputs["past_key_values"][i].append(torch.zeros(conv_shape, dtype=torch.float32))
                lang_inputs["past_key_values"][i].append(torch.zeros(recurrent_shape, dtype=torch.float32))

        #
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
            lang_inputs.pop("image_idx")
            inputs = lang_inputs

        return inputs

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
            return lang_output_names
        return output_names

    def get_inputs_info(self):
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="attention_mask", datatype=torch.int64, shape=("batch_size", "seq_len")),
            # IOInfo(name="pixel_values", datatype=torch.float32, shape=("batch_size", 3, "image_size", "image_size")),
        ]

    def prepare_inputs_for_generation(self, inputs, prefill_seq_len=32, batch_size=1):

        input_ids_length = inputs["input_ids"].shape[1]
        inputs["position_ids"] = torch.arange(input_ids_length).view(1, 1, input_ids_length).expand(-1, batch_size, -1)
        pos_ids, rope_deltas = self.model.get_rope_index(
            inputs["input_ids"],
            inputs["mm_token_type_ids"],
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

        inputs.pop("mm_token_type_ids")
        return inputs


class QEffQwen3_5MoeSparseMoeBlock(Qwen3_5MoeSparseMoeBlock):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, H = hidden_states.shape
        T = B * S
        x = hidden_states.view(T, H)
        prob, top_w, top_i = self.gate(hidden_states)
        idx = top_i.reshape(-1)

        w_up = self.experts.gate_up_proj[idx.flatten()]
        w_dn = self.experts.down_proj[idx.flatten()]

        xk = x.unsqueeze(1).expand(-1, self.gate.top_k, -1).contiguous()
        xk = xk.view(-1, 1, H)

        gate_proj, up_proj = torch.chunk(w_up, 2, dim=1)
        gate = torch.bmm(xk, gate_proj.transpose(1, 2))
        up = torch.bmm(xk, up_proj.transpose(1, 2))

        intermediate = up * self.experts.act_fn(gate)
        experts_out = torch.bmm(intermediate, w_dn.transpose(1, 2))
        experts_out = experts_out.view(T, self.gate.top_k, H) * top_w.unsqueeze(-1)
        experts_out = torch.einsum("bnd->bd", experts_out)

        shared_expert_output = self.shared_expert(x)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(x)) * shared_expert_output

        expert_output = experts_out + shared_expert_output
        return expert_output.reshape(B, S, H)


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


def _cumsum_scatter_gather_update_expert_blocked(
    x: torch.Tensor,
    T2Ei: torch.Tensor,
    W_g: torch.Tensor,
    W_u: torch.Tensor,
    W_d: torch.Tensor,
    routing_weight: torch.Tensor,
    experts_out: torch.Tensor,
    act_fn,
    T: int,
    packed_chunk_size: int,
) -> torch.Tensor:
    """Cumsum-scatter-gather-update expert helper for NSP-blocked dispatch.

    Accumulates one local expert's contribution in-place onto ``experts_out``.
    Uses a packed/cumsum layout so the MLP runs only over active rows, then
    scatters the weighted output back to original token positions.

    Shapes:
        x               : [T, H]
        T2Ei            : [num_nsp, T]            (bool)
        W_g, W_u        : [num_nsp, H, I]
        W_d             : [num_nsp, I, H]
        routing_weight  : [num_nsp, T]
        experts_out      : [num_nsp, T, H]         (accumulator, in-out)
    """
    batch_size, seq_len = T2Ei.shape
    packed_chunk_size = int(max(1, min(packed_chunk_size, seq_len)))

    matched_idx = _build_matched_idx_from_cumsum(T2Ei)
    valid_rows = T2Ei.to(torch.int32).sum(dim=1, keepdim=True)
    row_range = torch.arange(packed_chunk_size, dtype=torch.int32, device=x.device).unsqueeze(0)
    x_expanded = x.unsqueeze(0).expand(batch_size, -1, -1)
    rw_expanded = routing_weight.unsqueeze(-1)
    for packed_start in range(0, seq_len, packed_chunk_size):
        packed_stop = packed_start + packed_chunk_size
        chunk_matched_idx = matched_idx[:, packed_start:packed_stop]

        x_chunk = CtxGatherFunc3DGeneralized.apply(x_expanded, chunk_matched_idx)

        gate_prime = x_chunk @ W_g
        up_prime = x_chunk @ W_u
        down_chunk = (up_prime * act_fn(gate_prime)) @ W_d

        rw_chunk = CtxGatherFunc3DGeneralized.apply(rw_expanded, chunk_matched_idx)
        down_chunk = down_chunk * rw_chunk

        expert_out_chunk = CtxGatherFunc3DGeneralized.apply(experts_out, chunk_matched_idx)
        updated_chunk = expert_out_chunk + down_chunk

        chunk_valid_rows = torch.clamp(valid_rows - packed_start, min=0, max=packed_chunk_size)
        updated_chunk = torch.where(
            (row_range < chunk_valid_rows).unsqueeze(-1), updated_chunk, torch.zeros_like(updated_chunk)
        )
        experts_out = CtxScatterFunc3DGeneralized.apply(experts_out, chunk_matched_idx, updated_chunk)

    return experts_out


class QEffPrefillChunkedQwen3_5MoeSparseMoeBlock(Qwen3_5MoeSparseMoeBlock):
    def _forward_expert_blocked(self, x: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        act_fn = getattr(self.experts, "act_fn", F.silu)
        T, H = x.shape
        num_nsp = EXPERT_BLOCKING_NUM_NSP
        if self.gate.num_experts % num_nsp != 0:
            raise ValueError(
                f"num_experts ({self.gate.num_experts}) must be divisible by EXPERT_BLOCKING_NUM_NSP ({num_nsp})"
            )
        local_experts = self.gate.num_experts // num_nsp
        rw = routing_weights.transpose(0, 1).contiguous().view(local_experts, num_nsp, T).transpose(0, 1).contiguous()
        experts_out = x.new_zeros((num_nsp, T, H))
        inter = self.experts.gate_up_proj.shape[1] // 2

        # gate_up_proj is [E, 2I, H]. After split we get [E, I, H], so transpose to [E, H, I]
        # before grouping into [num_nsp, local_experts, H, I].
        wt_g, wt_u = torch.split(self.experts.gate_up_proj, inter, dim=1)
        wt_g = wt_g.transpose(1, 2).contiguous()
        wt_u = wt_u.transpose(1, 2).contiguous()
        W_g = wt_g.view(local_experts, num_nsp, H, -1).transpose(0, 1).contiguous()
        W_u = wt_u.view(local_experts, num_nsp, H, -1).transpose(0, 1).contiguous()

        # down_proj is [E, H, I]; blocked matmul expects [num_nsp, local_experts, I, H].
        W_d = self.experts.down_proj.transpose(1, 2).contiguous()
        W_d = W_d.view(local_experts, num_nsp, -1, H).transpose(0, 1).contiguous()

        for slot in range(local_experts):
            routing_weight = rw[:, slot, :]
            T2Ei = routing_weight > 0
            experts_out = _cumsum_scatter_gather_update_expert_blocked(
                x=x,
                T2Ei=T2Ei,
                W_g=W_g[:, slot],
                W_u=W_u[:, slot],
                W_d=W_d[:, slot],
                routing_weight=routing_weight,
                experts_out=experts_out,
                act_fn=act_fn,
                T=T,
                packed_chunk_size=EXPERT_BLOCKING_PACKED_CHUNK_SIZE,
            )
        return experts_out.sum(dim=0)

    # def orig_forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    #     B, S, H = hidden_states.shape
    #     T = B * S
    #     x = hidden_states.view(T, H)
    #     router_logits = self.gate(x)  # [T, E]
    #     prob = F.softmax(router_logits, -1, dtype=torch.float)
    #     top_w, top_i = torch.topk(prob, self.top_k, -1)
    #     if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
    #         top_w /= top_w.sum(-1, keepdim=True)
    #     top_w = top_w.to(hidden_states.dtype)
    #     masked_logits = torch.zeros_like(router_logits)
    #     masked_logits.scatter_(1, top_i, top_w)
    #     routing_weights = masked_logits
    #     experts_out = x.new_zeros((T, H))
    #     for e in range(self.gate.num_experts):
    #         routing_weight = routing_weights[:, e].unsqueeze(-1)
    #         W_g, W_u = self.experts[e].gate_proj.weight.T, self.experts[e].up_proj.weight.T
    #         W_d = self.experts[e].down_proj.weight.T
    #         gate = x @ W_g
    #         up = x @ W_u
    #         down = (up * self.experts[e].act_fn(gate)) @ W_d
    #         experts_out += down * routing_weight

    #     shared_expert_output = self.shared_expert(x)
    #     shared_expert_output = F.sigmoid(self.shared_expert_gate(x)) * shared_expert_output

    #     experts_out = experts_out + shared_expert_output
    #     return experts_out.view(B, S, H), router_logits

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, H = hidden_states.shape
        T = B * S
        x = hidden_states.view(T, H)
        act = getattr(self.experts, "act_fn", F.silu)

        prob, top_w, top_i = self.gate(hidden_states)
        routing_weights = torch.zeros((T, self.gate.num_experts), dtype=x.dtype)
        routing_weights.scatter_(1, top_i, top_w)

        if self.gate.num_experts % EXPERT_BLOCKING_NUM_NSP == 0:
            experts_out = self._forward_expert_blocked(x=x, routing_weights=routing_weights)

            shared_expert_output = self.shared_expert(x)
            shared_expert_output = F.sigmoid(self.shared_expert_gate(x)) * shared_expert_output
            expert_output = experts_out + shared_expert_output
            return expert_output.view(B, S, H)

        experts_out = torch.zeros_like(x, dtype=x.dtype)
        # breakpoint()
        for e in range(self.gate.num_experts):
            routing_weight = routing_weights[:, e].unsqueeze(-1)

            W_gate_up_e = self.experts.gate_up_proj[e]  # [H, 2I]
            W_dn_e = self.experts.down_proj[e]  # [I, H]
            #
            gate_up = x @ W_gate_up_e.T  # [T, 2I]

            I2 = gate_up.shape[-1] // 2
            gate = gate_up[:, :I2]  # [T, I]
            up = gate_up[:, I2:]  # [T, I]
            intermediate = up * act(gate)
            down = intermediate @ W_dn_e.T
            masked_down = torch.where(
                routing_weight > 0, down * routing_weight, torch.zeros_like(experts_out, dtype=down.dtype)
            )
            # masked_down = down * routing_weight
            experts_out += masked_down

        shared_expert_output = self.shared_expert(x)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(x)) * shared_expert_output

        expert_output = experts_out + shared_expert_output
        return expert_output.reshape(B, S, H)
