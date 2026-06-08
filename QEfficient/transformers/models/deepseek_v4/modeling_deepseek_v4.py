# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import math
from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4Config,
    DeepseekV4DecoderLayer,
    DeepseekV4Experts,
    DeepseekV4ForCausalLM,
    DeepseekV4HashRouter,
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
    DeepseekV4MLP,
    DeepseekV4Model,
    DeepseekV4RMSNorm,
    DeepseekV4RotaryEmbedding,
    DeepseekV4SparseMoeBlock,
    DeepseekV4TopKRouter,
    DeepseekV4UnweightedRMSNorm,
)

from QEfficient.customop.rms_norm import CustomRMSNormFunc
from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


def rotate_half_interleaved(x):
    """Rotate for interleaved RoPE (V4 style): pairs consecutive channels."""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def qeff_apply_rotary_pos_emb_v4(x, cos, sin, position_ids, rope_dim):
    """Apply interleaved RoPE to the rope-slice of x, leave the rest untouched.

    V4 uses partial rotary: only the first `rope_dim` channels get RoPE.
    cos/sin shape: [seq_len, rope_dim//2] (no duplication in V4's RotaryEmbedding).
    We repeat_interleave to get [seq_len, rope_dim] for the rotation.
    """
    cos_pos = cos[position_ids]  # [B, S, rope_dim//2]
    sin_pos = sin[position_ids]  # [B, S, rope_dim//2]
    cos_pos = cos_pos.repeat_interleave(2, dim=-1).unsqueeze(1)  # [B, 1, S, rope_dim]
    sin_pos = sin_pos.repeat_interleave(2, dim=-1).unsqueeze(1)  # [B, 1, S, rope_dim]

    x_rope = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]

    x_rotated = (x_rope * cos_pos) + (rotate_half_interleaved(x_rope) * sin_pos)
    if x_pass.shape[-1] == 0:
        return x_rotated.to(x.dtype)
    return torch.cat([x_rotated, x_pass], dim=-1).to(x.dtype)


class QEffDeepseekV4RotaryEmbedding(nn.Module):
    """Precomputed static sin/cos cache for V4's multi-type interleaved RoPE."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.config = config
        self.rope_dim = config.qk_rope_head_dim  # 64
        max_pos = config.max_position_embeddings

        rope_params = config.rope_parameters or {}
        self.layer_types = [k for k, v in rope_params.items() if isinstance(v, dict)]

        for layer_type in self.layer_types:
            rp = rope_params[layer_type]
            base = rp.get("rope_theta", config.rope_theta)
            dim = config.qk_rope_head_dim

            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
            t = torch.arange(max_pos, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)  # [max_pos, dim//2]
            cos_cached = freqs.cos()
            sin_cached = freqs.sin()

            rope_type = rp.get("rope_type", "default")
            if rope_type == "yarn":
                attention_factor = rp.get("attention_factor", 1.0)
                cos_cached = cos_cached * attention_factor
                sin_cached = sin_cached * attention_factor

            self.register_buffer(f"{layer_type}_cos_cached", cos_cached, persistent=False)
            self.register_buffer(f"{layer_type}_sin_cached", sin_cached, persistent=False)

    def forward(self, layer_type: str):
        cos = getattr(self, f"{layer_type}_cos_cached")
        sin = getattr(self, f"{layer_type}_sin_cached")
        return cos, sin


class QEffDeepseekV4CustomRMSNormAIC(nn.Module):
    """RMSNorm replaced with compiler-known custom-op."""

    def forward(self, hidden_states):
        return CustomRMSNormFunc.apply(hidden_states, self.weight, self.variance_epsilon)


class QEffDeepseekV4Attention(DeepseekV4Attention):
    """QEff-adapted V4 attention with static KV cache, batch_index, position_ids.

    V4 attention is Shared-KV MQA: a single KV head (K==V) broadcast to all query
    heads. The sliding-window branch uses standard causal mask; CSA/HCA compressor
    branches are disabled for the initial AI 100 decode path (compressor state is
    stateful and not ONNX-friendly). For decode, we keep only the sliding-window
    attention with explicit KV cache management.
    """

    def __qeff_init__(self):
        self.rope_dim = self.config.qk_rope_head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cos_main: Optional[torch.Tensor] = None,
        sin_main: Optional[torch.Tensor] = None,
        cos_compress: Optional[torch.Tensor] = None,
        sin_compress: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()
        hidden_shape = (bsz, q_len, -1, self.head_dim)

        # Q projection: low-rank then up-project
        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        q = self.q_b_norm(q)

        # KV projection: single shared head (K == V)
        kv = self.kv_norm(self.kv_proj(hidden_states))
        kv = kv.view(bsz, q_len, 1, self.head_dim).transpose(1, 2)

        # Select RoPE cos/sin based on layer type
        if self.rope_layer_type == "main":
            cos, sin = cos_main, sin_main
        else:
            cos, sin = cos_compress, sin_compress

        # Apply partial interleaved RoPE to Q
        q = qeff_apply_rotary_pos_emb_v4(q, cos, sin, position_ids, self.rope_dim)
        # Apply partial interleaved RoPE to KV
        kv = qeff_apply_rotary_pos_emb_v4(kv, cos, sin, position_ids, self.rope_dim)

        # KV cache update
        if past_key_value is not None:
            cache_kwargs = {"batch_index": batch_index, "position_ids": position_ids}
            # K == V in V4, store same tensor for both key and value slots
            kv_cached, _ = past_key_value.update(kv, kv, self.layer_idx, cache_kwargs)
            key_states = kv_cached
            value_states = kv_cached
        else:
            key_states = kv
            value_states = kv

        # Expand single KV head to match query heads (GQA expansion, num_kv_groups = num_heads)
        key_states = key_states.expand(-1, self.num_heads, -1, -1)
        value_states = value_states.expand(-1, self.num_heads, -1, -1)

        # Attention computation
        attn_weights = torch.matmul(q, key_states.transpose(2, 3)) * self.scaling

        # Add per-head learned sinks as bias to position 0
        sink_bias = self.sinks.view(1, self.num_heads, 1, 1)
        attn_weights = attn_weights + sink_bias * (attention_mask == 0).float()[:, :1, :, :1]

        if attention_mask is not None:
            attn_weights = torch.where(
                attention_mask,
                torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=attn_weights.dtype),
                attn_weights,
            )

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # Apply conjugate rotation to undo RoPE on the output rope-slice
        # (V4's K==V means V also has RoPE; undo via -sin at query position)
        attn_output = qeff_apply_rotary_pos_emb_v4(
            attn_output, cos, -sin, position_ids, self.rope_dim
        )

        # Grouped output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        grouped = attn_output.view(bsz, q_len, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped)
        # Explicit reshape to rank 3 for stable ONNX shape inference (avoid flatten()).
        grouped = grouped.reshape(bsz, q_len, -1)
        output = self.o_b_proj(grouped)

        return output, attn_weights


class QEffDeepseekV4HyperConnection(DeepseekV4HyperConnection):
    """QEff-adapted HyperConnection: same logic, explicit tensor ops for ONNX."""

    def forward(self, hidden_streams: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hc = self.hc_mult
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
        comb = comb / (torch.einsum("...ij->...j", comb).unsqueeze(-2) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (torch.einsum("...ij->...i", comb).unsqueeze(-1) + self.hc_eps)
            comb = comb / (torch.einsum("...ij->...j", comb).unsqueeze(-2) + self.hc_eps)

        collapsed = torch.einsum("bshd,bsh->bsd", hidden_streams, pre).to(hidden_streams.dtype)
        return post, comb, collapsed


class QEffDeepseekV4HyperHead(DeepseekV4HyperHead):
    """QEff-adapted HyperHead: final HC-stream collapse."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.input_norm(x.flatten(2).float())
        mixes = F.linear(flat, self.hc_fn.float())
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
        return torch.einsum("bshd,bsh->bsd", x, pre).to(x.dtype)


class QEffDeepseekV4SparseMoeBlock(DeepseekV4SparseMoeBlock):
    """QEff-adapted MoE block with einsum aggregation (M1 invariant)."""

    def __qeff_init__(self):
        experts = self.experts
        if hasattr(experts, "gate_up_proj"):
            gate_proj, up_proj = experts.gate_up_proj.chunk(2, dim=1)
            self.all_gate_proj = nn.Parameter(gate_proj.transpose(1, 2).contiguous())
            self.all_up_proj = nn.Parameter(up_proj.transpose(1, 2).contiguous())
            self.all_down_proj = nn.Parameter(experts.down_proj.transpose(1, 2).contiguous())
        self.act_fn = experts.act_fn
        self.num_experts = experts.num_experts
        self.limit = experts.limit

    def moe(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        """Batched-BMM MoE forward with einsum aggregation (M1, M3)."""
        bs, seq_len, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_states.shape[-1])
        top_k = topk_indices.shape[-1]

        gate_proj = self.all_gate_proj[topk_indices.flatten()]
        up_proj = self.all_up_proj[topk_indices.flatten()]
        down_proj = self.all_down_proj[topk_indices.flatten()]

        expert_in = (
            hidden_flat.unsqueeze(1)
            .expand(-1, top_k, -1)
            .contiguous()
            .view(-1, 1, hidden_size)
        )

        gate_out = torch.bmm(expert_in, gate_proj).clamp(max=self.limit)
        up_out = torch.bmm(expert_in, up_proj).clamp(min=-self.limit, max=self.limit)
        hidden = self.act_fn(gate_out) * up_out
        expert_output = torch.bmm(hidden, down_proj)

        experts_out = expert_output.view(bs * seq_len, top_k, hidden_size)
        experts_out = experts_out * topk_weights.unsqueeze(-1)
        # M1: einsum aggregation, NOT .sum(dim=1)
        final = torch.einsum("abc->ac", experts_out)
        return final.to(hidden_states.dtype)

    def forward(
        self, hidden_states: torch.Tensor, input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch, seq_len, hidden_dim = hidden_states.shape
        residual = hidden_states

        if self.is_hash:
            _, weights, indices = self.gate(hidden_states, input_ids)
        else:
            _, weights, indices = self.gate(hidden_states)

        routed = self.moe(hidden_states, indices, weights).view(batch, seq_len, hidden_dim)
        return routed + self.shared_experts(residual)


class QEffDeepseekV4DecoderLayer(DeepseekV4DecoderLayer):
    """QEff-adapted decoder layer with explicit KV cache threading."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        cos_main: Optional[torch.Tensor] = None,
        sin_main: Optional[torch.Tensor] = None,
        cos_compress: Optional[torch.Tensor] = None,
        sin_compress: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        dtype = hidden_states.dtype

        # Attention site: HC collapse → layernorm → attn → HC expand
        post, comb, collapsed = self.attn_hc(hidden_states)
        attn_output, _ = self.self_attn(
            self.input_layernorm(collapsed),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            batch_index=batch_index,
            cos_main=cos_main,
            sin_main=sin_main,
            cos_compress=cos_compress,
            sin_compress=sin_compress,
        )
        hidden_states = post.to(dtype).unsqueeze(3) * attn_output.unsqueeze(2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )

        # MLP site: HC collapse → layernorm → MoE → HC expand
        post, comb, collapsed = self.ffn_hc(hidden_states)
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
        hidden_states = post.to(dtype).unsqueeze(3) * mlp_output.unsqueeze(2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )

        return hidden_states


class QEffDeepseekV4Model(DeepseekV4Model):
    """QEff-adapted V4 model with precomputed RoPE and QEffDynamicCache."""

    def __qeff_init__(self):
        self.qeff_rotary_emb = QEffDeepseekV4RotaryEmbedding(self.config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and not isinstance(past_key_values, Cache) and past_key_values is not None:
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        attention_mask = _create_causal_mask(position_ids=position_ids, target_length=past_seen_tokens)

        # Expand input to hc_mult parallel streams
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()

        # Precomputed RoPE: index by position_ids
        cos_main, sin_main = self.qeff_rotary_emb("main")
        cos_compress, sin_compress = self.qeff_rotary_emb("compress")

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                batch_index=batch_index,
                input_ids=input_ids,
                cos_main=cos_main,
                sin_main=sin_main,
                cos_compress=cos_compress,
                sin_compress=sin_compress,
            )

        # Collapse hc_mult streams via HyperHead, then final norm
        hidden_states = self.norm(self.hc_head(hidden_states))

        # Convert Cache back to legacy list-of-tuples for ONNX trace (V3 pattern).
        next_cache = past_key_values.to_legacy_cache() if past_key_values is not None else None

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
        )


class QEffDeepseekV4ForCausalLM(DeepseekV4ForCausalLM):
    """QEff-adapted V4 CausalLM with INT32 logit gather (U2)."""

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffDeepseekV4DecoderLayer}

    def get_dummy_pkv_cache(self, config, batch_size, seq_len):
        # V4 shared-KV MQA: K == V, single head, full head_dim per layer.
        cache_shape = (batch_size, 1, seq_len, config.head_dim)
        dummy_cache = [[] for _ in range(config.num_hidden_layers)]
        for i in range(config.num_hidden_layers):
            dummy_cache[i].append(torch.zeros(cache_shape, dtype=config.torch_dtype))
            dummy_cache[i].append(torch.zeros(cache_shape, dtype=config.torch_dtype))
        return dummy_cache

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # U2: INT32 logit gather index
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = hidden_states[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]

        if self.config.tie_word_embeddings:
            logits = F.linear(hidden_states, self.model.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.to(hidden_states.dtype)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )
