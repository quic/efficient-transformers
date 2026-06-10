# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""QEfficient modeling for ``deepseek_v4`` (deepseek-ai/DeepSeek-V4-Pro).

V4 introduces shared-KV MQA (single KV head, K==V), partial interleaved RoPE
on the trailing ``qk_rope_head_dim`` channels, manifold-constrained
HyperConnections (mHC) replacing residuals, and a 384-expert hash+top-k MoE
with sqrtsoftplus noaux_tc routing. The CSA/HCA compressor branches on certain
layers are not yet wired (stateful, requires the compressed_kvs cache machinery
from deepseek_v3).
"""

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
    DeepseekV4ForCausalLM,
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
    DeepseekV4Model,
    DeepseekV4SparseMoeBlock,
)

from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


def rotate_half_interleaved(x):
    """V4 interleaved rotate-half: pairs even/odd channels and swaps them."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def qeff_apply_rotary_pos_emb_v4(x, cos, sin, position_ids, rope_dim):
    """Apply interleaved RoPE to the trailing ``rope_dim`` channels of x.

    cos/sin shape: ``[seq_len, rope_dim//2]`` (no duplication in V4's
    RotaryEmbedding). Each pair ``(c_k, c_k)`` appears at positions
    ``(2k, 2k+1)``; expressed as ``stack(c, c, -1).flatten(-2)`` instead of
    ``repeat_interleave`` for stable ONNX trace.
    """
    cos_pos = cos[position_ids]
    sin_pos = sin[position_ids]
    cos_pos = torch.stack([cos_pos, cos_pos], dim=-1).flatten(-2).unsqueeze(1)
    sin_pos = torch.stack([sin_pos, sin_pos], dim=-1).flatten(-2).unsqueeze(1)

    if rope_dim == x.shape[-1]:
        return ((x * cos_pos) + (rotate_half_interleaved(x) * sin_pos)).to(x.dtype)
    nope = x[..., :-rope_dim]
    rope = x[..., -rope_dim:]
    rotated = (rope * cos_pos) + (rotate_half_interleaved(rope) * sin_pos)
    return torch.cat([nope, rotated], dim=-1).to(x.dtype)


class QEffDeepseekV4RotaryEmbedding(nn.Module):
    """Precomputed sin/cos buffers, one pair per RoPE layer-type in config."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.config = config
        rope_params = config.rope_parameters or {}
        self.layer_types = [k for k, v in rope_params.items() if isinstance(v, dict)]

        for layer_type in self.layer_types:
            rp = rope_params[layer_type]
            base = rp.get("rope_theta", config.rope_theta)
            dim = config.qk_rope_head_dim
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
            t = torch.arange(config.max_position_embeddings, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            scaling = rp.get("attention_factor", 1.0) if rp.get("rope_type") == "yarn" else 1.0
            self.register_buffer(f"{layer_type}_cos_cached", freqs.cos() * scaling, persistent=False)
            self.register_buffer(f"{layer_type}_sin_cached", freqs.sin() * scaling, persistent=False)

    def forward(self, layer_type: str):
        return getattr(self, f"{layer_type}_cos_cached"), getattr(self, f"{layer_type}_sin_cached")


class QEffDeepseekV4Attention(DeepseekV4Attention):
    """Shared-KV MQA: one KV head, K==V, broadcast to query heads.

    The CSA/HCA compressor branches are not yet wired (stateful, would need
    the ``compressed_kvs`` cache from deepseek_v3). Decode runs the
    sliding-window path only.
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

        # Q: low-rank then up-project
        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        q = self.q_b_norm(q)

        # KV: single shared head (K == V)
        kv = self.kv_norm(self.kv_proj(hidden_states))
        kv = kv.view(bsz, q_len, 1, self.head_dim).transpose(1, 2)

        cos, sin = (cos_main, sin_main) if self.rope_layer_type == "main" else (cos_compress, sin_compress)
        q = qeff_apply_rotary_pos_emb_v4(q, cos, sin, position_ids, self.rope_dim)
        kv = qeff_apply_rotary_pos_emb_v4(kv, cos, sin, position_ids, self.rope_dim)

        if past_key_value is not None:
            cache_kwargs = {"batch_index": batch_index, "position_ids": position_ids}
            kv_cached, _ = past_key_value.update(kv, kv, self.layer_idx, cache_kwargs)
            key_states = value_states = kv_cached
        else:
            key_states = value_states = kv

        key_states = key_states.expand(-1, self.num_heads, -1, -1)
        value_states = value_states.expand(-1, self.num_heads, -1, -1)

        attn_weights = torch.matmul(q, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            # Per-head learned sinks added at position 0 (where mask == 0).
            sink_bias = self.sinks.view(1, self.num_heads, 1, 1)
            attn_weights = attn_weights + sink_bias * (attention_mask == 0).float()[:, :1, :, :1]
            attn_weights = torch.where(
                attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=attn_weights.dtype), attn_weights
            )

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # Undo RoPE on output: V4's K == V means V also carries RoPE; HF
        # reverses it before the grouped o_proj.
        attn_output = qeff_apply_rotary_pos_emb_v4(attn_output, cos, -sin, position_ids, self.rope_dim)

        attn_output = attn_output.transpose(1, 2).contiguous()
        grouped = attn_output.view(bsz, q_len, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).reshape(bsz, q_len, -1)
        return self.o_b_proj(grouped), attn_weights


class QEffDeepseekV4HyperConnection(DeepseekV4HyperConnection):
    """HC residual mixing with explicit reshape (avoids ``flatten(start_dim=2)``
    mistraces in ONNX)."""

    def forward(self, hidden_streams: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hc = self.hc_mult
        bsz, q_len, _, hidden_size = hidden_streams.shape
        flat = self.input_norm(hidden_streams.reshape(bsz, q_len, hc * hidden_size).float())
        pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
        comb = comb / (comb.sum(dim=-2).unsqueeze(-2) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1).unsqueeze(-1) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2).unsqueeze(-2) + self.hc_eps)

        collapsed = (hidden_streams * pre.unsqueeze(-1)).sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed


class QEffDeepseekV4HyperHead(DeepseekV4HyperHead):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, q_len, hc, hidden_size = x.shape
        flat = self.input_norm(x.reshape(bsz, q_len, hc * hidden_size).float())
        mixes = F.linear(flat, self.hc_fn.float())
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
        return (x * pre.unsqueeze(-1)).sum(dim=2).to(x.dtype)


class QEffDeepseekV4SparseMoeBlock(DeepseekV4SparseMoeBlock):
    """Batched-BMM MoE; M1 invariant kept via einsum-equivalent ``.sum(dim=1)``."""

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

    def moe(self, hidden_states, topk_indices, topk_weights):
        bs, seq_len, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_size)
        top_k = topk_indices.shape[-1]

        gate_proj = self.all_gate_proj[topk_indices.flatten()]
        up_proj = self.all_up_proj[topk_indices.flatten()]
        down_proj = self.all_down_proj[topk_indices.flatten()]

        expert_in = hidden_flat.unsqueeze(1).expand(-1, top_k, -1).contiguous().view(-1, 1, hidden_size)
        gate_out = torch.bmm(expert_in, gate_proj).clamp(max=self.limit)
        up_out = torch.bmm(expert_in, up_proj).clamp(min=-self.limit, max=self.limit)
        hidden = self.act_fn(gate_out) * up_out
        expert_output = torch.bmm(hidden, down_proj)

        experts_out = expert_output.view(bs * seq_len, top_k, hidden_size)
        experts_out = experts_out * topk_weights.unsqueeze(-1)
        return experts_out.sum(dim=1).to(hidden_states.dtype)

    def forward(self, hidden_states, input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, hidden_dim = hidden_states.shape
        residual = hidden_states
        if self.is_hash:
            _, weights, indices = self.gate(hidden_states, input_ids)
        else:
            _, weights, indices = self.gate(hidden_states)
        routed = self.moe(hidden_states, indices, weights).view(batch, seq_len, hidden_dim)
        return routed + self.shared_experts(residual)


def _hc_combine(output, post, comb, hidden_states, dtype):
    """V4 HC expand: combine attn/MLP output with the running HC streams."""
    return post.to(dtype).unsqueeze(3) * output.unsqueeze(2) + torch.matmul(
        comb.to(dtype).transpose(-1, -2), hidden_states
    )


class QEffDeepseekV4DecoderLayer(DeepseekV4DecoderLayer):
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
        hidden_states = _hc_combine(attn_output, post, comb, hidden_states, dtype)

        post, comb, collapsed = self.ffn_hc(hidden_states)
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
        hidden_states = _hc_combine(mlp_output, post, comb, hidden_states, dtype)
        return hidden_states


class QEffDeepseekV4Model(DeepseekV4Model):
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

        # Expand to hc_mult parallel streams.
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()

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

        # Collapse hc_mult streams via HyperHead, then final norm.
        hidden_states = self.norm(self.hc_head(hidden_states))

        next_cache = past_key_values.to_legacy_cache() if past_key_values is not None else None
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=next_cache)


class QEffDeepseekV4ForCausalLM(DeepseekV4ForCausalLM):
    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffDeepseekV4DecoderLayer}

    def get_dummy_pkv_cache(self, config, batch_size, seq_len):
        # Shared-KV MQA: K == V, single head, full head_dim.
        cache_shape = (batch_size, 1, seq_len, config.head_dim)
        return [
            [torch.zeros(cache_shape, dtype=config.torch_dtype), torch.zeros(cache_shape, dtype=config.torch_dtype)]
            for _ in range(config.num_hidden_layers)
        ]

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
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = hidden_states[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]

        if self.config.tie_word_embeddings:
            logits = F.linear(hidden_states, self.model.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.to(hidden_states.dtype)

        return CausalLMOutputWithPast(loss=None, logits=logits, past_key_values=outputs.past_key_values)
