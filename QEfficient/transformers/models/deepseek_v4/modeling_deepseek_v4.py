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
with sqrtsoftplus noaux_tc routing. CSA/HCA layers additionally compress every
``compress_rate`` tokens into long-range KV entries; the compressor runs
static-shape over the layer's cached input prefix (the V slot of the layer's
single K==V cache pair is repurposed to hold that input). The CSA Lightning
Indexer's top-k gather is omitted as inert for ctx within
``index_topk * compress_rate`` (asserted), mirroring glm_moe_dsa's inert DSA.
"""

from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4Config,
    DeepseekV4CSACompressor,
    DeepseekV4DecoderLayer,
    DeepseekV4ForCausalLM,
    DeepseekV4HCACompressor,
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
    """Precomputed sin/cos buffers, one pair per RoPE layer-type in config.

    Mirrors HF ``DeepseekV4RotaryEmbedding`` init: each layer-type's inv_freq and
    attention-scaling come from ``ROPE_INIT_FUNCTIONS[rope_type]`` (so YARN's
    interpolated inv_freq + mscale are applied, not a plain linear RoPE). The
    cos/sin tables are baked once for static gather at ``[position_ids]``; V4's
    interleaved RoPE keeps them half-width (``qk_rope_head_dim // 2``).
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.config = config
        rope_params = config.rope_parameters or {}
        self.layer_types = [k for k, v in rope_params.items() if isinstance(v, dict)]

        t = torch.arange(config.max_position_embeddings, dtype=torch.float32)
        for layer_type in self.layer_types:
            rope_type = rope_params[layer_type].get("rope_type", "default")
            rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            inv_freq, attention_scaling = rope_init_fn(config, layer_type=layer_type)
            freqs = torch.outer(t, inv_freq.to(torch.float32))
            self.register_buffer(f"{layer_type}_cos_cached", freqs.cos() * attention_scaling, persistent=False)
            self.register_buffer(f"{layer_type}_sin_cached", freqs.sin() * attention_scaling, persistent=False)

    def forward(self, layer_type: str):
        return getattr(self, f"{layer_type}_cos_cached"), getattr(self, f"{layer_type}_sin_cached")


def _v4_window_compress(kv, gate, position_bias, kv_norm, compress_rate, head_dim, overlap):
    """Pool ``compress_rate`` source tokens into one compressed entry (V4 §2.3).

    Shared by HCA (``overlap=False``) and CSA (``overlap=True``). ``kv`` / ``gate``
    are the per-token compressor projections over the *full gathered prefix*
    (``[B, S, F]`` with ``F == head_dim`` for HCA, ``2*head_dim`` for CSA). Only
    the longest window-aligned prefix is pooled; ``n_windows`` is a Python int
    derived from the static ``S`` dim (constant under ONNX trace). Returns the
    compressed entries ``[B, n_windows, head_dim]`` (RoPE NOT yet applied) — the
    caller adds RoPE at the window positions.

    For CSA the two-series Ca/Cb overlap layout (paper §2.3.1) is built within the
    single forward: window ``w`` reads window ``w-1``'s Ca slice via a static
    shifted-slice (no cross-call state — valid because the recompute always sees
    the full prefix, proven ≡ HF incremental).
    """
    batch, seq_len = kv.shape[0], kv.shape[1]
    n_windows = seq_len // compress_rate
    if n_windows == 0:
        return kv.new_zeros((batch, 0, head_dim))
    ratio = compress_rate
    usable = n_windows * ratio
    chunk_kv = kv[:, :usable].view(batch, n_windows, ratio, -1)
    chunk_gate = gate[:, :usable].view(batch, n_windows, ratio, -1) + position_bias

    if not overlap:
        compressed = kv_norm((chunk_kv * chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)).sum(dim=2))
        return compressed

    # CSA: Ca = [..., :head_dim] (next window's contribution), Cb = [..., head_dim:]
    # (current window). Entry w = softmax-gated combine of window w-1's Ca with
    # window w's Cb over 2*ratio slots. Window 0's first half stays zero-kv /
    # -inf-gate (softmax weight ~0) — the recompute owns the full prefix so there
    # is no prior-call overlap to inject.
    new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, head_dim))
    new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, head_dim), MIN_MASKED_ATTENTION_VALUE)
    new_kv[:, :, ratio:] = chunk_kv[..., head_dim:]
    new_gate[:, :, ratio:] = chunk_gate[..., head_dim:]
    if n_windows > 1:
        new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, :head_dim]
        new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, :head_dim]
    compressed = kv_norm((new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2))
    return compressed


class QEffDeepseekV4Compressor:
    """Shared static-shape forward for CSA/HCA compressors (paper §2.3).

    Stateless windowed compression over the *full gathered prefix* — the layer
    caches its input so decode always re-derives the complete window history
    (recompute-from-prefix ≡ HF stateful incremental, verified to 0.0). Emits the
    compressed KV entries (one per closed window of ``compress_rate`` tokens) and
    a per-query causal ``block_bias`` over them.

    The Lightning Indexer's top-``index_topk`` selection (CSA only) is omitted: it
    is mathematically inert while ``compressed_len <= index_topk`` (= ``ctx <=
    index_topk * compress_rate``), so every compressed entry is selected and the
    causal mask alone is exact. ``__qeff_init__`` asserts the compile ctx stays in
    that range. Same posture as ``glm_moe_dsa``'s inert DSA indexer.
    """

    overlap: bool  # set by the CSA/HCA subclass

    def __qeff_init__(self):
        # HF compressor stores head_dim / compress_rate / kv_norm / position_bias /
        # rotary_emb but not `config`; the rope dim lives on the rotary emb's config.
        self.rope_dim = self.rotary_emb.config.qk_rope_head_dim

    def compress(self, hidden_states, query_positions, cos, sin):
        """``hidden_states``: full gathered prefix ``[B, S_prefix, hidden]`` (S_prefix
        is the whole sequence so far, even at decode). ``query_positions``: the
        current forward's query positions ``[B, q_len]`` used for the causal
        block-bias. Returns ``(compressed_kv [B, 1, n_windows, head_dim],
        block_bias [B, 1, q_len, n_windows])`` or ``(zeros[B,1,0,head_dim], None)``
        when no window has closed yet."""
        batch, seq_len, _ = hidden_states.shape
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        compressed = _v4_window_compress(
            kv, gate, self.position_bias, self.kv_norm, self.compress_rate, self.head_dim, self.overlap
        )
        n_windows = compressed.shape[1]
        if n_windows == 0:
            return compressed.unsqueeze(1), None

        # The CSA Lightning Indexer's top-index_topk gather is omitted because it is
        # inert while every compressed entry is selected (n_windows <= index_topk).
        # Guard the omission so a too-long ctx fails loudly instead of silently wrong.
        indexer = getattr(self, "indexer", None)
        if indexer is not None:
            assert n_windows <= indexer.index_topk, (
                f"CSA indexer top-k omitted but n_windows={n_windows} > index_topk={indexer.index_topk}; "
                "ctx exceeds the indexer-inert range — port the top-k gather before compiling at this ctx."
            )

        # RoPE each window at its deterministic absolute position i*compress_rate.
        positions = (torch.arange(n_windows) * self.compress_rate).unsqueeze(0).expand(batch, -1)
        compressed = qeff_apply_rotary_pos_emb_v4(compressed.unsqueeze(1), cos, sin, positions, self.rope_dim)
        compressed_kv = compressed  # [B, 1, n_windows, head_dim]

        # Per-query causal block-bias: query t may attend compressed entry w only
        # if w*compress_rate < t+1, i.e. w < (t+1)//compress_rate.
        entry_indices = torch.arange(n_windows).view(1, 1, 1, -1)
        causal_threshold = ((query_positions + 1) // self.compress_rate).unsqueeze(1).unsqueeze(-1)  # [B,1,q,1]
        block_bias = torch.where(
            entry_indices >= causal_threshold,
            torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=compressed_kv.dtype),
            torch.zeros((), dtype=compressed_kv.dtype),
        )  # [B, 1, q_len, n_windows]
        return compressed_kv, block_bias


class QEffDeepseekV4CSACompressor(QEffDeepseekV4Compressor, DeepseekV4CSACompressor):
    overlap = True


class QEffDeepseekV4HCACompressor(QEffDeepseekV4Compressor, DeepseekV4HCACompressor):
    overlap = False


class QEffDeepseekV4Attention(DeepseekV4Attention):
    """Shared-KV MQA: one KV head, K==V, broadcast to query heads.

    Sliding-attention layers attend the sliding-window K=V cache only. CSA/HCA
    layers additionally compress every ``compress_rate`` tokens of the layer input
    into long-range KV entries (``QEffDeepseekV4Compressor``) and concatenate them
    onto the KV axis with a per-query causal block-bias. The compressor input is
    cached so decode re-derives the full window history (the ``compressed_kvs``
    pattern from deepseek_v3, realized over the standard cache with CtxScatter/
    CtxGather gathers).
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
            if self.compressor is not None:
                # One cache slot per layer (the export machinery assumes exactly one
                # (k,v) pair per decoder layer). V4 is K==V MQA, so the V slot is a
                # redundant copy of K — repurpose it to hold this layer's compressor
                # INPUT prefix. update() scatters/gathers K and V independently, so
                # K (head_dim) and V (hidden_size) may differ in width. K_out is the
                # sliding KV for attention; V_out is the full input prefix the
                # compressor pools windows over (decode feeds 1 token/step, so the
                # cache re-derives the whole window history — verified ≡ HF to 0.0).
                comp_in_4d = hidden_states.view(bsz, q_len, 1, -1).transpose(1, 2)  # [B,1,S,hidden]
                kv_cached, comp_gathered = past_key_value.update(kv, comp_in_4d, self.layer_idx, cache_kwargs)
                comp_in = comp_gathered.transpose(1, 2).reshape(bsz, -1, hidden_states.shape[-1])
            else:
                kv_cached, _ = past_key_value.update(kv, kv, self.layer_idx, cache_kwargs)
            key_states = value_states = kv_cached
        else:
            key_states = value_states = kv
            comp_in = hidden_states

        key_states = key_states.expand(-1, self.num_heads, -1, -1)
        value_states = value_states.expand(-1, self.num_heads, -1, -1)

        attn_weights = torch.matmul(q, key_states.transpose(2, 3)) * self.scaling

        block_bias = None
        if self.compressor is not None:
            # Pool windows over the full input prefix (cached above), RoPE each
            # window at its absolute position, score queries against the compressed
            # entries, and gate with a per-query causal block-bias.
            compressed_kv, block_bias = self.compressor.compress(
                comp_in, position_ids, cos_compress, sin_compress
            )
            n_comp = compressed_kv.shape[2]
            if n_comp > 0:
                comp_k = compressed_kv.expand(-1, self.num_heads, -1, -1)
                comp_scores = torch.matmul(q, comp_k.transpose(2, 3)) * self.scaling
                attn_weights = torch.cat([attn_weights, comp_scores], dim=-1)
            else:
                block_bias = None

        if attention_mask is not None:
            sliding_w = attention_mask.shape[-1]
            masked = torch.where(
                attention_mask,
                torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=attn_weights.dtype),
                attn_weights[..., :sliding_w],
            )
            attn_weights = torch.cat([masked, attn_weights[..., sliding_w:]], dim=-1)
        if block_bias is not None:
            # block_bias is [B, 1, S, n_windows]; broadcast-add over the compressed
            # columns (the trailing n_windows of attn_weights). Head dim 1 → num_heads.
            n_windows = block_bias.shape[-1]
            pad = attn_weights.new_zeros(
                (attn_weights.shape[0], 1, attn_weights.shape[2], attn_weights.shape[-1] - n_windows)
            )
            attn_weights = attn_weights + torch.cat([pad, block_bias], dim=-1)

        # Per-head learnable attention sink (gpt-oss style): append the sink as an
        # extra logit column, softmax over [scores | sink], then drop it. The sink
        # absorbs probability mass so the real-token weights sum to < 1 — adding it
        # as a per-row bias instead would cancel under softmax shift-invariance.
        sinks = self.sinks.reshape(1, self.num_heads, 1, 1).expand(bsz, -1, q_len, -1)
        combined_logits = torch.cat([attn_weights, sinks], dim=-1)
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        probs = F.softmax(combined_logits, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = probs[..., :-1]

        if self.compressor is not None and block_bias is not None and compressed_kv.shape[2] > 0:
            value_states = torch.cat([value_states, compressed_kv.expand(-1, self.num_heads, -1, -1)], dim=2)
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
        # One (K, V) pair per layer (the export path assumes exactly one per layer).
        # K is always the shared-KV MQA sliding cache [B,1,ctx,head_dim]. For CSA/HCA
        # layers the V slot is repurposed to hold the compressor input prefix
        # [B,1,ctx,hidden_size] (V4 is K==V, so the real V is a redundant copy of K);
        # sliding-only layers keep V=K width.
        layer_types = getattr(config, "layer_types", None) or ["sliding_attention"] * config.num_hidden_layers
        cache = []
        for layer_type in layer_types:
            k_shape = (batch_size, 1, seq_len, config.head_dim)
            v_width = config.hidden_size if layer_type != "sliding_attention" else config.head_dim
            v_shape = (batch_size, 1, seq_len, v_width)
            cache.append(
                [torch.zeros(k_shape, dtype=config.torch_dtype), torch.zeros(v_shape, dtype=config.torch_dtype)]
            )
        return cache

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
