# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
QEfficient modeling for MiniMaxAI/MiniMax-M3 (model_type=minimax_m3_vl).

Native multimodal sparse-MoE VLM:
  - Text decoder (60 layers; first 3 dense, rest MoE 128 + 1 shared expert).
  - MiniMax Sparse Attention (MSA) on layers 3-59; CPU-parity path falls back to
    dense full-attention via the indexer's `build_block_mask` (matches HF "eager"
    behaviour exactly), so QEff lands the standard SDPA path.
  - Gemma-style RMSNorm (`(x * w).to(dtype)` with `(weight + 1)`).
  - Partial RoPE (rotary_dim = head_dim * partial_rotary_factor).
  - swigluoai gated MLP (alpha, limit clamps).
  - Vision tower: CLIP-style ViT, Conv3d patch embed, 3D-RoPE, GELU+bias projector
    with patch-merge, image-token replacement.

Huge-tier handoff: this file targets CPU PyTorch parity + ONNX export. Compile/
hardware bring-up is the user's job from the handoff guide. The MSA selection
branch is left at HF semantics (it materializes a dense additive mask) so the
exported graph is the standard dense-attention shape; `Indexer` selection is a
no-op on the parity path.
"""

from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
    MiniMaxM3SparseForConditionalGeneration,
    MiniMaxM3VLAttention,
    MiniMaxM3VLDecoderLayer,
    MiniMaxM3VLModel,
    MiniMaxM3VLModelOutputWithPast,
    MiniMaxM3VLRMSNorm,
    MiniMaxM3VLRotaryEmbedding,
    MiniMaxM3VLSparseMoeBlock,
    MiniMaxM3VLTextModel,
    MiniMaxM3VLTopKRouter,
    MiniMaxM3VLVisionAttention,
    MiniMaxM3VLVisionModel,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
    repeat_kv,
)

from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo, get_padding_shape_from_config
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


# -----------------------------------------------------------------------------
# Text decoder
# -----------------------------------------------------------------------------


class QEffMiniMaxM3VLRotaryEmbedding(MiniMaxM3VLRotaryEmbedding):
    """Static cos/sin cache so torch.jit.trace sees a fixed table."""

    def __init__(self, config, device=None):
        super().__init__(config=config, device=device)
        self._set_cos_sin_cache(
            seq_len=self.original_max_seq_len,
            device=self.inv_freq.device,
            dtype=getattr(config, "torch_dtype", torch.float32),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", (emb.cos() * self.attention_scaling).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self.attention_scaling).to(dtype), persistent=False)


def qeff_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    **kwargs,
):
    """ONNX-friendly eager attention: matmul + torch.where mask + finite mask value."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        # Invariant U1: finite mask value (-1e9), torch.where for ONNX export.
        attn_weights = torch.where(
            attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=query.dtype), attn_weights
        )
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class QEffMiniMaxM3VLAttention(MiniMaxM3VLAttention):
    """M3 attention: per-head Gemma QK-norm + partial RoPE.

    On the CPU-parity / ONNX-export path the optional sparse `indexer` is bypassed:
    when present, HF's own `build_block_mask` would expand it into a dense additive
    mask, which is what we already provide via `_create_causal_mask` upstream. Block
    selection is a hardware-side optimisation that the huge-tier handoff guide will
    enable on the user's compile.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        sin_cached: Optional[torch.Tensor] = None,
        cos_cached: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Prefer precomputed cos/sin (driven from QEffMiniMaxM3VLTextModel).
        if cos_cached is not None and sin_cached is not None:
            cos, sin = cos_cached, sin_cached
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, {"position_ids": position_ids, "batch_index": batch_index}
            )

        attn_output, attn_weights = qeff_eager_attention_forward(
            self, query_states, key_states, value_states, attention_mask, scaling=self.scaling, **kwargs
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class QEffMiniMaxM3VLTopKRouter(MiniMaxM3VLTopKRouter):
    """Sigmoid scoring + bias-corrected top-k. ONNX-safe rewrite of the gather/sum."""

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states.to(self.weight.dtype), self.weight)
        routing_weights = torch.sigmoid(router_logits.float())
        scores_for_choice = routing_weights + self.e_score_correction_bias
        _, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_index)
        # einsum form is ONNX-safer than .sum(...,keepdim=True) in some QAIC stacks.
        denom = torch.einsum("bk->b", top_k_weights).unsqueeze(-1) + 1e-20
        top_k_weights = top_k_weights / denom
        return router_logits, top_k_weights, top_k_index


class QEffMiniMaxM3VLSparseMoeBlock(MiniMaxM3VLSparseMoeBlock):
    """Fused-experts MoE: BMM expert loop with index_select on the [E, ...] params.

    Mirrors the QEffQwen3VLMoeTextSparseMoeBlock shape (T x topk routing, fused
    bmm gate_up + down) but uses MiniMax's `(up + 1.0) * SwiGLU(gate)` activation
    and adds the shared-expert residual (`shared_experts(x) + routed_scaling_factor * routed`).
    """

    def __qeff_init__(self):
        # `MiniMaxM3VLExperts.gate_up_proj` is [E, 2*I, H] in HF; transpose so we can
        # bmm against [T, 1, H] inputs without index gymnastics.
        if self.experts.gate_up_proj.dim() == 3:
            self.all_gate_up_proj = nn.Parameter(self.experts.gate_up_proj.transpose(1, 2).contiguous())
            self.all_down_proj = nn.Parameter(self.experts.down_proj.transpose(1, 2).contiguous())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, H = hidden_states.shape
        T = B * S
        x = hidden_states.view(T, H)
        residual = x

        shared_output = self.shared_experts(hidden_states.view(B * S, H)).view(B, S, H)

        _, top_w, top_i = self.gate(x)  # weights [T, top_k], indices [T, top_k]

        idx = top_i.reshape(-1)  # [T * top_k]
        w_up = self.all_gate_up_proj.index_select(0, idx)  # [T*top_k, H, 2I]
        w_dn = self.all_down_proj.index_select(0, idx)  # [T*top_k, I, H]

        top_k = top_i.shape[-1]
        xk = x.unsqueeze(1).expand(-1, top_k, -1).contiguous().view(-1, 1, H)
        gate_up = torch.bmm(xk, w_up)  # [T*top_k, 1, 2I]
        I2 = gate_up.size(-1) // 2
        gate, up = gate_up[..., :I2], gate_up[..., I2:]
        # swigluoai: clamp(max=limit) for gate, clamp(min=-limit, max=limit) for up.
        gate = gate.clamp(max=self.experts.swiglu_limit)
        up = up.clamp(min=-self.experts.swiglu_limit, max=self.experts.swiglu_limit)
        glu = gate * torch.sigmoid(gate * self.experts.swiglu_alpha)
        intermediate = (up + 1.0) * glu

        experts_out = torch.bmm(intermediate, w_dn)  # [T*top_k, 1, H]
        experts_out = experts_out.view(T, top_k, H) * top_w.to(x.dtype).unsqueeze(-1)
        routed = torch.einsum("bnd->bd", experts_out)  # [T, H]

        out = routed.to(x.dtype) * self.routed_scaling_factor + residual * 0  # keep dtype lineage
        out = out.view(B, S, H) + shared_output
        return out


class QEffMiniMaxM3VLDecoderLayer(MiniMaxM3VLDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        sin_cached: Optional[torch.Tensor] = None,
        cos_cached: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            batch_index=batch_index,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            sin_cached=sin_cached,
            cos_cached=cos_cached,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class QEffMiniMaxM3VLTextModel(MiniMaxM3VLTextModel):
    """Text decoder with QEffDynamicCache and precomputed cos/sin at the model level."""

    def __qeff_init__(self):
        self.rotary_emb = QEffMiniMaxM3VLRotaryEmbedding(config=self.config)
        self.sin_cached = nn.Parameter(self.rotary_emb.sin_cached, requires_grad=False)
        self.cos_cached = nn.Parameter(self.rotary_emb.cos_cached, requires_grad=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Invariant V11: explicit causal mask (NOT SDPA-shaped attention_mask).
        # target_length is the kv-axis the mask must cover. With QEff's pre-allocated
        # KV buffer, past_key_values.update() does scatter+gather (no concat), so the
        # post-update kv-extent equals max(past_seen_tokens, seq_len) — NOT their sum.
        # Falling back to past_seen_tokens + seq_len double-counts and produces a 2x mask.
        if isinstance(attention_mask, torch.Tensor):
            target_length = attention_mask.shape[-1]
        else:
            target_length = max(past_seen_tokens, inputs_embeds.shape[1])
        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=target_length)

        # Partial-RoPE: only the first rotary_dim head dims are rotated; the model-level
        # cos/sin tables are already sized to rotary_dim. Shape stays [B, S, head_dim] so
        # the upstream apply_rotary_pos_emb (which unsqueezes once at unsqueeze_dim=1) can
        # broadcast against q/k of shape [B, heads, S, head_dim].
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]

        hidden_states = inputs_embeds
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                batch_index=batch_index,
                cache_position=cache_position,
                sin_cached=sin,
                cos_cached=cos,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


# -----------------------------------------------------------------------------
# Vision tower
# -----------------------------------------------------------------------------


class QEffMiniMaxM3VLVisionAttention(MiniMaxM3VLVisionAttention):
    """ONNX-friendly rewrite of the CLIP-style vision attention.

    Key deltas vs HF:
      - Replaces SDPA dispatch with explicit matmul + softmax (Invariants V2/V5).
      - Honours optional `attention_mask` via torch.where with -10000.0 (V5).
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        queries = self.q_proj(hidden_states).view(hidden_shape)
        keys = self.k_proj(hidden_states).view(hidden_shape)
        values = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        queries, keys = apply_rotary_pos_emb_vision(queries, keys, cos, sin)
        queries, keys = queries.transpose(1, 2), keys.transpose(1, 2)

        attn_weights = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_weights = torch.where(attention_mask, attn_weights, torch.tensor(-10000.0, dtype=attn_weights.dtype))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(queries.dtype)
        attn_output = torch.matmul(attn_weights, values)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        return self.out_proj(attn_output), attn_weights


class QEffMiniMaxM3VLVisionModel(MiniMaxM3VLVisionModel):
    """Vision tower forward: Conv3d patches + 3D-RoPE + LayerNorms + ViT blocks.

    Adds the FP16 vision-projector clamp (Invariant V1) at the projector boundary;
    see `QEffMiniMaxM3SparseForConditionalGeneration` (the projector lives one
    level up, so the clamp is applied there).
    """

    pass


# -----------------------------------------------------------------------------
# Wrapper (encoder + decoder + image-token replacement)
# -----------------------------------------------------------------------------


class QEffMiniMaxM3VLEncoderWrapper(nn.Module):
    """Vision QPC: pixel_values + image_grid_thw -> projected vision_embeds."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {self.model.model.vision_tower.layers[0].__class__}

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.model.model.vision_tower(
            pixel_values=pixel_values, image_grid_thw=image_grid_thw
        )
        # `last_hidden_state` is [1, N, hidden]; squeeze for the projector.
        features = vision_outputs.last_hidden_state.squeeze(0)
        vision_embeds = self.model.model.multi_modal_projector(features)
        # V1: FP16 vision-projector clamp.
        vision_embeds = vision_embeds.clamp(min=-60000.0, max=60000.0)
        # Add a leading batch axis for the language graph: [B, T, H].
        return vision_embeds.unsqueeze(0)


class QEffMiniMaxM3VLDecoderWrapper(nn.Module):
    """Language QPC: text + (already-projected) vision_embeds -> logits + KV.

    Image injection: `image_token_id` (200025) replacement, cumsum-derived gather
    (V4 / V7), decode-time no-op via `torch.where(input_ids.shape[1] == 1, ...)`.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.language_model = model.model.language_model
        self.lm_head = model.lm_head
        self.config = model.config

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffMiniMaxM3VLDecoderLayer}

    def forward(
        self,
        input_ids: torch.LongTensor,
        vision_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        image_idx: torch.Tensor,
        past_key_values,
        batch_index: Optional[torch.LongTensor] = None,
    ):
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        vision_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        mask = input_ids == self.config.image_token_id
        indices1 = mask.to(torch.int64).cumsum(1) - 1
        # V7: clamp negative indices before gather (pre-image text positions sit at -1).
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(mask.shape[0]).view(-1, 1)
        vision_embeds_expanded = vision_embeds[indices0, indices1.clamp(min=0)]
        # V4: torch.where image-text merge (replaces masked_scatter).
        merged = torch.where(mask.unsqueeze(-1), vision_embeds_expanded, inputs_embeds)
        # V10: decode-time merge no-op.
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, merged)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            batch_index=batch_index,
            use_cache=True,
        )

        next_image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        image_idx = torch.where(image_idx < next_image_idx, next_image_idx, image_idx)

        # U2: int32 last-token logit gather.
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states).float()
        return logits, vision_embeds, image_idx, outputs.past_key_values


class QEffMiniMaxM3SparseForConditionalGeneration(MiniMaxM3SparseForConditionalGeneration):
    """Top-level QEff wrapper. Registers the vision/language splits and exposes
    the standard QEff VLM interface (get_qeff_vision_encoder / _language_decoder /
    get_dummy_inputs / get_specializations / get_onnx_dynamic_axes / get_output_names)."""

    def get_qeff_vision_encoder(self):
        return QEffMiniMaxM3VLEncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEffMiniMaxM3VLDecoderWrapper(self)

    def get_submodules_for_export(self):
        return [QEffMiniMaxM3VLDecoderLayer]

    # Single-QPC fused forward (kv_offload=False).
    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.Tensor,
        image_idx: torch.Tensor,
        past_key_values,
        batch_index: Optional[torch.LongTensor] = None,
    ):
        inputs_embeds = self.model.language_model.embed_tokens(input_ids)
        vision_outputs = self.model.vision_tower(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        features = vision_outputs.last_hidden_state.squeeze(0)
        vision_embeds = self.model.multi_modal_projector(features).clamp(-60000.0, 60000.0).unsqueeze(0)
        vision_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        mask = input_ids == self.config.image_token_id
        indices1 = mask.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(mask.shape[0]).view(-1, 1)
        vision_embeds_expanded = vision_embeds[indices0, indices1.clamp(min=0)]
        merged = torch.where(mask.unsqueeze(-1), vision_embeds_expanded, inputs_embeds)
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, merged)

        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            batch_index=batch_index,
            use_cache=True,
        )

        next_image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        image_idx = torch.where(image_idx < next_image_idx, next_image_idx, image_idx)

        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states).float()
        return logits, pixel_values, image_idx, outputs.past_key_values

    # ----- dummy inputs / specializations / dynamic axes / output names -------

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        **kwargs,
    ):
        prefill_seq_len = int(kwargs.get("prefill_seq_len") or constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        fbs = constants.ONNX_EXPORT_EXAMPLE_FBS

        text_cfg = self.config.text_config
        vision_cfg = self.config.vision_config

        # MiniMax vision input shape: Conv3d on [-1, C, T_p, P, P]; HF flattens to
        # [num_patches, C * T_p * P * P]. For dummy export we pick a small grid.
        patch = vision_cfg.patch_size
        t_patch = vision_cfg.temporal_patch_size
        spatial_merge = vision_cfg.spatial_merge_size
        # Pick a tiny grid: 1 frame, 8x8 patches -> 64 raw, 16 merged tokens.
        grid_t, grid_h, grid_w = 1, 8, 8
        num_patches = grid_t * grid_h * grid_w
        merged = num_patches // (spatial_merge * spatial_merge)

        vision_inputs = {
            "pixel_values": torch.zeros(
                (num_patches, vision_cfg.num_channels * t_patch * patch * patch),
                dtype=self.config.torch_dtype,
            ),
            "image_grid_thw": torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.int64),
        }

        lang_inputs = {
            "input_ids": torch.ones((bs, prefill_seq_len), dtype=torch.int64),
            "vision_embeds": torch.ones(
                (bs, merged, text_cfg.hidden_size), dtype=self.config.torch_dtype
            ),
            "position_ids": torch.arange(prefill_seq_len, dtype=torch.int64).view(1, -1).repeat(bs, 1),
            "image_idx": torch.zeros((1, 1), dtype=torch.int64),
        }

        kv_cache_shape = get_padding_shape_from_config(
            config=text_cfg,
            batch_size=fbs if continuous_batching else bs,
            seq_len=prefill_seq_len,
        )
        lang_inputs["past_key_values"] = []
        for _ in range(text_cfg.num_hidden_layers):
            lang_inputs["past_key_values"].append(
                (
                    torch.zeros(kv_cache_shape, dtype=self.config.torch_dtype),
                    torch.zeros(kv_cache_shape, dtype=self.config.torch_dtype),
                )
            )

        if continuous_batching:
            lang_inputs["batch_index"] = torch.arange(bs).view(bs, 1)

        if comp_ctx_lengths is not None:
            lang_inputs["comp_ctx_lengths"] = torch.randint(0, 100, (40,), dtype=torch.int64)

        if kv_offload:
            return {"vision": vision_inputs, "lang": lang_inputs}
        lang_inputs.pop("vision_embeds")
        return {**vision_inputs, **lang_inputs}

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        img_size: Optional[int] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
        **compiler_options,
    ):
        prefill_seq_len = prefill_seq_len or constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
        ctx_len = ctx_len or 1024

        # Drain CCL kwargs out of compiler_options before they're passed verbatim to
        # qaic-compile (without this, a None default leaks through as the literal
        # string "-comp-ctx-lengths-prefill=None" and the compiler rejects it).
        comp_ctx_lengths_prefill = compiler_options.pop("comp_ctx_lengths_prefill", None)
        comp_ctx_lengths_decode = compiler_options.pop("comp_ctx_lengths_decode", None)

        # MiniMax-M3: image_seq_length=576 per tile, dynamic-res tiling.
        image_seq_length = getattr(self.config, "image_seq_length", 576)
        max_num_images = compiler_options.pop("max_num_images", 1)
        vision_size = image_seq_length * max_num_images

        # Vision-side symbols for the pixel_values / image_grid_thw inputs. Each tile
        # contributes (image_size/patch_size)**2 patches; with spatial_merge=2 those
        # collapse to image_seq_length tokens. So num_patches per tile is 4*image_seq_length.
        spatial_merge = getattr(self.config.vision_config, "spatial_merge_size", 2)
        num_patches = image_seq_length * (spatial_merge * spatial_merge) * max_num_images
        num_images = max_num_images

        vision = [
            {
                "batch_size": batch_size,
                "max_num_images": max_num_images,
                "vision_size": vision_size,
                "num_patches": num_patches,
                "num_images": num_images,
            }
        ]

        lang_prefill = {
            "batch_size": 1 if continuous_batching else batch_size,
            "seq_len": prefill_seq_len,
            "ctx_len": ctx_len,
            "vision_size": vision_size,
            "vision_batch_size": batch_size,
            "max_num_images": max_num_images,
            "num_patches": num_patches,
            "num_images": num_images,
        }
        if continuous_batching:
            lang_prefill["full_batch_size"] = kv_cache_batch_size
        else:
            lang_prefill["batch_size"] = kv_cache_batch_size
        if full_batch_size:
            lang_prefill["full_batch_exec_size"] = full_batch_size

        lang_decode = {
            "batch_size": full_batch_size if continuous_batching else batch_size,
            "seq_len": "1",
            "ctx_len": ctx_len,
            "vision_size": vision_size,
            "vision_batch_size": batch_size,
            "max_num_images": max_num_images,
            "num_patches": num_patches,
            "num_images": num_images,
        }
        if continuous_batching:
            lang_decode["full_batch_size"] = kv_cache_batch_size
        else:
            lang_decode["batch_size"] = kv_cache_batch_size

        lang = [lang_prefill, lang_decode]

        if kv_offload:
            return {"vision": vision, "lang": lang}, compiler_options
        lang[0].pop("vision_size")
        lang[1].pop("vision_size")
        return lang, compiler_options

    def get_onnx_dynamic_axes(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
    ):
        num_layers = self.config.text_config.num_hidden_layers
        vision_dynamic_axes = {
            "pixel_values": {0: "num_patches"},
            "image_grid_thw": {0: "num_images"},
        }
        lang_dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "vision_embeds": {0: "vision_batch_size", 1: "vision_size"},
        }
        if continuous_batching:
            lang_dynamic_axes["batch_index"] = {0: "batch_size"}
        for i in range(num_layers):
            lang_dynamic_axes[f"past_key.{i}"] = {
                0: "full_batch_size" if continuous_batching else "batch_size",
                2: "ctx_len",
            }
            lang_dynamic_axes[f"past_value.{i}"] = {
                0: "full_batch_size" if continuous_batching else "batch_size",
                2: "ctx_len",
            }
        if comp_ctx_lengths is not None:
            lang_dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}

        if kv_offload:
            return {"vision": vision_dynamic_axes, "lang": lang_dynamic_axes}
        return {**vision_dynamic_axes, **lang_dynamic_axes}

    def get_output_names(self, kv_offload: bool = False):
        vision_output_names = ["vision_embeds"]
        lang_output_names = ["logits"]
        for i in range(self.config.text_config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_output_names.append(f"past_{kv}.{i}_RetainedState")

        if kv_offload:
            lang_output_names.insert(1, "vision_embeds_RetainedState")
            lang_output_names.insert(2, "image_idx_output")
            return {"vision": vision_output_names, "lang": lang_output_names}
        lang_output_names.insert(1, "pixel_values_RetainedState")
        lang_output_names.insert(2, "image_idx_output")
        return lang_output_names

    def get_inputs_info(self):
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="attention_mask", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="pixel_values", datatype=self.config.torch_dtype, shape=("num_patches", "patch_dim")),
            IOInfo(name="image_grid_thw", datatype=torch.int64, shape=("num_images", 3)),
        ]
