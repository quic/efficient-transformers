# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""PyTorch Mixtral model."""

import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention,
    MixtralConfig,
    MixtralDecoderLayer,
    MixtralExperts,
    MixtralForCausalLM,
    MixtralModel,
    MixtralRotaryEmbedding,
    MixtralSparseMoeBlock,
    MixtralTopKRouter,
    load_balancing_loss_func,
    repeat_kv,
    rotate_half,
)

from QEfficient.blocking.attention_blocking import (
    AttentionBlockingConfig,
    BlockingMode,
    generic_blocked_attention_interface,
    past_key_value_update,
)
from QEfficient.customop import ctx_gather_3d_generalized, ctx_scatter_3d_generalized, ctx_scatter_3d_int
from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.transformers.moe import (
    MoEFlavour,
    MoEProfile,
    MoEWeights,
    QEffMoEBlockMixin,
    build_canonical_expert_weights,
    delete_module_attrs,
    stack_expert_linears,
)
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


class QEffMixtralRotaryEmbedding(MixtralRotaryEmbedding):
    """
    Copied from MixtralForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    The only differences are:
    - Add static sin/cos computations.
    """

    def __init__(self, config: MixtralConfig, device=None):
        super().__init__(config=config)
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=self.original_max_seq_len, device=self.inv_freq.device, dtype=config.torch_dtype
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def qeff_apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    # Cast back to original dtype
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = torch.where(
            attention_mask,
            torch.full_like(attn_weights, MIN_MASKED_ATTENTION_VALUE, dtype=attn_weights.dtype),
            attn_weights,
        )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _mixtral_silu_glu_mlp_dtype_safe(
    x: torch.Tensor,
    W_g: torch.Tensor,
    W_u: torch.Tensor,
    W_d: torch.Tensor,
    b_g: torch.Tensor | None = None,
    b_u: torch.Tensor | None = None,
    b_d: torch.Tensor | None = None,
    *,
    act_fn=F.silu,
) -> torch.Tensor:
    """Mixtral expert MLP with explicit dtype alignment for ONNX/runtime safety."""
    compute_dtype = x.dtype
    x = x.to(compute_dtype)
    W_g = W_g if W_g.dtype == compute_dtype else W_g.to(compute_dtype)
    W_u = W_u if W_u.dtype == compute_dtype else W_u.to(compute_dtype)
    W_d = W_d if W_d.dtype == compute_dtype else W_d.to(compute_dtype)

    gate = x @ W_g
    up = x @ W_u
    if b_g is not None:
        gate = gate + b_g.to(compute_dtype).unsqueeze(-2)
    if b_u is not None:
        up = up + b_u.to(compute_dtype).unsqueeze(-2)

    down = (up * act_fn(gate)) @ W_d
    if b_d is not None:
        down = down + b_d.to(compute_dtype).unsqueeze(-2)
    return down


def _build_matched_idx_from_cumsum(T2Ei: torch.Tensor) -> torch.Tensor:
    """Build packed->original token index."""
    _batch_size, seq_len = T2Ei.shape
    int32_max = torch.iinfo(torch.int32).max
    int32_max_scalar = torch.tensor(int32_max, dtype=torch.int32, device=T2Ei.device)
    token_idx = torch.arange(seq_len, dtype=torch.int32, device=T2Ei.device).unsqueeze(0).expand(T2Ei.shape[0], -1)
    valid_prefix = torch.cumsum(T2Ei.to(torch.int32), dim=1)
    valid_dest = valid_prefix - 1
    scatter_pos = torch.where(T2Ei, valid_dest, int32_max_scalar)
    matched_idx = torch.full_like(token_idx, int32_max)
    matched_idx = ctx_scatter_3d_int(
        matched_idx.unsqueeze(-1),
        scatter_pos,
        token_idx.unsqueeze(-1),
    ).squeeze(-1)
    return matched_idx


def _cumsum_scatter_gather_update_mixtral_expert_blocked(
    x: torch.Tensor,
    T2Ei: torch.Tensor,
    W_g: torch.Tensor,
    W_u: torch.Tensor,
    W_d: torch.Tensor,
    b_g: torch.Tensor | None,
    b_u: torch.Tensor | None,
    b_d: torch.Tensor | None,
    routing_weight: torch.Tensor,
    expert_out: torch.Tensor,
    act_fn,
    packed_chunk_size: int,
) -> torch.Tensor:
    """Cumsum-scatter-gather-update expert helper for Mixtral NSP-blocked dispatch."""
    _batch_size, seq_len = T2Ei.shape
    packed_chunk_size = max(1, min(packed_chunk_size, seq_len))

    matched_idx = _build_matched_idx_from_cumsum(T2Ei)
    valid_rows = torch.einsum("ij->i", T2Ei.to(torch.int32)).unsqueeze(1)
    row_range = torch.arange(packed_chunk_size, dtype=torch.int32, device=x.device).unsqueeze(0)
    x_expanded = x.unsqueeze(0).expand(T2Ei.shape[0], -1, -1)

    for packed_start in range(0, seq_len, packed_chunk_size):
        packed_stop = packed_start + packed_chunk_size
        chunk_matched_idx = matched_idx[:, packed_start:packed_stop]

        x_chunk = ctx_gather_3d_generalized(x_expanded, chunk_matched_idx)

        gate = x_chunk @ W_g
        up = x_chunk @ W_u
        if b_g is not None:
            gate = gate + b_g.unsqueeze(1)
        if b_u is not None:
            up = up + b_u.unsqueeze(1)
        down_chunk = (up * act_fn(gate)) @ W_d
        if b_d is not None:
            down_chunk = down_chunk + b_d.unsqueeze(1)

        rw_chunk = ctx_gather_3d_generalized(routing_weight, chunk_matched_idx)
        down_chunk = down_chunk * rw_chunk

        expert_out_chunk = ctx_gather_3d_generalized(expert_out, chunk_matched_idx)
        updated_chunk = expert_out_chunk + down_chunk

        chunk_valid_rows = torch.clamp(
            valid_rows - packed_start,
            min=torch.zeros_like(valid_rows),
            max=torch.full_like(valid_rows, packed_chunk_size),
        )
        updated_chunk = torch.where(
            (row_range < chunk_valid_rows).unsqueeze(-1), updated_chunk, torch.zeros_like(updated_chunk)
        )
        expert_out = ctx_scatter_3d_generalized(expert_out, chunk_matched_idx, updated_chunk)

    return expert_out


class QEffMixtralAttention(MixtralAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        cos_cached: torch.Tensor | None = None,
        sin_cached: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if past_key_values is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )

        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos_cached, sin_cached)

        past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
        blocking_config = getattr(self, "attn_blocking_config", AttentionBlockingConfig())
        use_blocking = blocking_config is not None and (blocking_config.mode != BlockingMode.NONE)
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
                position_ids=position_ids,
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
                position_ids=position_ids,
            )
            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                scaling=self.scaling,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


MIXTRAL_ATTENTION_CLASSES = {
    "eager": MixtralAttention,
}


class QEffMixtralTopKRouter(MixtralTopKRouter):
    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        compute_dtype = hidden_states.dtype
        router_weight = self.weight if self.weight.dtype == compute_dtype else self.weight.to(compute_dtype)
        router_logits = F.linear(hidden_states.to(compute_dtype), router_weight)
        router_probs = torch.softmax(router_logits.float(), dim=-1).to(router_logits.dtype)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if getattr(self, "norm_topk_prob", True):
            router_top_value = router_top_value / torch.einsum("bk->b", router_top_value).unsqueeze(-1)
        router_scores = router_top_value.to(router_logits.dtype)
        # Keep Mixtral semantics: first return value is softmaxed routing probabilities.
        return router_probs, router_scores, router_indices


class QEffMixtralExperts(MixtralExperts):
    def __qeff_init__(self):
        self.expert_dim = getattr(self, "intermediate_dim", self.gate_up_proj.shape[-2] // 2)
        gate_up_proj = self.gate_up_proj.detach()
        down_proj = self.down_proj.detach()
        self.gate_proj = nn.Parameter(
            gate_up_proj[:, : self.expert_dim, :].transpose(1, 2), requires_grad=False
        ).clone()
        self.up_proj = nn.Parameter(gate_up_proj[:, self.expert_dim :, :].transpose(1, 2), requires_grad=False).clone()
        self.down_proj_t = nn.Parameter(down_proj.transpose(1, 2), requires_grad=False)
        gate_up_proj_bias = getattr(self, "gate_up_proj_bias", None)
        self.gate_proj_bias = (
            nn.Parameter(gate_up_proj_bias[:, : self.expert_dim].detach(), requires_grad=False)
            if gate_up_proj_bias is not None
            else None
        )
        self.up_proj_bias = (
            nn.Parameter(gate_up_proj_bias[:, self.expert_dim :].detach(), requires_grad=False)
            if gate_up_proj_bias is not None
            else None
        )
        down_proj_bias = getattr(self, "down_proj_bias", None)
        self.down_proj_t_bias = (
            nn.Parameter(down_proj_bias.detach(), requires_grad=False) if down_proj_bias is not None else None
        )
        self.weights_transformed = False

    def transform_weights(self) -> MoEWeights:
        if getattr(self, "weights_transformed", False):
            return self.moe_weights

        if hasattr(self, "gate_up_proj"):
            self.moe_weights = build_canonical_expert_weights(
                gate_up=self.gate_up_proj,
                down=self.down_proj,
                fused=True,
                fused_split_dim=1,
                transpose_gate_up=True,
                transpose_down=True,
                clone=True,
            )
            delete_module_attrs(self, "gate_up_proj", "down_proj")
        else:
            self.moe_weights = MoEWeights(
                gate=stack_expert_linears(self, lambda expert: expert.w1.weight).clone(),
                up=stack_expert_linears(self, lambda expert: expert.w3.weight).clone(),
                down=stack_expert_linears(self, lambda expert: expert.w2.weight).clone(),
            )
            for expert in self:
                delete_module_attrs(expert, "w1", "w2", "w3")

        self.weights_transformed = True
        return self.moe_weights


class QEffMixtralSparseMoeBlock(QEffMoEBlockMixin, MixtralSparseMoeBlock):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    _moe_return_router_logits = True
    supports_moe_prefill_blocking = True
    supported_moe_flavours = (
        MoEFlavour.SIMPLE_LOOP,
        MoEFlavour.DECODE_BMM,
    )

    def __qeff_init__(self):
        super().__qeff_init__()
        self.top_k = getattr(self.gate, "top_k", getattr(self, "top_k", None))
        # Mixtral gate already returns normalized top-k weights.
        self.norm_topk_prob = getattr(self.gate, "norm_topk_prob", False)
        self.num_experts = getattr(self.gate, "num_experts", getattr(self.experts, "num_experts", None))
        if hasattr(self.experts, "act_fn"):
            self.act_fn = self.experts.act_fn
        else:
            self.act_fn = getattr(self.experts[0], "act_fn", F.silu)

    def transform_weights(self) -> MoEWeights:
        if getattr(self, "weights_transformed", False):
            return self.moe_weights
        # In some transform flows only the sparse block is remapped first, so
        # `self.experts` may still be HF MixtralExperts (without transform_weights()).
        if hasattr(self.experts, "transform_weights"):
            self.moe_weights = self.experts.transform_weights()
        else:
            if hasattr(self.experts, "gate_up_proj"):
                self.moe_weights = build_canonical_expert_weights(
                    gate_up=self.experts.gate_up_proj,
                    down=self.experts.down_proj,
                    fused=True,
                    fused_split_dim=1,
                    transpose_gate_up=True,
                    transpose_down=True,
                    clone=True,
                )
                delete_module_attrs(self.experts, "gate_up_proj", "down_proj")
            else:
                self.moe_weights = MoEWeights(
                    gate=stack_expert_linears(self.experts, lambda expert: expert.w1.weight).clone(),
                    up=stack_expert_linears(self.experts, lambda expert: expert.w3.weight).clone(),
                    down=stack_expert_linears(self.experts, lambda expert: expert.w2.weight).clone(),
                )
                for expert in self.experts:
                    delete_module_attrs(expert, "w1", "w2", "w3")
        if hasattr(self.experts, "act_fn"):
            self.act_fn = self.experts.act_fn
        else:
            self.act_fn = getattr(self.experts[0], "act_fn", F.silu)
        self.weights_transformed = True
        return self.moe_weights

    @property
    def moe_profile(self) -> MoEProfile:
        return MoEProfile(expert_mlp=partial(_mixtral_silu_glu_mlp_dtype_safe, act_fn=getattr(self, "act_fn", F.silu)))

    def route(self, x: torch.Tensor):
        gate_dtype = getattr(getattr(self.gate, "weight", None), "dtype", x.dtype)
        gate_out = self.gate(x.to(gate_dtype))
        if isinstance(gate_out, tuple) and len(gate_out) >= 3:
            router_logits, routing_weights, selected_experts = gate_out[0], gate_out[1], gate_out[2]
            routing_weights = routing_weights.to(x.dtype)
        else:
            router_logits = gate_out[0] if isinstance(gate_out, tuple) else gate_out
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            routing_weights = routing_weights / torch.einsum("bi->b", routing_weights)[:, None]
            routing_weights = routing_weights.to(x.dtype)
        return (selected_experts, routing_weights), router_logits

    def _get_prefill_expert_tensors(
        self, x_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if all(hasattr(self.experts, attr) for attr in ("gate_proj", "up_proj", "down_proj_t")):
            gate_proj = self.experts.gate_proj
            up_proj = self.experts.up_proj
            down_proj_t = self.experts.down_proj_t
            gate_proj_bias = getattr(self.experts, "gate_proj_bias", None)
            up_proj_bias = getattr(self.experts, "up_proj_bias", None)
            down_proj_t_bias = getattr(self.experts, "down_proj_t_bias", None)
            return (
                gate_proj.to(x_dtype),
                up_proj.to(x_dtype),
                down_proj_t.to(x_dtype),
                gate_proj_bias.to(x_dtype) if gate_proj_bias is not None else None,
                up_proj_bias.to(x_dtype) if up_proj_bias is not None else None,
                down_proj_t_bias.to(x_dtype) if down_proj_t_bias is not None else None,
            )

        if getattr(self, "weights_transformed", False) and hasattr(self, "moe_weights"):
            weights = self.moe_weights
            return (
                weights.gate.to(x_dtype),
                weights.up.to(x_dtype),
                weights.down.to(x_dtype),
                weights.gate_bias.to(x_dtype) if weights.gate_bias is not None else None,
                weights.up_bias.to(x_dtype) if weights.up_bias is not None else None,
                weights.down_bias.to(x_dtype) if weights.down_bias is not None else None,
            )

        if hasattr(self.experts, "gate_up_proj") and hasattr(self.experts, "down_proj"):
            expert_dim = self.experts.gate_up_proj.shape[1] // 2
            gate_proj = self.experts.gate_up_proj[:, :expert_dim, :].transpose(1, 2)
            up_proj = self.experts.gate_up_proj[:, expert_dim:, :].transpose(1, 2)
            down_proj_t = self.experts.down_proj.transpose(1, 2)
            return gate_proj.to(x_dtype), up_proj.to(x_dtype), down_proj_t.to(x_dtype), None, None, None

        raise RuntimeError("Mixtral prefill block cannot resolve expert weights for dispatch.")

    def _forward_prefill_blocked(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        T = batch_size * sequence_length
        x = hidden_states.view(T, hidden_dim)
        router_logits, top_w, top_i = self.gate(x)
        top_w = top_w.to(x.dtype)
        routing_weights = torch.zeros_like(router_logits)
        routing_weights.scatter_(1, top_i, top_w)

        num_experts = int(getattr(self.gate, "num_experts", getattr(self, "num_experts", 0)))
        num_nsp = getattr(self, "expert_blocking_num_nsp", num_experts)
        packed_chunk_size = getattr(self, "expert_blocking_packed_chunk_size", T)
        if num_experts % num_nsp != 0:
            num_nsp = max(1, math.gcd(num_experts, num_nsp))

        local_experts = num_experts // num_nsp
        rw = routing_weights.transpose(0, 1).contiguous().view(local_experts, num_nsp, T).transpose(0, 1).contiguous()
        gate_proj, up_proj, down_proj_t, gate_proj_bias, up_proj_bias, down_proj_bias = (
            self._get_prefill_expert_tensors(x.dtype)
        )

        W_g = gate_proj.view(local_experts, num_nsp, hidden_dim, -1).transpose(0, 1).contiguous().to(x.dtype)
        W_u = up_proj.view(local_experts, num_nsp, hidden_dim, -1).transpose(0, 1).contiguous().to(x.dtype)
        W_d = down_proj_t.view(local_experts, num_nsp, -1, hidden_dim).transpose(0, 1).contiguous().to(x.dtype)
        if gate_proj_bias is not None:
            b_g = gate_proj_bias.view(local_experts, num_nsp, -1).transpose(0, 1).contiguous().to(x.dtype)
            b_u = up_proj_bias.view(local_experts, num_nsp, -1).transpose(0, 1).contiguous().to(x.dtype)
        else:
            b_g = b_u = None
        b_d = (
            down_proj_bias.view(local_experts, num_nsp, hidden_dim).transpose(0, 1).contiguous().to(x.dtype)
            if down_proj_bias is not None
            else None
        )

        expert_out = x.new_zeros((num_nsp, T, hidden_dim))
        routing_weights_unsqueezed = rw.unsqueeze(-1)
        act_fn = getattr(self.experts, "act_fn", F.silu)
        for slot in range(local_experts):
            T2Ei = rw[:, slot, :] > 0
            expert_out = _cumsum_scatter_gather_update_mixtral_expert_blocked(
                x=x,
                T2Ei=T2Ei,
                W_g=W_g[:, slot],
                W_u=W_u[:, slot],
                W_d=W_d[:, slot],
                b_g=b_g[:, slot] if b_g is not None else None,
                b_u=b_u[:, slot] if b_u is not None else None,
                b_d=b_d[:, slot] if b_d is not None else None,
                routing_weight=routing_weights_unsqueezed[:, slot],
                expert_out=expert_out,
                act_fn=act_fn,
                packed_chunk_size=packed_chunk_size,
            )
        final_hidden_states = torch.einsum("nth->th", expert_out).reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    def forward(self, hidden_states: torch.Tensor):
        if getattr(self, "expert_blocking_num_nsp", None) is not None:
            return self._forward_prefill_blocked(hidden_states)
        return super().forward(hidden_states)


QEffPrefillChunkedMixtralSparseMoeBlock = QEffMixtralSparseMoeBlock


class QeffMixtralDecoderLayer(MixtralDecoderLayer):
    """
    Copied from MixtralForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py
    The only differences are:
    - add new args batch idx for the CB retention
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        output_router_logits: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,  # necessary, but kept here for BC
        sin_cached=None,
        cos_cached=None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
                should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
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
            sin_cached=sin_cached,
            cos_cached=cos_cached,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        moe_block = getattr(self, "block_sparse_moe", None)
        if moe_block is None:
            moe_block = getattr(self, "mlp", None)
        moe_out = moe_block(hidden_states)
        if isinstance(moe_out, tuple):
            hidden_states, _ = moe_out
        else:
            hidden_states, _ = moe_out, None
        hidden_states = residual + hidden_states

        return hidden_states


# Copied from transformers.models.mistral.modeling_mistral.MistralModel with MISTRAL->MIXTRAL,Mistral->Mixtral
class QEffMixtralModel(MixtralModel):
    """
    Copied from MixtralModel: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py
    The only differences are:
    - add new args position idx for the cache_kwargs for kv retention
    - update causal attention mask
    """

    def __qeff_init__(self):
        self.rotary_emb = QEffMixtralRotaryEmbedding(config=self.config)
        self.sin_cached = torch.nn.Parameter(self.rotary_emb.sin_cached * self.rotary_emb.attention_scaling)
        self.cos_cached = torch.nn.Parameter(self.rotary_emb.cos_cached * self.rotary_emb.attention_scaling)

    # Ignore copy
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | MoeModelOutputWithPast:
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        use_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            use_legacy_cache = True
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

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
            position_ids=position_ids, target_length=target_length, sliding_window=self.config.sliding_window
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        sin = self.sin_cached[position_ids].unsqueeze(1).to(device=hidden_states.device)
        cos = self.cos_cached[position_ids].unsqueeze(1).to(device=hidden_states.device)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                batch_index=batch_index,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                sin_cached=sin,
                cos_cached=cos,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if use_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
        )


class QEffMixtralForCausalLM(MixtralForCausalLM):
    """
    Copied from MixtralForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py
    The only differences are:
    - add new args position idx for the cache_kwargs for kv retention
    - update the hidden_states, and fix for onnx model
    """

    def get_submodules_for_export(self) -> type[nn.Module]:
        """
        Return the set of class used as the repeated layer across the model for subfunction extraction.
        Notes:
            This method should return the *class object* (not an instance).
            Downstream code can use this to find/build subfunctions for repeated blocks.
        """
        return {QeffMixtralDecoderLayer}

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | MoeCausalLMOutputWithPast:
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        # Cast to int32 to avoid ONNXRT issue
        logit_idx = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[
            torch.arange(position_ids.shape[0], device=position_ids.device).view(-1, 1), logit_idx
        ]
        lm_head_dtype = self.lm_head.weight.dtype
        logits = self.lm_head(hidden_states.to(lm_head_dtype)).float()

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1],
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )

        return MoeCausalLMOutputWithPast(
            loss=None,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )
