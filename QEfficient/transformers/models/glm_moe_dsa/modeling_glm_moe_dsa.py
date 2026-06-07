from typing import Optional, Type, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaAttention,
    GlmMoeDsaDecoderLayer,
    GlmMoeDsaForCausalLM,
    GlmMoeDsaIndexer,
    GlmMoeDsaModel,
    GlmMoeDsaMoE,
    GlmMoeDsaRMSNorm,
    GlmMoeDsaRotaryEmbedding,
    GlmMoeDsaTopkRouter,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from QEfficient.customop.ctx_scatter_gather import CtxGatherFuncBlockedKV, CtxScatterFunc3DGeneralized
from QEfficient.transformers.cache_utils import QEffDynamicCache, QEffDynamicCompressedKVRopeCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


class QEffGlmMoeDsaIndexer(GlmMoeDsaIndexer):
    def forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        indexer_key_cache: Optional[torch.Tensor] = None,
        blocked_indexer: bool = False,
        num_kv_blocks: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        q = self.wq_b(q_resid).view(batch_size, seq_len, self.n_heads, self.head_dim)
        q_pe = q[..., : self.qk_rope_head_dim]
        q_nope = q[..., self.qk_rope_head_dim :]
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_pe, q_nope], dim=-1)

        k = self.k_norm(self.wk(hidden_states))
        k_pe = k[..., : self.qk_rope_head_dim]
        k_nope = k[..., self.qk_rope_head_dim :]
        k_pe = apply_rotary_pos_emb(k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)

        if indexer_key_cache is not None:
            invalid_scatter_index = torch.iinfo(torch.int32).max
            cache_positions = torch.where(position_ids < 0, invalid_scatter_index, position_ids).to(torch.int32)
            indexer_key_cache = CtxScatterFunc3DGeneralized.apply(indexer_key_cache, cache_positions, k)
            k_cached = indexer_key_cache
        else:
            if seq_len > 1:
                self._cached_keys = None
            if self._cached_keys is not None:
                k_cached = torch.cat([self._cached_keys, k], dim=1)
            else:
                k_cached = k
            self._cached_keys = k_cached

        weights = self.weights_proj(hidden_states).float() * (self.n_heads**-0.5)
        scores = torch.einsum("bshd,btd->bsht", q.float(), k_cached.float()) * self.softmax_scale
        scores = F.relu(scores)
        index_scores = torch.einsum("bsht,bsh->bst", scores, weights).to(hidden_states.dtype)

        if attention_mask is not None and attention_mask.shape[-1] > 0:
            index_scores = index_scores + attention_mask[..., : index_scores.shape[-1]]
        ctx_indices = torch.arange(index_scores.shape[-1], device=hidden_states.device).view(1, 1, -1)
        future_mask = ctx_indices > position_ids.unsqueeze(-1)
        index_scores = index_scores.masked_fill(future_mask, MIN_MASKED_ATTENTION_VALUE)

        topk = min(self.index_topk, index_scores.shape[-1])
        topk_indices, valid_topk = _build_dsa_topk_indices(
            index_scores=index_scores,
            position_ids=position_ids,
            topk=topk,
            blocked_indexer=blocked_indexer,
            num_kv_blocks=num_kv_blocks,
            ctx_len_hint=getattr(self, "ctx_len_hint", None),
        )
        return topk_indices, valid_topk, indexer_key_cache


def _build_dsa_topk_indices(
    index_scores: torch.Tensor,
    position_ids: torch.Tensor,
    topk: int,
    blocked_indexer: bool = False,
    num_kv_blocks: int = 1,
    ctx_len_hint: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_position = position_ids.max(dim=-1).values
    if not blocked_indexer or num_kv_blocks <= 1:
        topk_indices = torch.topk(index_scores, k=topk, dim=-1).indices.to(torch.int32)
        valid_topk = topk_indices.to(position_ids.dtype) <= current_position.unsqueeze(-1)
        return topk_indices, valid_topk

    masked_score = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=index_scores.dtype, device=index_scores.device)
    ctx_len = index_scores.shape[-1]
    block_size = -(-(ctx_len_hint or ctx_len) // num_kv_blocks)
    candidate_scores = []
    candidate_indices = []

    for block_idx in range(num_kv_blocks):
        start_index = block_idx * block_size
        end_index = min(start_index + block_size, ctx_len)
        kv_len_block = end_index - start_index
        if kv_len_block <= 0:
            continue

        skip_future = (torch.tensor(start_index, device=index_scores.device) > current_position).all()
        if not torch.onnx.is_in_onnx_export() and not torch.jit.is_tracing():
            if skip_future.item():
                break

        scores_block = index_scores[..., start_index:end_index]
        if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
            scores_block = torch.where(skip_future, masked_score, scores_block)

        local_k = min(topk, block_size)
        local_topk = torch.topk(scores_block, k=local_k, dim=-1)
        candidate_scores.append(local_topk.values)
        candidate_indices.append(local_topk.indices.to(torch.int32) + start_index)

    merged_scores = torch.cat(candidate_scores, dim=-1)
    merged_indices = torch.cat(candidate_indices, dim=-1)
    final_k = min(topk, merged_scores.shape[-1])
    final_topk = torch.topk(merged_scores, k=final_k, dim=-1)
    topk_indices = torch.gather(merged_indices, -1, final_topk.indices.to(torch.int64)).to(torch.int32)
    valid_topk = topk_indices.to(position_ids.dtype) <= current_position.unsqueeze(-1)
    return topk_indices, valid_topk


def _gather_dsa_sparse_cache(
    compressed_kvs: QEffDynamicCompressedKVRopeCache,
    layer_idx: int,
    topk_indices: torch.Tensor,
    valid_topk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = compressed_kvs.layers[layer_idx]
    batch, num_kv_heads, _, _ = layer.ckv.shape
    gather_indices = topk_indices[:, -1, :].unsqueeze(1).expand(batch, num_kv_heads, -1)
    invalid_idx_value = torch.iinfo(torch.int32).max if torch.onnx.is_in_onnx_export() else 0
    gather_indices = torch.where(valid_topk[:, -1, :].unsqueeze(1), gather_indices, invalid_idx_value).to(torch.int32)

    ckv = CtxGatherFuncBlockedKV.apply(layer.ckv, gather_indices)
    k_pe = CtxGatherFuncBlockedKV.apply(layer.k_pe, gather_indices)
    invalid_mask = ~valid_topk[:, -1, :].unsqueeze(1).unsqueeze(-1)
    ckv = torch.where(invalid_mask, torch.zeros(1, dtype=ckv.dtype, device=ckv.device), ckv)
    k_pe = torch.where(invalid_mask, torch.zeros(1, dtype=k_pe.dtype, device=k_pe.device), k_pe)
    return ckv, k_pe


def _dsa_mla_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    per_head_v_up: torch.Tensor,
    scaling: float,
    layer_idx: int,
    compressed_kvs: QEffDynamicCompressedKVRopeCache,
    topk_indices: torch.Tensor,
    valid_topk: torch.Tensor,
    par_num_split: Optional[int] = None,
) -> torch.Tensor:
    batch, num_query_heads, query_len, d_abs = query.shape
    kv_lora_rank = module.config.kv_lora_rank
    ckv_sparse, k_pe_sparse = _gather_dsa_sparse_cache(compressed_kvs, layer_idx, topk_indices, valid_topk)
    num_kv_heads = ckv_sparse.shape[1]
    num_repeats = num_query_heads // num_kv_heads
    split = par_num_split or num_kv_heads
    key = torch.cat((ckv_sparse, k_pe_sparse), dim=-1)
    ckv_for_v = ckv_sparse
    topk = key.shape[2]
    valid_topk_mask = valid_topk[:, -1, :].to(query.dtype)
    pad = 0
    if topk % split != 0:
        pad = split - (topk % split)
        key = F.pad(key, (0, 0, 0, pad), value=0.0)
        ckv_for_v = F.pad(ckv_for_v, (0, 0, 0, pad), value=0.0)
        valid_topk_mask = F.pad(valid_topk_mask, (0, pad), value=0.0)
    topk += pad

    topk_per_split = topk // split
    q_fold = query.reshape(batch, num_kv_heads, query_len * num_repeats, d_abs)
    query_5d = q_fold.unsqueeze(2).expand(batch, num_kv_heads, split, query_len * num_repeats, d_abs)
    key_5d = key.view(batch, num_kv_heads, split, topk_per_split, d_abs)
    value_5d = ckv_for_v.view(batch, num_kv_heads, split, topk_per_split, kv_lora_rank)

    attn = torch.matmul(query_5d.float(), key_5d.transpose(-1, -2).float()) * scaling
    valid_mask = valid_topk_mask.view(batch, 1, split, 1, topk_per_split) > 0
    attn = attn.masked_fill(~valid_mask, MIN_MASKED_ATTENTION_VALUE)
    max_split = attn.max(dim=-1).values
    exp_attn = torch.exp(attn - max_split.unsqueeze(-1))
    exp_attn = torch.where(valid_mask, exp_attn, torch.zeros(1, dtype=exp_attn.dtype, device=exp_attn.device))
    sum_split = torch.einsum("bhsqt->bhsq", exp_attn)
    out_split = torch.matmul(exp_attn, value_5d.float())

    global_max = max_split.max(dim=2).values
    merge_weight = torch.exp(max_split - global_max.unsqueeze(2))
    denom = torch.einsum("bhsq,bhsq->bhq", merge_weight, sum_split) + 1e-20
    output = torch.einsum("bhsq,bhsqd->bhqd", merge_weight, out_split) / denom.unsqueeze(-1)
    output = output.to(query.dtype)
    output = output.view(batch, num_kv_heads, num_repeats, query_len, kv_lora_rank).reshape(
        batch, num_query_heads, query_len, kv_lora_rank
    )
    attn_output = torch.matmul(output, per_head_v_up.to(output.dtype))
    return attn_output.transpose(1, 2).contiguous()


def _ensure_compressed_cache_layer(
    compressed_kvs: QEffDynamicCompressedKVRopeCache,
    layer_idx: int,
    batch_size: int,
    ctx_len: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    while len(compressed_kvs.layers) <= layer_idx:
        compressed_kvs.add_new(
            torch.zeros((batch_size, 1, ctx_len, kv_lora_rank), dtype=dtype, device=device),
            torch.zeros((batch_size, 1, ctx_len, qk_rope_head_dim), dtype=dtype, device=device),
            layer_idx,
        )


class QEffGlmMoeDsaAttention(GlmMoeDsaAttention):
    def __qeff_init__(self):
        q_up, q_rope = self.q_b_proj.weight.T.view(
            -1, self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim
        ).split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        self.q_up = torch.nn.Parameter(q_up.reshape(-1, self.num_heads * self.qk_nope_head_dim).unsqueeze(0).detach())
        self.q_rope = torch.nn.Parameter(
            q_rope.reshape(-1, self.num_heads * self.qk_rope_head_dim).unsqueeze(0).detach()
        )

        k_up, v_up = self.kv_b_proj.weight.T.view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        self.k_up = torch.nn.Parameter(k_up.reshape(-1, self.num_heads * self.qk_nope_head_dim).unsqueeze(0).detach())
        self.v_up = torch.nn.Parameter(v_up.reshape(-1, self.num_heads * self.v_head_dim).unsqueeze(0).detach())

        per_head_q_up = self.q_up.squeeze(0).view(-1, self.num_heads, self.qk_nope_head_dim).transpose(0, 1)
        per_head_k_up = (
            self.k_up.squeeze(0).view(-1, self.num_heads, self.qk_nope_head_dim).transpose(0, 1).transpose(1, 2)
        )
        per_head_v_up = self.v_up.squeeze(0).view(-1, self.num_heads, self.v_head_dim).transpose(0, 1)
        self.per_head_q_up = torch.nn.Parameter(per_head_q_up.unsqueeze(0).detach())
        self.per_head_k_up = torch.nn.Parameter(per_head_k_up.unsqueeze(0).detach())
        self.per_head_v_up = torch.nn.Parameter(per_head_v_up.unsqueeze(0).detach())
        self.per_head_k_up_normal = torch.nn.Parameter(self.per_head_k_up.transpose(2, 3).detach())
        fusedqk = torch.bmm(per_head_q_up, per_head_k_up).reshape(
            -1, self.num_heads, self.q_lora_rank, self.kv_lora_rank
        )
        self.fusedqk = torch.nn.Parameter(fusedqk.detach())

    def _q_resid_and_query_pe(self, hidden_states, position_embeddings):
        batch_size, seq_len = hidden_states.shape[:-1]
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_pe = torch.matmul(q_resid, self.q_rope)
        q_pe = q_pe.view(batch_size, seq_len, self.num_heads, self.qk_rope_head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=1)
        return q_resid, q_pe

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        compressed_kvs: Optional[QEffDynamicCompressedKVRopeCache] = None,
        indexer_key_cache: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        mla_absorption: Optional[dict[str, bool]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if mla_absorption is not None and mla_absorption.get("cache_compressed", False):
            return self.forward_compressed_mla_dsa(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                compressed_kvs=compressed_kvs,
                indexer_key_cache=indexer_key_cache,
                position_ids=position_ids,
                batch_index=batch_index,
                mla_absorption=mla_absorption,
            )

        batch_size, seq_length = hidden_states.shape[:-1]
        cos, sin = position_embeddings
        q_resid, q_pe = self._q_resid_and_query_pe(hidden_states, position_embeddings)
        query_states = self.q_b_proj(q_resid).view(batch_size, seq_length, -1, self.qk_head_dim).transpose(1, 2)
        q_nope = query_states[..., : self.qk_nope_head_dim]

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_compressed = compressed_kv[..., : self.kv_lora_rank]
        k_pe = compressed_kv[..., self.kv_lora_rank :]
        k_compressed = self.kv_a_layernorm(k_compressed)
        kv_expanded = self.kv_b_proj(k_compressed).view(
            batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope = kv_expanded[..., : self.qk_nope_head_dim]
        value_states = kv_expanded[..., self.qk_nope_head_dim :]
        k_nope = k_nope.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        k_pe = apply_rotary_pos_emb(k_pe, cos, sin, unsqueeze_dim=1)
        k_pe = k_pe.expand(-1, k_nope.shape[1], -1, -1)

        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe], dim=-1)
        cache_kwargs = {"position_ids": position_ids, "batch_index": batch_index}
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        indexer_mask = (
            attention_mask[:, 0, :, :] if attention_mask is not None and attention_mask.dim() == 4 else attention_mask
        )
        topk_indices, valid_topk, indexer_key_cache = self.indexer.forward_with_cache(
            hidden_states=hidden_states,
            q_resid=q_resid,
            position_embeddings=position_embeddings,
            attention_mask=indexer_mask,
            position_ids=position_ids,
            indexer_key_cache=indexer_key_cache,
        )

        total_len = key_states.shape[2]
        safe_topk_indices = torch.where(valid_topk, topk_indices, torch.zeros_like(topk_indices))
        ctx_positions = torch.arange(total_len, device=hidden_states.device).view(1, 1, 1, total_len)
        selected_positions = (safe_topk_indices.unsqueeze(-1).to(ctx_positions.dtype) == ctx_positions).to(
            query_states.dtype
        )
        selected_positions = torch.einsum("bskt->bst", selected_positions)
        index_mask = torch.where(
            selected_positions > 0,
            torch.zeros(1, dtype=query_states.dtype, device=query_states.device),
            torch.full((1,), MIN_MASKED_ATTENTION_VALUE, dtype=query_states.dtype, device=query_states.device),
        )
        combined_mask = index_mask.unsqueeze(1)
        if attention_mask is not None and attention_mask.shape[-1] > 0:
            combined_mask = combined_mask + attention_mask[..., :total_len]

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            combined_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = self.o_proj(attn_output.reshape(batch_size, seq_length, -1).contiguous())
        return attn_output, attn_weights, indexer_key_cache

    def forward_compressed_mla_dsa(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        compressed_kvs: QEffDynamicCompressedKVRopeCache,
        indexer_key_cache: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        batch_index: Optional[torch.LongTensor],
        mla_absorption: dict[str, bool],
    ):
        if hidden_states.shape[1] != 1:
            raise ValueError("GLM-MoE-DSA compressed MLA path is decode-only; use sequence length 1.")
        if compressed_kvs is None:
            raise ValueError("compressed_kvs is required when cache_compressed=True.")

        batch_size, seq_length = hidden_states.shape[:-1]
        q_resid, q_pe = self._q_resid_and_query_pe(hidden_states, position_embeddings)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kva = compressed_kv[..., : self.kv_lora_rank]
        k_pe = compressed_kv[..., self.kv_lora_rank :]
        kva = self.kv_a_layernorm(kva).view(batch_size, seq_length, 1, self.kv_lora_rank).transpose(1, 2)
        k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        cos, sin = position_embeddings
        k_pe = apply_rotary_pos_emb(k_pe, cos, sin, unsqueeze_dim=1)
        if len(compressed_kvs.layers) <= self.layer_idx:
            if compressed_kvs.layers:
                ctx_len = compressed_kvs.layers[0].ckv.shape[2]
            elif position_ids is not None and not torch.onnx.is_in_onnx_export() and not torch.jit.is_tracing():
                ctx_len = max(int(position_ids.max().item()) + 1, seq_length)
            else:
                ctx_len = seq_length
            _ensure_compressed_cache_layer(
                compressed_kvs=compressed_kvs,
                layer_idx=self.layer_idx,
                batch_size=batch_size,
                ctx_len=ctx_len,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                dtype=kva.dtype,
                device=kva.device,
            )
        cache_kwargs = {"position_ids": position_ids, "batch_index": batch_index}
        compressed_kvs.write_only_ckv(kva, self.layer_idx, cache_kwargs)
        compressed_kvs.write_only_k_pe(k_pe, self.layer_idx, cache_kwargs)

        dsa_impl = getattr(self, "dsa_impl", "dsa_par")
        blocked_indexer = dsa_impl == "dsa_par_blocked"
        num_kv_blocks = int(getattr(self, "dsa_num_kv_blocks", 1) or 1)
        if blocked_indexer and num_kv_blocks <= 1:
            raise ValueError("GLM-MoE-DSA blocked DSA decode requires qaic_config['num_kv_blocks'] > 1.")

        indexer_mask = (
            attention_mask[:, 0, :, :] if attention_mask is not None and attention_mask.dim() == 4 else attention_mask
        )
        topk_indices, valid_topk, indexer_key_cache = self.indexer.forward_with_cache(
            hidden_states=hidden_states,
            q_resid=q_resid,
            position_embeddings=position_embeddings,
            attention_mask=indexer_mask,
            position_ids=position_ids,
            indexer_key_cache=indexer_key_cache,
            blocked_indexer=blocked_indexer,
            num_kv_blocks=num_kv_blocks,
        )

        if not mla_absorption.get("absorption", False):
            raise ValueError("GLM-MoE-DSA compressed decode currently requires MLA absorption.")
        if mla_absorption.get("online", False):
            qup_kupT = torch.matmul(self.per_head_q_up, self.per_head_k_up)
            dq_qup_kupT = torch.matmul(q_resid, qup_kupT)
        else:
            dq_qup_kupT = torch.matmul(q_resid, self.fusedqk)
        query = torch.cat((dq_qup_kupT, q_pe), dim=-1)
        attn_output = _dsa_mla_attention_forward(
            module=self,
            query=query,
            per_head_v_up=self.per_head_v_up,
            scaling=self.scaling,
            layer_idx=self.layer_idx,
            compressed_kvs=compressed_kvs,
            topk_indices=topk_indices,
            valid_topk=valid_topk,
            par_num_split=int(getattr(self, "dsa_par_num_split", 0) or 0),
        )
        attn_output = self.o_proj(attn_output.reshape(batch_size, seq_length, -1).contiguous())
        return attn_output, None, indexer_key_cache


class QEffGlmMoeDsaDecoderLayer(GlmMoeDsaDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        compressed_kvs: Optional[QEffDynamicCompressedKVRopeCache] = None,
        indexer_key_cache: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        mla_absorption: Optional[dict[str, bool]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, indexer_key_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            compressed_kvs=compressed_kvs,
            indexer_key_cache=indexer_key_cache,
            batch_index=batch_index,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            mla_absorption=mla_absorption,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, indexer_key_cache


class QEffGlmMoeDsaTopkRouter(GlmMoeDsaTopkRouter):
    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states, self.weight)
        router_scores = router_logits.sigmoid()
        scores_for_choice = router_scores + self.e_score_correction_bias.unsqueeze(0)
        group_scores = torch.einsum(
            "abc->ab",
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group).topk(2, dim=-1)[0],
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_range = torch.arange(self.n_group, device=hidden_states.device).view(1, 1, self.n_group)
        group_mask = (group_idx.unsqueeze(-1).to(group_range.dtype) == group_range).to(group_scores.dtype)
        group_mask = torch.einsum("tkg->tg", group_mask)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        topk_indices = torch.topk(
            scores_for_choice.masked_fill(~score_mask.bool(), 0.0), k=self.top_k, dim=-1, sorted=False
        )[1]
        topk_weights = router_scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (torch.einsum("ab->a", topk_weights).unsqueeze(-1) + 1e-20)
        return topk_indices, topk_weights * self.routed_scaling_factor


class QEffGlmMoeDsaMoE(GlmMoeDsaMoE):
    def __qeff_init__(self):
        gate_proj, up_proj = self.experts.gate_up_proj.chunk(2, dim=1)
        self.all_gate_proj = torch.nn.Parameter(gate_proj.transpose(1, 2).contiguous())
        self.all_up_proj = torch.nn.Parameter(up_proj.transpose(1, 2).contiguous())
        self.all_down_proj = torch.nn.Parameter(self.experts.down_proj.transpose(1, 2).contiguous())
        self.act_fn = self.experts.act_fn
        self.num_experts = self.experts.num_experts

    def moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        bs, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
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
        experts_out = expert_output.view(bs * seq_len, self.gate.top_k, self.config.hidden_size)
        experts_out = experts_out * topk_weights.unsqueeze(-1)
        return torch.einsum("abc->ac", experts_out).type(hidden_states.dtype)

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_output = self.gate(hidden_states)
        if isinstance(router_output, tuple):
            topk_indices, topk_weights = router_output
        else:
            topk_indices, topk_weights = self.route_tokens_to_experts(router_output)
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        return hidden_states + self.shared_experts(residuals)


QEffGlmMoeDsaRMSNorm = GlmMoeDsaRMSNorm
QEffGlmMoeDsaRotaryEmbedding = GlmMoeDsaRotaryEmbedding


class QEffGlmMoeDsaModel(GlmMoeDsaModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        compressed_kvs: Optional[list[torch.FloatTensor]] = None,
        indexer_key_cache: Optional[list[torch.FloatTensor]] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        mla_absorption: Optional[dict[str, bool]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        cache_compressed = bool(mla_absorption and mla_absorption.get("cache_compressed", False))
        if cache_compressed:
            past_key_values = QEffDynamicCompressedKVRopeCache.from_legacy_cache(compressed_kvs)
        elif use_cache and not isinstance(past_key_values, Cache):
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        if past_key_values is None:
            past_seen_tokens = 0
        elif cache_compressed:
            past_seen_tokens = past_key_values.layers[0].ckv.shape[2] if len(past_key_values.layers) > 0 else 0
        elif indexer_key_cache is not None:
            past_seen_tokens = indexer_key_cache[0].shape[1]
        elif hasattr(past_key_values, "layers") and len(past_key_values.layers) > 0:
            past_seen_tokens = past_key_values.layers[0].keys.shape[2]
        else:
            past_seen_tokens = past_key_values.get_seq_length()
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=past_seen_tokens)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        all_hidden_states = () if output_hidden_states else None
        next_indexer_key_cache = [] if indexer_key_cache is not None else None

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_indexer_cache = indexer_key_cache[layer_idx] if indexer_key_cache is not None else None
            hidden_states, layer_indexer_cache = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None if cache_compressed else past_key_values,
                compressed_kvs=past_key_values if cache_compressed else None,
                indexer_key_cache=layer_indexer_cache,
                batch_index=batch_index,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                mla_absorption=mla_absorption,
                use_cache=use_cache,
                **kwargs,
            )
            if next_indexer_key_cache is not None:
                next_indexer_key_cache.append(layer_indexer_cache)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values.to_legacy_cache() if past_key_values is not None else None,
            hidden_states=all_hidden_states,
            attentions=tuple(next_indexer_key_cache) if next_indexer_key_cache is not None else None,
        )


class QEffGlmMoeDsaForCausalLM(GlmMoeDsaForCausalLM):
    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffGlmMoeDsaDecoderLayer}

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        compressed_kvs: Optional[list[torch.FloatTensor]] = None,
        indexer_key_cache: Optional[list[torch.FloatTensor]] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        mla_absorption = getattr(self, "mla_absorption", None)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            compressed_kvs=compressed_kvs,
            indexer_key_cache=indexer_key_cache,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            mla_absorption=mla_absorption,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        if position_ids is not None:
            logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
            hidden_states = hidden_states[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states).to(hidden_states.dtype)
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_dummy_pkv_cache(self, config, batch_size, seq_len):
        cache_compressed = bool(
            getattr(self, "mla_absorption", None) and self.mla_absorption.get("cache_compressed", False)
        )
        dummy_cache = [[] for _ in range(config.num_hidden_layers)]
        if cache_compressed:
            cache_shape_1 = (batch_size, 1, seq_len, config.kv_lora_rank)
            cache_shape_2 = (batch_size, 1, seq_len, config.qk_rope_head_dim)
        else:
            cache_shape_1 = (batch_size, config.num_attention_heads, seq_len, config.qk_head_dim)
            cache_shape_2 = (batch_size, config.num_attention_heads, seq_len, config.v_head_dim)
        for i in range(config.num_hidden_layers):
            dummy_cache[i].append(torch.zeros(cache_shape_1, dtype=config.torch_dtype))
            dummy_cache[i].append(torch.zeros(cache_shape_2, dtype=config.torch_dtype))
        return dummy_cache


__all__ = [
    "QEffGlmMoeDsaAttention",
    "QEffGlmMoeDsaDecoderLayer",
    "QEffGlmMoeDsaForCausalLM",
    "QEffGlmMoeDsaIndexer",
    "QEffGlmMoeDsaModel",
    "QEffGlmMoeDsaMoE",
    "QEffGlmMoeDsaRMSNorm",
    "QEffGlmMoeDsaRotaryEmbedding",
    "QEffGlmMoeDsaTopkRouter",
]
