# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
    MiniMaxM3SparseForConditionalGeneration,
    MiniMaxM3VLAttention,
    MiniMaxM3VLDecoderLayer,
    MiniMaxM3VLDenseMLP,
    MiniMaxM3VLForCausalLM,
    MiniMaxM3VLIndexer,
    MiniMaxM3VLRotaryEmbedding,
    MiniMaxM3VLSparseMoeBlock,
    MiniMaxM3VLTextModel,
    MiniMaxM3VLTopKRouter,
    dynamic_rope_update,
    maybe_autocast,
    repeat_kv,
)

from QEfficient.transformers.cache_utils import (
    QEffDynamicCache,
    QEffMiniMaxSparseCache,
    read_kv_cache_with_indices,
    scatter_kv_into_cache,
    update_and_read_index_key_cache,
)
from QEfficient.blocking.attention_blocking import (
    AttentionBlockingConfig,
    BlockingMode,
    generic_blocked_attention_interface,
)
from QEfficient.customop.utils import ctx_gather_blocked_kv_dp, ctx_paged_scatter_dp

MASKED_ATTENTION_LOGIT = -3.0e4
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo, get_padding_shape_from_config
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def qeff_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotated_q = q[..., :rotary_dim]
    passthrough_q = q[..., rotary_dim:]
    rotated_k = k[..., :rotary_dim]
    passthrough_k = k[..., rotary_dim:]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotated_q = rotated_q * cos + rotate_half(rotated_q) * sin
    rotated_k = rotated_k * cos + rotate_half(rotated_k) * sin
    return torch.cat((rotated_q, passthrough_q), dim=-1), torch.cat((rotated_k, passthrough_k), dim=-1)

class QEffMiniMaxM3VLRotaryEmbedding(MiniMaxM3VLRotaryEmbedding):
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=x.device, dtype=torch.float32)
        position_ids_expanded = position_ids[..., None].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = position_ids_expanded.float() * inv_freq.view(1, 1, -1)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def qeff_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            attn_weights = torch.where(
                attention_mask,
                torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32, device=attn_weights.device),
                attn_weights,
            )
        else:
            attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class QEffMiniMaxM3VLIndexer(MiniMaxM3VLIndexer):
    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embeddings to an indexer tensor in standard or GP layout."""
        rotary_dim = cos.shape[-1]
        rotated = x[..., :rotary_dim]
        passthrough = x[..., rotary_dim:]
        head_axis = x.ndim - 3
        cos = cos.unsqueeze(head_axis)
        sin = sin.unsqueeze(head_axis)
        rotated = rotated * cos + rotate_half(rotated) * sin
        return torch.cat((rotated, passthrough), dim=-1)

    @staticmethod
    def _read_blocked_k_dp(
        index_key_cache: torch.Tensor,
        position_ids_dp: torch.Tensor,
        start_index: int,
        end_index: int,
        index_block_size: int,
        cp: int,
        hkv: int,
    ) -> torch.Tensor:
        """Gather one compact GP cache range while masking future positions."""
        batch_local, rows, _, _ = index_key_cache.shape
        block_len = end_index - start_index
        pos_max = position_ids_dp.max(dim=-1).values
        dp_lane_per_row = torch.arange(rows, device=index_key_cache.device) // (cp * hkv)
        pos_max_rows = pos_max[:, dp_lane_per_row]

        if cp == 1:
            gather_limit = pos_max_rows
        else:
            way = torch.arange(rows, device=index_key_cache.device).remainder(cp * hkv) // hkv
            cycle_size = index_block_size * cp
            offset_in_way = pos_max_rows.remainder(cycle_size) - way * index_block_size
            offset_in_way = torch.where(
                offset_in_way >= 0,
                offset_in_way.clamp_max(index_block_size - 1),
                torch.full_like(offset_in_way, -1),
            )
            gather_limit = (pos_max_rows // cycle_size) * index_block_size + offset_in_way

        ctx_indices = torch.arange(start_index, end_index, device=index_key_cache.device).view(1, 1, block_len)
        invalid_idx = torch.iinfo(torch.int32).max if torch.onnx.is_in_onnx_export() else 0
        ctx_indices = torch.where(ctx_indices > gather_limit.unsqueeze(-1), invalid_idx, ctx_indices)
        ctx_indices = ctx_indices.to(torch.int32).expand(batch_local, rows, block_len)
        return ctx_gather_blocked_kv_dp(index_key_cache, ctx_indices)

    def _select_blocks(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: "QEffMiniMaxSparseCache",
        layer_idx: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        blocking_config: Optional[AttentionBlockingConfig] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        batch, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        ctx_len = (blocking_config.ctx_len if (blocking_config and blocking_config.ctx_len) else None) or past_key_values.layers[layer_idx].keys.shape[2]
        num_blocks = (ctx_len + cfg.index_block_size - 1) // cfg.index_block_size

        dp = blocking_config.msa_indexer_dp if (blocking_config and blocking_config.msa_indexer_dp) else 1
        cp = blocking_config.msa_indexer_cp if (blocking_config and blocking_config.msa_indexer_cp) else 1
        hkv = blocking_config.indexer_n_head if (blocking_config and blocking_config.indexer_n_head) else 1
        num_kv_blocks = blocking_config.num_kv_blocks if (blocking_config and blocking_config.num_kv_blocks) else 1
        num_cores = (
            blocking_config.num_cores_per_device if (blocking_config and blocking_config.num_cores_per_device) else 1
        )

        if dp > 1:
            return self._select_blocks_dp(
                hidden_states, position_ids, past_key_values, layer_idx, cos, sin, blocking_config
            )

        idx_q = self.q_proj(hidden_states).view(batch, seq_len, cfg.index_n_heads, cfg.index_head_dim).transpose(1, 2)
        idx_k = self.k_proj(hidden_states).view(batch, seq_len, hkv, cfg.index_head_dim).transpose(1, 2)
        idx_q = self.q_norm(idx_q)
        idx_k = self.k_norm(idx_k)
        idx_q, idx_k = qeff_apply_rotary_pos_emb(
            idx_q,
            idx_k,
            cos[..., : cfg.index_head_dim],
            sin[..., : cfg.index_head_dim],
            int(cfg.head_dim * cfg.rope_parameters.get("partial_rotary_factor", 1.0)),
        )
        idx_k = past_key_values.update_index_key_cache(idx_k, layer_idx, position_ids)

        scores = torch.matmul(idx_q.float(), idx_k.float().transpose(-1, -2))
        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=ctx_len)
        scores = scores.masked_fill(causal_mask, -1.0e30)

        padded_len = num_blocks * cfg.index_block_size
        if padded_len != ctx_len:
            scores = F.pad(scores, (0, padded_len - ctx_len), value=-1.0e30)
        block_scores = scores.view(batch, cfg.index_n_heads, seq_len, num_blocks, cfg.index_block_size).amax(dim=-1)

        block_ids = torch.arange(num_blocks, device=hidden_states.device).view(1, 1, 1, -1)
        q_block = position_ids // cfg.index_block_size
        for local_offset in range(cfg.index_local_blocks):
            local_block = (q_block - local_offset).clamp(min=0)
            block_scores = torch.where(
                block_ids == local_block[:, None, :, None],
                torch.full_like(block_scores, 1.0e30),
                block_scores,
            )

        topk_scores, block_indices = torch.topk(block_scores, k=min(cfg.index_topk_blocks, num_blocks), dim=-1)
        block_valid = topk_scores > -1.0e29
        offsets = torch.arange(cfg.index_block_size, device=hidden_states.device).view(1, 1, 1, 1, -1)
        token_indices = block_indices.unsqueeze(-1) * cfg.index_block_size + offsets
        token_valid = block_valid.unsqueeze(-1) & (token_indices < ctx_len)
        token_valid = token_valid & (token_indices <= position_ids[:, None, :, None, None])
        token_indices = token_indices[:, :, 0]
        token_valid = token_valid[:, :, 0]
        safe_indices = torch.where(token_valid, token_indices, torch.zeros_like(token_indices)).to(torch.int32)
        return safe_indices, token_valid

    @staticmethod
    def _apply_rope_dp(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int
    ) -> torch.Tensor:
        """Apply RoPE to x in DP layout.

        x:   [..., H, seq_len, D]  where the leading dims include a DP axis
        cos: [..., seq_len, rotary_dim]  (no head axis yet)
        sin: [..., seq_len, rotary_dim]
        """
        rotated = x[..., :rotary_dim]
        passthrough = x[..., rotary_dim:]
        head_axis = x.ndim - 3  # insert singleton for H axis
        cos = cos.unsqueeze(head_axis)
        sin = sin.unsqueeze(head_axis)
        rotated = rotated * cos + rotate_half(rotated) * sin
        return torch.cat((rotated, passthrough), dim=-1)

    @staticmethod
    def _to_dp_cache_shape(
        cache: torch.Tensor,
        dp: int,
        rows: int,
        cp: int,
    ) -> torch.Tensor:
        """Reshape index_key_cache from standard (batch, 1, ctx_len, dim) to DP layout (batch//dp, rows, ctx_len//cp, dim).

        Uses cache.shape for batch and ctx_len so the reshape is dynamic-friendly for ONNX export.
        """
        B, _, T, D = cache.shape
        return cache.reshape(B // dp, rows, T // cp, D)

    @staticmethod
    def _from_dp_cache_shape(
        cache: torch.Tensor,
        dp: int,
        cp: int,
    ) -> torch.Tensor:
        """Revert index_key_cache from DP layout (batch//dp, rows, ctx_len//cp, dim) to standard (batch, 1, ctx_len, dim).

        Uses cache.shape for batch_local and cache_slots so the reshape is dynamic-friendly for ONNX export.
        """
        B, _, T, D = cache.shape
        return cache.reshape(B * dp, 1, T * cp, D)

    def _select_blocks_dp(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: "QEffMiniMaxSparseCache",
        layer_idx: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        blocking_config: Optional[AttentionBlockingConfig] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """DP-optimised block selection using the GP cache custom operations.

        Mirrors the benchmark's ``_select_blocks`` DP path while preserving
        QEfficient's standard cache serialization.
        """
        cfg = self.config
        batch, query_len = hidden_states.shape[0], hidden_states.shape[1]
        ctx_len = blocking_config.ctx_len

        dp = blocking_config.msa_indexer_dp
        cp = blocking_config.msa_indexer_cp if (blocking_config and blocking_config.msa_indexer_cp) else 1
        hkv = blocking_config.indexer_n_head if (blocking_config and blocking_config.indexer_n_head) else 1
        num_kv_blocks = blocking_config.num_kv_blocks if (blocking_config and blocking_config.num_kv_blocks) else 1
        num_cores = (
            blocking_config.num_cores_per_device if (blocking_config and blocking_config.num_cores_per_device) else 1
        )
        num_index_heads = cfg.index_n_heads

        num_blocks = (ctx_len + cfg.index_block_size - 1) // cfg.index_block_size
        selected_blocks = min(cfg.index_topk_blocks, num_blocks)

        if query_len != 1:
            raise ValueError("MSA indexer GP selection is decode-only; QL must be 1.")
        if dp < 1 or batch % dp:
            raise ValueError(
                "MSA indexer batch size must be divisible by msa_indexer_dp."
            )
        if cp < 1 or ctx_len % cp:
            raise ValueError(
                "MSA indexer ctx_len must be divisible by msa_indexer_cp."
            )
        if hkv < 1 or num_index_heads % hkv:
            raise ValueError(
                "index_n_heads must be divisible by indexer_n_head."
            )
        batch_local = batch // dp
        rows = dp * cp * hkv
        cache_slots = ctx_len // cp
        dim = cfg.index_head_dim
        index_key_cache = past_key_values.index_keys.get(layer_idx)
        if index_key_cache is None:
            index_key_cache = torch.zeros((batch, 1, ctx_len, dim), dtype=hidden_states.dtype, device=hidden_states.device)
        expected_cache_shape = (batch_local, rows, cache_slots, dim)
        index_key_cache = self._to_dp_cache_shape(index_key_cache, dp, rows, cp)

        # if tuple(index_key_cache.shape) != expected_cache_shape:
        #     raise ValueError(
        #         f"MSA index_key_cache shape {tuple(index_key_cache.shape)} != "
        #         f"{expected_cache_shape}."
        #     )

        if cache_slots % num_kv_blocks:
            raise ValueError(
                f"compact indexer cache slots ({cache_slots}) must be divisible "
                f"by num_kv_blocks ({num_kv_blocks})."
            )
        cache_block_size = cache_slots // num_kv_blocks
        if cache_block_size % num_cores:
            raise ValueError(
                f"compact indexer KV block length ({cache_block_size}) must be "
                f"divisible by num_cores_per_device ({num_cores})."
            )
        tokens_per_core = cache_block_size // num_cores
        if cfg.index_block_size % cp:
            raise ValueError(
                f"index_block_size ({cfg.index_block_size}) must be divisible "
                f"by msa_indexer_cp ({cp})."
            )
        if tokens_per_core % cfg.index_block_size:
            raise ValueError(
                f"compact indexer tokens per core ({tokens_per_core}) must be "
                f"divisible by index_block_size ({cfg.index_block_size})."
            )
        blocks_per_core = tokens_per_core // cfg.index_block_size

        # The external batch is DP-major: [dp0 local batches, dp1 local
        # batches, ...].  Convert once to the internal [B_local, DP, ...]
        # layout; keep the row axis [B_local, DP*cp*hkv] unchanged.
        position_ids_dp = position_ids.view(dp, batch_local, query_len).permute(1, 0, 2)
        cos_dp = cos.view(dp, batch_local, query_len, cos.shape[-1]).permute(1, 0, 2, 3)
        sin_dp = sin.view(dp, batch_local, query_len, sin.shape[-1]).permute(1, 0, 2, 3)
        idx_q = self.q_proj(hidden_states).view(
            dp, batch_local, query_len, num_index_heads, dim
        ).permute(1, 0, 3, 2, 4)
        idx_k = self.k_proj(hidden_states).view(
            dp, batch_local, query_len, hkv, dim
        ).permute(1, 0, 3, 2, 4)
        idx_q = self.q_norm(idx_q)
        idx_k = self.k_norm(idx_k)
        idx_q = self._apply_rope_dp(
            idx_q,
            cos_dp[..., : cfg.index_head_dim],
            sin_dp[..., : cfg.index_head_dim],
            int(cfg.head_dim * cfg.rope_parameters.get("partial_rotary_factor", 1.0)),
        )
        idx_k = self._apply_rope_dp(
            idx_k,
            cos_dp[..., : cfg.index_head_dim],
            sin_dp[..., : cfg.index_head_dim],
            int(cfg.head_dim * cfg.rope_parameters.get("partial_rotary_factor", 1.0)),
        )

        k_updates = idx_k.unsqueeze(2).expand(batch_local, dp, cp, hkv, query_len, dim).reshape(batch_local, rows, query_len, dim)
        logical_index_block = position_ids_dp // cfg.index_block_size
        live_way = (logical_index_block % cp)[:, :, None, None, :].expand(batch_local, dp, cp, hkv, query_len).reshape(batch_local, rows, query_len)
        row_idx = torch.arange(rows, device=hidden_states.device).view(1, rows, 1)
        row_way = row_idx.remainder(cp * hkv) // hkv
        row_live = row_way == live_way
        batch_id = torch.arange(batch_local, device=hidden_states.device).view(batch_local, 1, 1).expand(batch_local, rows, query_len)
        block_id = torch.where(row_live, batch_id, torch.iinfo(torch.int32).max).to(torch.int32)
        addr = (((logical_index_block // cp) * cfg.index_block_size + position_ids_dp % cfg.index_block_size).to(torch.int32)[:, :, None, None, :].expand(batch_local, dp, cp, hkv, query_len).reshape(batch_local, rows, query_len))
        index_key_cache = ctx_paged_scatter_dp(
            index_key_cache, block_id, addr, k_updates
        )

        q_heads_per_kv = num_index_heads // hkv
        ql_eff = q_heads_per_kv * query_len
        q_rows = idx_q.reshape(batch_local, dp, hkv, q_heads_per_kv, query_len, dim).unsqueeze(2).expand(batch_local, dp, cp, hkv, q_heads_per_kv, query_len, dim).reshape(batch_local, rows, ql_eff, dim)

        way_rows = (
            torch.arange(rows, device=hidden_states.device)
            .remainder(cp * hkv)
            // hkv
        )
        cache_addr = torch.arange(num_cores * tokens_per_core, device=hidden_states.device).view(1, 1, num_cores, tokens_per_core)
        t_pos = (cache_addr // cfg.index_block_size) * (cfg.index_block_size * cp) + way_rows.view(1, rows, 1, 1) * cfg.index_block_size + cache_addr.remainder(cfg.index_block_size)
        q_pos_rows_all = position_ids_dp[:, :, 0].view(batch_local, dp, 1, 1).expand(batch_local, dp, cp, hkv).reshape(batch_local, rows)

        block_ranges: list[tuple[int, int]] = []
        block_starts: list[int] = []
        for block_idx in range(num_kv_blocks):
            start = block_idx * cache_block_size
            end = min(start + cache_block_size, cache_slots)
            if start < end:
                block_ranges.append((start, end))
                block_starts.append(start)

        q_pos_shift_all = q_pos_rows_all[:, None, :, None, None] - torch.tensor(
            block_starts,
            device=hidden_states.device,
            dtype=q_pos_rows_all.dtype,
        ).view(1, len(block_starts), 1, 1, 1) * cp
        causal_masks: list[torch.Tensor] = []
        for block_idx in range(len(block_ranges)):
            q_pos_shift = q_pos_shift_all[:, block_idx]
            causal_masks.append(t_pos > q_pos_shift)

        candidate_score_groups: list[torch.Tensor] = []
        candidate_block_groups: list[torch.Tensor] = []
        local_batch_size = batch_local
        for batch_start in range(0, batch_local, local_batch_size):
            batch_end = min(batch_start + local_batch_size, batch_local)
            local = batch_end - batch_start
            q_local = q_rows[batch_start:batch_end]
            key_local = index_key_cache[batch_start:batch_end]
            pos_local = position_ids_dp[batch_start:batch_end]
            q_5d = q_local.unsqueeze(2).expand(local, rows, num_cores, ql_eff, dim)
            # q_5d: [local, rows, num_cores, ql_eff, dim]
            device_block_score_groups: list[torch.Tensor] = []
            device_block_id_groups: list[torch.Tensor] = []
            for block_idx, (start, end) in enumerate(block_ranges):
                # end - start == num_cores * tokens_per_core
                key_flat = self._read_blocked_k_dp(
                    key_local,
                    pos_local,
                    start,
                    end,
                    cfg.index_block_size,
                    cp,
                    hkv,
                )
                # key_flat: [local, rows, end - start, dim]
                key_5d = key_flat.view(
                    local, rows, num_cores, tokens_per_core, dim
                )
                # key_5d: [local, rows, num_cores, tokens_per_core, dim]
                score_block = torch.matmul(
                    q_5d.float(), key_5d.transpose(-1, -2).float()
                )
                # score_block:
                # [local, rows, num_cores, ql_eff, tokens_per_core]
                causal = causal_masks[block_idx][batch_start:batch_end]
                # causal: [local, rows, num_cores, tokens_per_core]
                score_block = score_block.masked_fill(
                    causal.unsqueeze(3), MASKED_ATTENTION_LOGIT
                )

                # Every row contains complete index blocks; ql_eff stays as
                # the query-head/query-length axis.
                score_block = score_block.view(
                    local,
                    rows,
                    num_cores,
                    ql_eff,
                    blocks_per_core,
                    cfg.index_block_size,
                )
                # score_block:
                # [local, DP*cp*hkv, C, ql_eff,
                #  blocks_per_core, block_size]
                block_scores_core = score_block.amax(dim=-1)
                # block_scores_core:
                # [local, DP*cp*hkv, C, ql_eff, blocks_per_core]

                block_start = start * cp // cfg.index_block_size
                block_ids_core = (
                    block_start
                    + torch.arange(
                        cp, device=hidden_states.device
                    ).view(cp, 1, 1)
                    + torch.arange(
                        num_cores, device=hidden_states.device
                    ).view(1, num_cores, 1)
                    * (cp * blocks_per_core)
                    + torch.arange(
                        blocks_per_core, device=hidden_states.device
                    ).view(1, 1, blocks_per_core)
                    * cp
                )
                # block_ids_core: [cp, C, blocks_per_core]
                block_ids_rows = block_ids_core.view(
                    1, 1, cp, 1, num_cores, blocks_per_core
                ).expand(
                    local,
                    dp,
                    cp,
                    hkv,
                    num_cores,
                    blocks_per_core,
                ).reshape(local, rows, num_cores, blocks_per_core)
                q_block_rows = (
                    position_ids_dp[batch_start:batch_end, :, 0]
                    // cfg.index_block_size
                ).view(local, dp, 1, 1).expand(
                    local, dp, cp, hkv
                ).reshape(local, rows)
                for local_offset in range(cfg.index_local_blocks):
                    local_block = (q_block_rows - local_offset).clamp(min=0)
                    local_mask = block_ids_rows == local_block.view(
                        local, rows, 1, 1
                    )
                    block_scores_core = torch.where(
                        local_mask.unsqueeze(3),
                        torch.full_like(
                            block_scores_core, -MASKED_ATTENTION_LOGIT
                        ),
                        block_scores_core,
                    )

                device_block_score_groups.append(block_scores_core)
                device_block_id_groups.append(block_ids_rows)
            if not device_block_score_groups:
                raise ValueError("MSA indexer selection produced no cache blocks.")

            device_block_scores = torch.cat(
                device_block_score_groups, dim=2
            ).permute(0, 1, 3, 2, 4).reshape(
                local, rows, ql_eff,
                num_kv_blocks * num_cores * blocks_per_core,
            )
            device_block_ids = torch.cat(
                device_block_id_groups, dim=2
            ).unsqueeze(2).expand(
                local, rows, ql_eff,
                num_kv_blocks * num_cores, blocks_per_core,
            ).reshape(
                local, rows, ql_eff,
                num_kv_blocks * num_cores * blocks_per_core,
            )
            # [local, DP*cp*indexer_n_head, ql_eff,
            #  num_kv_blocks*C*blocks_per_core]
            device_topk = min(selected_blocks, device_block_scores.shape[-1])
            device_topk_scores, device_topk_indices = torch.topk(
                device_block_scores, k=device_topk, dim=-1
            )
            device_topk_block_ids = torch.gather(
                device_block_ids, -1, device_topk_indices
            )
            candidate_score_groups.append(device_topk_scores)
            candidate_block_groups.append(device_topk_block_ids)

        candidate_scores = torch.cat(candidate_score_groups, dim=0).view(
            batch_local, dp, cp, hkv,
            q_heads_per_kv, query_len, device_topk
        ).permute(0, 1, 3, 4, 5, 2, 6).reshape(
            batch_local, dp, hkv, q_heads_per_kv, query_len,
            cp * device_topk
        )
        candidate_block_indices = torch.cat(
            candidate_block_groups, dim=0
        ).view(
            batch_local, dp, cp, hkv,
            q_heads_per_kv, query_len, device_topk
        ).permute(0, 1, 3, 4, 5, 2, 6).reshape(
            batch_local, dp, hkv, q_heads_per_kv, query_len,
            cp * device_topk
        )
        # candidate_*:
        # [batch_local, dp, indexer_n_head, q_heads_per_kv, QL,
        #  cp * device_topk]
        # Second Top-K: merge cp candidates on each DP/Hkv lane.
        topk_scores, candidate_topk = torch.topk(
            candidate_scores, k=selected_blocks, dim=-1
        )
        block_indices = torch.gather(
            candidate_block_indices, -1, candidate_topk
        )
        # topk_scores/block_indices: [batch_local, dp, num_index_heads, top_k]
        # Internally the candidates are [B_local, DP, H, ...].  The external
        # batch contract is DP-major, so move DP in front of B_local before
        # flattening; a direct reshape would produce local-batch-major order.
        topk_scores = topk_scores.permute(1, 0, 2, 3, 4, 5).reshape(
            batch, num_index_heads, query_len, selected_blocks
        )
        block_indices = block_indices.permute(1, 0, 2, 3, 4, 5).reshape(
            batch, num_index_heads, query_len, selected_blocks
        )
        block_valid = topk_scores > (MASKED_ATTENTION_LOGIT / 2)
        offsets = torch.arange(cfg.index_block_size, device=hidden_states.device).view(1, 1, 1, 1, -1)
        token_indices = block_indices.unsqueeze(-1) * cfg.index_block_size + offsets
        token_valid = block_valid.unsqueeze(-1) & (token_indices < ctx_len)
        token_valid = token_valid & (token_indices <= position_ids[:, None, :, None, None])
        token_indices = token_indices[:, :, 0]
        token_valid = token_valid[:, :, 0]
        safe_indices = torch.where(token_valid, token_indices, torch.zeros_like(token_indices)).to(torch.int32)
        if not torch.onnx.is_in_onnx_export():
            self.last_block_indices = block_indices.detach()
        # safe_indices:    [B, index_n_heads, selected_blocks, index_block_size]
        # token_valid:     [B, index_n_heads, selected_blocks, index_block_size]
        # index_key_cache: [B/DP, DP*cp*indexer_n_head, T/cp, D]
        index_key_cache = self._from_dp_cache_shape(index_key_cache, dp, cp)
        past_key_values.index_keys[layer_idx] = index_key_cache
        return safe_indices, token_valid


class QEffMiniMaxM3VLAttention(MiniMaxM3VLAttention):
    def _baseline_attention(
        self,
        query_states: torch.Tensor,
        selected_k: torch.Tensor,
        selected_v: torch.Tensor,
        flat_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len = query_states.shape[0], query_states.shape[2]
        num_heads = self.config.index_n_heads
        num_kv_groups = self.num_key_value_groups
        q = query_states.view(batch, num_heads, num_kv_groups, seq_len, self.head_dim)
        selected_k = selected_k.unsqueeze(2)
        selected_v = selected_v.unsqueeze(2)
        scores = torch.matmul(q.float(), selected_k.transpose(-1, -2).float()) * (self.head_dim**-0.5)
        scores = scores.masked_fill(~flat_valid[:, :, None, None, :], -1.0e30)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(selected_v.dtype)
        return torch.matmul(probs, selected_v)

    def _baseline_attention_gp(
        self,
        query_states: torch.Tensor,
        token_indices: torch.Tensor,
        token_valid: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        dp: int,
        blocking_config: Optional[AttentionBlockingConfig],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run selected-token attention in group-parallel layout.

        ``key_cache`` and ``value_cache`` have already been updated by the
        sparse cache writer and use ``[B_local, DP * Hkv, ctx_len, D]``. The
        selected indices may be either that layout or the external
        ``[batch, Hkv, blocks, block_size]`` layout returned by the indexer.
        """
        batch, _, query_len, head_dim = query_states.shape
        hkv = self.config.num_key_value_heads
        num_kv_groups = self.num_key_value_groups
        rows = dp * hkv

        if query_len != 1:
            raise ValueError("MSA attention GP is decode-only; QL must be 1.")
        if dp < 1 or batch % dp:
            raise ValueError(f"MSA batch size {batch} must be divisible by msa_attn_dp={dp}.")

        batch_local = batch // dp
        if blocking_config is not None and blocking_config.msa_attn_cp not in (None, 1):
            raise ValueError("MSA attention GP currently supports msa_attn_cp=1 only.")
        if tuple(key_cache.shape[:2]) != (batch_local, rows):
            raise ValueError(
                f"GP key cache shape {tuple(key_cache.shape)} must start with "
                f"({batch_local}, {rows})."
            )
        if tuple(value_cache.shape) != tuple(key_cache.shape):
            raise ValueError("GP key and value cache shapes must match.")

        selected_len: int
        if token_indices.ndim == 4:
            # External DP-major layout: [batch, Hkv, blocks, block_size].
            selected_len = token_indices.shape[-2] * token_indices.shape[-1]
            gather_idx = (
                token_indices.reshape(dp, batch_local, hkv, selected_len)
                .permute(1, 0, 2, 3)
                .reshape(batch_local, rows, selected_len)
            )
            valid_idx = (
                token_valid.reshape(dp, batch_local, hkv, selected_len)
                .permute(1, 0, 2, 3)
                .reshape(batch_local, rows, selected_len)
            )
        elif token_indices.ndim == 3:
            selected_len = token_indices.shape[-1]
            if tuple(token_indices.shape[:2]) == (batch_local, rows):
                gather_idx = token_indices
                valid_idx = token_valid
            elif tuple(token_indices.shape[:2]) == (batch, hkv):
                gather_idx = (
                    token_indices.reshape(dp, batch_local, hkv, selected_len)
                    .permute(1, 0, 2, 3)
                    .reshape(batch_local, rows, selected_len)
                )
                valid_idx = (
                    token_valid.reshape(dp, batch_local, hkv, selected_len)
                    .permute(1, 0, 2, 3)
                    .reshape(batch_local, rows, selected_len)
                )
            else:
                raise ValueError(
                    "MSA token indices must use [B_local, DP*Hkv, selected_len] "
                    "or [batch, Hkv, selected_len]."
                )
        else:
            raise ValueError("MSA token indices must be rank 3 or rank 4.")

        if tuple(valid_idx.shape) != tuple(gather_idx.shape):
            raise ValueError("MSA token_indices and token_valid shapes must match.")

        selected_k = ctx_gather_blocked_kv_dp(key_cache, gather_idx.to(torch.int32))
        selected_v = ctx_gather_blocked_kv_dp(value_cache, gather_idx.to(torch.int32))

        num_cores = (
            blocking_config.num_cores_per_device
            if blocking_config is not None and blocking_config.num_cores_per_device
            else 1
        )
        if num_kv_groups % num_cores:
            raise ValueError(
                f"MSA Q-head groups ({num_kv_groups}) must be divisible by "
                f"num_cores_per_device ({num_cores})."
            )
        groups_per_core = num_kv_groups // num_cores

        q = (
            query_states.reshape(dp, batch_local, hkv, num_kv_groups, query_len, head_dim)
            .permute(1, 0, 2, 3, 4, 5)
            .reshape(batch_local, rows, num_kv_groups, query_len, head_dim)
        )
        q = q.reshape(batch_local, rows, num_cores, groups_per_core * query_len, head_dim)
        selected_k = selected_k.unsqueeze(2).expand(-1, -1, num_cores, -1, -1)
        selected_v = selected_v.unsqueeze(2).expand(-1, -1, num_cores, -1, -1)
        valid = valid_idx.unsqueeze(2).expand(-1, -1, num_cores, -1)

        scores = torch.matmul(q.float(), selected_k.float().transpose(-1, -2)) * (head_dim**-0.5)
        scores = scores.masked_fill(~valid.unsqueeze(3), -1.0e30)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(selected_v.dtype)
        out = torch.matmul(probs, selected_v)
        out = (
            out.reshape(batch_local, rows, num_kv_groups, query_len, head_dim)
            .reshape(batch_local, dp, hkv, num_kv_groups, query_len, head_dim)
            .permute(1, 0, 2, 3, 4, 5)
            .reshape(batch, hkv * num_kv_groups, query_len, head_dim)
        )
        return out, key_cache, value_cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        query_shape = (*input_shape, self.config.num_attention_heads, self.head_dim)
        key_value_shape = (*input_shape, self.config.num_key_value_heads, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(query_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(key_value_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(key_value_shape).transpose(1, 2)

        cos, sin = position_embeddings
        # to do - don't use constant, should get from config
        query_states, key_states = qeff_apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            int(self.head_dim * self.config.rope_parameters.get("partial_rotary_factor", 1.0)),
        )

        cache_kwargs = {
            "position_ids": position_ids,
            "batch_index": kwargs.get("batch_index"),
        }
        if attention_mask is not None:
            cache_kwargs["CCL"] = attention_mask.shape[-1]

        if self.indexer is not None and isinstance(past_key_values, QEffMiniMaxSparseCache):
            blocking_config = getattr(self, "attn_blocking_config", None)
            if blocking_config:
                cache_kwargs["dp"] = getattr(blocking_config, "msa_attn_dp", 1)
                cache_kwargs["hkv"] =  self.config.num_key_value_heads

            token_indices, token_valid = self.indexer._select_blocks(
                hidden_states, position_ids, past_key_values, self.layer_idx, cos, sin,
                blocking_config=blocking_config,
            )

            past_key_values.write_only_sparse(key_states, value_states, self.layer_idx, cache_kwargs)
            attn_dp = blocking_config.msa_attn_dp if (blocking_config and blocking_config.msa_attn_dp) else 1
            if attn_dp > 1:
                dp = cache_kwargs["dp"]
                batch = input_shape[0]
                batch_local = batch // dp
                hkv = self.config.num_key_value_heads
                rows = dp * hkv
                key_cache = past_key_values.layers[self.layer_idx].keys.reshape(batch_local, rows, -1, self.head_dim)
                value_cache = past_key_values.layers[self.layer_idx].values.reshape(batch_local, rows, -1, self.head_dim)
                attn_output, key_cache, value_cache = self._baseline_attention_gp(
                    query_states, token_indices, token_valid, key_cache, value_cache, dp=attn_dp, blocking_config=blocking_config,
                )
                past_key_values.layers[self.layer_idx].keys = key_cache.reshape(batch, hkv, -1, self.head_dim)
                past_key_values.layers[self.layer_idx].values = value_cache.reshape(batch, hkv, -1, self.head_dim)
            else:
                selected_k, selected_v, flat_valid = past_key_values.read_kv_with_block_indices(
                    self.layer_idx, token_indices, token_valid
                )
                attn_output = self._baseline_attention(query_states, selected_k, selected_v, flat_valid)
            attn_output = attn_output.reshape(*input_shape, self.config.num_attention_heads * self.head_dim)
        else:
            blocking_config = getattr(self, "attn_blocking_config", AttentionBlockingConfig())
            use_blocking = blocking_config is not None and blocking_config.mode != BlockingMode.NONE
            if use_blocking:
                past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
                attn_output, _ = generic_blocked_attention_interface(
                    module=self,
                    query=query_states,
                    key=key_states,
                    value=value_states,
                    attention_mask=attention_mask,
                    scaling=self.scaling,
                    layer_idx=self.layer_idx,
                    past_key_value=past_key_values,
                    blocking_config=blocking_config,
                    comp_ctx_lengths=kwargs.get("comp_ctx_lengths"),
                    batch_index=kwargs.get("batch_index"),
                    position_ids=position_ids,
                    past_seen_tokens=past_seen_tokens,
                    prefill_only=blocking_config.mode.is_prefill,
                )
            else:
                if past_key_values is not None:
                    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
                attn_output, _ = qeff_eager_attention_forward(
                    self, query_states, key_states, value_states, attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                )
            attn_output = attn_output.reshape(*input_shape, self.config.num_attention_heads * self.head_dim)

        return self.o_proj(attn_output.contiguous()), None


def _qeff_minimax_clamp(hidden_states: torch.Tensor, min_value=None, max_value=None) -> torch.Tensor:
    if min_value is not None:
        min_tensor = torch.tensor(min_value, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = torch.maximum(hidden_states, min_tensor)
    if max_value is not None:
        max_tensor = torch.tensor(max_value, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = torch.minimum(hidden_states, max_tensor)
    return hidden_states


class QEffMiniMaxM3VLDenseMLP(MiniMaxM3VLDenseMLP):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        gate = _qeff_minimax_clamp(gate, max_value=self.swiglu_limit)
        up = _qeff_minimax_clamp(up, min_value=-self.swiglu_limit, max_value=self.swiglu_limit)
        glu = gate * torch.sigmoid(gate * self.swiglu_alpha)
        return self.down_proj((up + 1.0) * glu)


class QEffMiniMaxM3VLTopKRouter(MiniMaxM3VLTopKRouter):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = nn.functional.linear(hidden_states.to(self.weight.dtype), self.weight)
        routing_weights = nn.functional.sigmoid(router_logits.float())
        scores_for_choice = routing_weights + self.e_score_correction_bias
        _, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_index)
        denom = torch.einsum("tk->t", top_k_weights)
        top_k_weights = top_k_weights / denom[:, None]
        return router_logits, top_k_weights, top_k_index


class QEffMiniMaxM3VLSparseMoeBlock(MiniMaxM3VLSparseMoeBlock):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        tokens = batch_size * sequence_length
        hidden_states = hidden_states.view(tokens, hidden_dim)

        shared_output = self.shared_experts(hidden_states)
        _, top_k_weights, top_k_index = self.gate(hidden_states)
        top_k = self.gate.top_k

        expert_indices = top_k_index.flatten()
        gate_up_proj = self.experts.gate_up_proj.transpose(1, 2).index_select(0, expert_indices)
        down_proj = self.experts.down_proj.transpose(1, 2).index_select(0, expert_indices)

        expert_in = hidden_states.unsqueeze(1).expand(-1, top_k, -1).contiguous().view(-1, 1, hidden_dim)
        gate_up = torch.bmm(expert_in, gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        gate = _qeff_minimax_clamp(gate, max_value=self.experts.swiglu_limit)
        up = _qeff_minimax_clamp(up, min_value=-self.experts.swiglu_limit, max_value=self.experts.swiglu_limit)
        intermediate = (up + 1.0) * (gate * torch.sigmoid(gate * self.experts.swiglu_alpha))
        experts_out = torch.bmm(intermediate, down_proj)
        experts_out = experts_out.view(tokens, top_k, hidden_dim)
        experts_out = experts_out * top_k_weights.unsqueeze(-1).to(experts_out.dtype)
        experts_out = torch.einsum("tkh->th", experts_out)

        hidden_states = experts_out * self.routed_scaling_factor
        hidden_states = hidden_states + shared_output
        return hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class QEffMiniMaxM3VLDecoderLayer(MiniMaxM3VLDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class QEffMiniMaxM3VLTextModel(MiniMaxM3VLTextModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        index_keys = None,
        **kwargs,
    ) -> MoeModelOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        use_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            use_legacy_cache = True
            past_key_values = QEffMiniMaxSparseCache.from_legacy_cache(past_key_values, index_keys)
        elif use_cache and past_key_values is None:
            past_key_values = QEffMiniMaxSparseCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        target_length = past_seen_tokens + inputs_embeds.shape[1]
        if isinstance(attention_mask, torch.Tensor):
            target_length = attention_mask.shape[-1]
        elif past_key_values is not None and getattr(past_key_values, "layers", None):
            first_layer = past_key_values.layers[0]
            cached_keys = getattr(first_layer, "keys", None)
            if cached_keys is not None:
                target_length = cached_keys.shape[-2]
        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=target_length)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)

        if use_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class QEffMiniMaxM3VLForCausalLM(MiniMaxM3VLForCausalLM):
    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffMiniMaxM3VLDecoderLayer}

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        index_keys = None,
        **kwargs,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            index_keys=index_keys,
            **kwargs,
        )

        if position_ids is None:
            hidden_states = outputs.last_hidden_state[:, -1:, :]
            logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()
        else:
            logit_idx = position_ids.to(torch.int32).argmax(1, keepdim=True)
            hidden_states = outputs.last_hidden_state[torch.arange(position_ids.shape[0]).view(-1, 1), logit_idx]
            logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()
        return MoeCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=getattr(outputs, "router_logits", None),
        )


class QEffMiniMaxM3VLEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def get_submodules_for_export(self) -> Type[nn.Module]:
        layers = getattr(getattr(self.model.model, "vision_tower", None), "layers", None)
        if layers:
            return {layers[0].__class__}
        return set()

    def forward(self, pixel_values, image_grid_thw):
        image_outputs = self.model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        image_embeds = image_outputs.pooler_output if hasattr(image_outputs, "pooler_output") else image_outputs
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.cat(image_embeds, dim=0)
        image_embeds = image_embeds.to(pixel_values.device, pixel_values.dtype)
        bs = image_grid_thw.shape[0]
        split_size = torch.floor_divide(torch.tensor(image_embeds.size(0), device=image_embeds.device), bs)
        image_embeds = image_embeds.reshape(bs, split_size, image_embeds.size(-1))
        return image_embeds


class QEffMiniMaxM3VLDecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.language_model = self.model.model.language_model
        self.config = model.config

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffMiniMaxM3VLDecoderLayer}

    def get_onnx_past_key_value_names(self, layer_idx: int, layer_state=None) -> List[str]:
        return [f"past_key.{layer_idx}", f"past_value.{layer_idx}"]

    def get_onnx_index_key_names(self) -> List[str]:
        layer_types = getattr(self.config.text_config, "layer_types", None) or getattr(
            self.language_model.config, "layer_types", None
        )
        if layer_types is None:
            return []
        index_key_names = []
        for i in range(self.config.text_config.num_hidden_layers):
            if layer_types[i] == "minimax_m3_sparse":
                index_key_names.append(f"index_key.{i}" )
        return index_key_names

    def forward(
        self,
        input_ids=None,
        vision_embeds=None,
        position_ids=None,
        image_idx=None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        comp_ctx_lengths: Optional[List[int]] = None,
        index_keys=None,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Exactly one of input_ids or inputs_embeds must be provided.")

        if inputs_embeds is None:
            inputs_embeds = self.model.model.language_model.embed_tokens(input_ids)
            _, _, hidden_dim = inputs_embeds.shape
            selected = input_ids == self.config.image_token_index
            indices1 = selected.to(torch.int64).cumsum(1) - 1
            indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
            image_features_expanded = vision_embeds.reshape(-1, hidden_dim)[indices1]
            image_input_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
            inputs_embeds = torch.where(
                input_ids.shape[1] == torch.tensor(1, device=input_ids.device), inputs_embeds, image_input_embeds
            )
            image_idx_output = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        else:
            if image_idx is None:
                image_idx = torch.zeros((1, 1), dtype=torch.int64, device=inputs_embeds.device)
            image_idx_output = image_idx

        # Build a QEffMiniMaxSparseCache combining the KV cache (2-tuples) and the separate index keys.
        sparse_layer_indices = [int(name.split(".")[-1]) for name in self.get_onnx_index_key_names()]
        index_keys_dict = None
        if index_keys is not None:
            index_keys_dict = {sparse_layer_indices[j]: index_keys[j] for j in range(len(sparse_layer_indices))}
        cache = QEffMiniMaxSparseCache.from_legacy_cache(past_key_values, index_keys=index_keys_dict)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=cache,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=True,
        )
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[
            torch.arange(position_ids.shape[0], device=position_ids.device).view(-1, 1), logit_index
        ]
        logits = self.model.lm_head(hidden_states).float()

        result_cache = outputs.past_key_values
        if isinstance(result_cache, QEffMiniMaxSparseCache):
            past_kv_out = result_cache.to_kv_only_cache()
            index_keys_out = result_cache.get_index_keys_tuple()
        else:
            past_kv_out = tuple((t[0], t[1]) for t in result_cache)
            index_keys_out = tuple(t[2] for t in result_cache if len(t) == 3)

        return logits, vision_embeds, image_idx_output, past_kv_out, index_keys_out


class QEffMiniMaxM3SparseForConditionalGeneration(MiniMaxM3SparseForConditionalGeneration):
    def __qeff_init__(self):
        self.language_model = self.model.language_model
        self.config._attn_implementation = "eager"
        self.model.language_model.config._attn_implementation = "eager"
        self.model.vision_tower.config._attn_implementation = "eager"

    def get_qeff_vision_encoder(self):
        return QEffMiniMaxM3VLEncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEffMiniMaxM3VLDecoderWrapper(self)

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        pixel_values=None,
        image_grid_thw=None,
        image_idx=None,
        past_key_values=None,
        comp_ctx_lengths: Optional[List[int]] = None,
        batch_index: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if input_ids is None or position_ids is None or pixel_values is None or image_grid_thw is None:
            raise ValueError("input_ids, position_ids, pixel_values, and image_grid_thw must be provided.")
        if image_idx is None:
            image_idx = torch.zeros((1, 1), dtype=torch.int64, device=input_ids.device)

        image_features = self.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output
        image_features = image_features.to(device=input_ids.device, dtype=self.lm_head.weight.dtype)

        inputs_embeds = self.model.language_model.embed_tokens(input_ids)
        _, _, hidden_dim = inputs_embeds.shape
        selected = input_ids == self.config.image_token_index
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(selected.shape[0], device=selected.device).view(-1, 1)
        image_features_expanded = image_features.reshape(-1, hidden_dim).unsqueeze(0)[indices0, indices1]
        image_input_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
        inputs_embeds = torch.where(
            input_ids.shape[1] == torch.tensor(1, device=input_ids.device), inputs_embeds, image_input_embeds
        )

        if past_key_values is not None and not isinstance(past_key_values, Cache):
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=True,
        )
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states).float()
        image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)

        present = outputs.past_key_values
        if isinstance(present, Cache):
            if hasattr(present, "to_legacy_cache"):
                present = present.to_legacy_cache()
            elif hasattr(present, "layers"):
                legacy_cache = ()
                for layer in present.layers:
                    legacy_cache += ((getattr(layer, "keys", None), getattr(layer, "values", None)),)
                present = legacy_cache
        return logits, pixel_values, image_idx, present

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        comp_ctx_lengths_prefill: Optional[List[int]] = None,
        comp_ctx_lengths_decode: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
        **compiler_options,
    ):
        prefill_seq_len = prefill_seq_len if prefill_seq_len else constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
        ctx_len = ctx_len if ctx_len else constants.ONNX_EXPORT_CTX_LEN
        # img_size is accepted by generic VLM compile APIs, but MiniMax-M3 VLM
        # specialization derives language/vision shapes from patch settings.
        # Drop it to avoid leaking `-img-size=None` into qaic-compile flags.
        compiler_options.pop("img_size", None)
        num_image_patches = int(compiler_options.pop("num_image_patches", 4))
        num_images = int(compiler_options.pop("num_images", 1))
        vision_size = int(compiler_options.pop("vision_size", num_image_patches))

        qaic_config = getattr(self, "qaic_config", None) or {}
        msa_dp = max(
            int(qaic_config.get("msa_indexer_dp", 1) or 1),
            int(qaic_config.get("msa_attn_dp", 1) or 1),
        )
        export_batch_size = batch_size * msa_dp

        def _build_spec(seq_len, comp_ctx_lengths=None):
            spec = {
                "batch_size": full_batch_size if (continuous_batching and seq_len == 1) else export_batch_size,
                "seq_len": seq_len,
                "ctx_len": ctx_len,
                "num_image_patches": num_image_patches,
                "num_images": num_images,
            }
            if continuous_batching:
                spec["full_batch_size"] = kv_cache_batch_size
            if full_batch_size:
                spec["full_batch_exec_size"] = full_batch_size
            if comp_ctx_lengths is not None:
                spec["comp_ctx_lengths"] = comp_ctx_lengths
            return spec

        def _build_lang_spec(seq_len, comp_ctx_lengths=None):
            spec = {
                "batch_size": full_batch_size if (continuous_batching and seq_len == 1) else export_batch_size,
                "seq_len": seq_len,
                "ctx_len": ctx_len,
                "vision_size": vision_size,
                "vision_batch_size": batch_size,
            }
            if continuous_batching:
                spec["full_batch_size"] = kv_cache_batch_size
            if full_batch_size and seq_len != 1:
                spec["full_batch_exec_size"] = full_batch_size
            if comp_ctx_lengths is not None:
                spec["comp_ctx_lengths"] = comp_ctx_lengths
            return spec

        if comp_ctx_lengths_prefill and comp_ctx_lengths_decode:
            specs = [_build_spec(prefill_seq_len, c) for c in comp_ctx_lengths_prefill]
            specs.extend(_build_spec(1, c) for c in comp_ctx_lengths_decode)
        else:
            specs = [_build_spec(prefill_seq_len), _build_spec(1)]

        if kv_offload:
            vision = [{"batch_size": batch_size, "num_image_patches": num_image_patches, "num_images": num_images}]
            if comp_ctx_lengths_prefill and comp_ctx_lengths_decode:
                lang = [_build_lang_spec(prefill_seq_len, c) for c in comp_ctx_lengths_prefill]
                lang.extend(_build_lang_spec(1, c) for c in comp_ctx_lengths_decode)
            else:
                lang = [_build_lang_spec(prefill_seq_len), _build_lang_spec(1)]
            return {"vision": vision, "lang": lang}, compiler_options

        return specs, compiler_options

    def get_onnx_dynamic_axes(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
    ):
        vision_dynamic_axes = {
            "pixel_values": {0: "num_image_patches"},
            "image_grid_thw": {0: "num_images"},
        }

        lang_dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "vision_embeds": {0: "vision_batch_size", 1: "vision_size"},
        }

        lm_config = self.model.language_model.config
        layer_types = getattr(lm_config, "layer_types", None) or ["full_attention"] * lm_config.num_hidden_layers
        for i in range(lm_config.num_hidden_layers):
            lang_dynamic_axes[f"past_key.{i}"] = {
                0: "full_batch_size" if continuous_batching else "batch_size",
                2: "ctx_len",
            }
            lang_dynamic_axes[f"past_value.{i}"] = {
                0: "full_batch_size" if continuous_batching else "batch_size",
                2: "ctx_len",
            }
            if i < len(layer_types) and layer_types[i] == "minimax_m3_sparse":
                lang_dynamic_axes[f"index_key.{i}"] = {
                    0: "full_batch_size" if continuous_batching else "batch_size",
                    2: "ctx_len",
                }
        if continuous_batching:
            lang_dynamic_axes["batch_index"] = {0: "batch_size"}
        if comp_ctx_lengths is not None:
            lang_dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}

        if kv_offload:
            return {"vision": vision_dynamic_axes, "lang": lang_dynamic_axes}

        dynamic_axes = {**vision_dynamic_axes, **lang_dynamic_axes}
        dynamic_axes.pop("vision_embeds")
        return dynamic_axes

    def get_output_names(self, kv_offload: bool = False):
        lm_config = self.model.language_model.config
        layer_types = getattr(lm_config, "layer_types", None) or ["full_attention"] * lm_config.num_hidden_layers
        vision_output_names = ["vision_embeds"]
        output_names = ["logits", "pixel_values_RetainedState", "image_idx_output"]
        for i in range(lm_config.num_hidden_layers):
            output_names.append(f"past_key.{i}_RetainedState")
            output_names.append(f"past_value.{i}_RetainedState")
        for i in range(lm_config.num_hidden_layers):
            if i < len(layer_types) and layer_types[i] == "minimax_m3_sparse":
                output_names.append(f"index_key.{i}_RetainedState")
        if kv_offload:
            lang_output_names = ["logits", "vision_embeds_RetainedState", "image_idx_output"]
            for i in range(lm_config.num_hidden_layers):
                lang_output_names.append(f"past_key.{i}_RetainedState")
                lang_output_names.append(f"past_value.{i}_RetainedState")
            for i in range(lm_config.num_hidden_layers):
                if i < len(layer_types) and layer_types[i] == "minimax_m3_sparse":
                    lang_output_names.append(f"index_key.{i}_RetainedState")
            return {"vision": vision_output_names, "lang": lang_output_names}
        return output_names

    def get_dummy_pkv_cache(self, config, batch_size, seq_len, dtype=None):
        dtype = dtype or getattr(config, "torch_dtype", torch.float32) or torch.float32
        kv_cache_shape = get_padding_shape_from_config(config=config, batch_size=batch_size, seq_len=seq_len)
        past_key_values = []
        for _ in range(config.num_hidden_layers):
            k = torch.zeros(kv_cache_shape, dtype=dtype)
            v = torch.zeros(kv_cache_shape, dtype=dtype)
            past_key_values.append((k, v))
        return past_key_values

    def get_dummy_index_keys(self, config, batch_size, seq_len, dtype=None):
        dtype = dtype or getattr(config, "torch_dtype", torch.float32) or torch.float32
        layer_types = getattr(config, "layer_types", None) or ["full_attention"] * config.num_hidden_layers
        index_key_shape = (batch_size, 1, seq_len, config.index_head_dim)
        index_keys = []
        for i in range(config.num_hidden_layers):
            if layer_types[i] == "minimax_m3_sparse":
                index_keys.append(torch.zeros(index_key_shape, dtype=dtype))
        return index_keys

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        ctx_len: Optional[int] = None,
        **kwargs,
    ):
        ctx_len = constants.ONNX_EXPORT_CTX_LEN if ctx_len is None else int(ctx_len)
        prefill_seq_len = kwargs.get("prefill_seq_len")
        if prefill_seq_len is None:
            prefill_seq_len = constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
        prefill_seq_len = int(prefill_seq_len)
        qaic_config = getattr(self, "qaic_config", None) or {}
        msa_dp = max(
            int(qaic_config.get("msa_indexer_dp", 1) or 1),
            int(qaic_config.get("msa_attn_dp", 1) or 1),
        )
        requested_batch_size = (
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
            if kwargs.get("batch_size") is None
            else int(kwargs["batch_size"])
        )
        # Compile may already expand the language batch for DP before calling
        # this method.  Keep the adjustment idempotent so direct export and
        # compile-driven export produce the same tensor shapes.
        batch_size = requested_batch_size
        if batch_size % msa_dp:
            batch_size *= msa_dp
        fbs = constants.ONNX_EXPORT_EXAMPLE_FBS
        cache_batch_size = fbs if fbs % msa_dp == 0 else fbs * msa_dp
        dtype = getattr(self.config, "torch_dtype", torch.float32) or torch.float32
        patch_dim = (
            self.config.vision_config.num_channels
            * self.config.vision_config.temporal_patch_size
            * self.config.vision_config.patch_size
            * self.config.vision_config.patch_size
        )
        image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int64)
        num_image_patches = int(torch.prod(image_grid_thw).item())

        inputs = {
            "input_ids": torch.zeros((batch_size, prefill_seq_len), dtype=torch.int64),
            "pixel_values": torch.zeros((num_image_patches, patch_dim), dtype=dtype),
            "image_grid_thw": image_grid_thw,
            "position_ids": torch.arange(prefill_seq_len, dtype=torch.int64)
            .view(1, prefill_seq_len)
            .repeat(batch_size, 1),
            "image_idx": torch.zeros((1, 1), dtype=torch.int64),
        }
        inputs["input_ids"][:, 0] = self.config.image_token_index
        past_key_values = self.get_dummy_pkv_cache(
            config=self.model.language_model.config,
            batch_size=cache_batch_size if continuous_batching else batch_size,
            seq_len=prefill_seq_len,
            dtype=dtype,
        )
        index_keys = self.get_dummy_index_keys(
            config=self.model.language_model.config,
            batch_size=cache_batch_size if continuous_batching else batch_size,
            seq_len=ctx_len,
            dtype=dtype,
        )
        inputs["past_key_values"] = past_key_values
        inputs["index_keys"] = index_keys
        if continuous_batching:
            inputs["batch_index"] = torch.arange(batch_size).view(batch_size, 1)
        if comp_ctx_lengths is not None:
            inputs["comp_ctx_lengths"] = torch.randint(0, 100, (40,), dtype=torch.int64)
        if kv_offload:
            vision_inputs = {
                "pixel_values": inputs["pixel_values"],
                "image_grid_thw": inputs["image_grid_thw"],
            }
            lang_inputs = {
                "input_ids": inputs["input_ids"],
                "vision_embeds": torch.zeros(
                    (batch_size, num_image_patches, self.model.language_model.config.hidden_size), dtype=dtype
                ),
                "position_ids": inputs["position_ids"],
                "image_idx": inputs["image_idx"],
                "past_key_values": past_key_values,
                "index_keys": index_keys,
            }
            if continuous_batching:
                lang_inputs["batch_index"] = inputs["batch_index"]
            if comp_ctx_lengths is not None:
                lang_inputs["comp_ctx_lengths"] = inputs["comp_ctx_lengths"]
            return {"vision": vision_inputs, "lang": lang_inputs}
        return inputs

    def get_inputs_info(self):
        patch_dim = (
            self.config.vision_config.num_channels
            * self.config.vision_config.temporal_patch_size
            * self.config.vision_config.patch_size
            * self.config.vision_config.patch_size
        )
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="pixel_values", datatype=self.config.torch_dtype, shape=("num_image_patches", patch_dim)),
            IOInfo(name="image_grid_thw", datatype=torch.int64, shape=("num_images", 3)),
        ]
