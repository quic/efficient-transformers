# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
from torch import nn
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DisentangledSelfAttention,
)


def make_log_bucket_position_onnx(relative_pos, bucket_size: int, max_position: int):
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.abs(relative_pos)

    # Instead of torch.where with complex conditions, use mask-based approach
    # Original: torch.where((relative_pos < mid) & (relative_pos > -mid), mid-1, abs_pos)
    is_in_mid_range = abs_pos < mid
    abs_pos_clamped = torch.where(is_in_mid_range, torch.tensor(mid - 1).type_as(relative_pos), abs_pos)

    # Compute log position
    log_pos = (
        torch.ceil(torch.log(abs_pos_clamped / mid) / torch.log(torch.tensor((max_position - 1) / mid)) * (mid - 1))
        + mid
    )

    # Select between relative_pos and log_pos based on whether abs_pos <= mid
    bucket_pos = torch.where(abs_pos <= mid, relative_pos.type_as(log_pos), log_pos * sign)
    return bucket_pos


def build_relative_position_onnx(query_layer, key_layer, bucket_size: int = -1, max_position: int = -1):
    """
    Build relative position according to the query and key.
    """
    query_size = query_layer.size(-2)
    key_size = key_layer.size(-2)

    q_ids = torch.arange(query_size, dtype=torch.long, device=query_layer.device)
    k_ids = torch.arange(key_size, dtype=torch.long, device=key_layer.device)
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]

    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position_onnx(rel_pos_ids, bucket_size, max_position)

    rel_pos_ids = rel_pos_ids.to(torch.long)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


def c2p_dynamic_expand_onnx(c2p_pos, query_layer, relative_pos):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)])


def p2c_dynamic_expand_onnx(c2p_pos, query_layer, key_layer):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)])


def pos_dynamic_expand_onnx(pos_index, p2c_att, key_layer):
    return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2)))


def scaled_size_sqrt_onnx(query_layer: torch.Tensor, scale_factor: int):
    return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)


def build_rpos_onnx(query_layer, key_layer, relative_pos, position_buckets: int, max_relative_positions: int):
    """
    ONNX-compatible version of build_rpos.

    Removes @torch.jit.script and conditional logic that depends on tensor sizes.
    Instead, we always compute the relative position to avoid dynamic branching.
    """
    # Original had: if key_layer.size(-2) != query_layer.size(-2):
    # This creates a dynamic condition in ONNX. Instead, we'll always use relative_pos
    # if it's provided, otherwise compute it.
    if relative_pos is None:
        return build_relative_position_onnx(
            key_layer,
            key_layer,
            bucket_size=position_buckets,
            max_position=max_relative_positions,
        )
    else:
        return relative_pos


class QEffDisentangledSelfAttention(DisentangledSelfAttention):
    """
    ONNX-compatible version of DisentangledSelfAttention.

    Overrides methods to use ONNX-compatible helper functions without @torch.jit.script.
    """

    def disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        """
        Override to use ONNX-compatible functions.
        """
        if relative_pos is None:
            relative_pos = build_relative_position_onnx(
                query_layer,
                key_layer,
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        elif relative_pos.dim() != 4:
            raise ValueError(f"Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}")

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.to(device=query_layer.device, dtype=torch.long)

        rel_embeddings = rel_embeddings[0 : att_span * 2, :].unsqueeze(0)
        if self.share_att_key:
            pos_query_layer = self.transpose_for_scores(
                self.query_proj(rel_embeddings), self.num_attention_heads
            ).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
            pos_key_layer = self.transpose_for_scores(self.key_proj(rel_embeddings), self.num_attention_heads).repeat(
                query_layer.size(0) // self.num_attention_heads, 1, 1
            )
        else:
            if "c2p" in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(
                    self.pos_key_proj(rel_embeddings), self.num_attention_heads
                ).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
            if "p2c" in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(
                    self.pos_query_proj(rel_embeddings), self.num_attention_heads
                ).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)

        score = 0
        # content->position
        if "c2p" in self.pos_att_type:
            scale = scaled_size_sqrt_onnx(pos_key_layer, scale_factor)
            c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = torch.gather(
                c2p_att,
                dim=-1,
                index=c2p_pos.squeeze(0).expand([query_layer.size(0), query_layer.size(1), relative_pos.size(-1)]),
            )
            score += c2p_att / scale.to(dtype=c2p_att.dtype)

        # position->content
        if "p2c" in self.pos_att_type:
            scale = scaled_size_sqrt_onnx(pos_query_layer, scale_factor)
            r_pos = build_rpos_onnx(
                query_layer,
                key_layer,
                relative_pos,
                self.position_buckets,
                self.max_relative_positions,
            )
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = torch.gather(
                p2c_att,
                dim=-1,
                index=p2c_pos.squeeze(0).expand([query_layer.size(0), key_layer.size(-2), key_layer.size(-2)]),
            ).transpose(-1, -2)
            score += p2c_att / scale.to(dtype=p2c_att.dtype)

        return score

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        """
        Forward pass using ONNX-compatible attention bias computation.
        """
        if query_states is None:
            query_states = hidden_states
        query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)

        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        scale = scaled_size_sqrt_onnx(query_layer, scale_factor)
        attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_attention_bias(
                query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
            )

        if rel_att is not None:
            attention_scores = attention_scores + rel_att
        attention_scores = attention_scores
        attention_scores = attention_scores.view(
            -1, self.num_attention_heads, attention_scores.size(-2), attention_scores.size(-1)
        )

        attention_mask = attention_mask.bool()
        attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        # bsz x height x length x dimension
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)
        context_layer = torch.bmm(
            attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        )
        context_layer = (
            context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(new_context_layer_shape)
        if not output_attentions:
            return (context_layer, None)
        return (context_layer, attention_probs)
