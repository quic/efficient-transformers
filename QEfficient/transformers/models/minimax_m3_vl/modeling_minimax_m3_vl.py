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
    rotary_dim = cos.shape[-1]
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
    def _select_blocks(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: "QEffMiniMaxSparseCache",
        layer_idx: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        batch, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        ctx_len = past_key_values.layers[layer_idx].keys.shape[2]
        num_blocks = (ctx_len + cfg.index_block_size - 1) // cfg.index_block_size

        idx_q = self.q_proj(hidden_states).view(batch, seq_len, cfg.index_n_heads, cfg.index_head_dim).transpose(1, 2)
        idx_k = self.k_proj(hidden_states).view(batch, seq_len, 1, cfg.index_head_dim).transpose(1, 2)
        idx_q = self.q_norm(idx_q)
        idx_k = self.k_norm(idx_k)
        idx_q, idx_k = qeff_apply_rotary_pos_emb(
            idx_q,
            idx_k,
            cos[..., : cfg.index_head_dim],
            sin[..., : cfg.index_head_dim],
            cfg.index_head_dim // 2,
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
        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos, sin, self.config.index_head_dim // 2)

        cache_kwargs = {
            "position_ids": position_ids,
            "batch_index": kwargs.get("batch_index"),
        }
        if attention_mask is not None:
            cache_kwargs["CCL"] = attention_mask.shape[-1]

        if self.indexer is not None and isinstance(past_key_values, QEffMiniMaxSparseCache):
            past_key_values.write_only(key_states, value_states, self.layer_idx, cache_kwargs)
            token_indices, token_valid = self.indexer._select_blocks(
                hidden_states, position_ids, past_key_values, self.layer_idx, cos, sin
            )
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
        **kwargs,
    ) -> MoeModelOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        use_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            use_legacy_cache = True
            past_key_values = QEffMiniMaxSparseCache.from_legacy_cache(past_key_values)
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
        **kwargs,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
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
            indices0 = torch.arange(selected.shape[0], device=selected.device).view(-1, 1)
            image_features_expanded = vision_embeds.reshape(-1, hidden_dim).unsqueeze(0)[indices0, indices1]
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

        def _build_spec(seq_len, comp_ctx_lengths=None):
            spec = {
                "batch_size": full_batch_size if (continuous_batching and seq_len == 1) else batch_size,
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
                "batch_size": full_batch_size if (continuous_batching and seq_len == 1) else batch_size,
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
        **kwargs,
    ):
        prefill_seq_len = kwargs.get("prefill_seq_len")
        if prefill_seq_len is None:
            prefill_seq_len = constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
        prefill_seq_len = int(prefill_seq_len)

        batch_size = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        fbs = constants.ONNX_EXPORT_EXAMPLE_FBS
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
            batch_size=fbs if continuous_batching else batch_size,
            seq_len=prefill_seq_len,
            dtype=dtype,
        )
        index_keys = self.get_dummy_index_keys(
            config=self.model.language_model.config,
            batch_size=fbs if continuous_batching else batch_size,
            seq_len=prefill_seq_len,
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