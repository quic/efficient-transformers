# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Optional, Type, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForCausalLM,
    Gemma4TextAttention,
    Gemma4TextConfig,
    Gemma4TextDecoderLayer,
    Gemma4TextExperts,
    Gemma4TextModel,
    Gemma4TextRouter,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask


class QEffGemma4TextRouter(Gemma4TextRouter):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size

        router_probabilities = nn.functional.softmax(self.proj(hidden_states), dim=-1)
        top_k_weights, top_k_index = torch.topk(
            router_probabilities,
            k=self.config.top_k_experts,
            dim=-1,
        )

        top_k_weights = top_k_weights / torch.einsum("bk->b", top_k_weights).unsqueeze(-1)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]

        return router_probabilities, top_k_weights, top_k_index


class QEffGemma4TextExperts(Gemma4TextExperts):
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        tokens = hidden_states.shape[0]
        top_k = top_k_index.shape[-1]

        selected_gate_up = self.gate_up_proj[top_k_index.reshape(-1)].transpose(1, 2)
        gate_proj, up_proj = selected_gate_up.chunk(2, dim=-1)
        down_proj = self.down_proj[top_k_index.reshape(-1)].transpose(1, 2)

        expert_inputs = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, 1, self.hidden_dim)
        gate = torch.bmm(expert_inputs, gate_proj)
        up = torch.bmm(expert_inputs, up_proj)
        gated_output = self.act_fn(gate) * up

        experts_out = torch.bmm(gated_output, down_proj).view(tokens, top_k, self.hidden_dim)
        experts_out = experts_out * top_k_weights.unsqueeze(-1)
        return torch.einsum("tkh->th", experts_out)


class QEffGemma4TextAttention(Gemma4TextAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        if self.is_kv_shared_layer and past_key_values is not None:
            key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
            key_states = key_states.transpose(1, 2)

            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

        if past_key_values is not None:
            if not self.is_kv_shared_layer:
                key_states, value_states = past_key_values.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    {"position_ids": position_ids},
                )
            if self.store_full_length_kv:
                if not hasattr(past_key_values, "shared_layers"):
                    past_key_values.shared_layers = {}
                past_key_values.shared_layers[self.layer_idx] = key_states, value_states

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QEffGemma4TextDecoderLayer(Gemma4TextDecoderLayer):
    pass


class QEffGemma4TextModel(Gemma4TextModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(input_ids, inputs_embeds)
            per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)
        elif use_cache and past_key_values is None:
            past_key_values = QEffDynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        hidden_states = inputs_embeds

        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            per_layer_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            layer_attention_mask = attention_mask
            if not isinstance(attention_mask, dict):
                target_length = inputs_embeds.shape[1]
                if past_key_values is not None and len(past_key_values.layers) > i:
                    layer_keys = past_key_values.layers[i].keys
                    if layer_keys is not None and layer_keys.numel() > 0:
                        target_length = layer_keys.shape[-2]

                sliding_window = (
                    self.config.sliding_window if self.config.layer_types[i] == "sliding_attention" else None
                )
                layer_attention_mask = _create_causal_mask(
                    position_ids=position_ids,
                    target_length=target_length,
                    sliding_window=sliding_window,
                )
            else:
                layer_attention_mask = attention_mask[self.config.layer_types[i]]

            hidden_states = decoder_layer(
                hidden_states,
                per_layer_input,
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                attention_mask=layer_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        next_cache = past_key_values.to_legacy_cache() if use_cache else None
        output = BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=next_cache)
        return output if return_dict else output.to_tuple()


class QEffGemma4ForCausalLM(Gemma4ForCausalLM):
    def __qeff_init__(self):
        if hasattr(self.config, "_experts_implementation"):
            self.config._experts_implementation = "eager"

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffGemma4TextDecoderLayer}

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        del attention_mask, labels, logits_to_keep

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        if position_ids is not None:
            logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
            hidden_states = hidden_states[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        else:
            hidden_states = hidden_states[:, -1:, :]

        logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        logits = logits.float()

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_dummy_pkv_cache(self, config: Gemma4TextConfig, batch_size: int, seq_len: int):
        past_key_values = []
        for layer_type in config.layer_types:
            is_sliding = layer_type == "sliding_attention"
            head_dim = config.head_dim if is_sliding else (config.global_head_dim or config.head_dim)
            num_kv_heads = config.num_key_value_heads
            if not is_sliding and config.attention_k_eq_v and config.num_global_key_value_heads is not None:
                num_kv_heads = config.num_global_key_value_heads
            cache_len = min(config.sliding_window, seq_len) if is_sliding else seq_len
            cache_shape = [batch_size, num_kv_heads, cache_len, head_dim]
            past_key_values.append(
                (
                    torch.zeros(cache_shape, dtype=torch.float32),
                    torch.zeros(cache_shape, dtype=torch.float32),
                )
            )
        return past_key_values
