from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeConfig,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeSparseMoeBlock,
    repeat_kv,
    rotate_half,
)

from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask

# from QEfficient.transformers.models.llama.modeling_llama import qeff_apply_rotary_pos_emb


class QEffQwen3MoeRotaryEmbedding(Qwen3MoeRotaryEmbedding):
    def __init__(self, config: Qwen3MoeConfig, device=None):
        super().__init__(config=config)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=self.original_max_seq_len, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def qeff_apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # breakpoint()
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

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
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = torch.where(attention_mask, torch.tensor(-10000.0, dtype=torch.float32), attn_weights)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class QEffQwen3MoeSparseMoeBlock(Qwen3MoeSparseMoeBlock):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # # breakpoint()
        # B, S, D = hidden_states.shape  # [1, 8, 2304]
        # hidden_states = hidden_states.reshape(-1, D)  # [8, 2304]
        # T = hidden_states.size(0)  # 8 tokens
        # router_logits = self.gate(hidden_states)  # [8, 8]
        # probs = F.softmax(router_logits, dim=-1)  # [8, 8]

        # topk_scores, topk_indices = torch.topk(probs, self.top_k, dim=-1)  # [8, top_k] → topk_k is 2 for Grok1
        # topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)  # normalize per-token
        # topk_scores = topk_scores.to(hidden_states.dtype)  # [8, top_k]
        # route = torch.zeros((T, self.num_experts), dtype=hidden_states.dtype)
        # route.scatter_(1, topk_indices, topk_scores)  # [8, num_experts]
        # final_output = torch.zeros_like(hidden_states)  # [8, 2304]

        # for e, expert in enumerate(self.experts):
        #     scores = route[:, e].unsqueeze(1)  # [8, 1]
        #     masked_out = torch.where(
        #         scores > 0, expert(hidden_states) * scores, 0.0
        #     )  # # [8, 2304] × [8, 1] → [8, 2304]
        #     final_output += masked_out  # accumulate expert outputs
        # return final_output.reshape(B, S, D), router_logits  # ([1, 8, 2304], [8, num_experts])

        B, S, H = hidden_states.shape
        T = B * S
        x = hidden_states.view(T, H)

        router_logits = self.gate(x)  # [T, E]
        prob = F.softmax(router_logits, -1, dtype=torch.float)
        top_w, top_i = torch.topk(prob, self.top_k, -1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            top_w /= top_w.sum(-1, keepdim=True)
        top_w = top_w.to(x.dtype)

        # Create 2 expert idx based on the topk
        expert1_idx, expert2_idx, expert3_idx, expert4_idx, expert5_idx, expert6_idx, expert7_idx, expert8_idx = (
            top_i[:, 0],
            top_i[:, 1],
            top_i[:, 2],
            top_i[:, 3],
            top_i[:, 4],
            top_i[:, 5],
            top_i[:, 6],
            top_i[:, 7],
        )  # [T]
        weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8 = (
            top_w[:, 0],
            top_w[:, 1],
            top_w[:, 2],
            top_w[:, 3],
            top_w[:, 4],
            top_w[:, 5],
            top_w[:, 6],
            top_w[:, 7],
        )  # [T]

        Inter = 768
        upgate1 = x.new_zeros((T, Inter))
        upgate2 = x.new_zeros((T, Inter))
        upgate3 = x.new_zeros((T, Inter))
        upgate4 = x.new_zeros((T, Inter))
        upgate5 = x.new_zeros((T, Inter))
        upgate6 = x.new_zeros((T, Inter))
        upgate7 = x.new_zeros((T, Inter))
        upgate8 = x.new_zeros((T, Inter))

        expert_out1 = x.new_zeros((T, H))
        expert_out2 = x.new_zeros((T, H))
        expert_out3 = x.new_zeros((T, H))
        expert_out4 = x.new_zeros((T, H))
        expert_out5 = x.new_zeros((T, H))
        expert_out6 = x.new_zeros((T, H))
        expert_out7 = x.new_zeros((T, H))
        expert_out8 = x.new_zeros((T, H))

        for e in range(self.num_experts):
            exp = self.experts[e]
            mask1 = (expert1_idx == e).unsqueeze(1)  # [T, 1]
            mask2 = (expert2_idx == e).unsqueeze(1)  # [T, 1]
            mask3 = (expert3_idx == e).unsqueeze(1)  # [T, 1]
            mask4 = (expert4_idx == e).unsqueeze(1)  # [T, 1]
            mask5 = (expert5_idx == e).unsqueeze(1)  # [T, 1]
            mask6 = (expert6_idx == e).unsqueeze(1)  # [T, 1]
            mask7 = (expert7_idx == e).unsqueeze(1)  # [T, 1]
            mask8 = (expert8_idx == e).unsqueeze(1)  # [T, 1]

            # breakpoint()
            hidden_gate = (exp.act_fn(exp.gate_proj(x))) * exp.up_proj(x)
            # hidden_gate=exp.down_proj(hidden_gate)

            # Accumulate weighted contributions
            upgate1 += torch.where(mask1, hidden_gate, torch.zeros_like(upgate1))
            upgate2 += torch.where(mask2, hidden_gate, torch.zeros_like(upgate2))
            upgate3 += torch.where(mask3, hidden_gate, torch.zeros_like(upgate3))
            upgate4 += torch.where(mask4, hidden_gate, torch.zeros_like(upgate4))
            upgate5 += torch.where(mask5, hidden_gate, torch.zeros_like(upgate5))
            upgate6 += torch.where(mask6, hidden_gate, torch.zeros_like(upgate6))
            upgate7 += torch.where(mask7, hidden_gate, torch.zeros_like(upgate7))
            upgate8 += torch.where(mask8, hidden_gate, torch.zeros_like(upgate8))

        for e in range(self.num_experts):
            exp = self.experts[e]
            mask1 = (expert1_idx == e).unsqueeze(1)
            mask2 = (expert2_idx == e).unsqueeze(1)
            mask3 = (expert3_idx == e).unsqueeze(1)  # [T, 1]
            mask4 = (expert4_idx == e).unsqueeze(1)  # [T, 1]
            mask5 = (expert5_idx == e).unsqueeze(1)  # [T, 1]
            mask6 = (expert6_idx == e).unsqueeze(1)  # [T, 1]
            mask7 = (expert7_idx == e).unsqueeze(1)  # [T, 1]
            mask8 = (expert8_idx == e).unsqueeze(1)
            # breakpoint()
            expert_out1 += torch.where(
                mask1, exp.down_proj(upgate1) * weight1.unsqueeze(1), torch.zeros_like(expert_out1)
            )
            expert_out2 += torch.where(
                mask2, exp.down_proj(upgate2) * weight2.unsqueeze(1), torch.zeros_like(expert_out2)
            )
            expert_out3 += torch.where(
                mask3, exp.down_proj(upgate3) * weight3.unsqueeze(1), torch.zeros_like(expert_out3)
            )
            expert_out4 += torch.where(
                mask4, exp.down_proj(upgate4) * weight4.unsqueeze(1), torch.zeros_like(expert_out4)
            )
            expert_out5 += torch.where(
                mask5, exp.down_proj(upgate5) * weight5.unsqueeze(1), torch.zeros_like(expert_out5)
            )
            expert_out6 += torch.where(
                mask6, exp.down_proj(upgate6) * weight6.unsqueeze(1), torch.zeros_like(expert_out6)
            )
            expert_out7 += torch.where(
                mask7, exp.down_proj(upgate7) * weight7.unsqueeze(1), torch.zeros_like(expert_out7)
            )
            expert_out8 += torch.where(
                mask8, exp.down_proj(upgate8) * weight8.unsqueeze(1), torch.zeros_like(expert_out8)
            )

        expert_out = (
            expert_out1
            + expert_out2
            + expert_out3
            + expert_out4
            + expert_out5
            + expert_out6
            + expert_out7
            + expert_out8
        )
        return expert_out.view(B, S, H), router_logits


class QEffQwen3MoeAttention(Qwen3MoeAttention):
    def __qeff_init__(self):
        self.rotary_emb = QEffQwen3MoeRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # breakpoint()
        kv_seq_len = key_states.shape[-2]
        kv_seq_len = past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "batch_index": batch_index, "position_ids": position_ids}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value


class QEffQwen3MoeDecoderLayer(Qwen3MoeDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        # position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            batch_index=batch_index,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class QEffQwen3MoeModel(Qwen3MoeModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> MoeModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # if use_cache and past_key_values is None:
        #     past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            # seq_length_with_past = seq_length_with_past + past_key_values_length

        past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=past_key_values_length)

        hidden_states = inputs_embeds
        # breakpoint()
        # create position embeddings to be shared across the decoder layers
        # position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                batch_index=batch_index,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
                cache_position=cache_position,
                # position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_values = past_key_values.to_legacy_cache()

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


class QEffQwen3MoeForCausalLM(Qwen3MoeForCausalLM):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> MoeCausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3MoeForCausalLM

        >>> model = Qwen3MoeForCausalLM.from_pretrained("Qwen/Qwen3-MoE-15B-A2B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-MoE-15B-A2B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        # breakpoint()
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # breakpoint()
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            batch_index=batch_index,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # logits = self.lm_head(hidden_states[:, slice_indices, :])

        # loss = None
        # if labels is not None:
        #     loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        # aux_loss = None
        # if output_router_logits:
        #     aux_loss = load_balancing_loss_func(
        #         outputs.router_logits,
        #         self.num_experts,
        #         self.num_experts_per_tok,
        #         attention_mask,
        #     )
        #     if labels is not None:
        #         loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device
        # breakpoint()
        logit_idx = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs[0][torch.arange(position_ids.shape[0]).view(-1, 1), logit_idx]
        logits = self.lm_head(hidden_states)
        # logits = logits * self.output_multiplier_scale
        logits = logits.float()

        return MoeCausalLMOutputWithPast(
            # loss=loss,
            # aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )
