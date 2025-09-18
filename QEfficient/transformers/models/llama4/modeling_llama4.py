# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import math
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.llama4.modeling_llama4 import (
    Llama4ForCausalLM,
    Llama4ForConditionalGeneration,
    Llama4TextAttention,
    Llama4TextConfig,
    Llama4TextDecoderLayer,
    Llama4TextExperts,
    Llama4TextModel,
    Llama4TextMoe,
    Llama4VisionAttention,
    Llama4VisionModel,
    logger,
    repeat_kv,
)

from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo


def eager_attention_forward_vision(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) / math.sqrt(module.head_dim)
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    if attention_mask is not None:
        attn_weights = torch.where(attention_mask, torch.tensor(-10000.0, dtype=torch.float32), attn_weights)

    attn_weights = nn.functional.softmax(attn_weights.float(), dim=-1).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def complex_mul_onnx_safe(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_real, a_imag = a.unbind(dim=-1)
    b_real, b_imag = b.unbind(dim=-1)

    real = a_real * b_real - a_imag * b_imag
    imag = a_real * b_imag + a_imag * b_real

    return torch.stack((real, imag), dim=-1)


def qeff_vision_apply_rotary_emb(
    query: torch.Tensor, key: torch.Tensor, freqs_ci: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    query_in = query.view(*query.shape[:-1], -1, 2)
    key_in = key.view(*key.shape[:-1], -1, 2)

    # freqs_ci: [L, 1, D, 2] => [1, 1, L, D, 2] (broadcasted to query shape)
    freqs_ci = freqs_ci.unsqueeze(0).unsqueeze(0)  # [1,1,L,D,2]

    # Apply rotary: elementwise complex multiplication
    query_out = complex_mul_onnx_safe(query_in, freqs_ci)
    key_out = complex_mul_onnx_safe(key_in, freqs_ci)

    query_out = query_out.reshape(*query.shape)
    key_out = key_out.reshape(*key.shape)

    return query_out, key_out


class QEffLlama4VisionRotaryEmbedding(nn.Module):
    """
    Vision RoPE that
    • caches (cos, sin) tables as a real-valued buffer --► folds into an ONNX initializer
    • grows automatically if you feed larger images
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.theta = config.rope_theta
        self.patch_size = config.patch_size

        # Build the initial cache for the reference image resolution
        n_patches = (config.image_size // self.patch_size) ** 2
        self._build_cache(n_patches)  # registers `freqs_cis`
        self.max_tokens_cached = n_patches + 1  # +1 for CLS row

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, S, H]   (S = CLS + patches)
        Returns:      [S, H/num_heads, 2]  (cos,sin).
        """
        return self.freqs_cis.unsqueeze(1).to(hidden_states.device)

    def _build_cache(self, n_patches: int) -> None:
        """
        Pre-compute cos/sin for every patch plus the CLS token.
        Produces buffer `freqs_cis`:  [(n_patches+1), head_dim, 2]
        """
        # -- 2-D grid coordinates ---------------------------------------- #
        side = int(math.sqrt(n_patches))
        assert side * side == n_patches, "Vision RoPE expects a square grid of patches"

        coords = torch.arange(side)
        y, x = torch.meshgrid(coords, coords, indexing="ij")
        x = x.reshape(-1, 1)  # [n_patches,1]
        y = y.reshape(-1, 1)

        # -- rotary base frequencies ------------------------------------- #
        head_dim = self.hidden_size // self.n_heads // 2  # real+imag split
        rope_freq = 1.0 / (self.theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

        # angles along x / y; repeat_interleave = [freq0,freq0,freq1,freq1,…]
        ang_x = ((x + 1) * rope_freq).repeat_interleave(2, dim=-1)
        ang_y = ((y + 1) * rope_freq).repeat_interleave(2, dim=-1)
        freqs = torch.cat([ang_x, ang_y], dim=-1).float()[..., ::2]  # [n_patches, head_dim]

        # -- add CLS row = zeros  ---------------------------------------- #
        freqs = torch.cat([freqs, freqs.new_zeros((1, freqs.shape[1]))], dim=0)

        # -- stack real/imag → shape [tokens, head_dim, 2] --------------- #
        real, imag = torch.cos(freqs), torch.sin(freqs)
        freqs_cis = torch.stack([real, imag], dim=-1)

        #    freqs_cis = torch.view_as_complex(freqs_cis.contiguous())

        # store as non-persistent buffer
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)


class QEffLlama4VisionAttention(Llama4VisionAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_ci: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        query_states, key_states = qeff_vision_apply_rotary_emb(query_states, key_states, freqs_ci=freqs_ci)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attention_interface: Callable = eager_attention_forward_vision

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            None,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=None,
            is_causal=False,  # HAS TO BE ENFORCED
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QEffLlama4VisionModel(Llama4VisionModel):
    def __init__(self, config):
        super().__init__(config)
        # Define the general __qeff_init__() for any changes in the init calls
        # Set the init in the module mapping pytorch transforms
        self.config = config
        self.__qeff_init__()

    def __qeff_init__(self):
        self.rotary_embedding = QEffLlama4VisionRotaryEmbedding(config=self.config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:
        r"""

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MllamaVisionModel

        >>> checkpoint = "meta-llama/Llama-3.2-11B-Vision"
        >>> model = MllamaVisionModel.from_pretrained(checkpoint)
        >>> processor = AutoProcessor.from_pretrained(checkpoint)

        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")

        >>> output = model(**inputs)

        >>> print(output.last_hidden_state.shape)
        torch.Size([1, 1, 4, 1025, 7680])
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # num_concurrent_media and num_chunks are both currently 1
        max_num_tiles, num_channels, height, width = pixel_values.shape
        num_concurrent_media = 1
        num_chunks = 1
        hidden_state = self.patch_embedding(pixel_values)
        _, num_patches, hidden_dim = hidden_state.shape

        # Add cls token
        hidden_state = hidden_state.reshape(max_num_tiles * num_concurrent_media * num_chunks, num_patches, hidden_dim)
        class_embedding = self.class_embedding.expand(hidden_state.shape[0], 1, hidden_state.shape[-1])
        hidden_state = torch.cat([hidden_state, class_embedding], dim=1)
        num_patches += 1

        # Position embeddings
        hidden_state = hidden_state.reshape(max_num_tiles * num_concurrent_media, num_chunks, num_patches, hidden_dim)
        positional_embedding = self.positional_embedding_vlm.to(dtype=hidden_state.dtype, device=hidden_state.device)
        hidden_state = hidden_state + positional_embedding

        hidden_state = self.layernorm_pre(hidden_state)

        hidden_state = hidden_state.view(max_num_tiles, -1, hidden_dim)
        freqs_ci = self.rotary_embedding(pixel_values)

        output = self.model(
            hidden_state,
            attention_mask=None,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            freqs_ci=freqs_ci,
        )

        hidden_state = output.last_hidden_state

        hidden_state = self.layernorm_post(hidden_state)

        hidden_state = hidden_state[:, :-1, :]

        # now, we use Llama4VisionPixelShuffle + mlp to project embeddings
        hidden_state = self.vision_adapter(hidden_state)

        hidden_states = output.hidden_states if output_hidden_states else None

        if output_attentions:
            attentions = output[2]
        else:
            attentions = None

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states, attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
            attentions=attentions,
        )


class QEffLlama4TextRotaryEmbedding(nn.Module):
    def __init__(self, config: Llama4TextConfig, device=None):
        super().__init__()
        self.config = config
        self.rope_type = "llama3" if config.rope_scaling is not None else "default"
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        # self.max_seq_len_cached = config.max_position_embeddings
        # TODO: vbaddi Shouldn't for rope, the max posision_embeddings be original embeddings for rope,
        # chunk size 8192 always? and Revisit when >8K Chunked attention is enabled.
        self.max_seq_len_cached = constants.LLAMA4_MAX_POSITION_EMBEDDINGS

        # Get inverse frequency and scaling function (handles yarn/etc)
        inv_freq, self.attention_scaling = self.rope_init_fn(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute static cache
        self._set_freqs_cis_cache(self.max_seq_len_cached, device)

    def _set_freqs_cis_cache(self, seq_len, device):
        # Compute frequencies
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)  # [seq_len]
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim/2]

        # Convert to [real, imag] = [cos, sin]
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        freqs_cis = torch.stack([cos, sin], dim=-1)  # [seq_len, dim/2, 2]

        self.register_buffer("freqs_cis_cached", freqs_cis * self.attention_scaling, persistent=False)

    def forward(self, seq_len: Optional[int] = None, position_ids: Optional[torch.LongTensor] = None):
        """
        Returns: freqs_cis: [batch, seq_len, dim/2, 2] if position_ids given,
                           [seq_len, dim/2, 2] if only seq_len is given.
        """
        if position_ids is not None:
            # position_ids: [batch, seq_len]
            return self.freqs_cis_cached[position_ids]  # shape: [batch, seq_len, dim/2, 2]
        else:
            assert seq_len is not None, "Either seq_len or position_ids must be provided."
            return self.freqs_cis_cached[:seq_len]  # shape: [seq_len, dim/2, 2]


def qeff_apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.view(*xq.shape[:-1], -1, 2)
    xk_ = xk.view(*xk.shape[:-1], -1, 2)

    # freqs_cis is already in [..., 2] form (real, imag)
    freqs_cis_exp = freqs_cis.unsqueeze(2)  # [1,1,L,D,2]

    xq_out = complex_mul_onnx_safe(xq_, freqs_cis_exp)
    xk_out = complex_mul_onnx_safe(xk_, freqs_cis_exp)
    xq_out = xq_out.reshape(*xq.shape)
    xk_out = xk_out.reshape(*xk.shape)
    return xq_out, xk_out


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


class QEffLlama4TextExperts(Llama4TextExperts):
    def __qeff_init__(self):
        self.gate_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.expert_dim))
        self.up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.expert_dim))


class QEffLlama4TextMoe(Llama4TextMoe):
    def forward(self, hidden: torch.Tensor):
        B, S, H = hidden.shape
        T = B * S
        hidden = hidden.view(T, H)

        router_logits = self.router(hidden)
        # *top-k = 1*  → LLama4
        top_w, top_i = torch.topk(router_logits, self.top_k, dim=-1)  # both [T, K]
        masked_logits = torch.full_like(router_logits, float("-inf"))
        masked_logits.scatter_(1, top_i, top_w)

        # Here we multiply by scores before experts, different only for Llama4
        x = hidden * torch.sigmoid(top_w.float())

        # ── Book-keeping: create one boolean mask per expert once  ───────────────
        # routing_weights[e]  ==  True where token routed to that expert. Shape [E, T]
        routing_weights = torch.sigmoid(masked_logits.float()).to(hidden.dtype)

        # ────────────────── allocate the two big tensors ─────
        ffn_dim = self.experts.intermediate_size  # = 8/3 · H
        upgate = x.new_zeros((T, ffn_dim))
        expert_out = x.new_zeros((T, H))  # accum-out buffer

        # ───────────────────────── Stage-1 : Up-Gate ─────────────────────────────
        # Loop over experts
        for e in range(self.num_experts):
            W_g, W_u = self.experts.gate_proj[e], self.experts.up_proj[e]
            routing_weight = routing_weights[:, e].unsqueeze(-1)
            masked_up = torch.where(
                routing_weights[:, e].unsqueeze(-1) > 0,
                ((self.experts.act_fn(x @ W_g)) * (x @ W_u)),
                torch.zeros_like(upgate),
            )
            upgate += masked_up

        # At this point  upgate[t]  holds   UpGate(x_t)   for that token’s expert,
        # and arbitrary (zeros) data for tokens not routed to that expert.
        # ───────────────────────── Stage-2 : Down ────────────────────────────────
        for e in range(self.num_experts):
            routing_weight = routing_weights[:, e].unsqueeze(-1)
            masked_down = torch.where(
                routing_weight > 0, (upgate @ self.experts.down_proj[e]), torch.zeros_like(expert_out)
            )
            expert_out += masked_down

        # ───────────────────────── Stage-3 : Shared expert ───────────────────────
        shared_out = self.shared_expert(hidden)  # [T, H]
        final = shared_out + expert_out  # restore [B,S,H]
        return final.view(B, S, H), router_logits


class QEffLlama4TextAttention(Llama4TextAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(*input_shape, -1, self.head_dim)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]

        kv_seq_len = past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        ##
        if self.use_rope:  # the 16E model skips rope for long context on certain layers
            query_states, key_states = qeff_apply_rotary_emb(
                query_states, key_states, position_embeddings.to(query_states.device)
            )

        if hasattr(self, "qk_norm"):  # the 128E model does not use qk_norm
            query_states = self.qk_norm(query_states)
            key_states = self.qk_norm(key_states)

        # Use temperature tuning from https://arxiv.org/abs/2501.19399) to NoROPE layers
        if self.attn_temperature_tuning and not self.use_rope:
            attn_scales = (
                torch.log(torch.floor((position_ids.float() + 1.0) / self.floor_scale) + 1.0) * self.attn_scale + 1.0
            )
            attn_scales = attn_scales.view((*input_shape, 1, 1))
            query_states = (query_states * attn_scales).to(query_states.dtype)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        if past_key_value is not None:
            chunk_position_ids = position_ids

            if self.use_rope:
                chunk_position_ids = torch.where(
                    chunk_position_ids != -1, chunk_position_ids % self.config.attention_chunk_size, chunk_position_ids
                )

            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"batch_index": batch_index, "position_ids": chunk_position_ids}
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


class QEffLlama4TextDecoderLayer(Llama4TextDecoderLayer):
    """
    Copied from LlamaForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama.py
    The only differences are:
    - add new args batch idx for the CB models
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        chunk_causal_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        # use local attention mask for ROPE layers
        if self.use_chunked_attention:
            attention_mask = chunk_causal_mask

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_value=past_key_value,
            batch_index=batch_index,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        if self.is_moe_layer:
            # Change by VB
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        hidden_states = residual + hidden_states.view(residual.shape)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class QEffLlama4TextModel(Llama4TextModel):
    """
    Copied from LlamaForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama.py
    The only differences are:
    - add new args cache idx for the kv retention
    """

    def __init__(self, config: Llama4TextModel):
        super().__init__(config)
        # Define the general __qeff_init__() for any changes in the init calls
        # Set the init in the module mapping pytorch transforms
        self.config = config
        self.__qeff_init__()

    def __qeff_init__(self):
        self.rotary_emb = QEffLlama4TextRotaryEmbedding(config=self.config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = _create_causal_mask(
            position_ids=position_ids, target_length=past_key_values.key_cache[3].shape[-2]
        )
        chunk_position_ids = torch.where(
            position_ids != -1, position_ids % self.config.attention_chunk_size, position_ids
        )
        target_length = min(past_key_values.key_cache[0].shape[-2], torch.tensor(self.config.attention_chunk_size))
        chunk_causal_mask = _create_causal_mask(position_ids=chunk_position_ids, target_length=target_length)

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        freq_cis = self.rotary_emb(hidden_states, position_ids=position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                chunk_causal_mask=chunk_causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                batch_index=batch_index,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=freq_cis,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


class QEffLlama4ForCausalLM(Llama4ForCausalLM):
    """
    Copied from Llama4ForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama.py
    The only differences are:
    - add new args cache idx for the kv retention
    """

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
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
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        # Cast to INT32 to avoid issue while running in ONNXRT
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs[0][torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_dummy_pkv_cache(self, config, batch_size, seq_len):
        n_heads = config.num_key_value_heads
        d_head = config.head_dim
        is_chunked_attention = torch.tensor(
            [bool((i + 1) % 4) for i in range(config.num_hidden_layers)], dtype=torch.bool
        )
        attention_chunk_size = getattr(config, "attention_chunk_size", seq_len)
        global_cache_shape = [batch_size, n_heads, seq_len, d_head]
        chunked_cache_shape = [
            batch_size,
            n_heads,
            seq_len if seq_len < attention_chunk_size else attention_chunk_size,
            d_head,
        ]

        past_key_values = []
        for i in range(config.num_hidden_layers):
            cache_shape = global_cache_shape if not is_chunked_attention[i] else chunked_cache_shape
            new_layer_key_cache = torch.zeros(cache_shape, dtype=torch.float32)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=torch.float32)
            pkv = (new_layer_key_cache, new_layer_value_cache)
            past_key_values.append(pkv)
        return past_key_values


class QEffLlama4EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        vision_feature_layer = self.model.config.vision_config.vision_feature_layer
        vision_feature_select_strategy = self.model.config.vision_config.vision_feature_select_strategy
        image_features = self.model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            image_sizes=None,
        )
        vision_flat = image_features.view(-1, image_features.size(-1))
        projected_vision_flat = self.model.multi_modal_projector(vision_flat)
        return projected_vision_flat


# This wrapper utilizes the 'vision_embeds', which contains vision embeddings, and an 'image_idx' index starting at 0.
# Within the wrapper, a portion of vision_embeds is merged with inputs_embeds wherever it encounters the image_token_idx in input_ids.
# The image_idx is then updated to point to the next available index of vision_embeds.
# In successive iterations, merging starts from the image_idx of vision_embeds with inputs_embeds.


class QEffLlama4DecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.language_model = self.model.language_model
        self.config = self.model.config

    def forward(self, input_ids, vision_embeds, position_ids, image_idx, past_key_values):
        inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        selected = input_ids == self.model.config.image_token_index
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(selected.unsqueeze(0).shape[0]).view(-1, 1)
        image_features_expanded = vision_embeds.unsqueeze(0)[indices0, indices1]
        image_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_embeds)
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds, position_ids=position_ids, past_key_values=past_key_values, use_cache=True
        )
        next_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        image_idx = torch.where(image_idx < next_idx, next_idx, image_idx)
        return outputs.logits, vision_embeds, image_idx, outputs.past_key_values


class QEffLlama4ForConditionalGeneration(Llama4ForConditionalGeneration):
    def get_qeff_vision_encoder(self):
        return QEffLlama4EncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEffLlama4DecoderWrapper(self)

    def forward(self, input_ids, position_ids, pixel_values, image_idx, past_key_values):
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        vision_feature_layer = self.config.vision_config.vision_feature_layer
        vision_feature_select_strategy = self.config.vision_config.vision_feature_select_strategy
        image_features = self.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            image_sizes=None,
        )
        vision_flat = image_features.view(-1, image_features.size(-1))
        projected_vision_flat = self.multi_modal_projector(vision_flat)
        selected = input_ids == self.config.image_token_index
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(selected.unsqueeze(0).shape[0]).view(-1, 1)
        image_features_expanded = projected_vision_flat.unsqueeze(0)[indices0, indices1]
        image_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_embeds)
        outputs = self.language_model(
            inputs_embeds=inputs_embeds, position_ids=position_ids, past_key_values=past_key_values, use_cache=True
        )
        next_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        image_idx = torch.where(image_idx < next_idx, next_idx, image_idx)
        return outputs.logits, pixel_values, image_idx, outputs.past_key_values

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        img_size: int,
        kv_offload: bool = False,
        **compiler_options,
    ):
        # TODO: check if this should be named num_patches or something else
        max_num_tiles = compiler_options.pop("max_num_tiles", None)
        if max_num_tiles is None:
            logger.warning(
                "User should pass `max_num_tiles` to compile API to fix the dynamic axes `pixel_values`, you can get more info by calling get_inputs_info function!, Since its not found setting its value to 17"
            )
            max_num_tiles = 17

        prefill_seq_len = prefill_seq_len if prefill_seq_len else 32
        ctx_len = ctx_len if ctx_len else constants.INTERN_CTX_LEN
        chunk_ctx_len = min(
            ctx_len,
            (
                self.config.text_config.attention_chunk_size
                if hasattr(self, "config")
                else constants.LLAMA4_ATTENTION_CHUNK_SIZE
            ),
        )

        if img_size is None and hasattr(self.config.vision_config, "image_size"):
            img_size = getattr(self.config.vision_config, "image_size")
        elif img_size is None:
            img_size = 336  # FIXME based on llama4 Image size
            logger.warning("Setting img_size to be 336, as it was neither passed nor found in vision_config")

        downsample_ratio = int(round(1.0 / (self.config.vision_config.pixel_shuffle_ratio**2)))
        num_features_per_tile = int(
            (img_size // self.config.vision_config.patch_size)
            * (img_size // self.config.vision_config.patch_size)
            // downsample_ratio
        )
        vision_size = num_features_per_tile * max_num_tiles

        downsample_ratio = int(round(1.0 / (self.config.vision_config.pixel_shuffle_ratio**2)))
        num_features_per_tile = int(
            (img_size // self.config.vision_config.patch_size)
            * (img_size // self.config.vision_config.patch_size)
            // downsample_ratio
        )
        vision_size = num_features_per_tile * max_num_tiles

        vision = [
            {
                "batch_size": batch_size,
                "max_num_tiles": max_num_tiles,
                "img_size": img_size,
            }
        ]
        lang = [
            {
                "batch_size": batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "max_num_tiles": max_num_tiles,
                "img_size": img_size,
                "vision_size": vision_size,
                "chunk_length": prefill_seq_len,
                "chunk_ctx_len": chunk_ctx_len,
            },
            {
                "batch_size": batch_size,
                "seq_len": "1",
                "ctx_len": ctx_len,
                "max_num_tiles": max_num_tiles,
                "img_size": img_size,
                "vision_size": vision_size,
                "chunk_length": prefill_seq_len,
                "chunk_ctx_len": chunk_ctx_len,
            },
        ]

        specializations = {}

        if kv_offload:
            specializations["vision"] = vision
            specializations["lang"] = lang
            return specializations, compiler_options
        else:
            return lang, compiler_options

    def get_onnx_dynamic_axes(self, kv_offload: bool = False):
        # Define dynamic axes
        vision_dynamic_axes = {}
        lang_dynamic_axes = {}
        lang_dynamic_axes["input_ids"] = {0: "batch_size", 1: "seq_len"}
        lang_dynamic_axes["position_ids"] = {0: "batch_size", 1: "seq_len"}
        lang_dynamic_axes["vision_embeds"] = {0: "vision_size"}
        vision_dynamic_axes["pixel_values"] = {0: "max_num_tiles", 2: "img_size", 3: "img_size"}

        pkv_dynamic_axes = {0: "batch_size"}
        for i in range(self.language_model.config.num_hidden_layers):
            # switch between chunk_ctx_len and ctx_len for RoPE and NoPE layers.
            if int((i + 1) % 4 != 0):
                pkv_dynamic_axes[2] = "chunk_ctx_len"
            else:
                pkv_dynamic_axes[2] = "ctx_len"

            for kv in ["key", "value"]:
                lang_dynamic_axes[f"past_{kv}.{i}"] = pkv_dynamic_axes

        dynamic_axes = {}
        if kv_offload:
            dynamic_axes["vision"] = vision_dynamic_axes
            dynamic_axes["lang"] = lang_dynamic_axes
        else:
            lang_dynamic_axes.pop("vision_embeds")
            dynamic_axes = {**vision_dynamic_axes, **lang_dynamic_axes}
        return dynamic_axes

    def get_output_names(self, kv_offload: bool = False):
        vision_output_names = ["vision_embeds"]
        lang_output_names = ["logits"]
        for i in range(self.language_model.config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_output_names.append(f"past_{kv}.{i}_RetainedState")

        output_names = {}
        if kv_offload:
            lang_output_names.insert(1, "vision_embeds_RetainedState")
            lang_output_names.insert(2, "image_idx_output")
            output_names["vision"] = vision_output_names
            output_names["lang"] = lang_output_names
        else:
            lang_output_names.insert(1, "pixel_values_RetainedState")
            lang_output_names.insert(2, "image_idx_output")
            return lang_output_names
        return output_names

    def get_dummy_pkv_cache(self, config, batch_size, seq_len):
        n_heads = config.num_key_value_heads
        d_head = config.head_dim
        is_chunked_attention = torch.tensor(
            [bool((i + 1) % 4) for i in range(config.num_hidden_layers)], dtype=torch.bool
        )
        attention_chunk_size = getattr(config, "attention_chunk_size", seq_len)
        global_cache_shape = [batch_size, n_heads, seq_len, d_head]
        chunked_cache_shape = [
            batch_size,
            n_heads,
            seq_len if seq_len < attention_chunk_size else attention_chunk_size,
            d_head,
        ]

        past_key_values = []
        for i in range(config.num_hidden_layers):
            cache_shape = global_cache_shape if not is_chunked_attention[i] else chunked_cache_shape
            new_layer_key_cache = torch.zeros(cache_shape, dtype=torch.float32)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=torch.float32)
            pkv = (new_layer_key_cache, new_layer_value_cache)
            past_key_values.append(pkv)
        return past_key_values

    def get_dummy_inputs(self, kv_offload: bool = False):
        if vis_cfg := getattr(self.config, "vision_config", None):
            img_size = getattr(vis_cfg, "image_size", 336)
        else:
            img_size = 336

        # Define shapes
        inputs_shapes = {}
        inputs_shapes["input_ids"] = (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)
        max_num_tiles = 17
        downsample_ratio = int(round(1.0 / (self.config.vision_config.pixel_shuffle_ratio**2)))
        num_features_per_tile = int(
            (img_size // self.config.vision_config.patch_size)
            * (img_size // self.config.vision_config.patch_size)
            // downsample_ratio
        )
        vision_size = num_features_per_tile * max_num_tiles

        inputs_shapes["vision_embeds"] = (
            vision_size,
            self.language_model.config.hidden_size,  # 5120
        )
        inputs_shapes["position_ids"] = (
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN,
        )
        inputs_shapes["pixel_values"] = (
            max_num_tiles,  # constants.INTERN_NUM_PATCHES,
            constants.INTERN_NUM_CHANNELS,
            img_size,
            img_size,
        )
        inputs_shapes["image_idx"] = (1, 1)
        # Define inputs
        vision_inputs = {}
        lang_inputs = {}
        vision_inputs["pixel_values"] = torch.zeros((inputs_shapes["pixel_values"]), dtype=torch.float32)
        lang_inputs["input_ids"] = torch.zeros((inputs_shapes["input_ids"]), dtype=torch.int64)
        lang_inputs["vision_embeds"] = torch.zeros((inputs_shapes["vision_embeds"]), dtype=torch.float32)
        lang_inputs["position_ids"] = (
            torch.arange(constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN, dtype=torch.int64)
            .view(1, constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)
            .repeat(constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, 1)
        )
        lang_inputs["image_idx"] = torch.zeros((inputs_shapes["image_idx"]), dtype=torch.int64)
        # Add data for KV
        past_key_values = self.get_dummy_pkv_cache(
            config=self.language_model.config,
            batch_size=constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            seq_len=constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN,
        )

        lang_inputs["past_key_values"] = [[] for _ in range(self.language_model.config.num_hidden_layers)]
        for i in range(self.language_model.config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_inputs["past_key_values"][i].append(torch.zeros(past_key_values[0][0].shape, dtype=torch.float32))

        inputs = {}
        if kv_offload:
            inputs["vision"] = vision_inputs
            inputs["lang"] = lang_inputs
        else:
            lang_inputs.pop("vision_embeds")
            inputs = {**vision_inputs, **lang_inputs}

        return inputs

    def get_inputs_info(self):
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="attention_mask", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(
                name="pixel_values",
                datatype=torch.float32,
                shape=("max_num_tiles", 3, "img_size", "img_size"),
            ),
        ]
