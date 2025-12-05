# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput

from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo, get_padding_shape_from_config


def _non_meta_init_device(config) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eager_attention_forward(
    module: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout_p: float = 0.0,
    **kwargs,
):
    scale_factor = 1 / math.sqrt(q.size(-1))
    num_kv_heads = k.size(1)
    num_q_heads = q.size(1)

    if num_q_heads != num_kv_heads:
        assert num_q_heads % num_kv_heads == 0
        repeat_factor = num_q_heads // num_kv_heads
        B, _, S, D = k.shape
        k = k.unsqueeze(2)
        k = k.expand(-1, -1, repeat_factor, -1, -1)
        k = k.reshape(B, num_q_heads, S, D)

        v = v.unsqueeze(2)
        v = v.expand(-1, -1, repeat_factor, -1, -1)
        v = v.reshape(B, num_q_heads, S, D)

    attn_weights = torch.matmul(q, k.transpose(2, 3)) * scale_factor

    if attention_mask is not None:
        attn_weights = torch.where(attention_mask, torch.tensor(-10000.0, dtype=torch.float32), attn_weights)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v)

    return attn_output, attn_weights


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    B, nh, T, hs = x.size()
    x = x.view(B, nh, T, 2, hs // 2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    B, nh, T, hs = x.size()
    x = x.view(B, nh, T, hs // 2, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.view(B, nh, T, hs)


def qeff_apply_rotary_pos_emb(q, k, cos, sin, position_ids, config, unsqueeze_dim=1):
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
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    # Apply rotation
    if config.rope_impl == "interleave":
        q_embed = (q * cos) + (rotate_every_two(q) * sin)
        k_embed = (k * cos) + (rotate_every_two(k) * sin)
    else:
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
    # Cast back to original dtype
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class QEffMolmoRotaryEmbedding(nn.Module):
    """
    Copied from Olmo2RotaryEmbedding: https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmo2/modeling_olmo2.py
    The only differences are:
    - Add static sin/cos computations.
    """

    def __init__(self, config, device=None):
        super().__init__()
        dim = config.d_model // config.n_heads
        self.inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim))
        self.original_max_seq_len = config.max_position_embeddings or config.max_sequence_length
        self._set_cos_sin_cache(
            seq_len=self.original_max_seq_len, device=_non_meta_init_device(config), dtype=torch.get_default_dtype()
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


class QEffMultiHeadDotProductAttention(nn.Module):
    def forward(self, inputs_q: torch.Tensor, inputs_kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        if inputs_kv is not None:
            inputs_k = inputs_kv
            inputs_v = inputs_kv
        else:
            inputs_k = inputs_q
            inputs_v = inputs_q

        xq, xk, xv = self.wq(inputs_q), self.wk(inputs_k), self.wv(inputs_v)

        xq = self._split_heads(xq, self.num_heads)
        xk = self._split_heads(xk, self.num_key_value_heads)
        xv = self._split_heads(xv, self.num_key_value_heads)

        if self.num_heads != self.num_key_value_heads:
            xk = xk.repeat_interleave(self.num_key_value_groups, dim=2, output_size=self.num_heads)
            xv = xv.repeat_interleave(self.num_key_value_groups, dim=2, output_size=self.num_heads)

        og_dtype = xq.dtype

        if self.config.float32_attention:
            xq = xq.to(torch.float)
            xk = xk.to(torch.float)

        if self.config.attention_type == "direct":
            attn_weights = torch.einsum("...qhd,...khd->...hqk", xq / math.sqrt(xq.size(-1)), xk)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(xq.dtype)
            if self.attention_dropout is not None:
                attn_weights = self.attention_dropout(attn_weights)
            attn_output = torch.einsum("...hqk,...khd->...qhd", attn_weights.to(xv.dtype), xv)

        elif self.config.attention_type == "sdpa":
            if self.config.float32_attention and not torch.is_autocast_enabled():
                xv = xv.to(torch.float32)

            attn_output = scaled_dot_product_attention(
                xq.transpose(1, 2).contiguous(),
                xk.transpose(1, 2).contiguous(),
                xv.transpose(1, 2).contiguous(),
                is_causal=False,
                dropout_p=self.config.vision_backbone.attention_dropout,
            ).transpose(1, 2)
        else:
            raise NotImplementedError(self.config.attention_type)
        attn_output = attn_output.to(og_dtype)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.wo(attn_output)
        attn_output = self.residual_dropout(attn_output)

        return attn_output


class QEffMolmoBlock(nn.Module):
    def __qeff_init__(self):
        self.rotary_emb = QEffMolmoRotaryEmbedding(config=self.config)

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        layer_past: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

        if self.config.use_position_ids and self.config.rope:
            kv_seq_len = k.shape[-2]
            kv_seq_len = layer_past.get_seq_length(self.layer_id)
            # Apply rotary embeddings
            cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
            q, k = qeff_apply_rotary_pos_emb(q, k, cos, sin, position_ids, self.config)

        if not self.config.use_position_ids and self.config.rope:
            kv_seq_len = k.shape[-2]
            kv_seq_len = layer_past.get_seq_length(kv_seq_len, self.layer_id)
            # Apply rotary embeddings
            cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
            q, k = qeff_apply_rotary_pos_emb(q, k, cos, sin, position_ids, self.config)

        if layer_past is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "batch_index": batch_index,
                "position_ids": position_ids,
            }
            if comp_ctx_lengths is not None:
                attention_bias = attention_bias[:, :, :, : comp_ctx_lengths.shape[-1]]
                cache_kwargs["CCL"] = attention_bias.shape[-1]
            k, v = layer_past.update(k, v, self.layer_id, cache_kwargs)

        attention_interface: Callable = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            q,
            k,
            v,
            attention_bias,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            **kwargs,
        )

        # Re-assemble all head outputs side-by-side.
        att = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection.
        return self.attn_out(att), layer_past


class QEffMolmoSequentialBlock(nn.Module):
    def __qeff_init__(self):
        self.rotary_emb = QEffMolmoRotaryEmbedding(config=self.config)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        layer_past: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        if not self.config.norm_after:
            atten_in = self.attn_norm(x)
        else:
            atten_in = x
        qkv = self.att_proj(atten_in)

        if self.config.clip_qkv is not None:
            qkv.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        q, k, v = qkv.split(self.fused_dims, dim=-1)

        # Get attention scores.
        att, cache = self.attention(
            q,
            k,
            v,
            attention_bias,
            position_ids=position_ids,
            layer_past=layer_past,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=use_cache,
        )

        if self.config.norm_after:
            att = self.attn_norm(att)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x

        if not self.config.norm_after:
            x = self.ff_norm(x)

        x = self.ff_proj(x)

        x = self.act(x)
        x = self.ff_out(x)

        if self.config.norm_after:
            x = self.ff_norm(x)

        x = self.dropout(x)
        x = og_x + x

        return x, cache


class QEffMolmo(nn.Module):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        image_input_idx: Optional[torch.Tensor] = None,
        subsegment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        append_last_valid_logits: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ModelOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param input_embeddings: A tensor of shape `(batch_size, seq_len, d_model)` with input
            embeddings. When provided, it is treated as the output of the input embedding layer.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            which input IDs are masked. A `1` value in the mask means that
            the corresponding input ID should *not* be ignored. A `0` means
            that the corresponding input ID is masked.

            This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
            library.
        :param attention_bias: A tensor of shape `(batch_size, 1, seq_len, seq_len)`,
            `(1, 1, seq_len, seq_len)`, or `(seq_len, seq_len)`. This is used
            to introduce causal or other biases.

            If the tensor is a bool or byte tensor, a `True` or `1` at `attention_bias[:, :, i, j]`
            indicates that the i-th element in the sequence is allowed to attend to the j-th
            element in the sequence.

            If the tensor is a float tensor, it will just be added to the attention
            scores before the softmax.

            The default is causal, which corresponds to a lower-diagonal byte matrix of ones.
        :param response_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            the response mask. A `1` value in the mask means that the corresponding token
            is a response token. A `0` means that the corresponding token is not
            a response token.
        :param past_key_values: Pre-computed keys and values for each attention block.
            Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        :param use_cache: If `True`, return key and value tensors for each block.
        :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        """

        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        has_image = images is not None

        assert not (has_image and input_embeddings is not None), "Cannot provide both images and input embeddings."
        # assert not (has_image and past_key_values is not None), "Cached key and values should not be used with images."

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=past_seen_tokens)

        if self.config.use_position_ids and attention_mask is None:
            attention_mask = input_ids != -1

        if subsegment_ids is not None:
            assert not use_cache, "Subsegment_ids cannot be used with cache."
            subsegment_mask = subsegment_ids.unsqueeze(2) <= subsegment_ids.unsqueeze(1)
            attention_mask = (
                subsegment_mask.to(attention_mask.dtype) * attention_mask.unsqueeze(2) * attention_mask.unsqueeze(1)
            )
            if position_ids is None:
                raise ValueError("Positioned ids must be given if using subsegment_ids")
        else:
            if self.config.use_position_ids and position_ids is None:
                position_ids = torch.clamp(
                    torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1,
                    min=0,
                ).broadcast_to((batch_size, attention_mask.shape[-1]))

        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings  # type: ignore

        if not self.config.rope:
            # Get positional embeddings.
            # shape: (1, seq_len)
            pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            # shape: (1, seq_len, d_model)
            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = pos_emb + x

        # Add input + positional embeddings and apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        # normalized
        if self.config.normalize_input_embeds:
            x = x * (self.config.d_model**0.5)

        # decoder layers
        all_hidden_states = []

        # Apply blocks one-by-one.
        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layer_past = None if past_key_values is None else past_key_values
                x, past_key_values = block(
                    x,
                    attention_bias=causal_mask,
                    position_ids=position_ids,
                    layer_past=layer_past,
                    comp_ctx_lengths=comp_ctx_lengths,
                    batch_index=batch_index,
                    use_cache=use_cache,
                )

        else:
            for group_idx, block_group in enumerate(self.transformer.block_groups):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layers_past = (
                    None
                    if past_key_values is None
                    else past_key_values[
                        group_idx * self.config.block_group_size : (group_idx + 1) * self.config.block_group_size
                    ]
                )
                x, past_key_values = block_group(
                    x,
                    attention_bias=causal_mask,
                    position_ids=position_ids,
                    layers_past=layers_past,
                    comp_ctx_lengths=comp_ctx_lengths,
                    use_cache=use_cache,
                )

        # Cast to INT32 to avoid issue while running in ONNXRT
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = x[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]

        x = self.transformer.ln_f(hidden_states)  # type: ignore
        if output_hidden_states:
            # add final hidden state post-final-layernorm, following HuggingFace's convention
            all_hidden_states.append(x)

        # Get logits.
        # shape: (batch_size, seq_len or 1, vocab_size)
        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore
        else:
            logits = self.transformer.ff_out(x)  # type: ignore
        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))

        if use_cache:
            next_cache = past_key_values.to_legacy_cache()

        return ModelOutput(
            logits=logits,
            past_key_values=next_cache,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )  # type: ignore[arg-type]


class QEffMolmoEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, image_masks, image_input_idx, valid_idx):
        image_features, _ = self.model.model.vision_backbone(pixel_values, image_masks)
        num_image, num_patch = image_features.shape[1:3]
        batch_size = image_input_idx.shape[0]
        image_features = image_features.view(batch_size, num_image * num_patch, -1)
        image_input_idx = image_input_idx.view(batch_size, num_image * num_patch)

        image_input_idx = image_input_idx[0, valid_idx]
        sorted_indices = torch.argsort(image_input_idx)

        return image_features[0, valid_idx][0, sorted_indices]


class QEffMolmoDecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # self.language_model = self.model.language_model
        self.config = self.model.config

    def forward(
        self,
        input_ids,
        vision_embeds,
        position_ids,
        image_idx,
        past_key_values,
        comp_ctx_lengths: Optional[List[int]] = None,
        batch_index: Optional[torch.LongTensor] = None,
    ):
        if input_ids is not None:
            input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
        inputs_embeds = self.model.model.transformer.wte(input_ids)
        selected = input_ids == 152066
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(selected.unsqueeze(0).shape[0]).view(-1, 1)
        image_features_expanded = vision_embeds[indices0, indices1]
        image_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded + inputs_embeds, inputs_embeds)
        #
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_embeds)
        outputs = self.model.model.forward(
            input_embeddings=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=True,
        )
        next_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        image_idx = torch.where(image_idx < next_idx, next_idx, image_idx)
        return outputs.logits, vision_embeds, image_idx, outputs.past_key_values


class QEffMolmoModel(nn.Module):
    def get_qeff_vision_encoder(self):
        return QEffMolmoEncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEffMolmoDecoderWrapper(self)

    """
    Copied from Llama4ForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama.py
    The only differences are:
    - add new args cache idx for the kv retention
    """

    def forward(
        self,
        pixel_values,
        image_masks,
        image_input_idx,
        valid_idx,
        input_ids,
        position_ids,
        image_idx,
        past_key_values,
        comp_ctx_lengths: Optional[List[int]] = None,
    ):
        image_features, _ = self.model.vision_backbone(pixel_values, image_masks)
        num_image, num_patch = image_features.shape[1:3]
        batch_size = image_input_idx.shape[0]
        image_features = image_features.view(batch_size, num_image * num_patch, -1)
        image_input_idx = image_input_idx.view(batch_size, num_image * num_patch)

        valid = image_input_idx >= 0
        indices0 = torch.arange(valid.unsqueeze(0).shape[0]).view(-1, 1)

        image_input_idx = image_input_idx[0, valid_idx]
        sorted_indices = torch.argsort(image_input_idx)

        vision_embeds = image_features[0, valid_idx][0, sorted_indices]

        if input_ids is not None:
            input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)

        inputs_embeds = self.model.transformer.wte(input_ids)
        selected = input_ids == 152066
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(selected.unsqueeze(0).shape[0]).view(-1, 1)
        image_features_expanded = vision_embeds[indices0, indices1]
        image_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded + inputs_embeds, inputs_embeds)

        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_embeds)
        outputs = self.model.forward(
            input_embeddings=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            use_cache=True,
        )
        next_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        image_idx = torch.where(image_idx < next_idx, next_idx, image_idx)

        return outputs.logits, pixel_values, image_idx, outputs.past_key_values

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        num_images: int = None,
        img_size: int = None,
        comp_ctx_lengths_prefill: Optional[List[int]] = None,
        comp_ctx_lengths_decode: Optional[List[int]] = None,
        valid_size: int = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
        **compiler_options,
    ):
        prefill_seq_len = prefill_seq_len if prefill_seq_len else 1024
        ctx_len = ctx_len if ctx_len else constants.INTERN_CTX_LEN

        img_size = 588
        img_tile = 576
        num_patch = 144

        if None in (num_images, valid_size):
            num_images = 5
            valid_size = 544

        vision = [
            {
                "batch_size": batch_size,
                "img_size": img_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "img_tile": img_tile,
                "num_images": num_images,
                "num_patch": num_patch,
                "valid_size": valid_size,
            }
        ]

        if comp_ctx_lengths_prefill is not None and comp_ctx_lengths_decode is not None:
            lang = []

            for i in range(0, len(comp_ctx_lengths_prefill)):
                lang_prefill = {
                    "batch_size": 1 if continuous_batching else batch_size,
                    "seq_len": prefill_seq_len,
                    "ctx_len": ctx_len,
                    "comp_ctx_lengths": comp_ctx_lengths_prefill[i],
                    "valid_size": valid_size,
                    "vision_batch_size": batch_size,
                }
                if continuous_batching:
                    lang_prefill["full_batch_size"] = kv_cache_batch_size
                else:
                    lang_prefill["batch_size"] = kv_cache_batch_size

                if full_batch_size:
                    lang_prefill["full_batch_exec_size"] = full_batch_size
                if kv_offload:
                    values = {
                        "img_size": img_size,
                        "img_tile": img_tile,
                        "num_images": num_images,
                        "num_patch": num_patch,
                    }

                    for key, value in values.items():
                        lang_prefill[key] = value

                lang.append(lang_prefill)

            for i in range(0, len(comp_ctx_lengths_decode)):
                lang_decode = {
                    "batch_size": full_batch_size if continuous_batching else batch_size,
                    "seq_len": "1",
                    "ctx_len": ctx_len,
                    "comp_ctx_lengths": comp_ctx_lengths_decode[i],
                    "valid_size": valid_size,
                    "vision_batch_size": batch_size,
                }
                if continuous_batching:
                    lang_decode["full_batch_size"] = kv_cache_batch_size
                else:
                    lang_decode["batch_size"] = kv_cache_batch_size
                if kv_offload:
                    values = {
                        "img_size": img_size,
                        "img_tile": img_tile,
                        "num_images": num_images,
                        "num_patch": num_patch,
                    }

                    for key, value in values.items():
                        lang_decode[key] = value

                lang.append(lang_decode)

        else:
            lang_prefill = {
                "batch_size": 1 if continuous_batching else batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "valid_size": valid_size,
                "vision_batch_size": batch_size,
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
                "valid_size": valid_size,
                "vision_batch_size": batch_size,
            }

            if continuous_batching:
                lang_decode["full_batch_size"] = kv_cache_batch_size
            else:
                lang_decode["batch_size"] = kv_cache_batch_size

            if kv_offload:
                values = {
                    "img_size": img_size,
                    "img_tile": img_tile,
                    "num_images": num_images,
                    "num_patch": num_patch,
                }

                for key, value in values.items():
                    lang_prefill[key] = value
                    lang_decode[key] = value

            lang = [lang_prefill, lang_decode]

        specializations = {}

        if kv_offload:
            specializations["vision"] = vision
            specializations["lang"] = lang
            return specializations, compiler_options
        else:
            return lang, compiler_options

    def get_onnx_dynamic_axes(
        self, comp_ctx_lengths: Optional[List[int]] = None, kv_offload: bool = False, continuous_batching: bool = False
    ):
        # Define dynamic axes
        vision_dynamic_axes = {}
        lang_dynamic_axes = {}
        lang_dynamic_axes["input_ids"] = {0: "batch_size", 1: "seq_len"}
        lang_dynamic_axes["position_ids"] = {0: "batch_size", 1: "seq_len"}
        lang_dynamic_axes["vision_embeds"] = {0: "vision_batch_size", 1: "valid_size"}

        vision_dynamic_axes["pixel_values"] = {0: "batch_size", 1: "num_images", 2: "img_tile", 3: "img_size"}
        vision_dynamic_axes["image_input_idx"] = {0: "batch_size", 1: "num_images", 2: "num_patch"}
        vision_dynamic_axes["image_masks"] = {0: "batch_size", 1: "num_images", 2: "img_tile"}
        vision_dynamic_axes["valid_idx"] = {0: "batch_size", 1: "valid_size"}

        num_layers = self.model.config.n_layers

        for i in range(num_layers):
            lang_dynamic_axes[f"past_key.{i}"] = {
                0: "full_batch_size" if continuous_batching else "batch_size",
                2: "ctx_len",
            }
            lang_dynamic_axes[f"past_value.{i}"] = {
                0: "full_batch_size" if continuous_batching else "batch_size",
                2: "ctx_len",
            }

        if continuous_batching:
            lang_dynamic_axes["batch_index"] = {0: "batch_size"}

        if comp_ctx_lengths is not None:
            lang_dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}

        dynamic_axes = {}
        if kv_offload:
            dynamic_axes["vision"] = vision_dynamic_axes
            dynamic_axes["lang"] = lang_dynamic_axes
        else:
            dynamic_axes = {**vision_dynamic_axes, **lang_dynamic_axes}
        return dynamic_axes

    def get_output_names(self, kv_offload: bool = False):
        vision_output_names = ["vision_embeds"]
        lang_output_names = ["logits"]

        #
        for i in range(self.model.config.n_layers):
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

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        **kwargs,
    ):
        inputs_shapes = {}
        inputs_shapes_lang = {}
        inputs_shapes["input_ids"] = (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)

        inputs_shapes["vision_embeds"] = (
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            544,
            self.config.hidden_size,
        )
        inputs_shapes["position_ids"] = (
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN,
        )
        inputs_shapes["pixel_values"] = (
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            5,
            576,
            588,
        )

        inputs_shapes["image_masks"] = (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, 5, 576)

        inputs_shapes["image_input_idx"] = (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, 5, 144)

        inputs_shapes_lang["image_input_idx"] = (
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            720,
        )

        inputs_shapes["valid_idx"] = (1, 544)

        inputs_shapes["image_idx"] = (1, 1)
        inputs_shapes["image_sizes"] = (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, 2)
        # Define inputs
        vision_inputs = {}
        lang_inputs = {}
        vision_inputs["pixel_values"] = torch.zeros((inputs_shapes["pixel_values"]), dtype=torch.float32)
        vision_inputs["image_masks"] = torch.zeros((inputs_shapes["image_masks"]), dtype=torch.float32)
        vision_inputs["image_input_idx"] = torch.zeros((inputs_shapes["image_input_idx"]), dtype=torch.int32)

        vision_inputs["valid_idx"] = torch.zeros((inputs_shapes["valid_idx"]), dtype=torch.int64)

        lang_inputs["input_ids"] = torch.zeros((inputs_shapes["input_ids"]), dtype=torch.int64)
        lang_inputs["vision_embeds"] = torch.zeros((inputs_shapes["vision_embeds"]), dtype=torch.float32)
        lang_inputs["position_ids"] = (
            torch.arange(constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN, dtype=torch.int64)
            .view(1, constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)
            .repeat(constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, 1)
        )
        lang_inputs["image_idx"] = torch.zeros((inputs_shapes["image_idx"]), dtype=torch.int64)

        bs: int = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        fbs: int = constants.ONNX_EXPORT_EXAMPLE_FBS

        # Add data for KV
        kv_cache_shape = get_padding_shape_from_config(
            config=self.config,
            batch_size=fbs if continuous_batching else bs,
            seq_len=constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN,
        )

        lang_inputs["past_key_values"] = [[] for _ in range(self.model.config.n_layers)]
        for i in range(self.model.config.n_layers):
            for kv in ["key", "value"]:
                lang_inputs["past_key_values"][i].append(torch.zeros(kv_cache_shape, dtype=torch.float32))

        if comp_ctx_lengths is not None:
            lang_inputs["comp_ctx_lengths"] = torch.randint(0, 100, (40,), dtype=torch.long)
        if continuous_batching:
            lang_inputs["batch_index"] = torch.arange(bs).view(bs, 1)

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
                shape=("batch_size", "num_images", "img_tile", "img_size"),
            ),
            IOInfo(
                name="image_masks",
                datatype=torch.float32,
                shape=("batch_size", "num_images", "img_tile"),
            ),
            IOInfo(
                name="image_input_idx",
                datatype=torch.int32,
                shape=("batch_size", "num_images", "num_patches"),
            ),
            IOInfo(
                name="valid_idx",
                datatype=torch.int64,
                shape=("batch_size", "valid_size"),
            ),
        ]
