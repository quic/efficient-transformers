# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import math
from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import activations

try:
    from transformers.activations import PytorchGELUTanh
except ImportError:
    from transformers.activations import GELUTanh

    activations.PytorchGELUTanh = GELUTanh
    PytorchGELUTanh = GELUTanh
from transformers.activations import PytorchGELUTanh
from transformers.cache_utils import Cache
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast

from QEfficient.utils import constants


def eager_attention_forward(q, k, v, **kwargs):
    q_cu_seqlens = kwargs.get("q_cu_seqlens")
    seq_length = q.shape[0]
    if q_cu_seqlens is not None:
        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for idx in range(1, len(q_cu_seqlens)):
            attention_mask[
                ...,
                q_cu_seqlens[idx - 1] : q_cu_seqlens[idx],
                q_cu_seqlens[idx - 1] : q_cu_seqlens[idx],
            ] = True
    else:
        attention_mask = None

    q = q.transpose(0, 1)  # (num_heads, seq_len, head_dim)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if attention_mask is not None:
        attn_weight = attn_weight + attention_mask
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_out = attn_weight @ v
    attn_out = attn_out.transpose(0, 1)  # (seq_len, num_heads, head_dim)
    return attn_out.reshape(attn_out.shape[0], -1)


VL_VISION_ATTENTION_FUNCTIONS = {"eager": eager_attention_forward}


def get_rope_shape_decorate(func):
    _get_rope_shape_first_call_flag = set()

    def wrapper(org, interpolation_mode, shape):
        key = (org.requires_grad, torch.is_grad_enabled(), interpolation_mode)
        if key not in _get_rope_shape_first_call_flag:
            _get_rope_shape_first_call_flag.add(key)
            _ = func(org, interpolation_mode, shape=(64, 64))
        return func(org, interpolation_mode, shape)

    return wrapper


@get_rope_shape_decorate
def get_rope_shape(org, interpolation_mode, shape):
    return (
        F.interpolate(
            org.permute((2, 0, 1)).unsqueeze(0),
            size=shape,
            mode=interpolation_mode,
        )
        .squeeze(0)
        .permute((1, 2, 0))
        .flatten(end_dim=1)
    )


def apply_rope(xq, xk, freqs_cis):
    # Support both complex freqs (..., dim//2) and stacked real/imag (2, ..., dim//2)
    if torch.is_complex(freqs_cis):
        freqs_cis = freqs_cis.unsqueeze(-2)
        xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().view(*xk.shape[:-1], -1, 2))
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    # freqs_cis shape: (2, seq_len, dim//2)
    # xq shape: (seq_len, num_heads, head_dim)
    freqs_cos = freqs_cis[0].unsqueeze(-2)  # (seq_len, 1, dim//2)
    freqs_sin = freqs_cis[1].unsqueeze(-2)  # (seq_len, 1, dim//2)
    xq_r = xq.float().view(*xq.shape[:-1], -1, 2)
    xq_r0, xq_r1 = xq_r[..., 0], xq_r[..., 1]
    xq_out_r = xq_r0 * freqs_cos - xq_r1 * freqs_sin
    xq_out_i = xq_r0 * freqs_sin + xq_r1 * freqs_cos
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(-2)
    xk_r = xk.float().view(*xk.shape[:-1], -1, 2)
    xk_r0, xk_r1 = xk_r[..., 0], xk_r[..., 1]
    xk_out_r = xk_r0 * freqs_cos - xk_r1 * freqs_sin
    xk_out_i = xk_r0 * freqs_sin + xk_r1 * freqs_cos
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class QEffLearnable2DInterpPosEmbDivided_fixed(nn.Module):
    def __qeff_init__(self):
        self.interpolation_mode = "bilinear"


class Rope2DPosEmbRepeated(nn.Module):
    """2D rotary position embedding with multi-resolution support.
    This class is intended to be used in the following way:
    1. Before training, create an instance of Rope2DPosEmb. This instance will hold the precomputed cis.
    2. Before each forward pass, call `get_freqs_cis_by_*` to get the `freqs_cis` tensor for this iteration.
    3. During the forward pass, pass the `freqs_cis` tensor to each attention layer, and call `apply` just before each attention operation.
        The rope is shared across all attention layers and all heads.
    Refs:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - VisionLLaMA: https://arxiv.org/abs/2403.00522
    - https://github.com/Meituan-AutoML/VisionLLaMA/blob/main/dit/models.py
    Args:
        dim (int): usually the multi-head attention dimension, should be divisible by 4 (TODO: relax this constraint if needed)
        max_height (int): the maximum height of the 2D grid
        max_width (int): the maximum width of the 2D grid
        theta_base (float): the base of the theta
        device (str): the device to store the precomputed cis
    """

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base=10000):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def extra_repr(self):
        return f"dim={self.dim}, max_height={self.max_height}, max_width={self.max_width}, theta_base={self.theta_base}"

    def _ensure_precomputed_freqs(self, device: torch.device) -> None:
        if not hasattr(self, "freqs_cis"):
            self.register_buffer("freqs_cis", self._precompute_freqs_cis(device), persistent=False)
        elif self.freqs_cis.device != device:
            self.freqs_cis = self._precompute_freqs_cis(device)

        if not hasattr(self, "freqs_cos"):
            self.register_buffer("freqs_cos", self.freqs_cis.real.contiguous(), persistent=False)
        elif self.freqs_cos.device != device:
            self.freqs_cos = self.freqs_cis.real.contiguous()

        if not hasattr(self, "freqs_sin"):
            self.register_buffer("freqs_sin", self.freqs_cis.imag.contiguous(), persistent=False)
        elif self.freqs_sin.device != device:
            self.freqs_sin = self.freqs_cis.imag.contiguous()

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        """Calculate the cis(freqs) for each position in the 2D grid.
        Return: complex tensor of shape (max_height, max_width, dim//2) and value:
            height axis: ret[h, w, 2*i] = cis(h * theta_base**(-4*i/dim))
            weight axis: ret[h, w, 2*i+1] = cis(w * theta_base**(-4*i/dim))   with (i in [0, dim//4))
            note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,
        """
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(self, grid_thws: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Args:
            grid_thws (torch.Tensor): grid time, height and width
        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)
        """
        self._ensure_precomputed_freqs(device)

        shapes = grid_thws.tolist()
        assert all(1 <= h <= self.max_height and 1 <= w <= self.max_width for t, h, w in shapes), (
            shapes,
            self.max_height,
            self.max_width,
        )
        freqs_cis = torch.cat(
            [self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1) for t, h, w in shapes],
            dim=0,
        )
        return freqs_cis


class MLP2(nn.Module):
    """
    Args:
        dims: [in_dim, hidden_dim, out_dim]
        bias: whether to use bias in linear layer.
    """

    def __init__(self, dims: list[int], activation, bias=True):
        super().__init__()
        assert len(dims) == 3
        self.fc0 = nn.Linear(dims[0], dims[1], bias=bias)
        self.fc1 = nn.Linear(dims[1], dims[2], bias=bias)
        self.activation = activation
        for m in [self.fc0, self.fc1]:
            nn.init.trunc_normal_(m.weight, std=math.sqrt(2 / m.in_features))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc0(x)
        x = self.activation(x)
        return self.fc1(x)


class MoonViTEncoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        *,
        attn_implementation: str = "flash_attention_2",
        activation=F.gelu,
        attn_bias: bool = False,
        use_deterministic_attn: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size_per_attention_head = self.hidden_dim // self.num_heads
        self.attn_implementation = attn_implementation
        self.use_deterministic_attn = use_deterministic_attn

        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP2([hidden_dim, mlp_dim, hidden_dim], activation)
        self.wqkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=attn_bias)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)

    def attention_qkvpacked(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
    ):
        """
        Args:
            x (torch.Tensor): (batch_size, seqlen, hidden_dim)
            cu_seqlens (torch.Tensor):
        """
        xqkv = self.wqkv(x)

        qkv_shape = xqkv.size()[:-1] + (
            3,
            self.num_heads,
            self.hidden_size_per_attention_head,
        )
        # xqkv: (batch_size, seqlen, 3, nheads, headdim)
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)

        xq, xk = apply_rope(xq, xk, rope_freqs_cis)

        attn_func = VL_VISION_ATTENTION_FUNCTIONS[self.attn_implementation]
        attn_out = attn_func(
            xq,
            xk,
            xv,
            q_cu_seqlens=cu_seqlens,
            k_cu_seqlens=cu_seqlens,
            max_seqlen_k=max_seqlen,
            max_seqlen_q=max_seqlen,
            deterministic=self.use_deterministic_attn,
        )

        attn_out = self.wo(attn_out)
        return attn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rope_freqs_cis: torch.Tensor | None = None,
    ):
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)

        hidden_states = self.attention_qkvpacked(hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class QEffMoonViT3dEncoder(nn.Module):
    def __qeff_init__(self):
        if not hasattr(self, "blocks") or len(self.blocks) == 0:
            return

        old_blocks = list(self.blocks)
        first_block = old_blocks[0]
        self.block_cfg = {
            "num_heads": first_block.num_heads,
            "hidden_dim": first_block.hidden_dim,
            "mlp_dim": first_block.mlp.fc0.out_features,
            "activation": PytorchGELUTanh(),
            "attn_bias": first_block.wqkv.bias is not None,
            "attn_implementation": "eager",
        }

        head_dim = first_block.hidden_size_per_attention_head
        max_height = getattr(self.rope_2d, "max_height", 512)
        max_width = getattr(self.rope_2d, "max_width", 512)
        theta_base = getattr(self.rope_2d, "theta_base", 10000)
        self.rope_2d = Rope2DPosEmbRepeated(head_dim, max_height, max_width, theta_base=theta_base)

        new_blocks = []
        for old_block in old_blocks:
            new_block = MoonViTEncoderLayer(**self.block_cfg, use_deterministic_attn=False)
            new_block.load_state_dict(old_block.state_dict())
            new_blocks.append(new_block.to(device=old_block.wqkv.weight.device, dtype=old_block.wqkv.weight.dtype))
        self.blocks = nn.ModuleList(new_blocks)


class QEffKimiK25EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = self.model.config

    def get_submodules_for_export(self) -> Type[nn.Module]:
        """
        Return the set of class used as the repeated layer across the model for subfunction extraction.
        Notes:
            This method should return the *class object* (not an instance).
            Downstream code can use this to find/build subfunctions for repeated blocks.
        """
        return {self.model.vision_tower.encoder.blocks[0].__class__}

    def forward(self, pixel_values: torch.Tensor, h_shape: torch.Tensor, w_shape: torch.Tensor) -> torch.Tensor:
        """
        ONNX-exportable forward that runs only the vision tower and mm_projector.
        Uses h_shape and w_shape (int64 ones tensors of length h and w respectively)
        to derive spatial dimensions via .shape[0], enabling dynamic axis export.

        Args:
            pixel_values: Preprocessed image pixel values (num_patches, channels*patch_h*patch_w).
            h_shape: int64 ones tensor of shape (h,) encoding number of patch rows.
            w_shape: int64 ones tensor of shape (w,) encoding number of patch columns.

        Returns:
            vision_embeds: Projected image embeddings as a single tensor. For multiple
                same-sized images, embeddings are concatenated in input order.
        """

        h_shape = h_shape.to(pixel_values.device)
        w_shape = w_shape.to(pixel_values.device)

        # Keep them in ONNX graph
        dummy = (h_shape.float().sum() + w_shape.float().sum()) * 0.0
        pixel_values = pixel_values + dummy

        h = h_shape.shape[0]
        w = w_shape.shape[0]
        num_tokens_per_image = h * w
        if pixel_values.shape[0] % num_tokens_per_image != 0:
            raise ValueError(
                f"pixel_values first dimension ({pixel_values.shape[0]}) must be divisible by h*w ({num_tokens_per_image})."
            )
        num_images = pixel_values.shape[0] // num_tokens_per_image

        target_dtype = self.model.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.to(target_dtype)

        hidden_states = self.model.vision_tower.patch_embed.proj(pixel_values).view(pixel_values.size(0), -1)

        pos_emb_module = self.model.vision_tower.patch_embed.pos_emb
        pos_emb_2d = get_rope_shape(
            pos_emb_module.weight,
            interpolation_mode=pos_emb_module.interpolation_mode,
            shape=(h, w),
        )
        hidden_states = hidden_states + pos_emb_2d.repeat(num_images, 1)

        rope_2d = self.model.vision_tower.encoder.rope_2d
        rope_2d._ensure_precomputed_freqs(pixel_values.device)
        freqs_cos = rope_2d.freqs_cos[:h, :w].reshape(-1, rope_2d.dim // 2).repeat(num_images, 1)
        freqs_sin = rope_2d.freqs_sin[:h, :w].reshape(-1, rope_2d.dim // 2).repeat(num_images, 1)
        freqs_cis = torch.stack([freqs_cos, freqs_sin], dim=0)

        if num_images == 1:
            cu_seqlens = None
        else:
            lengths = torch.full(
                (num_images + 1,), num_tokens_per_image, dtype=torch.int64, device=hidden_states.device
            )
            lengths[0] = 0
            cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int64)
        max_seqlen = num_tokens_per_image

        encoder_dtype = self.model.vision_tower.encoder.blocks[0].wqkv.weight.dtype
        hidden_states = hidden_states.to(encoder_dtype)
        freqs_cis = freqs_cis.to(encoder_dtype)
        for block in self.model.vision_tower.encoder.blocks:
            hidden_states = block(hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis=freqs_cis)

        final_ln_dtype = self.model.vision_tower.encoder.final_layernorm.weight.dtype
        hidden_states = self.model.vision_tower.encoder.final_layernorm(hidden_states.to(final_ln_dtype))

        merge_kernel_size = self.model.vision_tower.merge_kernel_size
        kernel_height, kernel_width = merge_kernel_size
        new_height = h // kernel_height
        new_width = w // kernel_width
        d_model = hidden_states.size(-1)
        reshaped = hidden_states.view(num_images, new_height, kernel_height, new_width, kernel_width, d_model)
        reshaped = reshaped.permute(0, 1, 3, 2, 4, 5).contiguous()
        merged = reshaped.view(num_images * new_height * new_width, kernel_height * kernel_width, -1)

        pre_norm_dtype = self.model.mm_projector.pre_norm.weight.dtype
        merged = merged.to(pre_norm_dtype)
        vision_embeds = self.model.mm_projector.proj(self.model.mm_projector.pre_norm(merged).view(merged.shape[0], -1))
        return vision_embeds


class QEffKimiK25DecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.language_model = self.model.language_model
        self.lm_head = self.language_model.lm_head
        self.config = self.model.config

    def get_submodules_for_export(self) -> Type[nn.Module]:
        """
        Return the set of class used as the repeated layer across the model for subfunction extraction.
        Notes:
            This method should return the *class object* (not an instance).
            Downstream code can use this to find/build subfunctions for repeated blocks.
        """
        return {self.language_model.model.layers[0].__class__}

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_embeds: Optional[torch.FloatTensor] = None,
        image_idx: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        compressed_kvs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        batch_index: Optional[torch.LongTensor] = None,
        comp_ctx_lengths: Optional[List[int]] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Tuple:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        vision_embeds_for_state = None
        if vision_embeds is not None:
            if vision_embeds.dim() == 3:
                if vision_embeds.shape[0] != 1:
                    raise ValueError(
                        f"Expected vision_embeds batch dim to be 1, got shape {tuple(vision_embeds.shape)}"
                    )
                vision_embeds_for_state = vision_embeds[0].to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            elif vision_embeds.dim() == 2:
                vision_embeds_for_state = vision_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            else:
                raise ValueError(f"Expected vision_embeds rank 2 or 3, got {vision_embeds.dim()}.")

        if vision_embeds_for_state is not None and input_ids is not None:
            if attention_mask is None:
                if position_ids is None:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
                else:
                    attention_mask = (position_ids >= 0).to(dtype=torch.long, device=input_ids.device)
            if image_idx is None:
                image_idx = torch.zeros((input_ids.shape[0], 1), dtype=torch.int64, device=input_ids.device)
            selected = input_ids == self.config.media_placeholder_token_id
            selected_any = selected.any(dim=1, keepdim=True)
            selected_image_tokens = selected.to(torch.int64).sum(dim=1, keepdim=True)
            inputs_embeds = inputs_embeds.to(vision_embeds_for_state.dtype)

            merged_inputs_embeds, merged_attention_mask, _, merged_position_ids = (
                self.model._qeff_merge_single_image_symbolic(
                    image_features=vision_embeds_for_state,
                    inputs_embeds=inputs_embeds,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None,
                )
            )

            inputs_embeds = merged_inputs_embeds
            attention_mask = merged_attention_mask
            if position_ids is None:
                position_ids = merged_position_ids
            else:
                # Preserve caller-provided absolute position offset (needed for
                # chunked prefill/decode parity) while using merged sequence
                # positions for image-expanded tokens.
                position_offset = position_ids[:, :1] + image_idx.to(
                    device=position_ids.device, dtype=position_ids.dtype
                )
                position_ids = torch.where(
                    merged_attention_mask > 0,
                    merged_position_ids + position_offset,
                    torch.full_like(merged_position_ids, -1),
                )

            merged_image_tokens = (
                torch._shape_as_tensor(vision_embeds_for_state)[:1]
                .view(1, 1)
                .to(device=image_idx.device, dtype=torch.int64)
            )
            image_position_delta = torch.clamp(merged_image_tokens - selected_image_tokens, min=0)
            image_idx = image_idx + selected_any.to(torch.int64) * image_position_delta

        if position_ids is None and attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = torch.where(attention_mask > 0, position_ids, torch.full_like(position_ids, -1))
        elif position_ids is not None and attention_mask is not None:
            position_ids = torch.where(attention_mask > 0, position_ids, torch.full_like(position_ids, -1))

        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            compressed_kvs=compressed_kvs,
            past_key_values=past_key_values,
            batch_index=batch_index,
            comp_ctx_lengths=comp_ctx_lengths,
            use_cache=True if use_cache is None else use_cache,
            **kwargs,
        )
        logits = outputs[0].float()

        mla_absorption = getattr(self.model, "mla_absorption", None)
        if mla_absorption is not None:
            cache_compressed = mla_absorption.get("cache_compressed", False)
        else:
            cache_compressed = False

        if cache_compressed:
            output_kvs = getattr(outputs, "compressed_kvs", None)
        else:
            output_kvs = getattr(outputs, "past_key_values", None)
        return logits, vision_embeds_for_state, image_idx, output_kvs


# ref https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/llava/modeling_llava.py#L240
class QEffKimiK25ForConditionalGeneration(nn.Module):
    def get_qeff_vision_encoder(self):
        return QEffKimiK25EncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEffKimiK25DecoderWrapper(self)

    def _qeff_merge_input_ids_with_image_features(
        self,
        image_features: list[torch.Tensor],
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        """
        Args:
            image_features (:obj:`torch.Tensor` of shape :obj:`(num_image_tokens, embed_dim)`):
                The image features to merge with the input embeddings.
            inputs_embeds (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length, embed_dim)`):
                The input embeddings.
            input_ids (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`):
                The input ids.
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`):
                The attention mask.
            labels (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, *optional*):
                The labels.
        """
        if len(image_features) == 0:
            raise ValueError("At least one image_features tensor is required.")

        _, embed_dim = image_features[0].shape
        feature_lengths = [x.shape[0] for x in image_features]
        image_features_cat = torch.cat(image_features, dim=0)

        image_token_index: int = self.config.media_placeholder_token_id
        pad_token_id: int = self.config.pad_token_id

        batch_size, _ = input_ids.shape
        target_device = inputs_embeds.device

        left_padding = (~(input_ids[:, -1] == pad_token_id)).sum() == 0

        image_token_mask = input_ids == image_token_index
        non_image_mask = ~image_token_mask

        flat_image_token_mask = image_token_mask.reshape(-1).to(torch.long)
        flat_image_order = torch.cumsum(flat_image_token_mask, dim=0)
        feature_lengths_tensor = torch.tensor(feature_lengths, dtype=input_ids.dtype, device=input_ids.device)
        flat_image_lengths = torch.zeros_like(flat_image_order, dtype=input_ids.dtype)
        for idx in range(len(feature_lengths)):
            flat_image_lengths = (
                flat_image_lengths + (flat_image_order == (idx + 1)).to(input_ids.dtype) * (feature_lengths_tensor[idx])
            )

        token_occupation_table = torch.ones_like(input_ids)
        token_occupation_table = token_occupation_table.reshape(-1)
        token_occupation_table = torch.where(flat_image_token_mask.bool(), flat_image_lengths, token_occupation_table)
        token_occupation_table = token_occupation_table.reshape(input_ids.shape)

        max_embed_dim = int(token_occupation_table.sum(-1).max().item())

        new_token_positions = torch.cumsum(token_occupation_table, 1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if bool(left_padding):
            new_token_positions = new_token_positions + nb_image_pad[:, None]

        merged_positions = torch.arange(max_embed_dim, device=input_ids.device).view(1, 1, -1)
        text_position_one_hot = merged_positions == new_token_positions.unsqueeze(-1)
        text_position_one_hot = torch.logical_and(text_position_one_hot, non_image_mask.unsqueeze(-1))

        final_embedding = (
            text_position_one_hot.to(inputs_embeds.dtype).unsqueeze(-1) * inputs_embeds.unsqueeze(2)
        ).sum(dim=1)
        final_attention_mask = (text_position_one_hot.to(attention_mask.dtype) * attention_mask.unsqueeze(-1)).sum(
            dim=1
        )

        image_to_overwrite = torch.logical_not(text_position_one_hot.any(dim=1))
        image_to_overwrite_cumsum = torch.cumsum(image_to_overwrite.to(torch.int64), dim=1)
        image_to_overwrite = torch.logical_and(
            image_to_overwrite,
            image_to_overwrite_cumsum - 1 >= nb_image_pad[:, None].to(target_device),
        )

        image_slot_ids = torch.cumsum(image_to_overwrite.to(torch.int64), dim=1) - 1
        if image_features_cat.shape[0] > 0:
            valid_image_slots = torch.logical_and(image_to_overwrite, image_slot_ids < image_features_cat.shape[0])
            image_slot_ids = torch.clamp(image_slot_ids, min=0, max=image_features_cat.shape[0] - 1)
            gathered_image_embeddings = image_features_cat.to(target_device)[image_slot_ids]
            final_embedding = torch.where(valid_image_slots.unsqueeze(-1), gathered_image_embeddings, final_embedding)
            image_to_overwrite = valid_image_slots
        else:
            image_to_overwrite = torch.zeros_like(image_to_overwrite)

        final_attention_mask = torch.logical_or(final_attention_mask.bool(), image_to_overwrite).to(
            final_attention_mask.dtype
        )

        position_ids = torch.cumsum(final_attention_mask, dim=1) - 1
        position_ids = torch.where(final_attention_mask == 0, torch.ones_like(position_ids), position_ids)

        pad_token_mask = input_ids == pad_token_id
        pad_position_one_hot = torch.logical_and(
            merged_positions == new_token_positions.unsqueeze(-1), pad_token_mask.unsqueeze(-1)
        )
        pad_positions_mask = pad_position_one_hot.any(dim=1)
        final_embedding = torch.where(
            pad_positions_mask.unsqueeze(-1), torch.zeros_like(final_embedding), final_embedding
        )

        return final_embedding, final_attention_mask, labels, position_ids

    def _qeff_merge_single_image_symbolic(
        self,
        image_features: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        image_token_index: int = self.config.media_placeholder_token_id
        pad_token_id: int = self.config.pad_token_id

        target_device = inputs_embeds.device
        image_features = image_features.to(target_device)
        image_shape = torch._shape_as_tensor(image_features).to(device=input_ids.device, dtype=input_ids.dtype)
        num_image_tokens = image_shape[0]

        image_token_mask = input_ids == image_token_index
        non_image_mask = ~image_token_mask
        image_token_count = image_token_mask.to(input_ids.dtype).sum(dim=1, keepdim=True)
        has_image = image_token_count > 0

        token_occupation = torch.where(
            image_token_mask,
            num_image_tokens.view(1, 1).expand_as(input_ids),
            torch.ones_like(input_ids),
        )
        new_token_positions = torch.cumsum(token_occupation, dim=1) - 1

        final_seq_len = input_ids.shape[1] + image_features.shape[0] - 1
        merged_positions = torch.arange(final_seq_len, device=input_ids.device, dtype=input_ids.dtype).view(1, 1, -1)
        text_position_one_hot = merged_positions == new_token_positions.unsqueeze(-1)
        text_position_one_hot = torch.logical_and(text_position_one_hot, non_image_mask.unsqueeze(-1))

        final_embedding = (
            text_position_one_hot.to(inputs_embeds.dtype).unsqueeze(-1) * inputs_embeds.unsqueeze(2)
        ).sum(dim=1)
        final_attention_mask = (text_position_one_hot.to(attention_mask.dtype) * attention_mask.unsqueeze(-1)).sum(
            dim=1
        )

        image_start_positions = torch.where(
            image_token_mask,
            new_token_positions - num_image_tokens.view(1, 1) + 1,
            torch.zeros_like(new_token_positions),
        )
        image_start = image_start_positions.max(dim=1, keepdim=True).values
        image_positions = merged_positions.squeeze(1) - image_start
        max_image_index = num_image_tokens.view(1, 1) - 1
        safe_image_positions = torch.minimum(torch.clamp(image_positions, min=0), max_image_index)
        image_slots = torch.logical_and(image_positions >= 0, image_positions < num_image_tokens.view(1, 1))
        image_slots = torch.logical_and(image_slots, has_image)
        image_slots = torch.logical_and(image_slots, torch.logical_not(text_position_one_hot.any(dim=1)))

        gathered_image_embeddings = image_features[safe_image_positions.to(torch.long)]
        final_embedding = torch.where(image_slots.unsqueeze(-1), gathered_image_embeddings, final_embedding)
        final_attention_mask = torch.logical_or(final_attention_mask.bool(), image_slots).to(final_attention_mask.dtype)

        position_ids = torch.cumsum(final_attention_mask, dim=1) - 1
        position_ids = torch.where(final_attention_mask == 0, torch.full_like(position_ids, -1), position_ids)

        pad_token_mask = input_ids == pad_token_id
        pad_position_one_hot = torch.logical_and(
            merged_positions == new_token_positions.unsqueeze(-1), pad_token_mask.unsqueeze(-1)
        )
        pad_positions_mask = pad_position_one_hot.any(dim=1)
        final_embedding = torch.where(
            pad_positions_mask.unsqueeze(-1), torch.zeros_like(final_embedding), final_embedding
        )

        return final_embedding, final_attention_mask, labels, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | list[torch.FloatTensor] | None = None,
        grid_thws: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        compressed_kvs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | LlavaCausalLMOutputWithPast:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        ```"""
        assert self.vision_tower is not None, "vision_tower is not loaded"
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and len(pixel_values) > 0 and input_ids.shape[1] != 1:
                image_features = self._extract_image_features(pixel_values, grid_thws)
                if self.mm_projector:
                    image_features = self.mm_projector(image_features)

                inputs_embeds = inputs_embeds.to(image_features[0].dtype)  # num_tokens, embed_dim
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features,
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                    labels,
                )

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            compressed_kvs=compressed_kvs,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_output_names(self, kv_offload: bool = False):
        vision_output_names = ["vision_embeds"]
        lang_output_names = ["logits"]

        mla_absorption = getattr(self.language_model, "mla_absorption", None)
        if mla_absorption is not None:
            cache_compressed = mla_absorption.get("cache_compressed", False)
        else:
            cache_compressed = False

        if cache_compressed:
            for i in range(self.language_model.config.num_hidden_layers):
                lang_output_names.append(f"compressed_kv.{i}_RetainedState")
                lang_output_names.append(f"k_pe.{i}_RetainedState")
        else:
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

    def get_dummy_inputs(
        self,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        **kwargs,
    ):
        prefill_seq_len = kwargs.get("prefill_seq_len")
        if prefill_seq_len is None:
            prefill_seq_len = constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
        prefill_seq_len = int(prefill_seq_len)

        bs: int = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        fbs: int = constants.ONNX_EXPORT_EXAMPLE_FBS

        inputs_shapes = {}
        inputs_shapes["pixel_values"] = (
            constants.KIMI_NUM_PATCHES,
            constants.ONNX_EXPORT_IMAGE_DEPTH,
            constants.KIMI_PATCH_SIZE,
            constants.KIMI_PATCH_SIZE,
        )
        inputs_shapes["vision_embeds"] = (
            constants.KIMI_NUM_IMAGE_TOKENS,
            self.language_model.config.hidden_size,
        )
        inputs_shapes["image_idx"] = (1, 1)
        inputs_shapes["h"] = constants.KIMI_IMAGE_HEIGHT
        inputs_shapes["w"] = constants.KIMI_IMAGE_WIDTH

        vision_inputs = {
            "pixel_values": torch.zeros((inputs_shapes["pixel_values"]), dtype=self.config.torch_dtype),
            "h_shape": torch.zeros((inputs_shapes["h"]), dtype=torch.int64),
            "w_shape": torch.zeros((inputs_shapes["w"]), dtype=torch.int64),
        }

        input_ids = torch.zeros((bs, prefill_seq_len), dtype=torch.int64)
        input_ids[:, 0] = self.config.media_placeholder_token_id

        lang_inputs = {
            "input_ids": input_ids,
            "position_ids": torch.arange(prefill_seq_len, dtype=torch.int64).view(1, prefill_seq_len).repeat(bs, 1),
        }

        mla_absorption = getattr(self.language_model, "mla_absorption", None)
        if mla_absorption is not None:
            cache_compressed = mla_absorption.get("cache_compressed", False)
        else:
            cache_compressed = False

        pkv_cache = self.language_model.get_dummy_pkv_cache(
            config=self.language_model.config,
            batch_size=fbs if continuous_batching else bs,
            seq_len=constants.ONNX_EXPORT_CTX_LEN,
        )

        if cache_compressed:
            lang_inputs["compressed_kvs"] = [[] for _ in range(self.language_model.config.num_hidden_layers)]
            for i in range(self.language_model.config.num_hidden_layers):
                lang_inputs["compressed_kvs"][i].append(
                    torch.zeros(pkv_cache[0][0].shape, dtype=self.language_model.config.torch_dtype)
                )
                lang_inputs["compressed_kvs"][i].append(
                    torch.zeros(pkv_cache[0][1].shape, dtype=self.language_model.config.torch_dtype)
                )
        else:
            lang_inputs["past_key_values"] = [[] for _ in range(self.language_model.config.num_hidden_layers)]
            for i in range(self.language_model.config.num_hidden_layers):
                lang_inputs["past_key_values"][i].append(
                    torch.zeros(pkv_cache[0][0].shape, dtype=self.language_model.config.torch_dtype)
                )
                lang_inputs["past_key_values"][i].append(
                    torch.zeros(pkv_cache[0][1].shape, dtype=self.language_model.config.torch_dtype)
                )

        lang_inputs["vision_embeds"] = torch.zeros(
            inputs_shapes["vision_embeds"],
            dtype=self.language_model.config.torch_dtype,
        )
        lang_inputs["image_idx"] = torch.zeros(inputs_shapes["image_idx"], dtype=torch.int64)

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

    def get_onnx_dynamic_axes(
        self, comp_ctx_lengths: Optional[List[int]] = None, kv_offload: bool = False, continuous_batching: bool = False
    ):
        vision_dynamic_axes = {}
        lang_dynamic_axes = {}
        lang_dynamic_axes["input_ids"] = {0: "batch_size", 1: "seq_len"}
        lang_dynamic_axes["position_ids"] = {0: "batch_size", 1: "seq_len"}
        lang_dynamic_axes["vision_embeds"] = {0: "num_image_tokens"}
        if continuous_batching:
            lang_dynamic_axes["batch_index"] = {0: "batch_size"}
        vision_dynamic_axes = {
            "pixel_values": {0: "num_patches"},
            "h_shape": {0: "h"},
            "w_shape": {0: "w"},
            "vision_embeds": {0: "num_image_tokens"},
        }

        mla_absorption = getattr(self.language_model, "mla_absorption", None)
        if mla_absorption is not None:
            cache_compressed = mla_absorption.get("cache_compressed", False)
        else:
            cache_compressed = False

        cache_batch_axis = "full_batch_size" if continuous_batching else "batch_size"

        if cache_compressed:
            for i in range(self.language_model.config.num_hidden_layers):
                lang_dynamic_axes[f"compressed_kv.{i}"] = {0: cache_batch_axis, 2: "ctx_len"}
                lang_dynamic_axes[f"k_pe.{i}"] = {0: cache_batch_axis, 2: "ctx_len"}
        else:
            for i in range(self.language_model.config.num_hidden_layers):
                for kv in ["key", "value"]:
                    lang_dynamic_axes[f"past_{kv}.{i}"] = {0: cache_batch_axis, 2: "ctx_len"}

        if comp_ctx_lengths is not None:
            lang_dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}

        dynamic_axes = {}
        if kv_offload:
            dynamic_axes["vision"] = vision_dynamic_axes
            dynamic_axes["lang"] = lang_dynamic_axes
        else:
            lang_dynamic_axes.pop("vision_embeds")
            dynamic_axes = {**vision_dynamic_axes, **lang_dynamic_axes}
        return dynamic_axes

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
        **compiler_options,
    ):
        comp_ctx_lengths_prefill = compiler_options.pop("comp_ctx_lengths_prefill", None)
        comp_ctx_lengths_decode = compiler_options.pop("comp_ctx_lengths_decode", None)
        compiler_options.pop("img_size", None)
        num_patches = compiler_options.pop("num_patches", None)
        height = compiler_options.pop("height", None)
        width = compiler_options.pop("width", None)
        h = compiler_options.pop("h", None)
        w = compiler_options.pop("w", None)
        num_frames = compiler_options.pop("num_frames", 1)
        num_image_tokens = compiler_options.pop("num_image_tokens", None)
        mm_processor_kwargs = compiler_options.pop("mm_processor_kwargs", None) or {}

        prefill_seq_len = prefill_seq_len if prefill_seq_len else constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
        ctx_len = ctx_len if ctx_len else constants.ONNX_EXPORT_CTX_LEN

        def normalize_list(value, default):
            if value is None:
                return [default]
            if isinstance(value, int):
                return [value]
            return list(value)

        def normalize_sized_list(value, count, name):
            if value is None:
                return None
            if isinstance(value, int):
                return [value] * count
            value = list(value)
            if len(value) != count:
                raise ValueError(f"Expected {name} to contain {count} entries, got {len(value)}.")
            return value

        def validate_dimension_lists(heights, widths, height_name, width_name):
            if len(heights) != len(widths):
                raise ValueError(
                    f"Expected {height_name} and {width_name} to contain the same number of entries, "
                    f"got {len(heights)} and {len(widths)}."
                )

        patch_size = getattr(self.config.vision_config, "patch_size", constants.KIMI_PATCH_SIZE)
        merge_kernel_size = getattr(self.config.vision_config, "merge_kernel_size", (2, 2))
        if isinstance(merge_kernel_size, int):
            kernel_height = kernel_width = merge_kernel_size
            merge_kernel_size = (merge_kernel_size, merge_kernel_size)
        else:
            kernel_height, kernel_width = merge_kernel_size

        if h is not None or w is not None:
            heights = normalize_list(h, constants.KIMI_IMAGE_HEIGHT)
            widths = normalize_list(w, constants.KIMI_IMAGE_WIDTH)
            validate_dimension_lists(heights, widths, "h", "w")
        elif height is not None or width is not None:
            pixel_heights = normalize_list(height, constants.KIMI_IMAGE_HEIGHT * patch_size)
            pixel_widths = normalize_list(width, constants.KIMI_IMAGE_WIDTH * patch_size)
            validate_dimension_lists(pixel_heights, pixel_widths, "height", "width")

            in_patch_limit = mm_processor_kwargs.get("in_patch_limit", 16384)
            patch_limit_on_one_side = mm_processor_kwargs.get("patch_limit_on_one_side", 512)
            factor_height = kernel_height * patch_size
            factor_width = kernel_width * patch_size
            heights = []
            widths = []
            for pixel_height, pixel_width in zip(pixel_heights, pixel_widths):
                scale = min(
                    1.0,
                    math.sqrt(
                        in_patch_limit / (max(1.0, pixel_width // patch_size) * max(1.0, pixel_height // patch_size))
                    ),
                    patch_limit_on_one_side * patch_size / pixel_width,
                    patch_limit_on_one_side * patch_size / pixel_height,
                )
                resized_height = min(max(1, int(pixel_height * scale)), patch_limit_on_one_side * patch_size)
                resized_width = min(max(1, int(pixel_width * scale)), patch_limit_on_one_side * patch_size)
                pad_height = (factor_height - resized_height % factor_height) % factor_height
                pad_width = (factor_width - resized_width % factor_width) % factor_width
                heights.append((resized_height + pad_height) // patch_size)
                widths.append((resized_width + pad_width) // patch_size)
        else:
            heights = [constants.KIMI_IMAGE_HEIGHT]
            widths = [constants.KIMI_IMAGE_WIDTH]

        num_frames = normalize_sized_list(1 if num_frames is None else num_frames, len(heights), "num_frames")
        explicit_num_patches = normalize_sized_list(num_patches, len(heights), "num_patches")
        explicit_num_image_tokens = normalize_sized_list(num_image_tokens, len(heights), "num_image_tokens")

        vision = []
        max_num_image_tokens = 0
        for index, (height, width, frames) in enumerate(zip(heights, widths, num_frames)):
            if height % kernel_height != 0 or width % kernel_width != 0:
                raise ValueError(
                    f"Kimi image grid h={height}, w={width} must be divisible by merge_kernel_size={merge_kernel_size}."
                )

            computed_num_patches = height * width * frames
            computed_num_image_tokens = (height // kernel_height) * (width // kernel_width) * frames
            resolved_num_patches = (
                explicit_num_patches[index] if explicit_num_patches is not None else computed_num_patches
            )
            resolved_num_image_tokens = (
                explicit_num_image_tokens[index] if explicit_num_image_tokens is not None else computed_num_image_tokens
            )
            max_num_image_tokens = max(max_num_image_tokens, resolved_num_image_tokens)

            vision.append(
                {
                    "num_patches": resolved_num_patches,
                    "h": height,
                    "w": width,
                    "num_image_tokens": resolved_num_image_tokens,
                }
            )

        if comp_ctx_lengths_prefill is not None:
            lang = []

            for i in range(0, len(comp_ctx_lengths_prefill)):
                lang_prefill = {
                    "batch_size": 1 if continuous_batching else batch_size,
                    "seq_len": prefill_seq_len,
                    "ctx_len": ctx_len,
                    "num_image_tokens": max_num_image_tokens,
                }
                if continuous_batching:
                    lang_prefill["full_batch_size"] = kv_cache_batch_size
                else:
                    lang_prefill["batch_size"] = kv_cache_batch_size
                if full_batch_size:
                    lang_prefill["full_batch_exec_size"] = full_batch_size

                lang.append(lang_prefill)

            for i in range(0, len(comp_ctx_lengths_decode)):
                lang_decode = {
                    "batch_size": full_batch_size if continuous_batching else batch_size,
                    "seq_len": "1",
                    "ctx_len": ctx_len,
                    "num_image_tokens": max_num_image_tokens,
                }

                if continuous_batching:
                    lang_decode["full_batch_size"] = kv_cache_batch_size
                else:
                    lang_decode["batch_size"] = kv_cache_batch_size

                lang.append(lang_decode)

        else:
            lang_prefill = {
                "batch_size": 1 if continuous_batching else batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "num_image_tokens": max_num_image_tokens,
            }
            if continuous_batching:
                lang_prefill["full_batch_size"] = kv_cache_batch_size
            else:
                lang_prefill["batch_size"] = kv_cache_batch_size
            if full_batch_size:
                lang_prefill["full_batch_exec_size"] = full_batch_size

            lang_decode = {
                "batch_size": full_batch_size if continuous_batching else batch_size,
                "seq_len": 1,
                "ctx_len": ctx_len,
                "num_image_tokens": max_num_image_tokens,
            }

            if continuous_batching:
                lang_decode["full_batch_size"] = kv_cache_batch_size
            else:
                lang_decode["batch_size"] = kv_cache_batch_size

            lang = [lang_prefill, lang_decode]

        specializations = {}

        if kv_offload:
            specializations["vision"] = vision
            specializations["lang"] = lang
            return specializations, compiler_options
        else:
            return [{**vision_spec, **lang_spec} for vision_spec in vision for lang_spec in lang], compiler_options
