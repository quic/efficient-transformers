# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import math
import sys as _sys
from collections.abc import Sequence
from copy import deepcopy
from io import BytesIO
from pathlib import Path

# from QEfficient import QEFFAutoModelForImageTextToText
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, activations

from QEfficient.transformers.models.deepseek_v3.modeling_deepseek import QEffDeepseekV3ForCausalLM

try:
    from transformers.activations import PytorchGELUTanh
except ImportError:
    from transformers.activations import GELUTanh

    activations.PytorchGELUTanh = GELUTanh
    PytorchGELUTanh = GELUTanh
from transformers.activations import PytorchGELUTanh
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast

from QEfficient.utils import constants

from .configuration_kimi_k25 import KimiK25Config


def eager_attention_forward(q, k, v, **kwargs):
    q = q.transpose(0, 1)  # (num_heads, seq_len, head_dim)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
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


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    From:
    https://github.com/OpenGVLab/InternVideo/blob/421f6d2361fc8f61a3394244571f2601a4e99e29/InternVideo2/multi_modality/models/backbones/internvideo2/pos_embed.py#L86
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, t_size, cls_token=False):
    """
    t_size: int of the temporal size
    return:
    pos_embed: [t_size, embed_dim] or [1+t_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class QEffLearnable2DInterpPosEmbDivided_fixed(nn.Module):
    def __qeff_init__(self):
        self.interpolation_mode = "bilinear"

    """def __qeff_init__(self,
                 height: int,
                 width: int,
                 num_frames: int,
                 dim: int,
                 interpolation_mode: str = 'bilinear') -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        self.register_buffer('time_weight',
                             torch.from_numpy(
                                 get_1d_sincos_pos_embed(
                                     self.dim,
                                     self.num_frames)).float().unsqueeze(1),
                             persistent=False)

        self.reset_parameters()
    """

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for t, h, w in grid_thws.tolist():
            assert t <= self.num_frames, f"t:{t} > self.num_frames:{self.num_frames}"
            if (h, w) == self.weight.shape[:-1]:
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = get_rope_shape(
                    self.weight,
                    interpolation_mode=self.interpolation_mode,
                    shape=(h, w),
                )
            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[0:t]

            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))

        out = x + torch.cat(pos_embs)
        return out


class MoonVision3dPatchEmbed(nn.Module):
    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: int | tuple[int, int] = (14, 14),
        pos_emb_height: int = 14,
        pos_emb_width: int = 14,
        pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
    ):
        super().__init__()
        assert isinstance(patch_size, int | Sequence), f"Invalid patch_size type: {type(patch_size)}"
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert len(patch_size) == 2, f"Expected patch_size to be a tuple of 2, got {patch_size}"
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=patch_size)

        if pos_emb_type == "divided_fixed":
            self.pos_emb = Learnable2DInterpPosEmbDivided_fixed(
                height=pos_emb_height, width=pos_emb_width, num_frames=pos_emb_time, dim=out_dim
            )
        else:
            raise NotImplementedError(f"Not support pos_emb_type: {pos_emb_type}")

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (L, Channels): input tensor
            grid_hws (N, 3): temporal, height and width
        Returns:
            (L, Cout) tensor
        """
        x = self.proj(x).view(x.size(0), -1)
        # apply positional embedding
        x = self.pos_emb(x, grid_thws)
        return x


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

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thws: torch.Tensor,
    ) -> torch.Tensor:
        # if not hasattr(self.rope_2d, 'freqs_cos'):
        #    self.rope_2d.register_buffer('freqs_cos', self.rope_2d.freqs_cis.real.contiguous(), persistent=False)
        #    self.rope_2d.register_buffer('freqs_sin', self.rope_2d.freqs_cis.imag.contiguous(), persistent=False)
        rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_thws=grid_thws, device=hidden_states.device)

        lengths = torch.cat(
            (
                torch.zeros(1, dtype=grid_thws.dtype, device=grid_thws.device),
                grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
            )
        )

        max_seqlen = lengths.max()
        cu_seqlens = lengths.to(hidden_states.device).cumsum(dim=0, dtype=torch.int32)
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis=rope_freqs_cis)

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


def tpool_patch_merger(
    x: torch.Tensor,
    grid_thws: torch.Tensor,
    merge_kernel_size: tuple[int, int] = (2, 2),
) -> list[torch.Tensor]:
    d_model = x.size(-1)

    outputs = []
    pre_sum = 0
    for t, h, w in grid_thws.tolist():
        # Get the current sequence
        seq = x[pre_sum : pre_sum + t * h * w]
        # Reshape along self.merge_kernel_size and concat to the last dimension
        kernel_height, kernel_width = merge_kernel_size
        new_height, new_width = h // kernel_height, w // kernel_width
        reshaped_seq = seq.view(t, new_height, kernel_height, new_width, kernel_width, d_model)
        reshaped_seq = reshaped_seq.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)  # temporal pooling
        padded_seq = reshaped_seq.view(new_height * new_width, kernel_height * kernel_width, -1)
        outputs.append(padded_seq)
        pre_sum += t * h * w

    return outputs


class MoonViT3dPretrainedModel(PreTrainedModel):
    config_class = None
    model_type = "moonvit3d"
    _no_split_modules = ["PackingTransformer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        config = deepcopy(config)
        self.merge_kernel_size = config.merge_kernel_size
        self.patch_size = config.patch_size
        self.merge_type = config.merge_type

        self.patch_embed = MoonVision3dPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
            pos_emb_time=config.init_pos_emb_time,
            pos_emb_type=config.pos_emb_type,
        )

        self.encoder = MoonViT3dEncoder(
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            block_cfg={
                "num_heads": config.num_attention_heads,
                "hidden_dim": config.hidden_size,
                "mlp_dim": config.intermediate_size,
                "activation": PytorchGELUTanh(),
                "attn_bias": True,
                "attn_implementation": config._attn_implementation,
            },
            video_attn_type=config.video_attn_type,
        )

    def forward(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values (torch.Tensor): The input pixel values.
            grid_thws (torch.Tensor): Temporal, height and width.
        Returns:
            torch.Tensor: The output tokens.
        """
        # grid_thws = grid_thws.to('cpu')
        assert grid_thws.ndim == 2, f"grid_thws should be 2D, got {grid_thws.ndim}"
        assert grid_thws.size(1) == 3, f"No support for thw: {grid_thws}"
        hidden_states = self.patch_embed(pixel_values, grid_thws)
        hidden_states = self.encoder(hidden_states, grid_thws)
        if self.merge_type == "sd2_tpool":  # spatial downsampling 2x with temporal pooling all
            hidden_states = tpool_patch_merger(hidden_states, grid_thws, merge_kernel_size=self.merge_kernel_size)
        else:
            raise NotImplementedError(f"Not support {self.merge_type}")

        return hidden_states


# ============================================================================
# MM Projector Helper Classes (from mm_projector/modeling_mm_projectors.py)
# ============================================================================


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO, use faster LayerNorm
        self.pre_norm = nn.LayerNorm(config.mm_hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(config.mm_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, x, *args, **kwargs):
        assert isinstance(x, list | tuple), f"x is not a list or tuple: {type(x)}"
        lengths = [item.shape[0] for item in x]
        x = torch.cat(x, dim=0)
        x = self.pre_norm(x)
        x = self.proj(x)
        x = torch.split(x, lengths, dim=0)

        return x


class PatchMergerMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        eps = config.projector_ln_eps
        self.hidden_size = config.mm_hidden_size * (config.merge_kernel_size[0] * config.merge_kernel_size[1])
        self.pre_norm = nn.LayerNorm(config.mm_hidden_size, eps=eps)
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, config.hidden_size),
        )

    def forward(self, x, *args, **kwargs):
        if isinstance(x, list) or isinstance(x, tuple):
            x = [self.proj(self.pre_norm(item).view(item.shape[0], -1)) for item in x]
        else:
            # B, N, N_k, C = x.shape
            B = x.shape[0]
            x = self.proj(self.pre_norm(x).view(B, -1, self.hidden_size))
        return x


class KimiK25PreTrainedModel(PreTrainedModel):
    config_class = KimiK25Config
    base_model_prefix = "model"
    _no_split_modules = [
        "MoonViT3dPretrainedModel",
        "MoonViTEncoderLayer",
        "DeepseekDecoderLayer",
        "PatchMergerMLP",
    ]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False

    def _init_weights(self, module):
        # important: this ported version of Llava isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/llava should serve for that purpose
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class VisionTowerConfig(PretrainedConfig):
    model_type = "moonvit3d"

    def __init__(self, config: KimiK25Config, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = config.patch_size
        self.init_pos_emb_height = config.init_pos_emb_height
        self.init_pos_emb_width = config.init_pos_emb_width
        self.init_pos_emb_time = config.init_pos_emb_time
        self.pos_emb_type = config.pos_emb_type
        self.num_attention_heads = config.vt_num_attention_heads
        self.num_hidden_layers = config.vt_num_hidden_layers
        self.hidden_size = config.vt_hidden_size
        self.intermediate_size = config.vt_intermediate_size
        self.merge_kernel_size = config.merge_kernel_size
        self.video_attn_type = config.video_attn_type
        self.merge_type = config.merge_type
        self._attn_implementation = config._attn_implementation


class ProjectorConfig:
    def __init__(self, config: KimiK25Config):
        self.mm_projector_type = config.mm_projector_type
        self.mm_hidden_size = config.mm_hidden_size
        self.hidden_size = config.text_hidden_size
        self.merge_kernel_size = config.merge_kernel_size
        self.projector_hidden_act = config.projector_hidden_act
        self.projector_ln_eps = config.projector_ln_eps


class QEffKimiK25EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = self.model.config
        # _orig_apply_rope = _kimi_module.apply_rope
        # _kimi_module.apply_rope = _apply_rope_real
        # _kimi_module = _sys.modules[type(model).__module__]

        # Restore original apply_rope and attention functions
        # _kimi_module.apply_rope = _orig_apply_rope
        # _kimi_module.VL_VISION_ATTENTION_FUNCTIONS.update(_orig_attn_functions)

        # _orig_attn_functions = _kimi_module.VL_VISION_ATTENTION_FUNCTIONS.copy()
        # _kimi_module.VL_VISION_ATTENTION_FUNCTIONS["eager"] = _full_attention_forward

    def get_submodules_for_export(self) -> Type[nn.Module]:
        """
        Return the set of class used as the repeated layer across the model for subfunction extraction.
        Notes:
            This method should return the *class object* (not an instance).
            Downstream code can use this to find/build subfunctions for repeated blocks.
        """
        return {self.model.vision_model.model.layers[0].__class__}
        # return {self.model.layers[0].__class__}

    def forward_only_image(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> list[torch.Tensor]:
        """
        Run only the vision tower and mm_projector to extract image embeddings.

        Args:
            pixel_values: Preprocessed image pixel values.
            grid_thws: Grid temporal/height/width info for the images.

        Returns:
            image_embeds: List of projected image embedding tensors, one per image.
        """
        image_features = self._extract_image_features(pixel_values, grid_thws)
        if self.mm_projector:
            image_features = self.mm_projector(image_features)
        return image_features

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
            image_embeds: Projected image embeddings as a single tensor.
        """
        _kimi_module = _sys.modules[type(self).__module__]
        _get_rope_shape = _kimi_module.get_rope_shape

        h_shape = h_shape.to(pixel_values.device)
        w_shape = w_shape.to(pixel_values.device)

        # Keep them in ONNX graph
        dummy = (h_shape.float().sum() + w_shape.float().sum()) * 0.0
        pixel_values = pixel_values + dummy

        h = h_shape.shape[0]
        w = w_shape.shape[0]

        target_dtype = self.model.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.to(target_dtype)

        # --- Patch embedding ---
        x = self.model.vision_tower.patch_embed.proj(pixel_values).view(pixel_values.size(0), -1)

        # Positional embedding (single image, t=1)
        pos_emb_module = self.model.vision_tower.patch_embed.pos_emb
        pos_emb_2d = _get_rope_shape(
            pos_emb_module.weight,
            interpolation_mode=pos_emb_module.interpolation_mode,
            shape=(h, w),
        )
        x = x + pos_emb_2d

        # --- Encoder ---
        # For single image with t=1: cu_seqlens = [0, h*w]
        num_tokens = h * w
        cu_seqlens = torch.zeros(2, dtype=torch.int32, device=x.device)
        cu_seqlens[1] = num_tokens
        max_seqlen = num_tokens

        # RoPE frequencies for single image (stacked real/imag to avoid complex tensors in ONNX)
        # Shape: (2, num_tokens, dim//2) where [0]=cos, [1]=sin
        rope_2d = self.model.vision_tower.encoder.rope_2d
        rope_2d._ensure_precomputed_freqs(x.device)
        freqs_cos = rope_2d.freqs_cos[:h, :w].reshape(-1, rope_2d.dim // 2)
        freqs_sin = rope_2d.freqs_sin[:h, :w].reshape(-1, rope_2d.dim // 2)
        freqs_cis = torch.stack([freqs_cos, freqs_sin], dim=0)

        encoder_dtype = self.model.vision_tower.encoder.blocks[0].wqkv.weight.dtype
        x = x.to(encoder_dtype)
        freqs_cis = freqs_cis.to(encoder_dtype)

        for block in self.model.vision_tower.encoder.blocks:
            x = block(x, cu_seqlens, max_seqlen, rope_freqs_cis=freqs_cis)

        final_ln_dtype = self.model.vision_tower.encoder.final_layernorm.weight.dtype
        x = self.model.vision_tower.encoder.final_layernorm(x.to(final_ln_dtype))

        # --- tpool_patch_merger (single image, t=1) ---
        merge_kernel_size = self.model.vision_tower.merge_kernel_size
        kernel_height, kernel_width = merge_kernel_size
        d_model = x.size(-1)
        new_height = h // kernel_height
        new_width = w // kernel_width
        reshaped = x.view(1, new_height, kernel_height, new_width, kernel_width, d_model)
        reshaped = reshaped.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
        merged = reshaped.view(new_height * new_width, kernel_height * kernel_width, -1)

        # --- mm_projector (PatchMergerMLP on single tensor) ---
        pre_norm_dtype = self.model.mm_projector.pre_norm.weight.dtype
        merged = merged.to(pre_norm_dtype)
        image_embeds = self.model.mm_projector.proj(self.model.mm_projector.pre_norm(merged).view(merged.shape[0], -1))
        return image_embeds


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
        return {self.model.layers[0].__class__}

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
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
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if image_idx is None:
            image_idx = torch.zeros((1, 1), dtype=torch.int64, device=inputs_embeds.device)

        if image_embeds is not None and input_ids is not None and input_ids.shape[1] != 1:
            if image_embeds.dim() == 2:
                image_embeds = image_embeds.unsqueeze(0)

            image_embeds = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            _, _, hidden_size = inputs_embeds.shape
            selected = input_ids == self.config.media_placeholder_token_id
            indices1 = selected.to(torch.int64).cumsum(1) - 1
            indices1 = torch.where(indices1 != -1, indices1 + image_idx.to(indices1.device), indices1)
            indices0 = torch.arange(selected.shape[0], device=selected.device).view(-1, 1)
            safe_indices1 = torch.where(indices1 < 0, torch.zeros_like(indices1), indices1)
            image_features_expanded = image_embeds.reshape(-1, hidden_size).unsqueeze(0)[indices0, safe_indices1]
            image_input_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
            inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_input_embeds)
            image_idx = (indices1.max() + 1).reshape(1, 1)

        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            compressed_kvs=compressed_kvs,
            past_key_values=past_key_values,
            batch_index=batch_index,
            comp_ctx_lengths=comp_ctx_lengths,
            use_cache=True if use_cache is None else use_cache,
            **kwargs,
        )

        # QEff Deepseek language_model.forward already returns final logits.
        logits = outputs[0].float()

        output_compressed_kvs = getattr(outputs, "compressed_kvs", None)
        output_past_key_values = getattr(outputs, "past_key_values", None)
        return logits, image_embeds, image_idx, output_compressed_kvs, output_past_key_values


# ref https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/llava/modeling_llava.py#L240
class QEffKimiK25ForConditionalGeneration(KimiK25PreTrainedModel):
    def __init__(self, config: KimiK25Config):
        super().__init__(config)

        vt_config = VisionTowerConfig(config.vision_config)
        self.vision_tower = MoonViT3dPretrainedModel(vt_config)

        proj_config = ProjectorConfig(config.vision_config)
        if proj_config.mm_projector_type == "identity":
            self.mm_projector = IdentityMap()
        elif proj_config.mm_projector_type == "mlp":
            self.mm_projector = MLP(proj_config)
        elif proj_config.mm_projector_type == "patchmerger":
            self.mm_projector = PatchMergerMLP(proj_config)
        else:
            raise ValueError(f"Unsupported mm_projector_type: {proj_config.mm_projector_type}")

        self.language_model = QEffDeepseekV3ForCausalLM(config.text_config)
        self.post_init()

        if hasattr(self.language_model, "dtype"):
            target_dtype = self.language_model.dtype
            self.vision_tower = self.vision_tower.to(dtype=target_dtype)
            self.mm_projector = self.mm_projector.to(dtype=target_dtype)

    def get_qeff_vision_encoder(self):
        return QEffKimiK25EncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEffKimiK25DecoderWrapper(self)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: int | None = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def _merge_input_ids_with_image_features(
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
        _, embed_dim = image_features[0].shape
        feature_lengths = [x.shape[0] for x in image_features]
        image_features = torch.cat(image_features, dim=0)

        image_token_index: int = self.config.media_placeholder_token_id
        pad_token_id: int = self.config.pad_token_id
        ignore_index: int = self.config.ignore_index

        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(pad_token_id))

        # 1. Create a mask to know where special image tokens are
        _token_occupation_table = torch.ones_like(input_ids.flatten())
        _token_occupation_table[input_ids.flatten() == image_token_index] = torch.tensor(
            feature_lengths, dtype=torch.long, device=input_ids.device
        )
        _token_occupation_table = _token_occupation_table.reshape(input_ids.shape)

        max_embed_dim = _token_occupation_table.sum(-1).max().item()
        assert max_embed_dim >= sequence_length, (
            f"The maximum embedding dimension ({max_embed_dim}) is less than the sequence length ({sequence_length})"
        )
        batch_indices, non_image_indices = torch.where(input_ids != image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        new_token_positions = torch.cumsum(_token_occupation_table, -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size,
            max_embed_dim,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim),
                ignore_index,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask.
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        image_to_overwrite = torch.full(
            (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
        )
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {image_to_overwrite.sum()} while"
                f" the number of image features given to the model is {image_features.shape[:-1].numel()}. "
                "This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def _extract_image_features(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            pixel_values (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_channels, height, width)`):
                The pixel values of the images processed by image processor.
            grid_thws (:obj:`torch.Tensor` of shape :obj:`(batch_size, 3)`):
                The grid, height, width of the images.
        Returns:
            selected_image_feature (:obj:`torch.FloatTensor` of shape :obj:`(num_image_tokens, embed_dim)`):
                The selected image features to use as input to the projector head.
        """

        target_dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.to(target_dtype)

        image_features = self.vision_tower(pixel_values, grid_thws)
        return image_features

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | list[torch.FloatTensor] | None = None,
        grid_thws: torch.Tensor | None = None,
        # h_shape: torch.Tensor | None = None,
        # w_shape: torch.Tensor | None = None,
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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        grid_thws=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = getattr(past_key_values, "seen_tokens", cache_length)
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.media_placeholder_token_id in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "grid_thws": grid_thws,
            }
        )
        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)

    def get_output_names(self, kv_offload: bool = False):
        vision_output_names = ["image_embeds"]
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
            output_names["vision"] = vision_output_names
            output_names["lang"] = lang_output_names
        else:
            return lang_output_names
        return output_names

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

        bs: int = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        fbs: int = constants.ONNX_EXPORT_EXAMPLE_FBS

        model_path = Path(
            "/home/huggingface_hub/models--moonshotai--Kimi-K2.5/snapshots/4d01dfe0332d63057c186e0b262165819efb6611"
        )
        processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        image_url = "https://huggingface.co/moonshotai/Kimi-K2.5/resolve/main/figures/kimi-logo.png"
        image = Image.open(BytesIO(requests.get(image_url).content)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": image},
                    {"type": "text", "text": "Tell me about yourself."},
                ],
            }
        ]
        inputs = processor(
            messages=messages,
            add_generation_prompt=True,
            tokenize=False,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.language_model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # Build h_shape and w_shape from grid_thws (single image, t=1)
        grid_thws_val = inputs["grid_thws"]
        h_val = int(grid_thws_val[0, 1].item())
        w_val = int(grid_thws_val[0, 2].item())
        h_shape_tensor = torch.ones(h_val, dtype=torch.int64, device=self.language_model.device)
        w_shape_tensor = torch.ones(w_val, dtype=torch.int64, device=self.language_model.device)

        vision_inputs = {
            "pixel_values": inputs["pixel_values"],
            "h_shape": h_shape_tensor,
            "w_shape": w_shape_tensor,
        }

        lang_inputs = {
            "input_ids": torch.zeros((bs, prefill_seq_len), dtype=torch.int64),
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
            seq_len=prefill_seq_len,
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

        if continuous_batching:
            lang_inputs["batch_index"] = torch.arange(bs).view(bs, 1)

        if comp_ctx_lengths is not None:
            lang_inputs["comp_ctx_lengths"] = torch.randint(0, 100, (40,), dtype=torch.int64)

        inputs = {}
        if kv_offload:
            inputs["vision"] = vision_inputs
            inputs["lang"] = lang_inputs
        else:
            lang_inputs.pop("image_embeds")
            inputs = {**vision_inputs, **lang_inputs}

        return inputs

    def get_onnx_dynamic_axes(
        self, comp_ctx_lengths: Optional[List[int]] = None, kv_offload: bool = False, continuous_batching: bool = False
    ):
        vision_dynamic_axes = {}
        lang_dynamic_axes = {}
        lang_dynamic_axes["input_ids"] = {0: "batch_size", 1: "seq_len"}
        lang_dynamic_axes["position_ids"] = {0: "batch_size", 1: "seq_len"}
        lang_dynamic_axes["image_embeds"] = {0: "num_image_tokens"}
        if continuous_batching:
            lang_dynamic_axes["batch_index"] = {0: "batch_size"}
        vision_dynamic_axes = {
            "pixel_values": {0: "num_patches"},
            "h_shape": {0: "h"},
            "w_shape": {0: "w"},
            "image_embeds": {0: "num_image_tokens"},
        }

        mla_absorption = getattr(self.language_model, "mla_absorption", None)
        if mla_absorption is not None:
            cache_compressed = mla_absorption.get("cache_compressed", False)
        else:
            cache_compressed = False

        if cache_compressed:
            for i in range(self.language_model.config.num_hidden_layers):
                lang_dynamic_axes[f"compressed_kv.{i}"] = {0: "batch_size", 2: "ctx_len"}
                lang_dynamic_axes[f"k_pe.{i}"] = {0: "batch_size", 2: "ctx_len"}
        else:
            for i in range(self.language_model.config.num_hidden_layers):
                for kv in ["key", "value"]:
                    lang_dynamic_axes[f"past_{kv}.{i}"] = {0: "batch_size", 2: "ctx_len"}

        if comp_ctx_lengths is not None:
            lang_dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}

        dynamic_axes = {}
        if kv_offload:
            dynamic_axes["vision"] = vision_dynamic_axes
            dynamic_axes["lang"] = lang_dynamic_axes
        else:
            lang_dynamic_axes.pop("image_embeds")
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
        num_patches = compiler_options.pop("num_patches", None)
        h = compiler_options.pop("h", None)
        w = compiler_options.pop("w", None)
        num_image_tokens = compiler_options.pop("num_image_tokens", None)

        prefill_seq_len = prefill_seq_len if prefill_seq_len else 32  # constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
        ctx_len = ctx_len if ctx_len else 32  # constants.ONNX_EXPORT_EXAMPLE_CTX_LEN

        vision = [
            {
                "num_patches": 2400,  # num_patches
                "h": 30,  # h
                "w": 80,  # w
                "num_image_tokens": 600,  # num_image_tokens
            }
        ]

        if comp_ctx_lengths_prefill is not None:
            lang = []

            for i in range(0, len(comp_ctx_lengths_prefill)):
                lang_prefill = {
                    "batch_size": 1 if continuous_batching else batch_size,
                    "seq_len": prefill_seq_len,
                    "ctx_len": ctx_len,
                    "num_image_tokens": 600,  # num_image_tokens
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
                    "num_image_tokens": 600,  # num_image_tokens
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
                "num_image_tokens": 600,  # num_image_tokens
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
                "num_image_tokens": 600,  # num_image_tokens
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
            lang[0].pop("vision_size")
            lang[1].pop("vision_size")
            return lang, compiler_options
