# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
CPU-only unit tests for FLUX first-block-cache with attention blocking configs.
"""

import pytest
import torch
import torch.nn as nn
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from QEfficient.diffusers.first_block_cache.flux import enable_flux_first_block_cache
from QEfficient.diffusers.models.modeling_utils import compute_blocked_attention, get_attention_blocking_config


class _FluxNormOut(nn.Module):
    def forward(self, hidden_states: torch.Tensor, adaln_out: torch.Tensor) -> torch.Tensor:
        del adaln_out
        return hidden_states


class _BlockingAwareFluxBlock(nn.Module):
    def __init__(self, residual_scale: float):
        super().__init__()
        self.residual_scale = residual_scale
        self.last_config = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        joint_attention_kwargs=None,
    ):
        del temb, image_rotary_emb, joint_attention_kwargs

        mode, head_block_size, num_kv_blocks, num_q_blocks = get_attention_blocking_config()
        self.last_config = (mode, head_block_size, num_kv_blocks, num_q_blocks)

        bs, cl, hidden_size = hidden_states.shape
        num_heads = 2
        head_dim = hidden_size // num_heads
        q = hidden_states.view(bs, cl, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        k = q
        v = q
        blocked = compute_blocked_attention(
            q=q,
            k=k,
            v=v,
            head_block_size=head_block_size or num_heads,
            num_kv_blocks=num_kv_blocks or 1,
            num_q_blocks=num_q_blocks or 1,
            blocking_mode=mode,
        )
        blocked = blocked.permute(0, 2, 1, 3).reshape(bs, cl, hidden_size)
        return encoder_hidden_states, hidden_states + self.residual_scale * blocked


class _DummyFluxModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_dim = 8
        self.config = {"num_attention_heads": 2, "attention_head_dim": 4}
        self.x_embedder = nn.Identity()
        self.context_embedder = nn.Identity()
        self.transformer_blocks = nn.ModuleList(
            [
                _BlockingAwareFluxBlock(0.30),
                _BlockingAwareFluxBlock(0.20),
                _BlockingAwareFluxBlock(0.10),
            ]
        )
        self.single_transformer_blocks = nn.ModuleList(
            [
                _BlockingAwareFluxBlock(0.07),
                _BlockingAwareFluxBlock(0.05),
            ]
        )
        self.norm_out = _FluxNormOut()
        self.proj_out = nn.Identity()

    def pos_embed(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(ids.shape[0], 1)

    def forward(self, **kwargs):
        return Transformer2DModelOutput(sample=kwargs["hidden_states"])


class _DummyTransformerWrapper:
    def __init__(self):
        self.model = _DummyFluxModel()
        self.hash_params = {}
        self.compile_kwargs = None

    def get_onnx_params(self):
        return {"hidden_states": torch.randn(1, 6, 8)}, {"hidden_states": {0: "batch_size", 1: "cl"}}, ["sample"]

    def _compile(self, **kwargs):
        self.compile_kwargs = kwargs
        return "compiled"

    def compile(self, *args, **kwargs):
        return self._compile(*args, **kwargs)


def _set_blocking_env(monkeypatch, mode: str, head_block_size: int = 2, num_kv_blocks: int = 2, num_q_blocks: int = 2):
    monkeypatch.setenv("ATTENTION_BLOCKING_MODE", mode)
    monkeypatch.setenv("head_block_size", str(head_block_size))
    monkeypatch.setenv("num_kv_blocks", str(num_kv_blocks))
    monkeypatch.setenv("num_q_blocks", str(num_q_blocks))


def _make_forward_inputs():
    torch.manual_seed(13)
    hidden_states = torch.randn(1, 6, 8)
    encoder_hidden_states = torch.randn(1, 4, 8)
    timestep = torch.tensor([1], dtype=torch.long)
    img_ids = torch.zeros(6, 1)
    txt_ids = torch.zeros(4, 1)
    adaln_emb = torch.zeros(3, 1)
    adaln_single_emb = torch.zeros(2, 1)
    adaln_out = torch.zeros(1)
    return hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids, adaln_emb, adaln_single_emb, adaln_out


def _manual_cache_intermediates(
    model,
    hidden_states,
    encoder_hidden_states,
    img_ids,
    txt_ids,
    adaln_emb,
    adaln_single_emb,
):
    downsample_factor = model._qeff_first_block_cache_downsample_factor
    if downsample_factor <= 0:
        raise ValueError(f"downsample_factor must be > 0, got {downsample_factor}")

    try:
        hidden_states = model.x_embedder(hidden_states)
        encoder_hidden_states = model.context_embedder(encoder_hidden_states)
        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = model.pos_embed(ids)

        original_hidden_states = hidden_states
        encoder_hidden_states, hidden_states = model.transformer_blocks[0](
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=adaln_emb[0],
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=None,
        )
        current_first_hidden_states_residuals = hidden_states - original_hidden_states
    except (IndexError, RuntimeError, TypeError) as exc:
        raise AssertionError(f"Failed to compute FLUX cache intermediates: {exc}") from exc

    if current_first_hidden_states_residuals.shape[-1] % downsample_factor != 0:
        raise ValueError(
            "downsample_factor must divide hidden dimension exactly, "
            f"got hidden_size={current_first_hidden_states_residuals.shape[-1]}, "
            f"downsample_factor={downsample_factor}"
        )
    downsampled_first_hidden_states_residuals = current_first_hidden_states_residuals[
        ..., ::downsample_factor
    ].contiguous()

    try:
        remaining_hidden_states = hidden_states
        remaining_encoder_hidden_states = encoder_hidden_states
        for index_block, block in enumerate(model.transformer_blocks[1:], start=1):
            remaining_encoder_hidden_states, remaining_hidden_states = block(
                hidden_states=remaining_hidden_states,
                encoder_hidden_states=remaining_encoder_hidden_states,
                temb=adaln_emb[index_block],
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=None,
            )

        for index_block, block in enumerate(model.single_transformer_blocks):
            remaining_encoder_hidden_states, remaining_hidden_states = block(
                hidden_states=remaining_hidden_states,
                encoder_hidden_states=remaining_encoder_hidden_states,
                temb=adaln_single_emb[index_block],
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=None,
            )

        current_hidden_states_residuals = remaining_hidden_states - hidden_states
        return hidden_states, downsampled_first_hidden_states_residuals, current_hidden_states_residuals
    except (IndexError, RuntimeError, TypeError) as exc:
        raise AssertionError(f"Failed to compute FLUX cache intermediates: {exc}") from exc


@pytest.mark.diffusers
@pytest.mark.parametrize("mode", ["default", "kv", "q", "qkv"])
def test_flux_first_block_cache_runs_with_all_blocking_modes(mode, monkeypatch):
    _set_blocking_env(monkeypatch, mode)
    wrapper = _DummyTransformerWrapper()
    enable_flux_first_block_cache(wrapper, downsample_factor=2)

    hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids, adaln_emb, adaln_single_emb, adaln_out = (
        _make_forward_inputs()
    )
    _, expected_first_residuals, _ = _manual_cache_intermediates(
        wrapper.model,
        hidden_states,
        encoder_hidden_states,
        img_ids,
        txt_ids,
        adaln_emb,
        adaln_single_emb,
    )
    prev_hidden_states_residuals = torch.randn(1, 6, 8)

    output, retained_first, retained_hidden = wrapper.model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        adaln_emb=adaln_emb,
        adaln_single_emb=adaln_single_emb,
        adaln_out=adaln_out,
        prev_first_hidden_states_residuals=expected_first_residuals,
        prev_hidden_states_residuals=prev_hidden_states_residuals,
        cache_threshold=1.0,
    )

    assert isinstance(output, Transformer2DModelOutput)
    assert output.sample.shape == (1, 6, 8)
    assert torch.isfinite(output.sample).all()
    assert torch.allclose(retained_first, expected_first_residuals, atol=1e-6)
    assert torch.allclose(retained_hidden, prev_hidden_states_residuals, atol=1e-6)
    all_blocks = list(wrapper.model.transformer_blocks) + list(wrapper.model.single_transformer_blocks)
    assert all(block.last_config[0] == mode for block in all_blocks)


@pytest.mark.diffusers
def test_flux_first_block_cache_qkv_hit_uses_prev_hidden_residuals(monkeypatch):
    _set_blocking_env(monkeypatch, "qkv")
    wrapper = _DummyTransformerWrapper()
    enable_flux_first_block_cache(wrapper, downsample_factor=2)

    hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids, adaln_emb, adaln_single_emb, adaln_out = (
        _make_forward_inputs()
    )
    first_hidden_states, first_residuals, _ = _manual_cache_intermediates(
        wrapper.model,
        hidden_states,
        encoder_hidden_states,
        img_ids,
        txt_ids,
        adaln_emb,
        adaln_single_emb,
    )
    prev_hidden_states_residuals = torch.randn(1, 6, 8)

    output, _, retained_hidden = wrapper.model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        adaln_emb=adaln_emb,
        adaln_single_emb=adaln_single_emb,
        adaln_out=adaln_out,
        prev_first_hidden_states_residuals=first_residuals,
        prev_hidden_states_residuals=prev_hidden_states_residuals,
        cache_threshold=1.0,
    )

    assert torch.allclose(retained_hidden, prev_hidden_states_residuals, atol=1e-6)
    assert torch.allclose(output.sample, first_hidden_states + prev_hidden_states_residuals, atol=1e-6)


@pytest.mark.diffusers
def test_flux_first_block_cache_kv_miss_uses_current_hidden_residuals(monkeypatch):
    _set_blocking_env(monkeypatch, "kv")
    wrapper = _DummyTransformerWrapper()
    enable_flux_first_block_cache(wrapper, downsample_factor=2)

    hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids, adaln_emb, adaln_single_emb, adaln_out = (
        _make_forward_inputs()
    )
    first_hidden_states, first_residuals, current_hidden_residuals = _manual_cache_intermediates(
        wrapper.model,
        hidden_states,
        encoder_hidden_states,
        img_ids,
        txt_ids,
        adaln_emb,
        adaln_single_emb,
    )
    prev_first_residuals = first_residuals + 1.0
    prev_hidden_states_residuals = torch.randn(1, 6, 8)

    output, _, retained_hidden = wrapper.model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        adaln_emb=adaln_emb,
        adaln_single_emb=adaln_single_emb,
        adaln_out=adaln_out,
        prev_first_hidden_states_residuals=prev_first_residuals,
        prev_hidden_states_residuals=prev_hidden_states_residuals,
        cache_threshold=0.0,
    )

    assert torch.allclose(retained_hidden, current_hidden_residuals, atol=1e-6)
    assert torch.allclose(output.sample, first_hidden_states + current_hidden_residuals, atol=1e-6)


@pytest.mark.diffusers
def test_flux_first_block_cache_reads_blocking_env_config(monkeypatch):
    _set_blocking_env(monkeypatch, "qkv", head_block_size=4, num_kv_blocks=3, num_q_blocks=5)
    wrapper = _DummyTransformerWrapper()
    enable_flux_first_block_cache(wrapper, downsample_factor=2)

    hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids, adaln_emb, adaln_single_emb, adaln_out = (
        _make_forward_inputs()
    )
    _, first_residuals, _ = _manual_cache_intermediates(
        wrapper.model,
        hidden_states,
        encoder_hidden_states,
        img_ids,
        txt_ids,
        adaln_emb,
        adaln_single_emb,
    )

    wrapper.model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        adaln_emb=adaln_emb,
        adaln_single_emb=adaln_single_emb,
        adaln_out=adaln_out,
        prev_first_hidden_states_residuals=first_residuals,
        prev_hidden_states_residuals=torch.zeros(1, 6, 8),
        cache_threshold=1.0,
    )

    all_blocks = list(wrapper.model.transformer_blocks) + list(wrapper.model.single_transformer_blocks)
    assert all(block.last_config == ("qkv", 4, 3, 5) for block in all_blocks)


@pytest.mark.diffusers
def test_flux_first_block_cache_patch_is_idempotent_with_blocking(monkeypatch):
    _set_blocking_env(monkeypatch, "q")
    wrapper = _DummyTransformerWrapper()
    enable_flux_first_block_cache(wrapper, downsample_factor=2)

    patched_forward = wrapper.model.forward
    patched_get_onnx_params = wrapper.get_onnx_params
    patched_compile = wrapper.compile
    downsample_factor = wrapper.model._qeff_first_block_cache_downsample_factor
    hash_params = dict(wrapper.hash_params)

    enable_flux_first_block_cache(wrapper, downsample_factor=4)

    assert wrapper.model.forward is patched_forward
    assert wrapper.get_onnx_params is patched_get_onnx_params
    assert wrapper.compile is patched_compile
    assert wrapper.model._qeff_first_block_cache_downsample_factor == downsample_factor
    assert wrapper.hash_params == hash_params
