# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
CPU-only unit tests for WAN first-block-cache with attention blocking configs.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from QEfficient.diffusers.first_block_cache.wan import enable_wan_first_block_cache
from QEfficient.diffusers.models.modeling_utils import compute_blocked_attention, get_attention_blocking_config


class _BlockingAwareWanBlock(nn.Module):
    def __init__(self, residual_scale: float):
        super().__init__()
        self.residual_scale = residual_scale
        self.last_config = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        del encoder_hidden_states, timestep_proj, rotary_emb

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
        return hidden_states + self.residual_scale * blocked


class _DummyWanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.blocks = nn.ModuleList(
            [
                _BlockingAwareWanBlock(0.30),
                _BlockingAwareWanBlock(0.20),
                _BlockingAwareWanBlock(0.10),
            ]
        )
        self.norm_out = nn.Identity()
        self.proj_out = nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.zeros(2, 8), requires_grad=False)

    def patch_embedding(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def forward(self, **kwargs):
        hidden_states = self.patch_embedding(kwargs["hidden_states"]).flatten(2).transpose(1, 2)
        return Transformer2DModelOutput(sample=hidden_states)


class _DummyTransformerWrapper:
    def __init__(self):
        self.model = _DummyWanModel()
        self.hash_params = {}
        self.compile_kwargs = None

    def get_onnx_params(self):
        return {"hidden_states": torch.randn(1, 8, 1, 2, 3)}, {"hidden_states": {0: "batch_size"}}, ["sample"]

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
    torch.manual_seed(9)
    hidden_states = torch.randn(1, 8, 1, 2, 3)
    encoder_hidden_states = torch.randn(1, 4, 8)
    rotary_emb = torch.randn(2, 1, 1, 1)
    temb = torch.zeros(1, 8)
    timestep_proj = torch.zeros(1, 6, 8)
    return hidden_states, encoder_hidden_states, rotary_emb, temb, timestep_proj


def _manual_cache_intermediates(model, hidden_states, encoder_hidden_states, rotary_emb, timestep_proj):
    try:
        rotary_chunks = torch.split(rotary_emb, 1, dim=0)
        flattened = model.patch_embedding(hidden_states).flatten(2).transpose(1, 2)

        first_block_out = model.blocks[0](flattened, encoder_hidden_states, timestep_proj, rotary_chunks)
        first_block_residual = first_block_out - flattened

        downsample_factor = model._qeff_first_block_cache_downsample_factor
        if downsample_factor <= 0:
            raise ValueError(f"downsample_factor must be > 0, got {downsample_factor}")
        if first_block_residual.shape[-1] % downsample_factor != 0:
            raise ValueError(
                "downsample_factor must divide hidden dimension exactly, "
                f"got hidden_size={first_block_residual.shape[-1]}, downsample_factor={downsample_factor}"
            )

        downsampled_first_block_residual = first_block_residual[..., ::downsample_factor].contiguous()

        remain_hidden = first_block_out
        for block in model.blocks[1:]:
            remain_hidden = block(remain_hidden, encoder_hidden_states, timestep_proj, rotary_chunks)
        current_remain_block_residual = remain_hidden - first_block_out

        return first_block_out, downsampled_first_block_residual, current_remain_block_residual
    except (IndexError, RuntimeError, TypeError, ValueError) as exc:
        raise AssertionError(f"Failed to compute WAN cache intermediates: {exc}") from exc


@pytest.mark.diffusers
@pytest.mark.parametrize("mode", ["default", "kv", "q", "qkv"])
def test_wan_first_block_cache_runs_with_all_blocking_modes(mode, monkeypatch):
    _set_blocking_env(monkeypatch, mode)
    wrapper = _DummyTransformerWrapper()
    enable_wan_first_block_cache(wrapper, downsample_factor=2)

    hidden_states, encoder_hidden_states, rotary_emb, temb, timestep_proj = _make_forward_inputs()
    _, expected_first_residual, _ = _manual_cache_intermediates(
        wrapper.model, hidden_states, encoder_hidden_states, rotary_emb, timestep_proj
    )
    prev_remain_residual = torch.randn(1, 6, 8)

    output, retained_first, retained_remain = wrapper.model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        rotary_emb=rotary_emb,
        temb=temb,
        timestep_proj=timestep_proj,
        prev_first_block_residuals=expected_first_residual,
        prev_remain_block_residuals=prev_remain_residual,
        cache_threshold=1.0,
    )

    assert isinstance(output, Transformer2DModelOutput)
    assert output.sample.shape == (1, 6, 8)
    assert torch.isfinite(output.sample).all()
    assert torch.allclose(retained_first, expected_first_residual, atol=1e-6)
    assert torch.allclose(retained_remain, prev_remain_residual, atol=1e-6)
    assert all(block.last_config[0] == mode for block in wrapper.model.blocks)


@pytest.mark.diffusers
def test_wan_first_block_cache_qkv_hit_uses_prev_remain_residual(monkeypatch):
    _set_blocking_env(monkeypatch, "qkv")
    wrapper = _DummyTransformerWrapper()
    enable_wan_first_block_cache(wrapper, downsample_factor=2)

    hidden_states, encoder_hidden_states, rotary_emb, temb, timestep_proj = _make_forward_inputs()
    first_block_out, first_residual, _ = _manual_cache_intermediates(
        wrapper.model, hidden_states, encoder_hidden_states, rotary_emb, timestep_proj
    )
    prev_remain_residual = torch.randn(1, 6, 8)

    output, _, retained_remain = wrapper.model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        rotary_emb=rotary_emb,
        temb=temb,
        timestep_proj=timestep_proj,
        prev_first_block_residuals=first_residual,
        prev_remain_block_residuals=prev_remain_residual,
        cache_threshold=1.0,
    )

    assert torch.allclose(retained_remain, prev_remain_residual, atol=1e-6)
    assert torch.allclose(output.sample, first_block_out + prev_remain_residual, atol=1e-6)


@pytest.mark.diffusers
def test_wan_first_block_cache_kv_miss_uses_current_remain_residual(monkeypatch):
    _set_blocking_env(monkeypatch, "kv")
    wrapper = _DummyTransformerWrapper()
    enable_wan_first_block_cache(wrapper, downsample_factor=2)

    hidden_states, encoder_hidden_states, rotary_emb, temb, timestep_proj = _make_forward_inputs()
    first_block_out, first_residual, current_remain_residual = _manual_cache_intermediates(
        wrapper.model, hidden_states, encoder_hidden_states, rotary_emb, timestep_proj
    )
    prev_first_residual = first_residual + 1.0
    prev_remain_residual = torch.randn(1, 6, 8)

    output, _, retained_remain = wrapper.model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        rotary_emb=rotary_emb,
        temb=temb,
        timestep_proj=timestep_proj,
        prev_first_block_residuals=prev_first_residual,
        prev_remain_block_residuals=prev_remain_residual,
        cache_threshold=0.0,
    )

    assert torch.allclose(retained_remain, current_remain_residual, atol=1e-6)
    assert torch.allclose(output.sample, first_block_out + current_remain_residual, atol=1e-6)


@pytest.mark.diffusers
def test_wan_first_block_cache_reads_blocking_env_config(monkeypatch):
    _set_blocking_env(monkeypatch, "qkv", head_block_size=4, num_kv_blocks=3, num_q_blocks=5)
    wrapper = _DummyTransformerWrapper()
    enable_wan_first_block_cache(wrapper, downsample_factor=2)

    hidden_states, encoder_hidden_states, rotary_emb, temb, timestep_proj = _make_forward_inputs()
    _, expected_first_residual, _ = _manual_cache_intermediates(
        wrapper.model, hidden_states, encoder_hidden_states, rotary_emb, timestep_proj
    )

    wrapper.model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        rotary_emb=rotary_emb,
        temb=temb,
        timestep_proj=timestep_proj,
        prev_first_block_residuals=expected_first_residual,
        prev_remain_block_residuals=torch.zeros(1, 6, 8),
        cache_threshold=1.0,
    )

    assert all(block.last_config == ("qkv", 4, 3, 5) for block in wrapper.model.blocks)


@pytest.mark.diffusers
def test_wan_first_block_cache_patch_is_idempotent_with_blocking(monkeypatch):
    _set_blocking_env(monkeypatch, "q")
    wrapper = _DummyTransformerWrapper()
    enable_wan_first_block_cache(wrapper, downsample_factor=2)

    patched_forward = wrapper.model.forward
    patched_get_onnx_params = wrapper.get_onnx_params
    patched_compile = wrapper.compile
    downsample_factor = wrapper.model._qeff_first_block_cache_downsample_factor
    hash_params = dict(wrapper.hash_params)

    enable_wan_first_block_cache(wrapper, downsample_factor=4)

    assert wrapper.model.forward is patched_forward
    assert wrapper.get_onnx_params is patched_get_onnx_params
    assert wrapper.compile is patched_compile
    assert wrapper.model._qeff_first_block_cache_downsample_factor == downsample_factor
    assert wrapper.hash_params == hash_params
