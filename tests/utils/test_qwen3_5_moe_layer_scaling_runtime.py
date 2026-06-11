# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from types import SimpleNamespace

import pytest

from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import _resolve_layer_scale_runtime_config


def test_qwen3_5_moe_runtime_scaling_default_noop(monkeypatch):
    monkeypatch.delenv("QEFF_QWEN3_5_MOE_ENABLE_LAYER_SCALING", raising=False)
    monkeypatch.delenv("QEFF_QWEN3_5_MOE_LAYER_SCALE_YAML", raising=False)
    enabled, scales, default_scale, mode = _resolve_layer_scale_runtime_config(SimpleNamespace())
    assert enabled is False
    assert scales == {}
    assert default_scale == 1.0
    assert mode == "checkpoint_scaled_mlp_and_residual_branch"


def test_qwen3_5_moe_runtime_scaling_from_config(monkeypatch):
    monkeypatch.delenv("QEFF_QWEN3_5_MOE_ENABLE_LAYER_SCALING", raising=False)
    monkeypatch.delenv("QEFF_QWEN3_5_MOE_LAYER_SCALE_YAML", raising=False)
    cfg = SimpleNamespace(
        qeff_layer_scales={"22": 0.125},
        qeff_layer_scale_default=1.0,
        qeff_layer_scale_mode="checkpoint_scaled_mlp_and_residual_branch",
    )
    enabled, scales, default_scale, mode = _resolve_layer_scale_runtime_config(cfg)
    assert enabled is True
    assert scales == {22: 0.125}
    assert default_scale == 1.0
    assert mode == "checkpoint_scaled_mlp_and_residual_branch"


def test_qwen3_5_moe_runtime_scaling_rejects_unknown_mode(monkeypatch):
    monkeypatch.delenv("QEFF_QWEN3_5_MOE_ENABLE_LAYER_SCALING", raising=False)
    monkeypatch.delenv("QEFF_QWEN3_5_MOE_LAYER_SCALE_YAML", raising=False)
    cfg = SimpleNamespace(
        qeff_layer_scales={"22": 0.125},
        qeff_layer_scale_default=1.0,
        qeff_layer_scale_mode="unsupported_mode",
    )
    with pytest.raises(ValueError, match="Unsupported qeff layer-scale mode"):
        _resolve_layer_scale_runtime_config(cfg)


def test_qwen3_5_moe_runtime_scaling_env_disable_forces_noop(monkeypatch):
    monkeypatch.setenv("QEFF_QWEN3_5_MOE_ENABLE_LAYER_SCALING", "0")
    monkeypatch.delenv("QEFF_QWEN3_5_MOE_LAYER_SCALE_YAML", raising=False)
    cfg = SimpleNamespace(
        qeff_layer_scales={"22": 0.125},
        qeff_layer_scale_default=1.0,
        qeff_layer_scale_mode="checkpoint_scaled_mlp_and_residual_branch",
    )
    enabled, scales, default_scale, mode = _resolve_layer_scale_runtime_config(cfg)
    assert enabled is False
    assert scales == {}
    assert default_scale == 1.0
    assert mode == "checkpoint_scaled_mlp_and_residual_branch"
