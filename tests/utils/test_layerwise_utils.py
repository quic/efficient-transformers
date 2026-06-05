# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import pytest

from QEfficient.utils.custom_loader import CustomLoader
from QEfficient.utils.layerwise_utils import build_layer_windows


def test_build_layer_windows_divisible():
    assert build_layer_windows(4, 1) == [(3, 4), (2, 3), (1, 2), (0, 1)]
    assert build_layer_windows(4, 2) == [(2, 4), (0, 2)]


def test_build_layer_windows_non_divisible():
    # Last (lowest) window is the smaller remainder.
    assert build_layer_windows(5, 2) == [(3, 5), (1, 3), (0, 1)]


def test_build_layer_windows_invalid():
    with pytest.raises(ValueError):
        build_layer_windows(0, 1)
    with pytest.raises(ValueError):
        build_layer_windows(4, 0)


def _make_loader():
    # Build without touching the network: bypass __init__ download.
    loader = CustomLoader.__new__(CustomLoader)
    loader.hf_auto_class = None
    loader.pretrained_model_name_or_path = "dummy"
    loader.layer_prefixes = ("model.layers.",)
    loader.total_layers = 4
    loader.from_pretrained_kwargs = {}
    return loader


def test_shard_filter_keeps_window_layers_and_edges():
    loader = _make_loader()

    weight_map = {
        "model.embed_tokens.weight": "shard_a.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "shard_a.safetensors",
        "model.layers.1.self_attn.q_proj.weight": "shard_b.safetensors",
        "model.layers.2.self_attn.q_proj.weight": "shard_c.safetensors",
        "model.layers.3.self_attn.q_proj.weight": "shard_d.safetensors",
        "model.norm.weight": "shard_d.safetensors",
        "lm_head.weight": "shard_d.safetensors",
    }
    shard_files = [f"/tmp/{name}" for name in sorted(set(weight_map.values()))]

    import transformers

    original = transformers.modeling_utils.get_checkpoint_shard_files
    try:
        transformers.modeling_utils.get_checkpoint_shard_files = lambda *a, **k: (
            list(shard_files),
            {"weight_map": dict(weight_map)},
        )
        # Window [1, 2): keep layer 1 + all non-layer (edge) keys, drop layers 0/2/3.
        with loader._shard_filter(1, 2):
            files, meta = transformers.modeling_utils.get_checkpoint_shard_files()
    finally:
        transformers.modeling_utils.get_checkpoint_shard_files = original

    kept = set(meta["weight_map"].keys())
    assert "model.layers.1.self_attn.q_proj.weight" in kept
    assert "model.layers.0.self_attn.q_proj.weight" not in kept
    assert "model.layers.2.self_attn.q_proj.weight" not in kept
    # Non-layer (edge) weights are always kept; HF decides where to place them.
    assert {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"} <= kept


def test_shard_filter_noop_without_weight_map():
    loader = _make_loader()
    import transformers

    original = transformers.modeling_utils.get_checkpoint_shard_files
    try:
        transformers.modeling_utils.get_checkpoint_shard_files = lambda *a, **k: (["/tmp/x.safetensors"], {})
        with loader._shard_filter(0, 1):
            files, meta = transformers.modeling_utils.get_checkpoint_shard_files()
    finally:
        transformers.modeling_utils.get_checkpoint_shard_files = original

    assert files == ["/tmp/x.safetensors"]


def test_custom_loader_multi_prefix_selection():
    loader = CustomLoader.__new__(CustomLoader)
    loader.layer_prefixes = ("model.layers.", "model.language_model.layers.")
    loader.total_layers = 4

    weight_map = {
        "model.vision_model.encoder.0.weight": "v.safetensors",
        "model.multi_modal_projector.weight": "v.safetensors",
        "model.language_model.embed_tokens.weight": "a.safetensors",
        "model.language_model.layers.0.x": "a.safetensors",
        "model.language_model.layers.1.x": "b.safetensors",
        "model.language_model.layers.2.x": "c.safetensors",
        "model.language_model.norm.weight": "c.safetensors",
        "lm_head.weight": "c.safetensors",
    }
    shard_files = [f"/tmp/{name}" for name in sorted(set(weight_map.values()))]

    import transformers

    original = transformers.modeling_utils.get_checkpoint_shard_files
    try:
        transformers.modeling_utils.get_checkpoint_shard_files = lambda *a, **k: (
            list(shard_files),
            {"weight_map": dict(weight_map)},
        )
        with loader._shard_filter(1, 2):
            _, meta = transformers.modeling_utils.get_checkpoint_shard_files()
    finally:
        transformers.modeling_utils.get_checkpoint_shard_files = original

    kept = set(meta["weight_map"].keys())
    # Window [1,2): keep language layer 1, drop language layers 0/2.
    assert "model.language_model.layers.1.x" in kept
    assert "model.language_model.layers.0.x" not in kept
    assert "model.language_model.layers.2.x" not in kept
    # Vision encoder + projector + edges are always kept.
    assert "model.vision_model.encoder.0.weight" in kept
    assert "model.multi_modal_projector.weight" in kept
    assert {"model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight"} <= kept
