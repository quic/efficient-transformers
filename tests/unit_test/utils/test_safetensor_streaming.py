# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json

import torch
from safetensors.torch import save_file

from QEfficient.utils.safetensor_materializer import (
    load_safetensor_checkpoint_into_model,
    resolve_safetensor_checkpoint_for_streaming,
)


class TinyMetaModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(2, 3))
        self.register_buffer("scale", torch.empty(3))


class TinyPackedExpertModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.experts = torch.nn.Module()
        self.mlp.experts.gate_up_proj = torch.nn.Parameter(torch.empty(2, 4, 3))
        self.mlp.experts.down_proj = torch.nn.Parameter(torch.empty(2, 3, 2))


def _write_checkpoint(path):
    path.mkdir()
    save_file(
        {
            "weight": torch.ones(2, 3, dtype=torch.bfloat16),
            "scale": torch.arange(3, dtype=torch.float32),
        },
        path / "model.safetensors",
        metadata={"format": "pt"},
    )


def _write_sharded_checkpoint(path):
    path.mkdir()
    save_file(
        {"weight": torch.ones(2, 3, dtype=torch.bfloat16)},
        path / "model-00001-of-00002.safetensors",
        metadata={"format": "pt"},
    )
    save_file(
        {"scale": torch.arange(3, dtype=torch.float32)},
        path / "model-00002-of-00002.safetensors",
        metadata={"format": "pt"},
    )
    with (path / "model.safetensors.index.json").open("w") as handle:
        json.dump(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    "weight": "model-00001-of-00002.safetensors",
                    "scale": "model-00002-of-00002.safetensors",
                },
            },
            handle,
        )


def test_resolve_streaming_checkpoint_uses_source_path_without_copy(tmp_path):
    source_path = tmp_path / "source"
    _write_checkpoint(source_path)

    checkpoint = resolve_safetensor_checkpoint_for_streaming(source_path, torch.float16)

    assert checkpoint.path == source_path.resolve()
    assert checkpoint.target_dtype == torch.float16
    assert checkpoint.dtype_aligned is False
    assert list(tmp_path.iterdir()) == [source_path]


def test_streaming_loader_casts_floating_tensors_to_fp16(tmp_path):
    source_path = tmp_path / "source"
    _write_checkpoint(source_path)

    with torch.device("meta"):
        model = TinyMetaModule()

    result = load_safetensor_checkpoint_into_model(model, source_path, target_dtype=torch.float16)

    assert result.loaded_tensor_count == 2
    assert not result.missing_parameter_names
    assert model.weight.device.type == "cpu"
    assert model.weight.dtype == torch.float16
    assert model.scale.dtype == torch.float16
    assert torch.equal(model.weight, torch.ones(2, 3, dtype=torch.float16))


def test_streaming_loader_casts_sharded_checkpoint_with_thread_pool(tmp_path, monkeypatch):
    source_path = tmp_path / "source"
    _write_sharded_checkpoint(source_path)
    monkeypatch.setattr("QEfficient.utils.safetensor_materializer._SAFETENSOR_PARALLEL_WORKERS", 2)

    with torch.device("meta"):
        model = TinyMetaModule()

    result = load_safetensor_checkpoint_into_model(model, source_path, target_dtype=torch.float16)

    assert result.loaded_tensor_count == 2
    assert not result.missing_parameter_names
    assert model.weight.dtype == torch.float16
    assert model.scale.dtype == torch.float16
    assert torch.equal(model.scale, torch.arange(3, dtype=torch.float16))


def test_streaming_loader_packs_legacy_expert_tensors(tmp_path):
    source_path = tmp_path / "source"
    source_path.mkdir()
    save_file(
        {
            "mlp.experts.0.gate_proj.weight": torch.full((2, 3), 1, dtype=torch.bfloat16),
            "mlp.experts.0.up_proj.weight": torch.full((2, 3), 2, dtype=torch.bfloat16),
            "mlp.experts.0.down_proj.weight": torch.arange(6, dtype=torch.bfloat16).reshape(3, 2),
            "mlp.experts.1.gate_proj.weight": torch.full((2, 3), 3, dtype=torch.bfloat16),
            "mlp.experts.1.up_proj.weight": torch.full((2, 3), 4, dtype=torch.bfloat16),
            "mlp.experts.1.down_proj.weight": torch.arange(6, 12, dtype=torch.bfloat16).reshape(3, 2),
        },
        source_path / "model.safetensors",
        metadata={"format": "pt"},
    )

    with torch.device("meta"):
        model = TinyPackedExpertModule()

    result = load_safetensor_checkpoint_into_model(model, source_path, target_dtype=torch.float16)

    assert result.loaded_tensor_count == 2
    assert not result.missing_parameter_names
    assert not result.unexpected_tensor_names
    assert model.mlp.experts.gate_up_proj.dtype == torch.float16
    assert torch.equal(
        model.mlp.experts.gate_up_proj[0],
        torch.tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2]], dtype=torch.float16),
    )
    assert torch.equal(
        model.mlp.experts.gate_up_proj[1],
        torch.tensor([[3, 3, 3], [3, 3, 3], [4, 4, 4], [4, 4, 4]], dtype=torch.float16),
    )
    assert torch.equal(model.mlp.experts.down_proj[1], torch.arange(6, 12, dtype=torch.float16).reshape(3, 2))
