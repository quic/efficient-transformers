# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from QEfficient.utils.layer_scale_checkpoint import (
    apply_layer_scale_recipe_to_snapshot,
    build_layer_scale_recipe_from_recovery_result,
    dump_layer_scale_recipe_yaml,
    load_layer_scale_recipe,
)


def _write_recipe(recipe_path):
    recipe_path.write_text(
        "\n".join(
            [
                "ModelID: Qwen/Qwen3.5-397B-A17B",
                "LayerScales:",
                "  default_scale: 1.0",
                "  per_layer:",
                "    22: 0.5",
                "TensorScalingRules:",
                "  - name: experts_gate_up_proj_up_half",
                "    tensor_template: model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj",
                "    operation: scale_second_half_dim1",
                "  - name: shared_expert_up_proj",
                "    tensor_template: model.language_model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight",
                "    operation: scale_all",
            ]
        ),
        encoding="utf-8",
    )


def test_apply_layer_scale_recipe_to_snapshot(tmp_path):
    source_dir = tmp_path / "src_snapshot"
    source_dir.mkdir(parents=True)

    gate_up = torch.arange(2 * 6 * 4, dtype=torch.float16).view(2, 6, 4)
    shared_up = torch.arange(12, dtype=torch.float16).view(3, 4)
    save_file(
        {
            "model.language_model.layers.22.mlp.experts.gate_up_proj": gate_up,
            "model.language_model.layers.22.mlp.shared_expert.up_proj.weight": shared_up,
            "untouched.tensor": torch.ones(5, dtype=torch.float16),
        },
        str(source_dir / "model-00001-of-00001.safetensors"),
    )

    index = {
        "metadata": {"total_size": 0},
        "weight_map": {
            "model.language_model.layers.22.mlp.experts.gate_up_proj": "model-00001-of-00001.safetensors",
            "model.language_model.layers.22.mlp.shared_expert.up_proj.weight": "model-00001-of-00001.safetensors",
            "untouched.tensor": "model-00001-of-00001.safetensors",
        },
    }
    (source_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    (source_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_moe", "text_config": {"model_type": "qwen3_5_moe_text"}}),
        encoding="utf-8",
    )

    recipe_path = tmp_path / "recipe.yaml"
    _write_recipe(recipe_path)

    out_dir = tmp_path / "scaled_snapshot"
    audit = apply_layer_scale_recipe_to_snapshot(
        source_dir=source_dir,
        output_dir=out_dir,
        recipe_path=recipe_path,
        allow_hardlink_unchanged=False,
    )

    assert audit["scaled_tensor_count"] == 2
    assert audit["patched_shards"] == ["model-00001-of-00001.safetensors"]
    assert audit["touched_layers"] == [22]

    with safe_open(str(out_dir / "model-00001-of-00001.safetensors"), framework="pt", device="cpu") as sf:
        scaled_gate_up = sf.get_tensor("model.language_model.layers.22.mlp.experts.gate_up_proj")
        scaled_shared_up = sf.get_tensor("model.language_model.layers.22.mlp.shared_expert.up_proj.weight")
        untouched = sf.get_tensor("untouched.tensor")

    assert torch.equal(scaled_gate_up[:, :3, :], gate_up[:, :3, :])
    assert torch.equal(scaled_gate_up[:, 3:, :], gate_up[:, 3:, :] * 0.5)
    assert torch.equal(scaled_shared_up, shared_up * 0.5)
    assert torch.equal(untouched, torch.ones(5, dtype=torch.float16))

    cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg["qeff_layer_scales"] == {"22": 0.5}
    assert cfg["qeff_layer_scale_default"] == 1.0
    assert cfg["text_config"]["qeff_layer_scales"] == {"22": 0.5}


def test_load_layer_scale_recipe_rejects_mode_mismatch(tmp_path):
    recipe_path = tmp_path / "bad_recipe.yaml"
    recipe_path.write_text(
        "\n".join(
            [
                "ModelID: Qwen/Qwen3.5-397B-A17B",
                "LayerScales:",
                "  default_scale: 1.0",
                "  per_layer:",
                "    22: 0.5",
                "RuntimeEquivalence:",
                "  mode: checkpoint_scaled_mlp_and_residual_branch",
                "ConfigMetadata:",
                "  mode_value: some_other_mode",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mode_value must match RuntimeEquivalence.mode"):
        load_layer_scale_recipe(recipe_path)


def test_build_layer_scale_recipe_from_recovery_result(tmp_path):
    recovery_result = {
        "config": {"model_id": "Qwen/Qwen3.5-397B-A17B"},
        "unresolved": None,
        "layers": [
            {
                "layer_idx": 22,
                "resolved": True,
                "selected_scale": 0.125,
                "final_variant": {"layer_out_finite_ratio": {"hf_fp16": 1.0, "qeff_fp16": 1.0}},
            },
            {
                "layer_idx": 23,
                "resolved": True,
                "selected_scale": 1.0,
                "final_variant": {"layer_out_finite_ratio": {"hf_fp16": 1.0, "qeff_fp16": 1.0}},
            },
        ],
    }

    recipe = build_layer_scale_recipe_from_recovery_result(
        recovery_result=recovery_result,
        default_scale=1.0,
    )
    assert recipe.model_id == "Qwen/Qwen3.5-397B-A17B"
    assert recipe.layer_scales == {22: 0.125}
    assert recipe.config_mode_value == "checkpoint_scaled_mlp_and_residual_branch"

    out_yaml = tmp_path / "generated_recipe.yaml"
    dump_layer_scale_recipe_yaml(recipe, out_yaml, include_expanded_specs=True)
    loaded = load_layer_scale_recipe(out_yaml)
    assert loaded.layer_scales == {22: 0.125}
