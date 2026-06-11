# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
from types import SimpleNamespace

import pytest

from QEfficient.utils.precision_recovery_agent import (
    PrecisionRecoveryAgentRequest,
    needs_scale_search,
    parse_scale_candidate_schedules,
    resolve_model_card_from_loaded_qeff_model,
    resolve_model_id_from_card,
    run_precision_recovery_agent_from_loaded_qeff_model,
)


def test_parse_scale_candidate_schedules_defaults():
    schedules = parse_scale_candidate_schedules(None)
    assert len(schedules) == 1
    assert schedules[0].startswith("1.0")


def test_parse_scale_candidate_schedules_semicolon_separated():
    schedules = parse_scale_candidate_schedules("1.0,0.5;1.0,0.25")
    assert schedules == ("1.0,0.5", "1.0,0.25")


def test_parse_scale_candidate_schedules_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        parse_scale_candidate_schedules(" ; ")


def test_resolve_model_id_from_card_direct_id_passthrough():
    assert resolve_model_id_from_card("Qwen/Qwen3.5-397B-A17B") == "Qwen/Qwen3.5-397B-A17B"


def test_resolve_model_id_from_card_json(tmp_path):
    card_path = tmp_path / "model_card.json"
    card_path.write_text(json.dumps({"model_id": "Qwen/Qwen3.5-397B-A17B"}), encoding="utf-8")
    assert resolve_model_id_from_card(str(card_path)) == "Qwen/Qwen3.5-397B-A17B"


def test_resolve_model_id_from_card_yaml(tmp_path):
    card_path = tmp_path / "model_card.yaml"
    card_path.write_text("model_name: Qwen/Qwen3.5-397B-A17B\n", encoding="utf-8")
    assert resolve_model_id_from_card(str(card_path)) == "Qwen/Qwen3.5-397B-A17B"


def test_resolve_model_id_from_card_markdown(tmp_path):
    card_path = tmp_path / "README.md"
    card_path.write_text("This card targets Qwen/Qwen3.5-397B-A17B for eval.\n", encoding="utf-8")
    assert resolve_model_id_from_card(str(card_path)) == "Qwen/Qwen3.5-397B-A17B"


def test_needs_scale_search_false_when_no_matmul_or_mlp_promotions():
    summary = {
        "promoted_layers": [
            {
                "layer_idx": 22,
                "selected_patches": ["sparse_moe_post_ops_fp32"],
            }
        ]
    }
    assert needs_scale_search(summary) is False


def test_needs_scale_search_true_when_matmul_or_mlp_promoted():
    summary = {
        "promoted_layers": [
            {
                "layer_idx": 22,
                "selected_patches": ["shared_expert_matmul_fp32_keep_fp32"],
            }
        ]
    }
    assert needs_scale_search(summary) is True


def test_resolve_model_card_from_loaded_qeff_model_uses_wrapper_attr(tmp_path):
    model_dir = tmp_path / "cached_model"
    model_dir.mkdir()
    qeff_model = SimpleNamespace(_pretrained_model_name_or_path=str(model_dir))

    resolved = resolve_model_card_from_loaded_qeff_model(qeff_model)

    assert resolved == str(model_dir.resolve())


def test_resolve_model_card_from_loaded_qeff_model_uses_nested_model_attr():
    qeff_model = SimpleNamespace(model=SimpleNamespace(pretrained_path="Qwen/Qwen3.5-397B-A17B"))

    resolved = resolve_model_card_from_loaded_qeff_model(qeff_model)

    assert resolved == "Qwen/Qwen3.5-397B-A17B"


def test_resolve_model_card_from_loaded_qeff_model_errors_when_unavailable():
    with pytest.raises(ValueError, match="Could not resolve model card/model id"):
        resolve_model_card_from_loaded_qeff_model(SimpleNamespace())


def test_run_precision_recovery_agent_from_loaded_qeff_model_builds_request(monkeypatch):
    captured = {}

    def _fake_runner(request):
        captured["request"] = request
        return {"ok": True, "model_id": "Qwen/Qwen3.5-397B-A17B"}

    monkeypatch.setattr("QEfficient.utils.precision_recovery_agent.run_precision_recovery_agent", _fake_runner)
    qeff_model = SimpleNamespace(model=SimpleNamespace(pretrained_path="Qwen/Qwen3.5-397B-A17B"))

    report = run_precision_recovery_agent_from_loaded_qeff_model(
        qeff_model,
        prompt="Hello",
        max_layers=2,
        output_dir="scripts/debug/artifacts/agent_test",
    )

    assert report["ok"] is True
    request = captured["request"]
    assert isinstance(request, PrecisionRecoveryAgentRequest)
    assert request.model_card == "Qwen/Qwen3.5-397B-A17B"
    assert request.prompt == "Hello"
    assert request.max_layers == 2
