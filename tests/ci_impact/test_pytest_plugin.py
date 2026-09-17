# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
from unittest.mock import Mock, patch

from scripts.ci_impact.pytest_plugin import (
    _matches,
    _omitted_by_profile,
    _omitted_from_selective_ci,
    _write_catalog,
    pytest_collection_modifyitems,
)

__test__ = False


def test_node_matching_requires_exact_callspecs() -> None:
    prefix = "tests/test_models.py::test_model"

    assert _matches("tests/test_models.py::test_model[case0]", ["tests/test_models.py::test_model[case0]"])
    assert not _matches("tests/test_models.py::test_model[case0]", [prefix])
    assert not _matches("tests/test_models.py::test_model_extra", [prefix])


def test_profile_omission_uses_layer_markers() -> None:
    item = Mock()
    marker = Mock()
    marker.name = "full_layers"
    item.iter_markers.return_value = [marker]

    assert _omitted_by_profile(item, "dummy_layers_model")
    assert not _omitted_by_profile(item, "full_layers_model")
    assert _omitted_from_selective_ci(item)


def test_profile_omission_selects_llm_override_nodeids() -> None:
    selected = Mock(nodeid="tests/test_models.py::test_llm_choice")
    selected_marker = Mock()
    selected_marker.name = "few_layers"
    selected.iter_markers.return_value = [selected_marker]
    deterministic_only = Mock(nodeid="tests/test_models.py::test_deterministic")
    deterministic_marker = Mock()
    deterministic_marker.name = "few_layers"
    deterministic_only.iter_markers.return_value = [deterministic_marker]
    config = Mock()
    config.getoption.side_effect = lambda option: {
        "--impact-plan": None,
        "--impact-stage": None,
        "--impact-profile-omissions": "dummy_layers_model",
    }[option]
    payload = {
        "mode": "selective",
        "stages": {
            "export_compile": {
                "nodeids": [selected.nodeid, deterministic_only.nodeid],
                "profile_override_nodeids": [selected.nodeid],
            }
        },
    }
    items = [selected, deterministic_only]

    with patch("scripts.ci_impact.pytest_plugin._load", return_value=(payload, "export_compile")):
        pytest_collection_modifyitems(config, items)

    assert items == [selected]
    config.hook.pytest_deselected.assert_called_once_with(items=[deterministic_only])


def test_collect_only_catalog_contains_exact_nodeids_and_runtime_stages(tmp_path) -> None:
    catalog_path = tmp_path / "catalog.json"
    config = Mock()
    config.getoption.side_effect = lambda option: {
        "--impact-catalog": catalog_path,
        "--impact-head": "abc123",
    }[option]
    llm_item = Mock(nodeid="tests/test_models.py::test_model[case0]")
    marker = Mock()
    marker.name = "llm_model"
    llm_item.iter_markers.return_value = [marker]
    full_layers_item = Mock(nodeid="tests/test_models.py::test_full_model[case0]")
    full_layers_marker = Mock()
    full_layers_marker.name = "full_layers"
    full_layers_item.iter_markers.return_value = [marker, full_layers_marker]
    docs_item = Mock(nodeid="tests/unit_test/test_helper.py::test_helper")
    docs_item.iter_markers.return_value = []

    _write_catalog(config, [llm_item, full_layers_item, docs_item])

    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert payload == {
        "head": "abc123",
        "schema_version": 2,
        "tests": [{"nodeid": "tests/test_models.py::test_model[case0]", "stages": ["export_compile", "qaic_llm"]}],
    }
