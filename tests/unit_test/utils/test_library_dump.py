# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Lean unit tests for QEfficient/utils/library_dump.py.

Focus: manifest shape, path behavior, deduplication, basic resilience.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from transformers import LlamaConfig, LlamaForCausalLM


def make_tiny_llama():
    cfg = LlamaConfig(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=500,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture(autouse=True)
def _reset_written_registry():
    import QEfficient.utils.library_dump as ld

    ld._written.clear()
    yield
    ld._written.clear()


def _read_manifest(path):
    return json.loads(path.read_text())


class TestManifestBasics:
    def test_manifest_has_required_top_level_keys(self, tmp_path):
        from QEfficient.utils import library_dump as ld

        with patch.object(ld, "QEFF_HOME", tmp_path):
            path = ld.dump_qeff_library_once("LlamaForCausalLM", "LlamaForCausalLM")

        manifest = _read_manifest(path)
        for key in ("schema_version", "created_at", "model", "qeff", "runtime", "qaic_sdk", "dependencies"):
            assert key in manifest

    def test_schema_version_is_one(self, tmp_path):
        from QEfficient.utils import library_dump as ld

        with patch.object(ld, "QEFF_HOME", tmp_path):
            path = ld.dump_qeff_library_once(None, "LlamaForCausalLM")

        assert _read_manifest(path)["schema_version"] == 1

    def test_created_at_is_timezone_aware(self, tmp_path):
        from QEfficient.utils import library_dump as ld

        with patch.object(ld, "QEFF_HOME", tmp_path):
            path = ld.dump_qeff_library_once(None, "LlamaForCausalLM")

        ts = _read_manifest(path)["created_at"]
        assert "+" in ts or ts.endswith("Z")

    def test_model_block_fields(self, tmp_path):
        from QEfficient.utils import library_dump as ld

        with patch.object(ld, "QEFF_HOME", tmp_path):
            path = ld.dump_qeff_library_once("LlamaForCausalLM", "LlamaForCausalLM")

        model_block = _read_manifest(path)["model"]
        assert model_block["model_name"] == "LlamaForCausalLM"
        assert model_block["model_architecture"] == "LlamaForCausalLM"

    def test_runtime_block_has_pid(self, tmp_path):
        from QEfficient.utils import library_dump as ld

        with patch.object(ld, "QEFF_HOME", tmp_path):
            path = ld.dump_qeff_library_once(None, "LlamaForCausalLM")

        runtime = _read_manifest(path)["runtime"]
        assert runtime["pid"] == os.getpid()

    def test_dependencies_is_dict_with_core_packages(self, tmp_path):
        from QEfficient.utils import library_dump as ld

        with patch.object(ld, "QEFF_HOME", tmp_path):
            path = ld.dump_qeff_library_once(None, "LlamaForCausalLM")

        deps = _read_manifest(path)["dependencies"]
        assert isinstance(deps, dict)
        for pkg in ("torch", "transformers", "numpy"):
            assert pkg in deps


class TestPathAndDedup:
    def test_path_uses_arch_or_model(self, tmp_path):
        from QEfficient.utils import library_dump as ld

        with patch.object(ld, "QEFF_HOME", tmp_path):
            p1 = ld.dump_qeff_library_once("LlamaForCausalLM", "LlamaForCausalLM")
            p2 = ld.dump_qeff_library_once(None, "GPT2LMHeadModel")

        assert p1.parent == tmp_path / "LlamaForCausalLM" / "LlamaForCausalLM"
        assert p2.parent == tmp_path / "GPT2LMHeadModel" / "GPT2LMHeadModel"
        assert p1.name.startswith("qeff_library-") and p1.name.endswith(".json")

    def test_same_model_returns_same_path(self, tmp_path):
        from QEfficient.utils import library_dump as ld

        with patch.object(ld, "QEFF_HOME", tmp_path):
            p1 = ld.dump_qeff_library_once("LlamaForCausalLM", "LlamaForCausalLM")
            p2 = ld.dump_qeff_library_once("LlamaForCausalLM", "LlamaForCausalLM")

        assert p1 == p2


class TestResilience:
    def test_unwritable_location_returns_none(self, tmp_path):
        from QEfficient.utils import library_dump as ld

        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0o444)
        try:
            with patch.object(ld, "QEFF_HOME", locked):
                result = ld.dump_qeff_library_once("LlamaForCausalLM", "LlamaForCausalLM")
            assert result is None
        finally:
            locked.chmod(0o755)


class TestIntegration:
    def test_manifest_created_during_model_construction(self, tmp_path):
        import QEfficient.utils.library_dump as ld
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        with patch.object(ld, "QEFF_HOME", tmp_path):
            QEFFAutoModelForCausalLM(make_tiny_llama())

        manifests = list(tmp_path.rglob("qeff_library-*.json"))
        assert len(manifests) == 1


class TestQaicSdkVersions:
    def test_qaic_sdk_parses_aic_strings_or_none(self):
        from QEfficient.utils import library_dump as ld

        fake_output = "platform:AIC.1.23.0.18\napps:AIC.1.23.0.18\n"
        mock_proc = MagicMock()
        mock_proc.stdout = fake_output

        with patch("subprocess.run", return_value=mock_proc):
            result = ld._qaic_sdk_versions()

        for key in ("apps", "platform"):
            val = result[key]
            assert val is None or val.startswith("AIC")

    def test_qaic_sdk_not_found_becomes_none(self):
        from QEfficient.utils import library_dump as ld

        fake_output = "platform:not found\napps:not found\n"
        mock_proc = MagicMock()
        mock_proc.stdout = fake_output

        with patch("subprocess.run", return_value=mock_proc):
            result = ld._qaic_sdk_versions()

        assert result == {"apps": None, "platform": None}
