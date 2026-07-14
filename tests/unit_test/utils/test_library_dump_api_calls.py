# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
CPU-only tests that ensure library dumps are written for key API entry points
(even when failures happen) and deduplication holds for multi-module pipelines.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from transformers import GPT2Config, GPT2LMHeadModel

from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from QEfficient.utils.test_utils import load_vlm_model_from_config

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "image_text_model_configs.json"


def _read_manifest_paths(root: Path):
    return list(root.rglob("qeff_library-*.json"))


def _load_qwen3_0_6b_custom_vlm_config():
    data = json.loads(CONFIG_PATH.read_text())
    # Use qwen3_vl custom config from tests (no downloads).
    model_entry = next(cfg for cfg in data["image_text_models"] if cfg["model_type"] == "qwen3_vl")
    custom = model_entry.get("additional_params", {})
    from transformers import AutoConfig

    return AutoConfig.for_model("qwen3_vl", trust_remote_code=True, **custom)


class TestCausalLmDumpFailure:
    def test_causal_lm_dump_survives_transform_failure(self, tmp_path):
        model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=64, vocab_size=256)).eval()

        first_transform = QEFFAutoModelForCausalLM._pytorch_transforms[0]

        def exploding_apply(cls, model, *args, **kwargs):
            raise RuntimeError("forced transform failure")

        with (
            patch("QEfficient.utils.library_dump.QEFF_HOME", tmp_path),
            patch.object(first_transform, "apply", classmethod(exploding_apply)),
        ):
            with pytest.raises(RuntimeError, match="forced transform failure"):
                QEFFAutoModelForCausalLM(model)

        assert _read_manifest_paths(tmp_path), "expected manifest to exist despite failure"


class TestVlmDumpFailure:
    def test_vlm_dump_survives_transform_failure(self, tmp_path):
        config = _load_qwen3_0_6b_custom_vlm_config()
        model_hf = load_vlm_model_from_config(config)

        from QEfficient.transformers.models.modeling_auto import _QEFFAutoModelForImageTextToTextSingleQPC

        first_transform = _QEFFAutoModelForImageTextToTextSingleQPC._pytorch_transforms[0]

        def exploding_apply(cls, model, *args, **kwargs):
            raise RuntimeError("forced transform failure")

        with (
            patch("QEfficient.utils.library_dump.QEFF_HOME", tmp_path),
            patch.object(first_transform, "apply", classmethod(exploding_apply)),
        ):
            with pytest.raises(RuntimeError, match="forced transform failure"):
                QEFFAutoModelForImageTextToText(model_hf, kv_offload=False)

        assert _read_manifest_paths(tmp_path), "expected manifest to exist despite failure"


class TestDiffusersDumpDedup:
    def test_flux_pipeline_exports_only_one_dump_per_module(self, tmp_path, monkeypatch):
        # Ensure HF downloads use the configured cache.
        monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
        monkeypatch.setenv("HF_HUB_CACHE", os.environ.get("HF_HUB_CACHE", "/home/huggingface_hub"))

        from tests.diffusers.test_flux import _build_flux_pipeline

        with patch("QEfficient.utils.library_dump.QEFF_HOME", tmp_path):
            pipeline, _ = _build_flux_pipeline(enable_first_block_cache=False)

        # Stub export calls to keep the test fast while exercising pipeline iteration.
        def _noop_export(*args, **kwargs):
            return str(tmp_path / "dummy.onnx")

        for module in pipeline.modules.values():
            monkeypatch.setattr(module, "export", _noop_export, raising=True)

        with patch("QEfficient.utils.library_dump.QEFF_HOME", tmp_path):
            pipeline.export(export_dir=str(tmp_path))
            pipeline.export(export_dir=str(tmp_path))

        manifests = _read_manifest_paths(tmp_path)
        # One manifest per module, no duplicates across repeated exports.
        assert len(manifests) == len(pipeline.modules)
