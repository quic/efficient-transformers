# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Unit tests for the generic diffusion pipeline infra.

Coverage in this single file:
1. Generic dispatch (`QEffDiffusionPipeline.from_pretrained`) returns concrete child classes.
2. Error handling for unsupported or malformed HF pipeline config.
3. `register_pipeline` extension path for new HF pipeline mappings.
4. Child pipeline inheritance from `QEffDiffusionPipeline`.
5. Shared `_export_modules` behavior:
   - honors `skip_if_qpc_exists`
   - enables ONNX subfunction flag only for supported modules
6. Shared `_compile_modules` behavior:
   - calls config + execute setup
   - exports only when ONNX is missing
   - runs optional pre-compile hook
   - routes to sequential vs parallel compile path
7. Example scripts use only generic `QEffDiffusionPipeline.from_pretrained(...)`.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from diffusers import DiffusionPipeline

from QEfficient import QEffDiffusionPipeline, QEffFluxPipeline, QEffWanImageToVideoPipeline, QEffWanPipeline
from QEfficient.diffusers.pipelines.pipeline_utils import ONNX_SUBFUNCTION_MODULE

pytestmark = pytest.mark.diffusers

EXAMPLE_SCRIPTS = [
    "examples/diffusers/flux/flux_1_schnell.py",
    "examples/diffusers/flux/flux_1_schnell_first_block_cache.py",
    "examples/diffusers/flux/flux_1_shnell_custom.py",
    "examples/diffusers/wan/wan_first_block_cache.py",
    "examples/diffusers/wan/wan_lightning.py",
    "examples/diffusers/wan/wan_lightning_custom.py",
    "examples/diffusers/wan_i2v/wan_i2v_custom.py",
    "examples/diffusers/wan_i2v/wan_lightning_i2v.py",
]


def _patch_load_config(monkeypatch, hf_class_name):
    def _fake_load_config(cls, pretrained_model_name_or_path, cache_dir=None):
        return {"_class_name": hf_class_name}

    monkeypatch.setattr(DiffusionPipeline, "load_config", classmethod(_fake_load_config))


def _patch_child_factory(monkeypatch, qeff_cls):
    captured = {}

    def _fake_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        captured["model_id"] = pretrained_model_name_or_path
        captured["kwargs"] = kwargs
        return cls.__new__(cls)

    monkeypatch.setattr(qeff_cls, "from_pretrained", classmethod(_fake_from_pretrained))
    return captured


class _DummyModule:
    def __init__(self, *, qpc_path: Optional[str] = None, onnx_path: Optional[str] = None):
        self.qpc_path = qpc_path
        self.onnx_path = onnx_path
        self.export_calls = []

    def get_onnx_params(self):
        return ({"x": 1}, {"x": {0: "batch_size"}}, ["y"])

    def export(self, **kwargs):
        self.export_calls.append(kwargs)


class _DummyPipeline(QEffDiffusionPipeline):
    def __init__(self, modules: Dict[str, _DummyModule]):
        self.modules = modules
        self.custom_config = {"dummy": True}
        self.export_calls = []

    def export(self, use_onnx_subfunctions: bool = False):
        self.export_calls.append(use_onnx_subfunctions)


def test_qeff_diffusion_pipeline_returns_concrete_child_for_supported_models(monkeypatch):
    """Factory should return the exact concrete child class for supported HF pipeline configs."""
    for hf_class_name, qeff_cls in [
        ("FluxPipeline", QEffFluxPipeline),
        ("WanPipeline", QEffWanPipeline),
        ("WanImageToVideoPipeline", QEffWanImageToVideoPipeline),
    ]:
        _patch_load_config(monkeypatch, hf_class_name)
        captured = _patch_child_factory(monkeypatch, qeff_cls)

        pipeline = QEffDiffusionPipeline.from_pretrained(
            "dummy-model-id",
            cache_dir="/tmp/hf-cache",
            use_unified=False,
        )

        assert isinstance(pipeline, qeff_cls)
        assert type(pipeline) is qeff_cls
        assert captured["model_id"] == "dummy-model-id"
        assert captured["kwargs"]["use_unified"] is False


def test_qeff_diffusion_pipeline_forwards_kwargs_to_load_config_and_child(monkeypatch):
    """`from_pretrained` should pass load-config kwargs and preserve child kwargs."""
    seen = {}

    def _fake_load_config(cls, pretrained_model_name_or_path, cache_dir=None):
        seen["model_id"] = pretrained_model_name_or_path
        seen["cache_dir"] = cache_dir
        return {"_class_name": "FluxPipeline"}

    monkeypatch.setattr(DiffusionPipeline, "load_config", classmethod(_fake_load_config))
    captured = _patch_child_factory(monkeypatch, QEffFluxPipeline)

    QEffDiffusionPipeline.from_pretrained(
        "dummy-model-id",
        cache_dir="/tmp/hf-cache",
        use_unified=False,
        enable_first_block_cache=True,
    )

    assert seen["model_id"] == "dummy-model-id"
    assert seen["cache_dir"] == "/tmp/hf-cache"
    assert captured["kwargs"]["use_unified"] is False
    assert captured["kwargs"]["enable_first_block_cache"] is True


def test_qeff_diffusion_pipeline_raises_for_unsupported_hf_pipeline(monkeypatch):
    """Unsupported HF `_class_name` must fail fast."""
    _patch_load_config(monkeypatch, "StableDiffusionPipeline")

    with pytest.raises(NotImplementedError):
        QEffDiffusionPipeline.from_pretrained("dummy-model-id")


def test_qeff_diffusion_pipeline_raises_for_missing_hf_class_name(monkeypatch):
    """Missing HF `_class_name` in config should raise clear ValueError."""

    def _fake_load_config(cls, pretrained_model_name_or_path, cache_dir=None):
        return {}

    monkeypatch.setattr(DiffusionPipeline, "load_config", classmethod(_fake_load_config))

    with pytest.raises(ValueError):
        QEffDiffusionPipeline.from_pretrained("dummy-model-id")


def test_qeff_diffusion_pipeline_register_pipeline_adds_new_mapping(monkeypatch):
    """`register_pipeline` should make new HF class names dispatchable."""
    _patch_load_config(monkeypatch, "DummyPipeline")
    captured = _patch_child_factory(monkeypatch, QEffFluxPipeline)

    mapping_key = "DummyPipeline"
    original = QEffDiffusionPipeline._HF_PIPELINE_TO_QEFF_CLASS.get(mapping_key)
    try:
        QEffDiffusionPipeline.register_pipeline(
            "DummyPipeline",
            "QEfficient.diffusers.pipelines.flux.pipeline_flux",
            "QEffFluxPipeline",
        )
        pipeline = QEffDiffusionPipeline.from_pretrained("dummy-model-id")
        assert isinstance(pipeline, QEffFluxPipeline)
        assert captured["model_id"] == "dummy-model-id"
    finally:
        if original is None:
            QEffDiffusionPipeline._HF_PIPELINE_TO_QEFF_CLASS.pop(mapping_key, None)
        else:
            QEffDiffusionPipeline._HF_PIPELINE_TO_QEFF_CLASS[mapping_key] = original


def test_qeff_child_pipelines_inherit_qeff_diffusion_pipeline():
    """Concrete diffusion pipelines should inherit from the generic base."""
    assert issubclass(QEffFluxPipeline, QEffDiffusionPipeline)
    assert issubclass(QEffWanPipeline, QEffDiffusionPipeline)
    assert issubclass(QEffWanImageToVideoPipeline, QEffDiffusionPipeline)


def test_export_modules_respects_skip_qpc_and_subfunction_flag():
    """Shared export should skip modules with precompiled QPC and set ONNX subfunction flag where supported."""
    subfn_module_name = next(iter(ONNX_SUBFUNCTION_MODULE))
    module_with_qpc = _DummyModule(qpc_path="/tmp/precompiled.qpc")
    module_without_qpc = _DummyModule()

    pipeline = _DummyPipeline(
        {
            subfn_module_name: module_without_qpc,
            "module_with_qpc": module_with_qpc,
        }
    )

    pipeline._export_modules(export_dir="/tmp/export", use_onnx_subfunctions=True, skip_if_qpc_exists=True)

    assert len(module_without_qpc.export_calls) == 1
    assert module_without_qpc.export_calls[0]["export_dir"] == "/tmp/export"
    assert module_without_qpc.export_calls[0]["use_onnx_subfunctions"] is True
    assert len(module_with_qpc.export_calls) == 0


def test_export_modules_can_force_export_even_if_qpc_exists():
    """Shared export should support forced export when requested."""
    module_with_qpc = _DummyModule(qpc_path="/tmp/precompiled.qpc")
    pipeline = _DummyPipeline({"module": module_with_qpc})

    pipeline._export_modules(skip_if_qpc_exists=False)

    assert len(module_with_qpc.export_calls) == 1


def test_compile_modules_sequential_runs_config_export_hook_and_compile(monkeypatch):
    """Sequential compile path should run setup, export-if-needed, pre-hook, then sequential compile."""
    module = _DummyModule(onnx_path=None)
    pipeline = _DummyPipeline({"module": module})
    specialization_updates: Dict[str, Any] = {"module": {"height": 512}}
    calls = {"config": [], "execute": 0, "seq": [], "par": [], "hook": 0}

    def _mock_config_manager(self, config_source=None, use_onnx_subfunctions=False):
        calls["config"].append((self, config_source, use_onnx_subfunctions))

    def _mock_set_execute_params(self):
        calls["execute"] += 1

    def _mock_compile_modules_sequential(modules, custom_config, specs):
        calls["seq"].append((modules, custom_config, specs))

    def _mock_compile_modules_parallel(modules, custom_config, specs):
        calls["par"].append((modules, custom_config, specs))

    monkeypatch.setattr("QEfficient.diffusers.pipelines.pipeline_diffusion.config_manager", _mock_config_manager)
    monkeypatch.setattr(
        "QEfficient.diffusers.pipelines.pipeline_diffusion.set_execute_params", _mock_set_execute_params
    )
    monkeypatch.setattr(
        "QEfficient.diffusers.pipelines.pipeline_diffusion.compile_modules_sequential",
        _mock_compile_modules_sequential,
    )
    monkeypatch.setattr(
        "QEfficient.diffusers.pipelines.pipeline_diffusion.compile_modules_parallel",
        _mock_compile_modules_parallel,
    )

    def _hook():
        calls["hook"] += 1

    pipeline._compile_modules(
        compile_config="cfg.json",
        parallel=False,
        use_onnx_subfunctions=True,
        specialization_updates=specialization_updates,
        required_module_names=["module"],
        pre_compile_hook=_hook,
    )

    assert calls["config"] == [(pipeline, "cfg.json", True)]
    assert calls["execute"] == 1
    assert calls["hook"] == 1
    assert pipeline.export_calls == [True]
    assert len(calls["seq"]) == 1
    assert calls["seq"][0] == (pipeline.modules, pipeline.custom_config, specialization_updates)
    assert calls["par"] == []


def test_compile_modules_parallel_skips_export_when_onnx_exists(monkeypatch):
    """Parallel compile path should not call export when ONNX path already exists."""
    module = _DummyModule(onnx_path="/tmp/model.onnx")
    pipeline = _DummyPipeline({"module": module})
    calls = {"seq": 0, "par": 0}

    monkeypatch.setattr(
        "QEfficient.diffusers.pipelines.pipeline_diffusion.config_manager", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "QEfficient.diffusers.pipelines.pipeline_diffusion.set_execute_params", lambda *args, **kwargs: None
    )

    def _mock_compile_modules_sequential(*args, **kwargs):
        calls["seq"] += 1

    def _mock_compile_modules_parallel(*args, **kwargs):
        calls["par"] += 1

    monkeypatch.setattr(
        "QEfficient.diffusers.pipelines.pipeline_diffusion.compile_modules_sequential",
        _mock_compile_modules_sequential,
    )
    monkeypatch.setattr(
        "QEfficient.diffusers.pipelines.pipeline_diffusion.compile_modules_parallel",
        _mock_compile_modules_parallel,
    )

    pipeline._compile_modules(parallel=True, specialization_updates={"module": {"width": 512}})

    assert pipeline.export_calls == []
    assert calls["par"] == 1
    assert calls["seq"] == 0


def test_diffusers_examples_use_generic_qeff_diffusion_pipeline():
    """All maintained diffusers example scripts must use the generic entrypoint."""
    repo_root = Path(__file__).resolve().parents[3]

    for rel_path in EXAMPLE_SCRIPTS:
        file_path = repo_root / rel_path
        source = file_path.read_text(encoding="utf-8")

        assert "from QEfficient import QEffDiffusionPipeline" in source
        assert "QEffDiffusionPipeline.from_pretrained(" in source
        assert "QEffFluxPipeline.from_pretrained(" not in source
        assert "QEffWanPipeline.from_pretrained(" not in source
        assert "QEffWanImageToVideoPipeline.from_pretrained(" not in source
