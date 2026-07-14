# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Weight-free export lane.

Tests every supported architecture using the weight-free dynamo export path
(use_weight_free_export=True). The model is built on meta device — no weights
in RAM — and the ONNX graph is exported without embedded weight tensors.
The QAIC compiler loads weights from the prepared checkpoint at compile time.

Four lanes are validated here (all CPU-only, no QAIC hardware required):

  1. test_weight_free_export
       Meta-device model → export() → ONNX + weight_spec.json produced.

  2. test_weight_free_no_embedded_weights
       The exported ONNX must not contain large embedded initializers.
       All weight tensors must be external (referenced via weight_spec.json).

  3. test_weight_free_spec_coverage
       Every key in weight_spec.json must resolve to a real tensor in the
       prepared_checkpoint/ directory produced alongside the ONNX.

  4. test_weight_free_cb_export
       Same as lane 1 but with continuous_batching=True (full_batch_size pool).

Architectures with weight_free_supported=False are skipped with a clear
message so coverage gaps surface in the report rather than silently passing.
"""

from __future__ import annotations

import json
from pathlib import Path

import onnx
import pytest

from QEfficient.exporter.weight_free.spec import load_weight_spec
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ..utils.report_generator import attach_dynamo_case
from .model_registry import DynamoModelSpec, spec_ids, specs_with

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LARGE_INITIALIZER_THRESHOLD_BYTES = 1024  # 1 KB — anything larger should be external


def _build_meta_qeff_model(spec: DynamoModelSpec, *, continuous_batching: bool = False) -> QEFFAutoModelForCausalLM:
    """Build a weight-free QEff model using meta device (no weights in RAM)."""
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(spec.model_id)
    config.dtype = torch.float32
    if not hasattr(config, "max_position_embeddings"):
        config.max_position_embeddings = getattr(config, "n_positions", 2048)

    with init_empty_weights():
        meta_model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")

    return QEFFAutoModelForCausalLM(
        meta_model,
        pretrained_model_name_or_path=spec.model_id,
        continuous_batching=continuous_batching,
    )


def _onnx_embedded_initializer_sizes(onnx_path: Path):
    """Return sizes (bytes) of embedded ONNX initializers (external data excluded)."""
    model = onnx.load(str(onnx_path), load_external_data=False)
    sizes = []
    for init in model.graph.initializer:
        # external_data means the tensor body is NOT embedded
        if init.data_location == onnx.TensorProto.EXTERNAL:
            continue
        try:
            import onnx.numpy_helper as nph

            sizes.append(nph.to_array(init).nbytes)
        except Exception:
            pass
    return sizes


def _prepared_checkpoint_dir(export_dir: Path) -> Path | None:
    """Find the prepared_checkpoint/ directory created under export_dir."""
    candidates = list(export_dir.rglob("prepared_checkpoint"))
    return candidates[0] if candidates else None


def _all_checkpoint_keys(prepared_dir: Path) -> set:
    """Return every tensor key in the prepared_checkpoint safetensors shards."""
    from safetensors import safe_open

    keys = set()
    index_path = prepared_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        for shard_name in set(weight_map.values()):
            shard = prepared_dir / shard_name
            if shard.exists():
                with safe_open(str(shard), framework="pt") as f:
                    keys.update(f.keys())
    else:
        for shard in prepared_dir.glob("*.safetensors"):
            with safe_open(str(shard), framework="pt") as f:
                keys.update(f.keys())
    return keys


# ---------------------------------------------------------------------------
# Lane 1 — Export produces expected artifacts
# ---------------------------------------------------------------------------

_WF_EXPORT_SPECS = specs_with(weight_free=True)


@pytest.mark.dynamo
@pytest.mark.dynamo_export
@pytest.mark.llm_model
@pytest.mark.regular
@pytest.mark.parametrize("spec", _WF_EXPORT_SPECS, ids=spec_ids(_WF_EXPORT_SPECS))
def test_weight_free_export(spec: DynamoModelSpec, dynamo_workdir, request):
    """Weight-free export must produce model.onnx and weight_spec.json."""
    attach_dynamo_case(
        request,
        category=spec.category,
        task="weight_free_export",
        architecture=spec.architecture,
        family=spec.family,
        supported_model=spec.model_id,
        coverage_columns=("WeightFree_Export", "WeightFree_Coverage"),
        notes=spec.notes,
    )
    if not spec.weight_free_supported:
        pytest.skip(spec.notes or "Architecture does not yet support weight-free export.")

    workdir = dynamo_workdir(architecture=spec.architecture, feature="wf_export", precision="fp32")
    qeff_model = _build_meta_qeff_model(spec)
    onnx_path = Path(
        qeff_model.export(
            export_dir=str(workdir),
            use_dynamo=True,
            use_weight_free_export=True,
        )
    )

    assert onnx_path.is_file(), f"ONNX not produced at {onnx_path}"
    spec_path = onnx_path.parent / "weight_spec.json"
    assert spec_path.is_file(), "weight_spec.json not produced alongside ONNX"
    spec_entries = load_weight_spec(spec_path)
    assert len(spec_entries) > 0, "weight_spec.json is empty — no weight mappings were built"


# ---------------------------------------------------------------------------
# Lane 2 — No embedded weight tensors in the ONNX
# ---------------------------------------------------------------------------


@pytest.mark.dynamo
@pytest.mark.dynamo_export
@pytest.mark.llm_model
@pytest.mark.regular
@pytest.mark.parametrize("spec", _WF_EXPORT_SPECS, ids=spec_ids(_WF_EXPORT_SPECS))
def test_weight_free_no_embedded_weights(spec: DynamoModelSpec, dynamo_workdir, request):
    """Weight-free ONNX must not contain large embedded initializers.

    All weight tensors must be external (referenced via weight_spec.json and
    loaded from the prepared_checkpoint at compile time). Embedded tensors >
    1 KB indicate that the weight-free promotion step missed some initializers.
    """
    attach_dynamo_case(
        request,
        category=spec.category,
        task="weight_free_no_embedded_weights",
        architecture=spec.architecture,
        family=spec.family,
        supported_model=spec.model_id,
        coverage_columns=("WeightFree_NoEmbeddedWeights",),
        notes=spec.notes,
    )
    if not spec.weight_free_supported:
        pytest.skip(spec.notes or "Architecture does not yet support weight-free export.")

    workdir = dynamo_workdir(architecture=spec.architecture, feature="wf_export", precision="fp32")
    qeff_model = _build_meta_qeff_model(spec)
    onnx_path = Path(
        qeff_model.export(
            export_dir=str(workdir),
            use_dynamo=True,
            use_weight_free_export=True,
        )
    )

    large = [s for s in _onnx_embedded_initializer_sizes(onnx_path) if s > _LARGE_INITIALIZER_THRESHOLD_BYTES]
    assert not large, (
        f"{spec.architecture}: {len(large)} embedded ONNX initializer(s) larger than "
        f"{_LARGE_INITIALIZER_THRESHOLD_BYTES} B — weight-free export should have made all "
        f"weight tensors external. Sizes: {sorted(large, reverse=True)[:5]}"
    )


# ---------------------------------------------------------------------------
# Lane 3 — weight_spec.json covers the prepared checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.dynamo
@pytest.mark.dynamo_export
@pytest.mark.llm_model
@pytest.mark.regular
@pytest.mark.parametrize("spec", _WF_EXPORT_SPECS, ids=spec_ids(_WF_EXPORT_SPECS))
def test_weight_free_spec_coverage(spec: DynamoModelSpec, dynamo_workdir, request):
    """Every key in weight_spec.json must resolve in the prepared_checkpoint.

    The QAIC compiler reads weight_spec.json to locate tensors at compile time.
    A missing key means the compiler would fail to find a weight, causing a
    compile-time error. This test catches mismatches between the ONNX graph
    initializer names and the checkpoint key space early, on CPU.
    """
    attach_dynamo_case(
        request,
        category=spec.category,
        task="weight_free_spec_coverage",
        architecture=spec.architecture,
        family=spec.family,
        supported_model=spec.model_id,
        coverage_columns=("WeightFree_SpecCoverage",),
        notes=spec.notes,
    )
    if not spec.weight_free_supported:
        pytest.skip(spec.notes or "Architecture does not yet support weight-free export.")

    workdir = dynamo_workdir(architecture=spec.architecture, feature="wf_export", precision="fp32")
    qeff_model = _build_meta_qeff_model(spec)
    onnx_path = Path(
        qeff_model.export(
            export_dir=str(workdir),
            use_dynamo=True,
            use_weight_free_export=True,
        )
    )

    spec_entries = load_weight_spec(onnx_path.parent / "weight_spec.json")
    prepared_dir = _prepared_checkpoint_dir(workdir)
    assert prepared_dir is not None, "prepared_checkpoint/ directory not found after export"

    checkpoint_keys = _all_checkpoint_keys(prepared_dir)
    missing = [e.location.checkpoint_key for e in spec_entries if e.location.checkpoint_key not in checkpoint_keys]
    assert not missing, (
        f"{spec.architecture}: {len(missing)} weight_spec key(s) not found in prepared_checkpoint — "
        f"the compiler would fail to resolve these weights at compile time. "
        f"First 5 missing: {missing[:5]}"
    )


# ---------------------------------------------------------------------------
# Lane 4 — Continuous batching weight-free export
# ---------------------------------------------------------------------------

_WF_CB_SPECS = specs_with(weight_free_cb=True)


@pytest.mark.dynamo
@pytest.mark.dynamo_export
@pytest.mark.llm_model
@pytest.mark.regular
@pytest.mark.parametrize("spec", _WF_CB_SPECS, ids=spec_ids(_WF_CB_SPECS))
def test_weight_free_cb_export(spec: DynamoModelSpec, dynamo_workdir, request):
    """Weight-free export with continuous_batching=True must produce ONNX + weight_spec."""
    attach_dynamo_case(
        request,
        category=spec.category,
        task="weight_free_cb_export",
        architecture=spec.architecture,
        family=spec.family,
        supported_model=spec.model_id,
        coverage_columns=("WeightFree_CB_Export",),
        notes=spec.notes,
    )
    if not spec.weight_free_cb_supported:
        pytest.skip(spec.notes or "Architecture does not yet support weight-free CB export.")

    workdir = dynamo_workdir(architecture=spec.architecture, feature="wf_cb_export", precision="fp32")
    qeff_model = _build_meta_qeff_model(spec, continuous_batching=True)
    onnx_path = Path(
        qeff_model.export(
            export_dir=str(workdir),
            use_dynamo=True,
            use_weight_free_export=True,
        )
    )

    assert onnx_path.is_file(), f"CB weight-free ONNX not produced at {onnx_path}"
    spec_path = onnx_path.parent / "weight_spec.json"
    assert spec_path.is_file(), "weight_spec.json not produced for CB weight-free export"
    assert len(load_weight_spec(spec_path)) > 0, "weight_spec.json is empty for CB export"
