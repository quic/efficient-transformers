# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Dynamo export lane.

Exports every architecture with ``use_dynamo=True, use_onnx_subfunctions=True``.
ONNX subfunctions are the core of the dynamo export feature; every supported arch
must emit them correctly. Architectures with ``subfunctions_supported=False``
skip this lane and are covered by the compile, parity, and generate lanes using
flat dynamo.

Registry integrity is also validated here so coverage gaps surface loudly.
"""

from __future__ import annotations

import pytest

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ..utils.report_generator import attach_dynamo_case
from ._helpers import assert_has_subfunctions, exported_onnx_path, load_model
from .model_registry import DYNAMO_MODEL_SPECS, DynamoModelSpec, spec_ids


@pytest.mark.dynamo
@pytest.mark.dynamo_export
@pytest.mark.llm_model
def test_dynamo_matrix_tracks_unique_architectures():
    """Registry integrity: every architecture is listed exactly once."""
    architectures = [spec.architecture for spec in DYNAMO_MODEL_SPECS]
    assert len(architectures) == len(set(architectures)), "duplicate architecture entries in registry"
    assert {"gpt2", "llama", "qwen2", "gemma3", "qwen2_5_vl", "molmo"}.issubset(set(architectures))


@pytest.mark.dynamo
@pytest.mark.dynamo_export
@pytest.mark.llm_model
@pytest.mark.regular
@pytest.mark.parametrize("spec", DYNAMO_MODEL_SPECS, ids=spec_ids(DYNAMO_MODEL_SPECS))
def test_dynamo_subfunction_export(spec: DynamoModelSpec, dynamo_workdir, request):
    """dynamo + ONNX subfunctions export — must produce a model with ONNX local functions."""
    attach_dynamo_case(
        request,
        category=spec.category,
        task="dynamo_subfunction_export",
        architecture=spec.architecture,
        family=spec.family,
        supported_model=spec.model_id,
        coverage_columns=("Export_Compile", "Subfunction_Coverage", "FP32_Coverage"),
        notes=spec.notes,
    )
    if not spec.subfunctions_supported:
        pytest.skip(spec.notes or "Architecture does not yet support the subfunction export lane.")

    workdir = dynamo_workdir(architecture=spec.architecture, feature="export_subfn", precision="fp32")
    model_hf = load_model(spec)
    qeff_model = QEFFAutoModelForCausalLM(model_hf, pretrained_model_name_or_path=spec.model_id)
    onnx_path = exported_onnx_path(qeff_model.export(workdir, use_dynamo=True, use_onnx_subfunctions=True))
    assert_has_subfunctions(onnx_path)
