# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
BF16 dynamo export lane (export-only).

Loads each tiny HF model with ``torch_dtype=torch.bfloat16`` and runs
``.export(use_dynamo=True, use_onnx_subfunctions=True)``. No compile, no
generate — compressed-precision on-device paths are covered separately.

This lane feeds the ``BF16_Coverage`` column so the report reflects the
real state of BF16 export, not a tracked-gap placeholder.
"""

from __future__ import annotations

import pytest
import torch

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ..utils.report_generator import attach_dynamo_case
from ._helpers import exported_onnx_path, load_model
from .model_registry import DYNAMO_MODEL_SPECS, DynamoModelSpec, spec_ids


@pytest.mark.dynamo
@pytest.mark.dynamo_export
@pytest.mark.llm_model
@pytest.mark.regular
@pytest.mark.precision("bf16")
@pytest.mark.parametrize("spec", DYNAMO_MODEL_SPECS, ids=spec_ids(DYNAMO_MODEL_SPECS))
def test_dynamo_subfunction_bf16_export(spec: DynamoModelSpec, dynamo_workdir, request):
    """BF16 export-only lane. Compile / generate intentionally out of scope."""
    attach_dynamo_case(
        request,
        category=spec.category,
        task="dynamo_subfunction_bf16_export",
        architecture=spec.architecture,
        family=spec.family,
        supported_model=spec.model_id,
        coverage_columns=("BF16_Coverage",),
        notes="BF16 lane exercises export only; compile / generate deliberately skipped.",
    )
    if not spec.subfunctions_supported:
        pytest.skip(spec.notes or "Architecture opts out of the subfunction lane.")

    workdir = dynamo_workdir(architecture=spec.architecture, feature="export_subfn_bf16", precision="bf16")
    model_hf = load_model(spec, torch_dtype=torch.bfloat16)
    qeff_model = QEFFAutoModelForCausalLM(model_hf, pretrained_model_name_or_path=spec.model_id)
    onnx_path = exported_onnx_path(qeff_model.export(workdir, use_dynamo=True, use_onnx_subfunctions=True))
    assert onnx_path.name.endswith(".onnx")
