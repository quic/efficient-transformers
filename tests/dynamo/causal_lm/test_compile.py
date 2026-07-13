# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Dynamo QAIC compile lane.

FP16 compile with ``use_dynamo=True``. Subfunctions are used when the arch
supports them (``spec.subfunctions_supported``); otherwise falls back to flat
dynamo so every architecture gets compile coverage regardless.
"""

from __future__ import annotations

import pytest

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ..utils.report_generator import attach_dynamo_case
from ._helpers import CTX_LEN, PROMPT_LEN, assert_qconfig_exists, load_model
from .model_registry import DYNAMO_MODEL_SPECS, DynamoModelSpec, spec_ids


@pytest.mark.dynamo
@pytest.mark.dynamo_compile
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.regular
@pytest.mark.parametrize("spec", DYNAMO_MODEL_SPECS, ids=spec_ids(DYNAMO_MODEL_SPECS))
def test_dynamo_subfunction_compile_fp16(spec: DynamoModelSpec, dynamo_workdir, request):
    """FP16 QAIC compile — subfunctions enabled when supported, flat dynamo otherwise."""
    use_subfn = spec.subfunctions_supported
    coverage_cols = ("Export_Compile", "FP16_Coverage")
    if use_subfn:
        coverage_cols = ("Export_Compile", "Subfunction_Coverage", "FP16_Coverage")

    attach_dynamo_case(
        request,
        category=spec.category,
        task="dynamo_compile_fp16",
        architecture=spec.architecture,
        family=spec.family,
        supported_model=spec.model_id,
        coverage_columns=coverage_cols,
        notes=spec.notes,
    )

    workdir = dynamo_workdir(architecture=spec.architecture, feature="compile_fp16", precision="fp16")
    model_hf = load_model(spec)
    qeff_model = QEFFAutoModelForCausalLM(model_hf, pretrained_model_name_or_path=spec.model_id)
    qpc_path = qeff_model.compile(
        compile_dir=str(workdir),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_devices=1,
        num_cores=16,
        use_dynamo=True,
        use_onnx_subfunctions=use_subfn,
    )
    assert_qconfig_exists(qpc_path)
