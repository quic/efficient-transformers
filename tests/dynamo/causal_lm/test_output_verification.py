# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Output verification lane — CPU parity (no hardware).

Validates three-way token equality: HF PyTorch baseline == QEff PyTorch
(post-transform) == OnnxRuntime on the exported model. Feeds ``CPU_Parity``.

HW-side parity (QEff PT vs QAIC FP16) is measured in ``test_generate_execute.py``
which feeds ``HW_Parity_FP16`` — giving full E2E confidence independently.

Subfunctions are used when the arch supports them; otherwise falls back to
flat dynamo. The ``Subfunction_Coverage`` column is only credited when
subfunctions are enabled. This ensures ``gpt_oss`` and similar archs still
get parity coverage even though they sit outside the subfunctions lane.
"""

from __future__ import annotations

import numpy as np
import pytest

from ..utils.report_generator import attach_dynamo_case
from ._helpers import api_runner, exported_onnx_path, prepare_runtime_model
from .model_registry import DYNAMO_MODEL_SPECS, DynamoModelSpec, spec_ids


@pytest.mark.dynamo
@pytest.mark.dynamo_export
@pytest.mark.llm_model
@pytest.mark.regular
@pytest.mark.parametrize("spec", DYNAMO_MODEL_SPECS, ids=spec_ids(DYNAMO_MODEL_SPECS))
def test_dynamo_hf_qeff_ort_runtime_parity(spec: DynamoModelSpec, dynamo_workdir, request):
    """HF PyTorch tokens == modified QEff PyTorch tokens == ORT tokens.

    Uses ``use_onnx_subfunctions=spec.subfunctions_supported`` so every arch
    runs this lane regardless of subfunction support.
    """
    use_subfn = spec.subfunctions_supported
    coverage_cols = ("CPU_Parity", "FP32_Coverage")
    if use_subfn:
        coverage_cols = ("CPU_Parity", "Subfunction_Coverage", "FP32_Coverage")

    attach_dynamo_case(
        request,
        category=spec.category,
        task="dynamo_hf_qeff_ort_parity",
        architecture=spec.architecture,
        family=spec.family,
        supported_model=spec.model_id,
        coverage_columns=coverage_cols,
        notes=spec.notes,
    )

    tokenizer, model_hf, build_qeff = prepare_runtime_model(spec)
    runner = api_runner(tokenizer, model_hf.config)
    hf_tokens = runner.run_hf_model_on_pytorch(model_hf)
    qeff_model = build_qeff()
    kv_tokens = runner.run_kv_model_on_pytorch(qeff_model.model)

    workdir = dynamo_workdir(architecture=spec.architecture, feature="parity_ort", precision="fp32")
    onnx_path = exported_onnx_path(qeff_model.export(workdir, use_dynamo=True, use_onnx_subfunctions=use_subfn))
    ort_tokens = runner.run_kv_model_on_ort(str(onnx_path))

    assert np.array_equal(hf_tokens, kv_tokens.squeeze(0)), "HF vs QEff PyTorch parity failed"
    assert np.array_equal(kv_tokens, ort_tokens), "QEff PyTorch vs ORT parity failed"
