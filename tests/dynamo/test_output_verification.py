# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Dynamo ORT parity tests.

Every architecture is tested twice:
  - dynamo=True, use_onnx_subfunctions=False
  - dynamo=True, use_onnx_subfunctions=True  (skipped for DYNAMO_NO_SUBFUNCTION_ARCHS)

Validates HF PT == QEff PT == ORT token parity via ApiRunner.
CPU-only. No QAIC hardware required.
"""

from __future__ import annotations

import pytest

from ._helpers import (
    DYNAMO_CAUSAL_LM_MODEL_IDS,
    DYNAMO_NO_SUBFUNCTION_ARCHS,
    run_dynamo_ort_parity,
    skip_on_model_fetch_error,
)


@pytest.mark.dynamo
@pytest.mark.dynamo_export
@pytest.mark.regular
@pytest.mark.parametrize("use_onnx_subfunctions", [False, True], ids=["flat", "subfn"])
@pytest.mark.parametrize(
    "model_type,model_id", sorted(DYNAMO_CAUSAL_LM_MODEL_IDS.items()), ids=sorted(DYNAMO_CAUSAL_LM_MODEL_IDS)
)
def test_dynamo_hf_qeff_ort_parity(model_type, model_id, use_onnx_subfunctions, tmp_export_dir):
    """HF PT == QEff PT == ORT parity for both subfunction modes."""
    if use_onnx_subfunctions and model_type in DYNAMO_NO_SUBFUNCTION_ARCHS:
        pytest.skip(f"{model_type} does not support subfunctions under dynamo")

    subfn_label = "subfn" if use_onnx_subfunctions else "flat"
    try:
        run_dynamo_ort_parity(model_id, tmp_export_dir / subfn_label, use_onnx_subfunctions=use_onnx_subfunctions)
    except Exception as exc:
        if "unavailable" in str(exc).lower() or "download" in str(exc).lower():
            skip_on_model_fetch_error(exc, model_id)
        raise
