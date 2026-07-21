# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Dynamo ORT parity tests.

All tests run with dynamo=True and use_onnx_subfunctions=True.
Validates HF PT == ORT token parity via ApiRunner.
CPU-only. No QAIC hardware required.
"""

from __future__ import annotations

import pytest

from ._helpers import (
    DYNAMO_CAUSAL_LM_MODEL_IDS,
    load_hf_model,
    load_tokenizer,
    run_dynamo_ort_parity,
    skip_on_model_fetch_error,
)


@pytest.mark.dynamo
@pytest.mark.dynamo_export
@pytest.mark.parametrize(
    "model_type,model_id", sorted(DYNAMO_CAUSAL_LM_MODEL_IDS.items()), ids=sorted(DYNAMO_CAUSAL_LM_MODEL_IDS)
)
def test_dynamo_hf_ort_parity(model_type, model_id, tmp_export_dir):
    """HF PT == ORT parity with dynamo=True and use_onnx_subfunctions=True."""
    if model_type == "gpt_oss":
        pytest.xfail()

    try:
        model_hf = load_hf_model(model_id)
        tokenizer = load_tokenizer(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    run_dynamo_ort_parity(model_hf, tokenizer, tmp_export_dir, use_onnx_subfunctions=True)
