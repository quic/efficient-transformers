# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Dynamo continuous-batching export tests.

Every architecture is tested twice:
  - dynamo=True, use_onnx_subfunctions=False
  - dynamo=True, use_onnx_subfunctions=True  (skipped for DYNAMO_NO_SUBFUNCTION_ARCHS)

CPU-only. No QAIC hardware required.
"""

from __future__ import annotations

import pytest

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ._helpers import (
    DYNAMO_CAUSAL_LM_MODEL_IDS,
    DYNAMO_NO_SUBFUNCTION_ARCHS,
    assert_has_subfunctions,
    assert_retained_state_outputs,
    exported_onnx_path,
    load_hf_model,
    skip_on_model_fetch_error,
)


@pytest.mark.dynamo
@pytest.mark.dynamo_export
@pytest.mark.regular
@pytest.mark.parametrize("use_onnx_subfunctions", [False, True], ids=["flat", "subfn"])
@pytest.mark.parametrize(
    "model_type,model_id", sorted(DYNAMO_CAUSAL_LM_MODEL_IDS.items()), ids=sorted(DYNAMO_CAUSAL_LM_MODEL_IDS)
)
def test_dynamo_cb_export(model_type, model_id, use_onnx_subfunctions, tmp_export_dir):
    """continuous_batching=True + dynamo=True for both subfunction modes."""
    if use_onnx_subfunctions and model_type in DYNAMO_NO_SUBFUNCTION_ARCHS:
        pytest.skip(f"{model_type} does not support subfunctions under dynamo")

    try:
        model_hf = load_hf_model(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    subfn_label = "subfn" if use_onnx_subfunctions else "flat"
    qeff_model = QEFFAutoModelForCausalLM(model_hf, continuous_batching=True)
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / subfn_label,
            dynamo=True,
            use_onnx_subfunctions=use_onnx_subfunctions,
            offload_pt_weights=not use_onnx_subfunctions,
        )
    )

    num_layers = model_hf.config.num_hidden_layers
    assert_retained_state_outputs(onnx_path, expected_count=2 * num_layers)

    if use_onnx_subfunctions:
        assert_has_subfunctions(onnx_path)
