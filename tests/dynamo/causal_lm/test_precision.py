# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Precision compile lanes (compile-only, no parity check).

The compressed compile lanes always pair ``use_dynamo=True`` with
``use_onnx_subfunctions=True`` — that's the shipping shape. Parity is
intentionally not checked for compressed modes.
"""

from __future__ import annotations

import pytest

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ..utils.report_generator import attach_dynamo_case
from ._helpers import CTX_LEN, PROMPT_LEN, assert_qconfig_exists, load_model
from .model_registry import DYNAMO_MODEL_SPECS, DynamoModelSpec, spec_ids

COMPRESSED_MODES = ("mxfp6", "mxfp6_mxint8")


@pytest.mark.dynamo
@pytest.mark.dynamo_compile
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.regular
@pytest.mark.precision("mxfp6")
@pytest.mark.parametrize("mode", COMPRESSED_MODES)
@pytest.mark.parametrize("spec", DYNAMO_MODEL_SPECS, ids=spec_ids(DYNAMO_MODEL_SPECS))
def test_dynamo_subfunction_compile_compressed(spec: DynamoModelSpec, mode: str, dynamo_workdir, request):
    """Compile-only validation for MXFP6 / MXFP6+MXINT8 KV-cache modes."""
    coverage_column = "MXINT8_Coverage" if mode == "mxfp6_mxint8" else "MXFP6_Coverage"
    attach_dynamo_case(
        request,
        category=spec.category,
        task=f"dynamo_subfunction_compile_{mode}",
        architecture=spec.architecture,
        family=spec.family,
        supported_model=spec.model_id,
        coverage_columns=("Export_Compile", "Subfunction_Coverage", coverage_column),
        notes=f"dynamo + subfunctions + {mode} compile-only; parity intentionally skipped.",
    )
    if not spec.subfunctions_supported:
        pytest.skip(spec.notes or "Architecture opts out of the subfunction lane.")

    workdir = dynamo_workdir(architecture=spec.architecture, feature=f"compile_subfn_{mode}", precision=mode)
    model_hf = load_model(spec)
    qeff_model = QEFFAutoModelForCausalLM(model_hf, pretrained_model_name_or_path=spec.model_id)
    compile_kwargs = dict(
        compile_dir=str(workdir),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_devices=1,
        num_cores=16,
        use_dynamo=True,
        use_onnx_subfunctions=True,
        mxfp6_matmul=True,
    )
    if mode == "mxfp6_mxint8":
        compile_kwargs["mxint8_kv_cache"] = True
    qpc_path = qeff_model.compile(**compile_kwargs)
    assert_qconfig_exists(qpc_path)
