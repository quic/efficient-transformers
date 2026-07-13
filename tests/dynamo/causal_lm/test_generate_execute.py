# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
QAIC generate/execute lane.

End-to-end: dynamo + subfunctions export -> compile -> ORT reference tokens
-> QAIC generate -> assert QAIC == ORT token stream. Serialised through the
``qaic-runtime`` xdist group and single-device acquisition from the
``DevicePool`` so multiple runtime lanes cannot double-book a physical
device.
"""

from __future__ import annotations

import numpy as np
import pytest

from ..utils.report_generator import attach_dynamo_case
from ._helpers import (
    CTX_LEN,
    PROMPT_LEN,
    api_runner,
    assert_qconfig_exists,
    exported_onnx_path,
    generate_without_sampler,
    prepare_runtime_model,
)
from .model_registry import DYNAMO_MODEL_SPECS, DynamoModelSpec, spec_ids


@pytest.mark.dynamo
@pytest.mark.dynamo_runtime
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.nightly
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.parametrize("spec", DYNAMO_MODEL_SPECS, ids=spec_ids(DYNAMO_MODEL_SPECS))
def test_dynamo_subfunction_qaic_generate_runtime_parity(spec: DynamoModelSpec, dynamo_workdir, device_pool, request):
    use_subfn = spec.subfunctions_supported
    coverage_cols = (
        "End_To_End_E2E",
        "QAIC_Generate_Execute",
        "HW_Parity_FP16",
        "FP16_Coverage",
    )
    if use_subfn:
        coverage_cols = (
            "End_To_End_E2E",
            "QAIC_Generate_Execute",
            "HW_Parity_FP16",
            "Subfunction_Coverage",
            "FP16_Coverage",
        )

    attach_dynamo_case(
        request,
        category=spec.category,
        task="dynamo_qaic_generate_parity",
        architecture=spec.architecture,
        family=spec.family,
        supported_model=spec.model_id,
        coverage_columns=coverage_cols,
        notes=spec.notes,
    )

    use_subfn = spec.subfunctions_supported

    tokenizer, model_hf, build_qeff = prepare_runtime_model(spec)
    runner = api_runner(tokenizer, model_hf.config)

    export_dir = dynamo_workdir(architecture=spec.architecture, feature="qaic_generate_export", precision="fp32")
    compile_dir = dynamo_workdir(architecture=spec.architecture, feature="qaic_generate_compile", precision="fp16")

    qeff_model = build_qeff()
    onnx_path = exported_onnx_path(qeff_model.export(export_dir, use_dynamo=True, use_onnx_subfunctions=use_subfn))
    ort_tokens = runner.run_kv_model_on_ort(str(onnx_path))

    with device_pool.acquire(1):
        qpc_path = qeff_model.compile(
            onnx_path=str(onnx_path),
            compile_dir=str(compile_dir),
            prefill_seq_len=PROMPT_LEN,
            ctx_len=CTX_LEN,
            num_devices=1,
            num_cores=16,
            use_dynamo=True,
            use_onnx_subfunctions=use_subfn,
        )
        assert_qconfig_exists(qpc_path)
        exec_info = generate_without_sampler(tokenizer, qeff_model)

    qaic_tokens = exec_info.generated_ids[0][:, : ort_tokens.shape[-1]]
    assert np.array_equal(ort_tokens, qaic_tokens), "QAIC vs ORT token divergence in dynamo runtime lane"
