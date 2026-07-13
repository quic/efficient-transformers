# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Multi-device (MDP) compile lane.

Compiles a core-tier CausalLM with ``num_devices > 1``, ``use_dynamo=True``,
``use_onnx_subfunctions=True``. Atomically acquires the requested device
count from the shared pool so we don't double-book physical devices across
xdist workers.
"""

from __future__ import annotations

import pytest

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ..utils.report_generator import attach_dynamo_case
from ._helpers import CTX_LEN, PROMPT_LEN, assert_qconfig_exists, load_model
from .model_registry import DynamoModelSpec, spec_ids, specs_with

_MULTI_DEVICE_SPECS = specs_with(continuous_batching=True, subfunctions=True)


@pytest.mark.dynamo
@pytest.mark.dynamo_compile
@pytest.mark.multi_device
@pytest.mark.on_qaic
@pytest.mark.nightly
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.parametrize("num_devices", (2,))
@pytest.mark.parametrize("spec", _MULTI_DEVICE_SPECS, ids=spec_ids(_MULTI_DEVICE_SPECS))
def test_dynamo_subfunction_multi_device_compile(
    spec: DynamoModelSpec, num_devices: int, dynamo_workdir, device_pool, request
):
    attach_dynamo_case(
        request,
        category=spec.category,
        task=f"dynamo_subfunction_multi_device_compile_x{num_devices}",
        architecture=spec.architecture,
        family=spec.family,
        supported_model=spec.model_id,
        coverage_columns=("CCL_Coverage", "Subfunction_Coverage", "FP16_Coverage"),
        notes=f"Multi-device compile with num_devices={num_devices}, subfunctions=on.",
    )
    if num_devices > len(device_pool.device_ids):
        pytest.skip(f"Only {len(device_pool.device_ids)} devices in pool; test requires {num_devices}.")

    model_hf = load_model(spec)
    qeff_model = QEFFAutoModelForCausalLM(model_hf, pretrained_model_name_or_path=spec.model_id)
    compile_dir = dynamo_workdir(
        architecture=spec.architecture,
        feature="multi_device_compile_subfn",
        precision="fp16",
        extras=(f"nd{num_devices}",),
    )

    with device_pool.acquire(num_devices):
        qpc_path = qeff_model.compile(
            compile_dir=str(compile_dir),
            prefill_seq_len=PROMPT_LEN,
            ctx_len=CTX_LEN,
            num_devices=num_devices,
            num_cores=16,
            use_dynamo=True,
            use_onnx_subfunctions=True,
        )
    assert_qconfig_exists(qpc_path)
