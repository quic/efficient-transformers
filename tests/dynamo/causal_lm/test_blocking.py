# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
KV blocking (user-tiled) runtime lane.

Exercises ``BlockingMode.KV`` end to end on QAIC with dynamo + subfunctions
+ CB. Only architectures marked ``blocking_kv_supported=True`` participate;
the rest skip so the coverage column reflects the real support surface.
Blocking modes other than KV are out of scope for now — Chunked / KV+Chunked
lanes can be added later next to this test.
"""

from __future__ import annotations

import numpy as np
import pytest

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ..utils.report_generator import attach_dynamo_case
from ._helpers import (
    CTX_LEN,
    FULL_BATCH_SIZE,
    GENERATION_LEN,
    PROMPT,
    PROMPT_LEN,
    api_runner,
    prepare_runtime_model,
)
from .model_registry import DYNAMO_MODEL_SPECS, DynamoModelSpec, spec_ids

_BLOCKING_SPECS = [spec for spec in DYNAMO_MODEL_SPECS if spec.blocking_kv_supported and spec.subfunctions_supported]


@pytest.mark.dynamo
@pytest.mark.dynamo_runtime
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.nightly
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.parametrize("spec", DYNAMO_MODEL_SPECS, ids=spec_ids(DYNAMO_MODEL_SPECS))
def test_dynamo_subfunction_blocking_kv_runtime(spec: DynamoModelSpec, dynamo_workdir, device_pool, request):
    """dynamo + subfunctions + CB + BlockingMode.KV (user-tiled) end to end."""
    attach_dynamo_case(
        request,
        category=spec.category,
        task="dynamo_subfunction_blocking_kv_runtime",
        architecture=spec.architecture,
        family=spec.family,
        supported_model=spec.model_id,
        coverage_columns=(
            "Blocking_KV_Coverage",
            "CB_Dynamo_Subfn",
            "Subfunction_Coverage",
            "FP16_Coverage",
        ),
        notes=spec.notes,
    )
    if not spec.blocking_kv_supported:
        pytest.skip("KV blocking is not enabled for this wrapper in the registry.")
    if not spec.subfunctions_supported:
        pytest.skip(spec.notes or "Architecture opts out of the subfunction lane.")

    tokenizer, model_hf, _ = prepare_runtime_model(spec, continuous_batching=True)
    runner = api_runner(tokenizer, model_hf.config, continuous_batching=True)
    hf_tokens = runner.run_hf_model_on_pytorch_CB(model_hf)

    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        spec.model_id,
        continuous_batching=True,
        qaic_config={"enable_blocking": True, "blocking_mode": "kv", "num_kv_blocks": 2},
    )
    compile_dir = dynamo_workdir(architecture=spec.architecture, feature="blocking_kv_compile_subfn", precision="fp16")

    with device_pool.acquire(1):
        qeff_model.compile(
            compile_dir=str(compile_dir),
            prefill_seq_len=PROMPT_LEN,
            ctx_len=CTX_LEN * 8,
            batch_size=1,
            full_batch_size=FULL_BATCH_SIZE,
            num_devices=1,
            num_cores=16,
            use_dynamo=True,
            use_onnx_subfunctions=True,
            user_tiled=True,
        )
        exec_info = qeff_model.generate(
            tokenizer=tokenizer,
            prompts=PROMPT * FULL_BATCH_SIZE,
            generation_len=GENERATION_LEN,
        )

    qaic_tokens = exec_info.generated_ids
    hf_stack = np.vstack(hf_tokens)
    horizon = min(hf_stack.shape[-1], qaic_tokens.shape[-1])
    assert np.array_equal(hf_stack[:, :horizon], qaic_tokens[:, :horizon]), "HF vs QAIC KV-blocking token divergence"
