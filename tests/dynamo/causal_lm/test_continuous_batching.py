# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Continuous-batching runtime lane.

Dynamo + subfunctions + CB compile + generate. HF CB vs QAIC CB tokens.
The blocking (user-tiled) variant lives in ``test_blocking.py`` so the
coverage matrix has one dedicated Blocking_KV column.
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
from .model_registry import DynamoModelSpec, spec_ids, specs_with

_CB_SPECS = specs_with(continuous_batching=True, subfunctions=True)


@pytest.mark.dynamo
@pytest.mark.dynamo_runtime
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.nightly
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.parametrize("spec", _CB_SPECS, ids=spec_ids(_CB_SPECS))
def test_dynamo_subfunction_continuous_batching_runtime(spec: DynamoModelSpec, dynamo_workdir, device_pool, request):
    """dynamo + subfunctions + CB compile + generate. HF CB vs QAIC CB tokens."""
    attach_dynamo_case(
        request,
        category=spec.category,
        task="dynamo_subfunction_continuous_batching_runtime",
        architecture=spec.architecture,
        family=spec.family,
        supported_model=spec.model_id,
        coverage_columns=(
            "End_To_End_E2E",
            "QAIC_Generate_Execute",
            "CB_Dynamo_Subfn",
            "Subfunction_Coverage",
            "FP16_Coverage",
        ),
        notes=spec.notes,
    )

    tokenizer, model_hf, _ = prepare_runtime_model(spec, continuous_batching=True)
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(spec.model_id, continuous_batching=True)
    runner = api_runner(tokenizer, model_hf.config, continuous_batching=True)
    hf_tokens = runner.run_hf_model_on_pytorch_CB(model_hf)

    compile_dir = dynamo_workdir(architecture=spec.architecture, feature="cb_compile_subfn", precision="fp16")
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
        )
        exec_info = qeff_model.generate(
            tokenizer=tokenizer,
            prompts=PROMPT * FULL_BATCH_SIZE,
            generation_len=GENERATION_LEN,
        )

    qaic_tokens = exec_info.generated_ids
    hf_stack = np.vstack(hf_tokens)
    horizon = min(hf_stack.shape[-1], qaic_tokens.shape[-1])
    assert np.array_equal(hf_stack[:, :horizon], qaic_tokens[:, :horizon]), "HF vs QAIC CB token divergence"
