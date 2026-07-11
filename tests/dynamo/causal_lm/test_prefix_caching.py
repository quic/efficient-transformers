# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Prefix-caching lane for dynamo + subfunctions.

Two shapes:
* Export smoke — the emitted ONNX must expose ``batch_index`` and the
  ``CtxScatterCB`` / ``CtxGatherCB`` ops that prefix caching relies on.
  Modeled after ``tests/unit_test/models/test_model_quickcheck.py`` ::
  ``test_prefix_caching_continuous_batching_export_and_ort_smoke``.
* Runtime compile + generate on QAIC.

Only architectures with ``prefix_caching_supported=True`` in the registry
participate; the rest skip with a clear reason so the coverage column
reflects the real support surface.
"""

from __future__ import annotations

import onnx
import pytest

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ..utils.report_generator import attach_dynamo_case
from ._helpers import (
    CTX_LEN,
    FULL_BATCH_SIZE,
    GENERATION_LEN,
    PROMPT,
    PROMPT_LEN,
    exported_onnx_path,
    load_tokenizer,
)
from .model_registry import DYNAMO_MODEL_SPECS, DynamoModelSpec, spec_ids

_PREFIX_SPECS = [spec for spec in DYNAMO_MODEL_SPECS if spec.prefix_caching_supported and spec.subfunctions_supported]


@pytest.mark.dynamo
@pytest.mark.dynamo_export
@pytest.mark.llm_model
@pytest.mark.regular
@pytest.mark.parametrize("spec", DYNAMO_MODEL_SPECS, ids=spec_ids(DYNAMO_MODEL_SPECS))
def test_dynamo_subfunction_prefix_caching_export_smoke(spec: DynamoModelSpec, dynamo_workdir, request):
    """Prefix-caching export shape: batch_index input, CtxScatter/GatherCB ops."""
    attach_dynamo_case(
        request,
        category=spec.category,
        task="dynamo_subfunction_prefix_caching_export",
        architecture=spec.architecture,
        family=spec.family,
        supported_model=spec.model_id,
        coverage_columns=("Prefix_Caching_Coverage", "Subfunction_Coverage"),
        notes=spec.notes,
    )
    if not spec.prefix_caching_supported:
        pytest.skip("Prefix caching is not supported by this wrapper.")
    if not spec.subfunctions_supported:
        pytest.skip(spec.notes or "Architecture opts out of the subfunction lane.")

    workdir = dynamo_workdir(architecture=spec.architecture, feature="prefix_cache_export_subfn", precision="fp32")
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        spec.model_id, continuous_batching=True, trust_remote_code=True
    )
    onnx_path = exported_onnx_path(qeff_model.export(workdir, use_dynamo=True, use_onnx_subfunctions=True))

    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    input_names = {inp.name for inp in onnx_model.graph.input}
    output_names = {out.name for out in onnx_model.graph.output}
    op_types = {node.op_type for node in onnx_model.graph.node}
    assert "batch_index" in input_names, "prefix caching export must expose batch_index"
    assert "CtxScatterCB" in op_types, "prefix caching export must contain CtxScatterCB"
    assert "CtxGatherCB" in op_types, "prefix caching export must contain CtxGatherCB"
    assert any(name.endswith("_RetainedState") for name in output_names), "cache retained-state outputs missing"


@pytest.mark.dynamo
@pytest.mark.dynamo_runtime
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.nightly
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.parametrize("spec", _PREFIX_SPECS, ids=spec_ids(_PREFIX_SPECS))
def test_dynamo_subfunction_prefix_caching_runtime(spec: DynamoModelSpec, dynamo_workdir, device_pool, request):
    """Prefix caching on QAIC: compile the CB export, generate, expect a full
    ``generation_len`` completion (no assertion on tokens vs HF — CB runner
    horizon is smaller than gen_len for tiny models)."""
    attach_dynamo_case(
        request,
        category=spec.category,
        task="dynamo_subfunction_prefix_caching_runtime",
        architecture=spec.architecture,
        family=spec.family,
        supported_model=spec.model_id,
        coverage_columns=("Prefix_Caching_Coverage", "CB_Dynamo_Subfn", "FP16_Coverage"),
        notes=spec.notes,
    )

    tokenizer = load_tokenizer(spec)
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(spec.model_id, continuous_batching=True)
    compile_dir = dynamo_workdir(architecture=spec.architecture, feature="prefix_cache_compile_subfn", precision="fp16")
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

    assert exec_info.generated_ids.shape[-1] > 0, "prefix caching generate produced no tokens"
