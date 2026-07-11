# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
On-device sampler lane for dynamo + subfunctions.

Only architectures with ``sampler_supported=True`` in the registry
participate. The test compiles both a sampler-enabled and a sampler-disabled
QPC, runs generation with a deterministic sampling configuration
(temperature=0, top_k=vocab_size), and asserts the two outputs match token
for token — the on-device sampler under greedy decoding must be equivalent
to the classic argmax path.
"""

from __future__ import annotations

import numpy as np
import pytest

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ..utils.report_generator import attach_dynamo_case
from ._helpers import CTX_LEN, GENERATION_LEN, PROMPT, PROMPT_LEN, load_tokenizer
from .model_registry import DYNAMO_MODEL_SPECS, DynamoModelSpec, spec_ids

_SAMPLER_SPECS = [spec for spec in DYNAMO_MODEL_SPECS if spec.sampler_supported and spec.subfunctions_supported]


@pytest.mark.dynamo
@pytest.mark.dynamo_runtime
@pytest.mark.llm_model
@pytest.mark.on_qaic
@pytest.mark.nightly
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.parametrize("spec", DYNAMO_MODEL_SPECS, ids=spec_ids(DYNAMO_MODEL_SPECS))
def test_dynamo_subfunction_sampler_runtime(spec: DynamoModelSpec, dynamo_workdir, device_pool, request):
    """On-device sampler under greedy decoding matches the argmax path."""
    attach_dynamo_case(
        request,
        category=spec.category,
        task="dynamo_subfunction_sampler_runtime",
        architecture=spec.architecture,
        family=spec.family,
        supported_model=spec.model_id,
        coverage_columns=("Sampler_Coverage", "CB_Dynamo_Subfn", "FP16_Coverage"),
        notes=spec.notes,
    )
    if not spec.sampler_supported:
        pytest.skip("Sampler is not supported by this wrapper.")
    if not spec.subfunctions_supported:
        pytest.skip(spec.notes or "Architecture opts out of the subfunction lane.")

    tokenizer = load_tokenizer(spec)

    common_compile = dict(
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN * 8,
        full_batch_size=1,
        num_devices=1,
        num_cores=16,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        use_dynamo=True,
        use_onnx_subfunctions=True,
    )

    with_sampler_dir = dynamo_workdir(architecture=spec.architecture, feature="sampler_with", precision="mxfp6_mxint8")
    without_sampler_dir = dynamo_workdir(
        architecture=spec.architecture, feature="sampler_without", precision="mxfp6_mxint8"
    )

    model_with = QEFFAutoModelForCausalLM.from_pretrained(
        spec.model_id,
        continuous_batching=True,
        qaic_config={"include_sampler": True, "return_pdfs": False, "max_top_k_ids": 512},
    )
    model_without = QEFFAutoModelForCausalLM.from_pretrained(
        spec.model_id,
        continuous_batching=True,
        qaic_config={"include_sampler": False, "return_pdfs": False},
    )

    with device_pool.acquire(1):
        model_with.compile(compile_dir=str(with_sampler_dir), **common_compile)
        model_without.compile(compile_dir=str(without_sampler_dir), **common_compile)

        sampling_params = {
            "repetition_penalties": np.array([[1.0]], dtype=np.float32),
            "presence_penalties": np.array([[0.0]], dtype=np.float32),
            "temperatures": np.array([[0.0]], dtype=np.float32),
            "top_ks": np.array([[512]], dtype=np.int32),
            "top_ps": np.array([[1.0]], dtype=np.float32),
            "min_ps": np.array([[0.0]], dtype=np.float32),
            "random_numbers": np.zeros((1, 512), dtype=np.float32),
        }
        with_out = model_with.generate(
            tokenizer=tokenizer,
            prompts=PROMPT,
            generation_len=GENERATION_LEN,
            include_sampler=True,
            return_pdfs=False,
            sampling_params=sampling_params,
        )
        without_out = model_without.generate(
            tokenizer=tokenizer,
            prompts=PROMPT,
            generation_len=GENERATION_LEN,
            include_sampler=False,
            return_pdfs=False,
        )

    assert with_out.generated_texts == without_out.generated_texts, "sampler vs argmax text divergence"
    assert np.array_equal(with_out.generated_ids, without_out.generated_ids), "sampler vs argmax token divergence"
