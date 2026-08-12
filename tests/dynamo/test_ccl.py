# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Dynamo + Compute-Context-Length (CCL) compile/generate tests.
"""

from __future__ import annotations

import pytest

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ._helpers import (
    BATCH_SIZE,
    DYNAMO_CAUSAL_LM_MODEL_IDS,
    FULL_BATCH_SIZE,
    load_hf_model,
    load_tokenizer,
    skip_on_model_fetch_error,
)

# CCL's specialization validation floors small comp_ctx_lengths values up to
# CCL_MIN_CTX_LEN (1024) then clamps back down to ctx_len; with the shared tiny
# CTX_LEN=16 used elsewhere in tests/dynamo/, both prefill and decode collapse to
# the same value and the collision-repair step (walking down by CCL_UNIQNE_STEP=32)
# lands on 0, which the compiler rejects. Use ctx_len/prefill_seq_len large enough
# to avoid that collision (matches values validated in manual CCL automation runs).
CCL_PREFILL_SEQ_LEN = 32
CCL_CTX_LEN = 128
CCL_LENGTHS = [1024, 2048]


def _generate(qeff_model, tokenizer, prompts):
    """Reuses the QPC the preceding compile() call just produced (qeff_model.qpc_path) -- no recompile."""
    output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        device_id=[0],
    )
    assert output is not None
    assert output.generated_texts is not None
    return output


@pytest.mark.dynamo
@pytest.mark.on_qaic
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_type,model_id", list(DYNAMO_CAUSAL_LM_MODEL_IDS.items()), ids=list(DYNAMO_CAUSAL_LM_MODEL_IDS)
)
def test_dynamo_ccl_compile_and_generate(model_type, model_id, tmp_export_dir):
    """Compile the shared ccl_enabled export normally, compile with CCL, generate on each QPC."""
    if model_type == "gpt_oss":
        pytest.skip("gpt_oss CB scatter op has shape mismatch with dynamo subfunctions — pending fix")

    try:
        model_hf = load_hf_model(model_id)
        tokenizer = load_tokenizer(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    qeff_model = QEFFAutoModelForCausalLM(model_hf, qaic_config={"ccl_enabled": True})
    # --- Compile normally (auto-generated CCL lists; no explicit comp_ctx_lengths) ---
    qeff_model.compile(
        compile_dir=str(tmp_export_dir / "normal_compile"),
        prefill_seq_len=CCL_PREFILL_SEQ_LEN,
        ctx_len=CCL_CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
        dynamo=True,
    )
    _generate(qeff_model, tokenizer, ["hello world"])

    # --- Compile with explicit CCL specializations, from the same export ---
    qeff_model.compile(
        compile_dir=str(tmp_export_dir / "ccl_compile"),
        prefill_seq_len=CCL_PREFILL_SEQ_LEN,
        ctx_len=CCL_CTX_LEN,
        comp_ctx_lengths_prefill=CCL_LENGTHS,
        comp_ctx_lengths_decode=CCL_LENGTHS,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
        dynamo=True,
    )
    _generate(qeff_model, tokenizer, ["hello world"])


@pytest.mark.dynamo
@pytest.mark.on_qaic
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_type,model_id", list(DYNAMO_CAUSAL_LM_MODEL_IDS.items()), ids=list(DYNAMO_CAUSAL_LM_MODEL_IDS)
)
def test_dynamo_cb_ccl_compile_and_generate(model_type, model_id, tmp_export_dir):
    """Continuous-batching + CCL: compile the shared CB+ccl_enabled export normally, compile with CCL,
    generate on each QPC."""
    # TODO: fix gpt_oss CB scatter op shape mismatch with dynamo subfunctions (see test_dynamo_cb_generate).
    if model_type == "gpt_oss":
        pytest.skip("gpt_oss CB scatter op has shape mismatch with dynamo subfunctions — pending fix")

    try:
        model_hf = load_hf_model(model_id)
        tokenizer = load_tokenizer(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    qeff_model = QEFFAutoModelForCausalLM(model_hf, continuous_batching=True, qaic_config={"ccl_enabled": True})

    prompts = ["hello world"] * FULL_BATCH_SIZE

    # --- Compile normally (auto-generated CCL lists; no explicit comp_ctx_lengths) ---
    qeff_model.compile(
        compile_dir=str(tmp_export_dir / "cb_normal_compile"),
        prefill_seq_len=CCL_PREFILL_SEQ_LEN,
        ctx_len=CCL_CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        full_batch_size=FULL_BATCH_SIZE,
        use_onnx_subfunctions=True,
        dynamo=True,
    )
    _generate(qeff_model, tokenizer, prompts)

    # --- Compile with explicit CCL specializations, from the same export ---
    qeff_model.compile(
        compile_dir=str(tmp_export_dir / "cb_ccl_compile"),
        prefill_seq_len=CCL_PREFILL_SEQ_LEN,
        ctx_len=CCL_CTX_LEN,
        comp_ctx_lengths_prefill=CCL_LENGTHS,
        comp_ctx_lengths_decode=CCL_LENGTHS,
        num_cores=16,
        batch_size=BATCH_SIZE,
        full_batch_size=FULL_BATCH_SIZE,
        use_onnx_subfunctions=True,
        dynamo=True,
    )
    _generate(qeff_model, tokenizer, prompts)
