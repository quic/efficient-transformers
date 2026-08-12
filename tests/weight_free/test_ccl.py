# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Weight-free + Compute-Context-Length (CCL) compile/generate tests.
"""

from __future__ import annotations

import pytest

from QEfficient.utils.device_utils import get_available_device_id

from ._helpers import (
    BATCH_SIZE,
    WEIGHT_FREE_CAUSAL_LM_MODEL_IDS,
    build_meta_qeff_model,
    load_tokenizer,
    skip_on_model_fetch_error,
)

# Mirrors tests/dynamo/test_ccl.py: CCL's specialization validation floors small
# comp_ctx_lengths values up to CCL_MIN_CTX_LEN (1024) then clamps back down to
# ctx_len; with a tiny CTX_LEN, both prefill and decode collapse to the same value
# and the collision-repair step lands on 0, which the compiler rejects. Use
# ctx_len/prefill_seq_len large enough to avoid that collision.
CCL_PREFILL_SEQ_LEN = 32
CCL_CTX_LEN = 128
CCL_LENGTHS = [1024, 2048]


def _generate(qeff_model, tokenizer, prompts):
    """Reuses the QPC the preceding compile() call just produced (qeff_model.qpc_path) -- no recompile."""
    output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        device_id=get_available_device_id(),
    )
    assert output is not None
    assert output.generated_texts is not None
    return output


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.xdist_group(name="qaic-runtime")
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_type,model_id",
    sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS.items()),
    ids=sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS),
)
def test_weight_free_ccl_compile_and_generate(model_type, model_id, tmp_export_dir):
    """Compile the shared weight-free + ccl_enabled export normally, compile with CCL, generate on each QPC."""
    if model_type == "gpt_oss":
        pytest.xfail()

    try:
        qeff_model = build_meta_qeff_model(model_id, qaic_config={"ccl_enabled": True})
        tokenizer = load_tokenizer(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    # --- Compile normally (auto-generated CCL lists; no explicit comp_ctx_lengths) ---
    qeff_model.compile(
        compile_dir=str(tmp_export_dir / "wf_normal_compile"),
        prefill_seq_len=CCL_PREFILL_SEQ_LEN,
        ctx_len=CCL_CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
        use_weight_free_export=True,
        offload_pt_weights=False,
    )
    _generate(qeff_model, tokenizer, ["hello world"])

    # --- Compile with explicit CCL specializations, from the same export ---
    qeff_model.compile(
        compile_dir=str(tmp_export_dir / "wf_ccl_compile"),
        prefill_seq_len=CCL_PREFILL_SEQ_LEN,
        ctx_len=CCL_CTX_LEN,
        comp_ctx_lengths_prefill=CCL_LENGTHS,
        comp_ctx_lengths_decode=CCL_LENGTHS,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
        use_weight_free_export=True,
        offload_pt_weights=False,
    )
    _generate(qeff_model, tokenizer, ["hello world"])
