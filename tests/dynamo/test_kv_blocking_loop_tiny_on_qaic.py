# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Run tiny CausalLM KV-headpar Dynamo Loop models on QAIC.

This mirrors examples/dynamo/causal_lm/basic_dynamo_inference.py, but keeps the
coverage in pytest form and pins the KV-loop dimensions requested for regression:
PL=64, CL=128, num_kv_blocks=2.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ._helpers import DYNAMO_CAUSAL_LM_MODEL_IDS, load_hf_model, load_tokenizer, skip_on_model_fetch_error

PREFILL_SEQ_LEN = 64
CTX_LEN = 128
NUM_KV_BLOCKS = 2
HEADPAR_SPLIT = 2
BATCH_SIZE = 1
GENERATION_LEN = 2
NUM_CORES = 2
PROMPT = "My name is"

KV_HEADPAR_LOOP_TINY_MODEL_TYPES = {
    "gpt_oss",
    "granite",
    "llama",
    "mistral",
    "mixtral",
    "mpt",
    "qwen2",
    "starcoder2",
}
KV_HEADPAR_LOOP_TINY_MODEL_IDS = {
    model_type: model_id
    for model_type, model_id in DYNAMO_CAUSAL_LM_MODEL_IDS.items()
    if model_type in KV_HEADPAR_LOOP_TINY_MODEL_TYPES
}


@pytest.mark.dynamo
@pytest.mark.dynamo_export
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_type,model_id",
    sorted(KV_HEADPAR_LOOP_TINY_MODEL_IDS.items()),
    ids=sorted(KV_HEADPAR_LOOP_TINY_MODEL_IDS),
)
def test_kv_headpar_dynamo_loop_subfunctions_tiny_generate_on_qaic(model_type, model_id, tmp_export_dir):
    del model_type
    qaic_config = {
        "blocking_mode": "kv_headpar",
        "num_kv_blocks": NUM_KV_BLOCKS,
        "headpar_split": HEADPAR_SPLIT,
    }

    try:
        model_hf = load_hf_model(model_id)
        tokenizer = load_tokenizer(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    qeff_model = QEFFAutoModelForCausalLM(model_hf, continuous_batching=False, qaic_config=qaic_config)
    qpc_path = qeff_model.compile(
        compile_dir=str(tmp_export_dir / "kv_headpar_loop_tiny_compile"),
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        num_cores=NUM_CORES,
        batch_size=BATCH_SIZE,
        qaic_config=qaic_config,
        dynamo=True,
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
    )

    exec_info = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=[PROMPT],
        generation_len=GENERATION_LEN,
    )

    assert Path(qpc_path).is_dir()
    assert exec_info is not None
    assert exec_info.generated_texts is not None
    assert len(exec_info.generated_texts) == BATCH_SIZE
