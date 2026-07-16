# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Dynamo on-QAIC tests.

All tests require QAIC hardware (marked @pytest.mark.on_qaic).
Every test is run twice: use_onnx_subfunctions=False and use_onnx_subfunctions=True.
The subfunction=True variant is skipped for DYNAMO_NO_SUBFUNCTION_ARCHS.

Covers:
  - FP16 compile
  - FP32 compile (torch_dtype=float32 model)
  - Multi-device compile (num_devices=4)
  - Generate FP16
  - HW ORT parity (ORT CPU tokens == QAIC FP16 top-1 token)
  - Continuous-batching generate
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ._helpers import (
    BATCH_SIZE,
    CTX_LEN,
    DYNAMO_CAUSAL_LM_MODEL_IDS,
    DYNAMO_NO_SUBFUNCTION_ARCHS,
    FULL_BATCH_SIZE,
    PROMPT_LEN,
    exported_onnx_path,
    load_hf_model,
    load_tokenizer,
    skip_on_model_fetch_error,
)

# Small subset used for compile+generate tests to keep hardware time reasonable
_HW_SUBSET = ("gpt2", "llama", "qwen2")


@pytest.mark.dynamo
@pytest.mark.dynamo_on_qaic
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("use_onnx_subfunctions", [False, True], ids=["flat", "subfn"])
@pytest.mark.parametrize(
    "model_type,model_id", sorted(DYNAMO_CAUSAL_LM_MODEL_IDS.items()), ids=sorted(DYNAMO_CAUSAL_LM_MODEL_IDS)
)
def test_dynamo_fp16_compile(model_type, model_id, use_onnx_subfunctions, tmp_export_dir):
    """Export with dynamo=True and compile to FP16 QPC."""
    if use_onnx_subfunctions and model_type in DYNAMO_NO_SUBFUNCTION_ARCHS:
        pytest.skip(f"{model_type} does not support subfunctions under dynamo")

    try:
        model_hf = load_hf_model(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    subfn_label = "subfn" if use_onnx_subfunctions else "flat"
    qeff_model = QEFFAutoModelForCausalLM(model_hf)
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / f"fp16_export_{subfn_label}",
            dynamo=True,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )
    )
    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / f"fp16_compile_{subfn_label}"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=use_onnx_subfunctions,
    )
    assert (tmp_export_dir / f"fp16_compile_{subfn_label}").is_dir()


@pytest.mark.dynamo
@pytest.mark.dynamo_on_qaic
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("use_onnx_subfunctions", [False, True], ids=["flat", "subfn"])
@pytest.mark.parametrize(
    "model_type,model_id", sorted(DYNAMO_CAUSAL_LM_MODEL_IDS.items()), ids=sorted(DYNAMO_CAUSAL_LM_MODEL_IDS)
)
def test_dynamo_fp32_compile(model_type, model_id, use_onnx_subfunctions, tmp_export_dir):
    """Export with dynamo=True and compile with fp32 model (no precision compression)."""
    if use_onnx_subfunctions and model_type in DYNAMO_NO_SUBFUNCTION_ARCHS:
        pytest.skip(f"{model_type} does not support subfunctions under dynamo")

    try:
        model_hf = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=False,
            torch_dtype=torch.float32,
        )
        model_hf.eval()
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    subfn_label = "subfn" if use_onnx_subfunctions else "flat"
    qeff_model = QEFFAutoModelForCausalLM(model_hf)
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / f"fp32_export_{subfn_label}",
            dynamo=True,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )
    )
    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / f"fp32_compile_{subfn_label}"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        use_onnx_subfunctions=use_onnx_subfunctions,
    )
    assert (tmp_export_dir / f"fp32_compile_{subfn_label}").is_dir()


@pytest.mark.dynamo
@pytest.mark.dynamo_on_qaic
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("use_onnx_subfunctions", [False, True], ids=["flat", "subfn"])
@pytest.mark.parametrize(
    "model_type,model_id",
    [(k, DYNAMO_CAUSAL_LM_MODEL_IDS[k]) for k in _HW_SUBSET if k in DYNAMO_CAUSAL_LM_MODEL_IDS],
    ids=list(_HW_SUBSET),
)
def test_dynamo_multi_device_compile(model_type, model_id, use_onnx_subfunctions, tmp_export_dir):
    """Export with dynamo=True and compile for 4 devices."""
    if use_onnx_subfunctions and model_type in DYNAMO_NO_SUBFUNCTION_ARCHS:
        pytest.skip(f"{model_type} does not support subfunctions under dynamo")

    try:
        model_hf = load_hf_model(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    subfn_label = "subfn" if use_onnx_subfunctions else "flat"
    qeff_model = QEFFAutoModelForCausalLM(model_hf)
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / f"mdp_export_{subfn_label}",
            dynamo=True,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )
    )
    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / f"mdp_compile_{subfn_label}"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        num_devices=4,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=use_onnx_subfunctions,
    )
    assert (tmp_export_dir / f"mdp_compile_{subfn_label}").is_dir()


@pytest.mark.dynamo
@pytest.mark.dynamo_on_qaic
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("use_onnx_subfunctions", [False, True], ids=["flat", "subfn"])
@pytest.mark.parametrize(
    "model_type,model_id",
    [(k, DYNAMO_CAUSAL_LM_MODEL_IDS[k]) for k in _HW_SUBSET if k in DYNAMO_CAUSAL_LM_MODEL_IDS],
    ids=list(_HW_SUBSET),
)
def test_dynamo_generate_fp16(model_type, model_id, use_onnx_subfunctions, tmp_export_dir):
    """End-to-end export → compile → generate with dynamo=True."""
    if use_onnx_subfunctions and model_type in DYNAMO_NO_SUBFUNCTION_ARCHS:
        pytest.skip(f"{model_type} does not support subfunctions under dynamo")

    try:
        model_hf = load_hf_model(model_id)
        tokenizer = load_tokenizer(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    subfn_label = "subfn" if use_onnx_subfunctions else "flat"
    qeff_model = QEFFAutoModelForCausalLM(model_hf)
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / f"gen_export_{subfn_label}",
            dynamo=True,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )
    )
    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / f"gen_compile_{subfn_label}"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=use_onnx_subfunctions,
    )
    output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=["hello world"],
        device_id=[0],
    )
    assert output is not None
    assert output.generated_texts is not None


@pytest.mark.dynamo
@pytest.mark.dynamo_on_qaic
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("use_onnx_subfunctions", [False, True], ids=["flat", "subfn"])
@pytest.mark.parametrize(
    "model_type,model_id",
    [(k, DYNAMO_CAUSAL_LM_MODEL_IDS[k]) for k in _HW_SUBSET if k in DYNAMO_CAUSAL_LM_MODEL_IDS],
    ids=list(_HW_SUBSET),
)
def test_dynamo_hw_ort_parity(model_type, model_id, use_onnx_subfunctions, tmp_export_dir):
    """ORT CPU tokens == QAIC FP16 top-1 token (greedy argmax parity)."""
    from QEfficient.utils.run_utils import ApiRunner

    if use_onnx_subfunctions and model_type in DYNAMO_NO_SUBFUNCTION_ARCHS:
        pytest.skip(f"{model_type} does not support subfunctions under dynamo")

    try:
        tokenizer = load_tokenizer(model_id)
        model_hf = load_hf_model(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    api_runner = ApiRunner(
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        config=model_hf.config,
        prompt=["hello world"],
        prompt_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        full_batch_size=None,
    )

    subfn_label = "subfn" if use_onnx_subfunctions else "flat"
    qeff_model = QEFFAutoModelForCausalLM(model_hf)
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / f"hw_parity_export_{subfn_label}",
            dynamo=True,
            use_onnx_subfunctions=use_onnx_subfunctions,
            offload_pt_weights=False,
        )
    )

    ort_tokens = api_runner.run_kv_model_on_ort(str(onnx_path))
    assert ort_tokens is not None, "ORT inference returned None"

    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / f"hw_parity_compile_{subfn_label}"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=use_onnx_subfunctions,
    )

    qaic_output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=["hello world"],
        device_id=[0],
    )

    assert qaic_output is not None, "QAIC generate returned None"
    if hasattr(qaic_output, "generated_ids") and qaic_output.generated_ids is not None:
        ort_first_token = int(ort_tokens[0][0])
        qaic_first_token = int(qaic_output.generated_ids[0][0][0])
        assert ort_first_token == qaic_first_token, (
            f"HW ORT parity failed for {model_id} (subfunctions={use_onnx_subfunctions}): "
            f"ORT={ort_first_token}, QAIC={qaic_first_token}"
        )


@pytest.mark.dynamo
@pytest.mark.dynamo_on_qaic
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("use_onnx_subfunctions", [False, True], ids=["flat", "subfn"])
@pytest.mark.parametrize(
    "model_type,model_id",
    [(k, DYNAMO_CAUSAL_LM_MODEL_IDS[k]) for k in _HW_SUBSET if k in DYNAMO_CAUSAL_LM_MODEL_IDS],
    ids=list(_HW_SUBSET),
)
def test_dynamo_cb_generate(model_type, model_id, use_onnx_subfunctions, tmp_export_dir):
    """Continuous-batching export → compile → generate with dynamo=True."""
    if use_onnx_subfunctions and model_type in DYNAMO_NO_SUBFUNCTION_ARCHS:
        pytest.skip(f"{model_type} does not support subfunctions under dynamo")

    try:
        model_hf = load_hf_model(model_id)
        tokenizer = load_tokenizer(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    subfn_label = "subfn" if use_onnx_subfunctions else "flat"
    qeff_model = QEFFAutoModelForCausalLM(model_hf, continuous_batching=True)
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / f"cb_gen_export_{subfn_label}",
            dynamo=True,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )
    )
    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / f"cb_gen_compile_{subfn_label}"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        full_batch_size=FULL_BATCH_SIZE,
        use_onnx_subfunctions=use_onnx_subfunctions,
    )
    prompts = ["hello world"] * FULL_BATCH_SIZE
    output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        device_id=[0],
    )
    assert output is not None
    assert output.generated_texts is not None
    assert len(output.generated_texts) == FULL_BATCH_SIZE
