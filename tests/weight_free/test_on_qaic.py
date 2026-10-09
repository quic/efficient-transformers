# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Weight-free on-QAIC tests.

All tests require QAIC hardware (marked @pytest.mark.on_qaic).
All tests run with weight_free=True (set on the QEff model, which forces
dynamo=True internally). Whisper runs with ONNX subfunctions disabled because
they are not supported for that model.

Covers:
  - Generate smoke test (weight-free export -> compile -> generate on QAIC)
  - HF PT vs QAIC HW parity (HF PT tokens == weight-free QAIC top-1 token)
  - Weight-free vs legacy/dynamo QAIC parity (two independently compiled QPCs
    of the same model, generated output compared directly)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import AutoConfig

from QEfficient.exporter.weight_free import resolve_weight_spec_path
from QEfficient.transformers.models.modeling_auto import (
    QEFFAutoModelForCausalLM,
    QEFFAutoModelForCTC,
    QEFFAutoModelForSpeechSeq2Seq,
)
from QEfficient.utils import get_num_layers_from_config
from QEfficient.utils.constants import WAV2VEC2_MAX_SEQ_LEN

from ._helpers import (
    BATCH_SIZE,
    CTX_LEN,
    PROMPT_LEN,
    WEIGHT_FREE_ASR_MODEL_IDS,
    WEIGHT_FREE_QAIC_MODEL_PARAMS,
    exported_onnx_path,
    load_hf_model,
    load_tokenizer,
    skip_on_model_fetch_error,
)

WEIGHT_FREE_ASR_QEFF_CLASSES = {
    "wav2vec2": QEFFAutoModelForCTC,
    "whisper": QEFFAutoModelForSpeechSeq2Seq,
}
WEIGHT_FREE_ASR_USE_SUBFUNCTIONS = {
    "wav2vec2": True,
    "whisper": False,
}
ASR_GENERATION_LEN = 8
ASR_WAV2VEC2_SEQ_LEN = WAV2VEC2_MAX_SEQ_LEN
WEIGHT_FREE_ASR_QAIC_MODEL_PARAMS = [
    pytest.param(
        model_type,
        model_id,
        marks=pytest.mark.xdist_group(name=f"qaic-runtime-asr-{model_type}"),
        id=model_type,
    )
    for model_type, model_id in sorted(WEIGHT_FREE_ASR_MODEL_IDS.items())
]


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_type,model_id", WEIGHT_FREE_QAIC_MODEL_PARAMS)
def test_weight_free_generate_fp16(model_type, model_id, tmp_export_dir):
    """End-to-end weight-free export -> compile -> generate on real QAIC hardware."""
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.num_hidden_layers = 2
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, config=config, weight_free=True)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / "wf_gen_export",
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / "wf_gen_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    tokenizer = load_tokenizer(model_id)
    output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=["hello world"],
    )
    assert output is not None
    assert output.generated_texts is not None


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_type,model_id", WEIGHT_FREE_QAIC_MODEL_PARAMS)
def test_weight_free_hw_hf_parity(model_type, model_id, tmp_export_dir):
    """HF PT tokens == weight-free QAIC FP16 tokens (exact equality)."""
    from QEfficient.utils.run_utils import ApiRunner

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
    hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
    assert hf_tokens is not None, "HF PT inference returned None"

    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.num_hidden_layers = get_num_layers_from_config(model_hf.config)
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, config=config, weight_free=True)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / "wf_hw_parity_export",
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / "wf_hw_parity_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    qaic_output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=["hello world"],
    )

    assert qaic_output is not None, "QAIC generate returned None"
    if hasattr(qaic_output, "generated_ids") and qaic_output.generated_ids is not None:
        gen_len = CTX_LEN - PROMPT_LEN
        qaic_tokens = qaic_output.generated_ids[0].flatten()[:gen_len]
        assert np.array_equal(hf_tokens, qaic_tokens), (
            f"Weight-free HW/HF parity failed for {model_id}: HF={hf_tokens.tolist()}, QAIC={qaic_tokens.tolist()}"
        )


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_type,model_id", WEIGHT_FREE_QAIC_MODEL_PARAMS)
def test_weight_free_vs_legacy_qaic_parity(model_type, model_id, tmp_export_dir):
    """Weight-free-compiled and legacy dynamo-compiled QPCs produce identical tokens on QAIC."""
    try:
        tokenizer = load_tokenizer(model_id)
        model_hf = load_hf_model(model_id)
        if model_type == "gpt_oss":
            model_hf = model_hf.to(torch.float32)
            model_hf.config.torch_dtype = torch.float32
            model_hf.config.dtype = torch.float32
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    # Legacy/dynamo leg — real weights, no weight-free export.
    qeff_legacy = QEFFAutoModelForCausalLM(model_hf)

    legacy_onnx_path = exported_onnx_path(
        qeff_legacy.export(
            tmp_export_dir / "legacy_export",
            dynamo=True,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    qeff_legacy.compile(
        onnx_path=str(legacy_onnx_path),
        compile_dir=str(tmp_export_dir / "legacy_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    legacy_output = qeff_legacy.generate(
        tokenizer=tokenizer,
        prompts=["hello world"],
    )
    assert legacy_output is not None, "Legacy QAIC generate returned None"

    # Weight-free leg — meta-device model, matching layer count.
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.num_hidden_layers = get_num_layers_from_config(model_hf.config)
        qeff_weight_free = QEFFAutoModelForCausalLM.from_pretrained(model_id, config=config, weight_free=True)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    weight_free_onnx_path = exported_onnx_path(
        qeff_weight_free.export(
            tmp_export_dir / "wf_vs_legacy_export",
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    qeff_weight_free.compile(
        onnx_path=str(weight_free_onnx_path),
        compile_dir=str(tmp_export_dir / "wf_vs_legacy_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    weight_free_output = qeff_weight_free.generate(
        tokenizer=tokenizer,
        prompts=["hello world"],
    )
    assert weight_free_output is not None, "Weight-free QAIC generate returned None"

    if (
        hasattr(legacy_output, "generated_ids")
        and legacy_output.generated_ids is not None
        and hasattr(weight_free_output, "generated_ids")
        and weight_free_output.generated_ids is not None
    ):
        legacy_tokens = legacy_output.generated_ids[0].flatten()
        weight_free_tokens = weight_free_output.generated_ids[0].flatten()
        assert np.array_equal(legacy_tokens, weight_free_tokens), (
            f"Weight-free vs legacy QAIC parity failed for {model_id}: "
            f"legacy={legacy_tokens.tolist()}, weight_free={weight_free_tokens.tolist()}"
        )


def _load_weight_free_asr(model_type, model_id):
    try:
        model_hf, processor = load_hf_model(model_id, model_type=model_type)
        qeff_model = WEIGHT_FREE_ASR_QEFF_CLASSES[model_type].from_pretrained(
            model_id,
            trust_remote_code=True,
            weight_free=True,
        )
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)
    return model_hf, processor, qeff_model


def _export_weight_free_asr(qeff_model, model_type, export_dir):
    onnx_path = exported_onnx_path(
        qeff_model.export(
            export_dir,
            use_onnx_subfunctions=WEIGHT_FREE_ASR_USE_SUBFUNCTIONS[model_type],
            offload_pt_weights=False,
        )
    )
    assert resolve_weight_spec_path(onnx_path).is_file()
    return onnx_path


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_type,model_id", WEIGHT_FREE_ASR_QAIC_MODEL_PARAMS)
def test_weight_free_asr_generate_fp16(model_type, model_id, tmp_export_dir):
    """Weight-free ASR export, QAIC compile, and execution smoke test."""
    _, processor, qeff_model = _load_weight_free_asr(model_type, model_id)
    onnx_path = _export_weight_free_asr(qeff_model, model_type, tmp_export_dir / "wf_asr_export")

    if model_type == "wav2vec2":
        qeff_model.compile(
            onnx_path=str(onnx_path),
            compile_dir=str(tmp_export_dir / "wf_asr_compile"),
            seq_len=ASR_WAV2VEC2_SEQ_LEN,
            batch_size=BATCH_SIZE,
            num_cores=16,
            use_onnx_subfunctions=True,
        )
        output = qeff_model.generate(processor, inputs=np.zeros(16000, dtype=np.float32))
        assert output is not None
        assert len(output) == BATCH_SIZE
    else:
        qeff_model.compile(
            onnx_path=str(onnx_path),
            compile_dir=str(tmp_export_dir / "wf_asr_compile"),
            ctx_len=ASR_GENERATION_LEN,
            batch_size=BATCH_SIZE,
            num_cores=16,
            use_onnx_subfunctions=False,
        )
        inputs = processor(np.zeros(16000, dtype=np.float32), sampling_rate=16000, return_tensors="pt")
        output = qeff_model.generate(inputs=inputs, generation_len=ASR_GENERATION_LEN)
        assert output is not None
        assert output.generated_ids is not None


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_type,model_id", [pytest.param("wav2vec2", WEIGHT_FREE_ASR_MODEL_IDS["wav2vec2"])])
def test_weight_free_asr_hw_hf_parity(model_type, model_id, tmp_export_dir):
    """Weight-free QAIC ASR output matches the HF PyTorch reference output."""
    model_hf, processor, qeff_model = _load_weight_free_asr(model_type, model_id)
    audio = np.zeros(16000, dtype=np.float32)

    if model_type == "wav2vec2":
        hf_inputs = processor(
            audio,
            return_tensors="pt",
            max_length=ASR_WAV2VEC2_SEQ_LEN,
            truncation=True,
            padding="max_length",
        )
        with torch.no_grad():
            hf_logits = model_hf(input_values=hf_inputs.input_values).logits
        hf_output = processor.batch_decode(hf_logits.argmax(dim=-1))
    else:
        input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        decoder_input_ids = torch.full(
            (BATCH_SIZE, 1),
            model_hf.config.decoder_start_token_id,
            dtype=torch.int64,
        )
        hf_output = model_hf.generate(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=ASR_GENERATION_LEN,
            do_sample=False,
        )

    onnx_path = _export_weight_free_asr(qeff_model, model_type, tmp_export_dir / "wf_asr_parity_export")
    if model_type == "wav2vec2":
        qeff_model.compile(
            onnx_path=str(onnx_path),
            compile_dir=str(tmp_export_dir / "wf_asr_parity_compile"),
            seq_len=ASR_WAV2VEC2_SEQ_LEN,
            batch_size=BATCH_SIZE,
            num_cores=16,
            use_onnx_subfunctions=True,
        )
        qaic_output = qeff_model.generate(processor, inputs=audio)
        assert hf_output == qaic_output
    else:
        qeff_model.compile(
            onnx_path=str(onnx_path),
            compile_dir=str(tmp_export_dir / "wf_asr_parity_compile"),
            ctx_len=ASR_GENERATION_LEN,
            batch_size=BATCH_SIZE,
            num_cores=16,
            use_onnx_subfunctions=False,
        )
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        qaic_output = qeff_model.generate(inputs=inputs, generation_len=ASR_GENERATION_LEN)
        assert np.array_equal(hf_output.numpy(), qaic_output.generated_ids)
