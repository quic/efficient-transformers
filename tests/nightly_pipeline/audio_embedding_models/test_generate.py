# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import json
import os
from pathlib import Path

import pytest
import torch
from datasets import load_dataset
from transformers import AutoModelForCTC, AutoProcessor

from QEfficient import QEFFAutoModelForCTC

from ..model_age_utils import filter_models_for_nightly
from ..nightly_utils import (
    compare_with_golden,
    make_golden_key,
    measure_peak_ram,
    pre_generate_utils,
    run_or_load_golden,
)

model_config_path = os.path.join(os.path.dirname(__file__), "../configs/validated_models.json")
with open(model_config_path, "r") as f:
    config = json.load(f)

PIPELINE_CONFIG_FP = os.path.join(os.path.dirname(__file__), "../configs/pipeline_configs.json")
test_models = filter_models_for_nightly(config["audio_embedding_models"], "audio_embedding_models")


def _generate_audio_embedding_model(
    model_name, get_pipeline_config, audio_embedding_model_artifacts, torch_dtype, dtype_key="fp32"
):
    """Common generate logic for CTC audio embedding models."""
    compile_params, generate_params = pre_generate_utils(
        model_name,
        "audio_embedding_model_configs",
        get_pipeline_config,
        audio_embedding_model_artifacts,
        dtype_key=dtype_key,
    )

    qeff_model = QEFFAutoModelForCTC.from_pretrained(model_name, torch_dtype=torch_dtype)
    qeff_model.qpc_path = Path(audio_embedding_model_artifacts[model_name][dtype_key]["qpc_path"])

    print("Loading audio sample from dataset...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    data = ds[0]["audio"]["array"]
    sample_rate = ds[0]["audio"]["sampling_rate"]
    processor = AutoProcessor.from_pretrained(model_name)

    # QPC inference
    with measure_peak_ram() as ram:
        model_output = qeff_model.generate(processor, inputs=data)
    print(f"\nTranscription: {model_output}")

    # Golden output: run PyTorch reference if not already stored, then compare
    golden_key = make_golden_key(
        dtype=dtype_key,
        config_params=compile_params,
        extra_tags={"seq_len": compile_params.get("seq_len", 480000)},
    )

    def _run_pytorch():
        """Run HF PyTorch CTC inference and return golden output dict."""
        hf_model = AutoModelForCTC.from_pretrained(model_name, attn_implementation="eager", low_cpu_mem_usage=False)
        hf_model.eval()
        input_values = processor(
            data,
            sampling_rate=sample_rate,
            return_tensors="pt",
            max_length=compile_params.get("seq_len", 480000),
            truncating=True,
            padding="max_length",
        ).input_values

        model_dtype = getattr(hf_model.config, "torch_dtype", next(hf_model.parameters()).dtype)
        input_values = input_values.to(dtype=model_dtype)
        with torch.no_grad():
            logits = hf_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        hf_transcription = processor.batch_decode(predicted_ids)
        return {
            "transcription": hf_transcription,
            "pytorch_hf_tokens": predicted_ids.tolist(),
        }

    golden = run_or_load_golden(
        category="audio_embedding_models",
        model_name=model_name,
        golden_key=golden_key,
        run_pytorch_fn=_run_pytorch,
        config_fp=PIPELINE_CONFIG_FP,
    )

    # Compare only transcription — normalize both to string
    qpc_transcription = model_output[0] if isinstance(model_output, list) else str(model_output)
    golden_transcription = golden.get("transcription", "")
    if isinstance(golden_transcription, list):
        golden_transcription = golden_transcription[0] if golden_transcription else ""
    comparison = compare_with_golden(
        qpc_output={"transcription": qpc_transcription},
        golden={"transcription": golden_transcription},
        tolerance=0.0,
    )
    print(f"\n[GOLDEN COMPARISON] passed={comparison['passed']} details={comparison['per_key']}")

    audio_embedding_model_artifacts[model_name][dtype_key].update(
        {
            "transcription": model_output,
            "generate_peak_ram_mb": round(ram["peak_mb"], 2),
            "golden_comparison": comparison,
        }
    )

    assert comparison["passed"], f"QPC output differs from golden PyTorch: {comparison['per_key']}"


# Config 1: FP32
@pytest.mark.parametrize("model_name", test_models)
def test_generate_audio_embedding_model(model_name, get_pipeline_config, audio_embedding_model_artifacts):
    """FP32 generate with golden output comparison."""
    _generate_audio_embedding_model(model_name, get_pipeline_config, audio_embedding_model_artifacts, torch.float32)


# Config 1: FP16
@pytest.mark.parametrize("model_name", test_models)
def test_generate_audio_embedding_model_fp16(model_name, get_pipeline_config, audio_embedding_model_artifacts):
    """FP16 generate with golden output comparison."""
    _generate_audio_embedding_model(
        model_name, get_pipeline_config, audio_embedding_model_artifacts, torch.float16, dtype_key="fp16"
    )
