# -----------------------------------------------------------------------------

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# -----------------------------------------------------------------------------


import json
import os
from pathlib import Path

import pytest
import torch
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from QEfficient import QEFFAutoModelForSpeechSeq2Seq

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
test_models = filter_models_for_nightly(config["audio_models"], "audio_models")


def _generate_audio_model(model_name, get_pipeline_config, audio_model_artifacts, torch_dtype, dtype_key="fp32"):
    """Common generate logic for audio seq2seq models."""
    compile_params, generate_params = pre_generate_utils(
        model_name, "audio_model_configs", get_pipeline_config, audio_model_artifacts, dtype_key=dtype_key
    )

    qeff_model = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(model_name, torch_dtype=torch_dtype)
    processor = AutoProcessor.from_pretrained(model_name)
    qeff_model.qpc_path = Path(audio_model_artifacts[model_name][dtype_key]["qpc_path"])

    print("Loading audio sample from dataset...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample_rate = ds[0]["audio"]["sampling_rate"]
    data = ds[0]["audio"]["array"]
    data = data.reshape(-1)

    # QPC inference
    with measure_peak_ram() as ram:
        exec_info = qeff_model.generate(
            inputs=processor(data, sampling_rate=sample_rate, return_tensors="pt"), **generate_params
        )
    transcription = processor.batch_decode(exec_info.generated_ids, skip_special_tokens=True)[0]
    print(f"\nTranscription: {transcription}")

    # Golden output: run PyTorch reference if not already stored, then compare
    golden_key = make_golden_key(
        dtype=dtype_key,
        config_params=compile_params,
        extra_tags={"ctx_len": compile_params.get("ctx_len", 32)},
    )

    def _run_pytorch():
        """Run HF PyTorch inference and return golden output dict."""
        hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, attn_implementation="eager", low_cpu_mem_usage=False
        )
        hf_model.eval()

        model_weight_dtype = next(hf_model.parameters()).dtype
        hf_tokens = hf_model.generate(
            processor(data, sampling_rate=sample_rate, return_tensors="pt").input_features.to(model_weight_dtype),
            max_new_tokens=generate_params.get("generation_len", 25),
            language="en",
        )
        hf_transcription = processor.batch_decode(hf_tokens, skip_special_tokens=True)[0]
        return {
            "pytorch_hf_tokens": hf_tokens.tolist(),
            "transcription": hf_transcription,
            "gen_len": generate_params.get("generation_len", 25),
        }

    golden = run_or_load_golden(
        category="audio_models",
        model_name=model_name,
        golden_key=golden_key,
        run_pytorch_fn=_run_pytorch,
        config_fp=PIPELINE_CONFIG_FP,
    )

    # Compare QPC output against golden
    comparison = compare_with_golden(
        qpc_output={"transcription": transcription},
        golden={"transcription": golden.get("transcription", "")},
        tolerance=0.0,
    )
    print(f"\n[GOLDEN COMPARISON] passed={comparison['passed']} details={comparison['per_key']}")

    audio_model_artifacts[model_name][dtype_key].update(
        {
            "batch_size": exec_info.batch_size,
            "transcription": transcription,
            "generated_ids": exec_info.generated_ids,
            "generate_peak_ram_mb": round(ram["peak_mb"], 2),
            "golden_comparison": comparison,
            "perf_metrics": {
                "prefill_time": exec_info.perf_metrics.prefill_time,
                "decode_perf": exec_info.perf_metrics.decode_perf,
                "total_perf": exec_info.perf_metrics.total_perf,
                "total_time": exec_info.perf_metrics.total_time,
            },
        }
    )

    assert comparison["passed"], f"QPC output differs from golden PyTorch: {comparison['per_key']}"


# Config 1: FP32
@pytest.mark.parametrize("model_name", test_models)
def test_generate_audio_model(model_name, get_pipeline_config, audio_model_artifacts):
    """FP32 generate with golden output comparison."""
    _generate_audio_model(model_name, get_pipeline_config, audio_model_artifacts, torch.float32)


# Config 2: FP16
@pytest.mark.parametrize("model_name", test_models)
def test_generate_audio_model_fp16(model_name, get_pipeline_config, audio_model_artifacts):
    """FP16 generate with golden output comparison."""
    _generate_audio_model(model_name, get_pipeline_config, audio_model_artifacts, torch.float16, dtype_key="fp16")
