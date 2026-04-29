# -----------------------------------------------------------------------------

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# -----------------------------------------------------------------------------


import json
import os

import pytest
from datasets import load_dataset
from transformers import AutoProcessor

from QEfficient import QEFFAutoModelForSpeechSeq2Seq

from ..nightly_utils import get_onnx_and_qpc_size, human_readable

model_config_path = os.path.join(os.path.dirname(__file__), "../configs/validated_models.json")
with open(model_config_path, "r") as f:
    config = json.load(f)

test_models = config["audio_models"]


@pytest.mark.parametrize("model_name", test_models[:1])
def test_generate_audio_model(model_name, get_model_config, audio_model_artifacts):
    config, pipeline_configs = get_model_config
    compile_params = pipeline_configs["audio_model_configs"][0].get("compile_params", {})
    generate_params = pipeline_configs["audio_model_configs"][0].get("generate_params", {})

    # Retrieve onnx_path from previous stage
    if model_name not in audio_model_artifacts or "onnx_path" not in audio_model_artifacts[model_name]:
        pytest.skip(f"ONNX path not available for {model_name}. Run test_export.py first.")

    # Retrieve qpc_path from previous stage
    if model_name not in audio_model_artifacts or "qpc_path" not in audio_model_artifacts[model_name]:
        pytest.skip(f"QPC path not available for {model_name}. Run test_compile.py first.")

    qeff_model = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    onnx_path = audio_model_artifacts[model_name].get("onnx_path")
    _ = qeff_model.compile(onnx_path=onnx_path, **compile_params)

    print("Loading audio sample from dataset...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample_rate = ds[0]["audio"]["sampling_rate"]
    data = ds[0]["audio"]["array"]
    # Reshape so shape corresponds to data with batch size 1
    data = data.reshape(-1)
    print(generate_params)
    exec_info = qeff_model.generate(
        inputs=processor(data, sampling_rate=sample_rate, return_tensors="pt"), **generate_params
    )
    print(exec_info)
    transcription = processor.batch_decode(exec_info.generated_ids)[0]
    print(f"\nTranscription: {transcription}")

    onnx_and_qpc_dir = os.path.dirname(onnx_path)
    size = get_onnx_and_qpc_size(onnx_and_qpc_dir)
    size = human_readable(size)
    # Store all metrics and execution info
    audio_model_artifacts[model_name].update(
        {
            "batch_size": exec_info.batch_size,
            "transcription": transcription,
            "generated_ids": exec_info.generated_ids,
            "onnx_and_qpc_dir": onnx_and_qpc_dir,
            "size": size,
            "perf_metrics": {
                "prefill_time": exec_info.perf_metrics.prefill_time,
                "decode_perf": exec_info.perf_metrics.decode_perf,
                "total_perf": exec_info.perf_metrics.total_perf,
                "total_time": exec_info.perf_metrics.total_time,
            },
        }
    )
