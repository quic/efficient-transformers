# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import json
import os

import pytest
import torch
from datasets import load_dataset
from transformers import AutoProcessor

from QEfficient import QEFFAutoModelForCTC

from ..nightly_utils import NIGHTLY_SKIPPED_MODELS, get_onnx_and_qpc_size

model_config_path = os.path.join(os.path.dirname(__file__), "../configs/validated_models.json")
with open(model_config_path, "r") as f:
    config = json.load(f)

test_models = config["audio_embedding_models"]


@pytest.mark.parametrize("model_name", test_models[:1])
def test_generate_audio_embedding_model(model_name, get_model_config, audio_embedding_model_artifacts):

    if model_name in NIGHTLY_SKIPPED_MODELS:
        pytest.skip(f"Skipping {model_name} as it is in nightly skipped models list.")

    config, pipeline_configs = get_model_config
    compile_params = pipeline_configs["audio_embedding_model_configs"][0].get("compile_params", {})

    # Retrieve onnx_path from previous stage
    if (
        model_name not in audio_embedding_model_artifacts
        or "onnx_path" not in audio_embedding_model_artifacts[model_name]
    ):
        pytest.skip(f"ONNX path not available for {model_name}. Run test_export.py first.")

    # Retrieve qpc_path from previous stage
    if (
        model_name not in audio_embedding_model_artifacts
        or "qpc_path" not in audio_embedding_model_artifacts[model_name]
    ):
        pytest.skip(f"QPC path not available for {model_name}. Run test_compile.py first.")

    onnx_path = audio_embedding_model_artifacts[model_name].get("onnx_path")

    qeff_model = QEFFAutoModelForCTC.from_pretrained(model_name, torch_dtype=torch.float32)
    _ = qeff_model.compile(onnx_path=onnx_path, **compile_params)

    print("Loading audio sample from dataset...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    data = ds[0]["audio"]["array"]
    processor = AutoProcessor.from_pretrained(model_name)

    model_output = qeff_model.generate(processor, inputs=data)
    print(f"\nTranscription: {model_output}")

    onnx_and_qpc_dir = os.path.dirname(onnx_path)
    size = get_onnx_and_qpc_size(onnx_and_qpc_dir)
    # Store all metrics and execution info
    audio_embedding_model_artifacts[model_name].update(
        {
            "transcription": model_output,
            "onnx_and_qpc_dir": onnx_and_qpc_dir,
            "size": size,
        }
    )
