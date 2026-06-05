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

from ..nightly_utils import get_onnx_and_qpc_size, pre_generate_utils

model_config_path = os.path.join(os.path.dirname(__file__), "../configs/validated_models.json")
with open(model_config_path, "r") as f:
    config = json.load(f)

test_models = config["audio_embedding_models"]


@pytest.mark.parametrize("model_name", test_models)
def test_generate_audio_embedding_model(model_name, get_pipeline_config, audio_embedding_model_artifacts):

    compile_params, generate_params = pre_generate_utils(
        model_name, "audio_embedding_model_configs", get_pipeline_config, audio_embedding_model_artifacts
    )

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
