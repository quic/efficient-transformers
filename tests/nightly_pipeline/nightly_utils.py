# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import pytest
import torch


def human_readable(size):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024


def get_onnx_and_qpc_size(dir):
    total_size = 0
    for root, dirs, files in os.walk(dir):
        for name in files:
            file_path = os.path.join(root, name)
            if not os.path.islink(file_path):  # avoid counting symlinks
                total_size += os.path.getsize(file_path)
    print(f"Total size of {dir}: {total_size} bytes")
    return human_readable(total_size)


def pre_export_compile_utils(model_name, model_class, get_pipeline_config):
    if model_name in NIGHTLY_SKIPPED_MODELS:
        pytest.skip(f"Skipping {model_name} as it is in nightly skipped models list.")

    pipeline_configs = get_pipeline_config
    export_params = pipeline_configs[model_class][0].get("export_params", {})
    compile_params = pipeline_configs[model_class][0].get("compile_params", {})

    return export_params, compile_params


def pre_generate_utils(model_name, model_class, get_pipeline_config, model_artifacts):
    if model_name in NIGHTLY_SKIPPED_MODELS:
        pytest.skip(f"Skipping {model_name} as it is in nightly skipped models list.")

    pipeline_configs = get_pipeline_config
    compile_params = pipeline_configs[model_class][0].get("compile_params", {})
    generate_params = pipeline_configs[model_class][0].get("generate_params", {})

    # Retrieve onnx_path from previous stage
    if model_name not in model_artifacts or "onnx_path" not in model_artifacts[model_name]:
        pytest.skip(f"ONNX path not available for {model_name}. Run export and compile first.")

    # Retrieve qpc_path from previous stage
    if model_name not in model_artifacts or "qpc_path" not in model_artifacts[model_name]:
        pytest.skip(f"QPC path not available for {model_name}. Run export and compile first.")

    return compile_params, generate_params


def max_pooling(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Apply max pooling to the last hidden states."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    last_hidden_states[input_mask_expanded == 0] = -1e9
    return torch.max(last_hidden_states, 1)[0]


NIGHTLY_SKIPPED_MODELS = {
    # Vision Models
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "meta-llama/Llama-3.2-90B-Vision-Instruct",
    "allenai/Molmo-7B-D-0924",
    # Causal Models
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "hpcai-tech/grok-1",
    # Audio Embedding Models
    "facebook/wav2vec2-large",
}
