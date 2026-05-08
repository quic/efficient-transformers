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
    # "Qwen/Qwen3-VL-2B-Instruct",
    # "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    # "google/gemma-3-4b-it",
    # "Qwen/Qwen2.5-VL-3B-Instruct",
    "llava-hf/llava-1.5-7b-hf",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "meta-llama/Llama-3.2-90B-Vision-Instruct",
    "ibm-granite/granite-vision-3.2-2b",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "allenai/Molmo-7B-D-0924",
    "OpenGVLab/InternVL2_5-1B",
    "OpenGVLab/InternVL3_5-1B",
    # Causal Models
    "openai-community/gpt2",
    "allenai/OLMo-2-0425-1B",
    "tiiuae/falcon-40b",
    # "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "google/codegemma-2b",
    "google/codegemma-7b",
    "google/gemma-2b",
    "google/gemma-7b",
    "google/gemma-2-2b",
    "google/gemma-2-9b",
    "google/gemma-2-27b",
    # "openai/gpt-oss-20b",
    "bigcode/starcoder",
    "bigcode/starcoder2-15b",
    "EleutherAI/gpt-j-6b",
    "ibm-granite/granite-3.1-8b-instruct",
    "ibm-granite/granite-guardian-3.1-8b",
    "ibm-granite/granite-20b-code-base-8k",
    "ibm-granite/granite-20b-code-instruct-8k",
    "codellama/CodeLlama-7b-hf",
    "codellama/CodeLlama-13b-hf",
    "codellama/CodeLlama-34b-hf",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "inceptionai/jais-adapted-7b",
    "inceptionai/jais-adapted-13b-chat",
    "inceptionai/jais-adapted-70b",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "lmsys/vicuna-13b-delta-v0",
    "lmsys/vicuna-13b-v1.3",
    "lmsys/vicuna-13b-v1.5",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Codestral-22B-v0.1",
    "mistralai/Mixtral-8x7B-v0.1",
    "microsoft/Phi-3-mini-4k-instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "Qwen/Qwen2-1.5B-Instruct",
    "Snowflake/Llama-3.1-SwiftKV-8B-Instruct",
    "hpcai-tech/grok-1",
    # Embedding Models
    "intfloat/multilingual-e5-large",
    # Audio Embedding Models
    "facebook/wav2vec2-large",
}
