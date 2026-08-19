# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy
import os

import pytest
import torch

from .model_age_utils import MODEL_AGE_ENV_VAR

MODEL_CLASS_SKIP_ENV_VARS = {
    "causal_pipeline_configs": "SKIP_CAUSAL_LM_MODELS",
    "image_text_to_text_model_configs": "SKIP_IMAGE_TEXT_MODELS",
    "embedding_model_configs": "SKIP_EMBEDDING_MODELS",
    "audio_model_configs": "SKIP_AUDIO_MODELS",
    "audio_embedding_model_configs": "SKIP_AUDIO_EMBEDDING_MODELS",
    "sequence_model_configs": "SKIP_SEQUENCE_MODELS",
}

CB_DEFAULT_MODEL_CLASSES = {"causal_pipeline_configs", "image_text_to_text_model_configs"}
ENABLE_NON_CB_MODE_ENV_VAR = "NIGHTLY_PIPELINE_ENABLE_NON_CB"
ENABLE_MULTI_SPECIALIZATION_ENV_VAR = "NIGHTLY_PIPELINE_ENABLE_MULTI_SPECIALIZATION"
MULTI_SPECIALIZATION_MODEL_CLASSES = {"image_text_to_text_model_configs"}
TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


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
    skip_reason = get_nightly_skip_reason(model_name, model_class)
    if skip_reason:
        pytest.skip(skip_reason)

    export_params, compile_params, _ = _resolve_mode_pipeline_params(get_pipeline_config, model_class, "non_cb")

    return export_params, compile_params


def pre_export_compile_utils_with_mode(model_name, model_class, get_pipeline_config, execution_mode):
    skip_reason = get_nightly_skip_reason(model_name, model_class)
    if skip_reason:
        pytest.skip(skip_reason)

    export_params, compile_params, _ = _resolve_mode_pipeline_params(get_pipeline_config, model_class, execution_mode)

    return export_params, compile_params


def pre_generate_utils(model_name, model_class, get_pipeline_config, model_artifacts, execution_mode="non_cb"):
    skip_reason = get_nightly_skip_reason(model_name, model_class)
    if skip_reason:
        pytest.skip(skip_reason)

    _, compile_params, generate_params = _resolve_mode_pipeline_params(get_pipeline_config, model_class, execution_mode)

    # Retrieve onnx_path from previous stage
    if model_name not in model_artifacts or "onnx_path" not in model_artifacts[model_name]:
        pytest.skip(f"ONNX path not available for {model_name}. Run export and compile first.")

    # Retrieve qpc_path from previous stage
    if model_name not in model_artifacts or "qpc_path" not in model_artifacts[model_name]:
        pytest.skip(f"QPC path not available for {model_name}. Run export and compile first.")

    return compile_params, generate_params


def get_execution_modes(get_pipeline_config, model_class):
    model_family_config = _get_model_family_config(get_pipeline_config, model_class)
    execution_modes = model_family_config.get("execution_modes", ["non_cb"])
    if not execution_modes:
        execution_modes = ["non_cb"]

    if (
        model_class in MULTI_SPECIALIZATION_MODEL_CLASSES
        and _is_truthy_env_var(ENABLE_MULTI_SPECIALIZATION_ENV_VAR)
        and "cb" in execution_modes
    ):
        return ["cb", "multi_spec"]

    if model_class not in CB_DEFAULT_MODEL_CLASSES:
        return execution_modes

    if "cb" not in execution_modes:
        return execution_modes

    if _is_truthy_env_var(ENABLE_NON_CB_MODE_ENV_VAR):
        return [mode for mode in execution_modes if mode != "multi_spec"]

    return [mode for mode in execution_modes if mode == "cb"]


def _is_truthy_env_var(env_var_name):
    return os.environ.get(env_var_name, "").strip().lower() in TRUTHY_ENV_VALUES


def is_continuous_batching_mode(execution_mode):
    return execution_mode == "cb"


def is_multi_specialization_mode(execution_mode):
    return execution_mode == "multi_spec"


def _get_model_family_config(get_pipeline_config, model_class):
    pipeline_configs = get_pipeline_config
    model_family_configs = pipeline_configs.get(model_class, [])
    if not model_family_configs:
        raise KeyError(f"Missing pipeline config for model class: {model_class}")
    return model_family_configs[0]


def _resolve_mode_pipeline_params(get_pipeline_config, model_class, execution_mode):
    model_family_config = _get_model_family_config(get_pipeline_config, model_class)
    valid_modes = set(get_execution_modes(get_pipeline_config, model_class))
    if execution_mode not in valid_modes:
        raise KeyError(f"Unsupported execution mode '{execution_mode}' for model class '{model_class}'")

    export_params = copy.deepcopy(model_family_config.get("export_params", {}))
    compile_params = copy.deepcopy(model_family_config.get("compile_params", {}))
    generate_params = copy.deepcopy(model_family_config.get("generate_params", {}))

    if execution_mode == "cb":
        cb_config = model_family_config.get("continuous_batching", {})
        export_params.update(copy.deepcopy(cb_config.get("export_params", {})))
        compile_params.update(copy.deepcopy(cb_config.get("compile_params", {})))
        generate_params.update(copy.deepcopy(cb_config.get("generate_params", {})))

        full_batch_size = cb_config.get("full_batch_size")
        if full_batch_size is not None and "full_batch_size" not in compile_params:
            compile_params["full_batch_size"] = full_batch_size
    elif execution_mode == "multi_spec":
        multispec_config = model_family_config.get("multi_specialization", {})
        export_params.update(copy.deepcopy(multispec_config.get("export_params", {})))
        compile_params.update(copy.deepcopy(multispec_config.get("compile_params", {})))
        generate_params.update(copy.deepcopy(multispec_config.get("generate_params", {})))

    return export_params, compile_params, generate_params


def max_pooling(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Apply max pooling to the last hidden states."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    last_hidden_states[input_mask_expanded == 0] = -1e9
    return torch.max(last_hidden_states, 1)[0]


def get_nightly_skip_reason(model_name, model_class):
    """Return a skip reason when a model is globally or dynamically skipped."""
    if model_name in NIGHTLY_SKIPPED_MODELS:
        return f"Skipping {model_name} as it is in nightly skipped models list."

    env_var = MODEL_CLASS_SKIP_ENV_VARS.get(model_class)
    if env_var and model_name in parse_skipped_models(os.environ.get(env_var, "")):
        return f"Skipping {model_name} as it is listed in {env_var}."

    return None


def parse_skipped_models(raw_value):
    """Parse comma-separated Jenkins skip parameters into exact model names."""
    if not raw_value:
        return set()
    return {model_name.strip() for model_name in raw_value.split(",") if model_name.strip()}


def nightly_pytest_id(model_name):
    model_age = os.environ.get(MODEL_AGE_ENV_VAR, "all")
    return f"{model_age}:{model_name}"


NIGHTLY_SKIPPED_MODELS = {
    # Vision Models (skipped due to large size or long runtime)
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "meta-llama/Llama-3.2-90B-Vision-Instruct",
    "allenai/Molmo-7B-D-0924",
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "Qwen/Qwen3.5-122B-A10B",
    # Causal Models
    "zai-org/GLM-4.5",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "mistralai/Mixtral-8x7B-v0.1",
    "hpcai-tech/grok-1",
    # Audio Embedding Models
    "facebook/wav2vec2-large",
}
