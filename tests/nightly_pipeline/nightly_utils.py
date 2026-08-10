# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import psutil
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


def get_file_or_dir_size(path):
    """Return human-readable size of a single file or directory."""
    path = str(path)
    if os.path.isfile(path):
        return human_readable(os.path.getsize(path))
    elif os.path.isdir(path):
        return get_onnx_and_qpc_size(path)
    return "N/A"


@contextmanager
def measure_peak_ram():
    """Context manager that tracks peak RSS memory (MB) for the enclosed block.

    Usage:
        with measure_peak_ram() as ram:
            do_work()
        print(ram["peak_mb"])
    """
    process = psutil.Process(os.getpid())
    result = {"peak_mb": process.memory_info().rss / (1024 ** 2)}
    stop = threading.Event()

    def _monitor():
        while not stop.is_set():
            current_mb = process.memory_info().rss / (1024 ** 2)
            if current_mb > result["peak_mb"]:
                result["peak_mb"] = current_mb
            stop.wait(0.1)

    t = threading.Thread(target=_monitor, daemon=True)
    t.start()
    try:
        yield result
    finally:
        stop.set()
        t.join()


def pre_export_compile_utils(model_name, model_class, get_pipeline_config):
    skip_reason = get_nightly_skip_reason(model_name, model_class)
    if skip_reason:
        pytest.skip(skip_reason)

    pipeline_configs = get_pipeline_config
    export_params = pipeline_configs[model_class][0].get("export_params", {})
    compile_params = pipeline_configs[model_class][0].get("compile_params", {})

    return export_params, compile_params


def pre_generate_utils(model_name, model_class, get_pipeline_config, model_artifacts, dtype_key=None):
    skip_reason = get_nightly_skip_reason(model_name, model_class)
    if skip_reason:
        pytest.skip(skip_reason)

    pipeline_configs = get_pipeline_config
    compile_params = pipeline_configs[model_class][0].get("compile_params", {})
    generate_params = pipeline_configs[model_class][0].get("generate_params", {})

    # Check artifacts exist - support both nested (dtype_key) and flat structure
    if model_name not in model_artifacts:
        pytest.skip(f"No artifacts for {model_name}. Run export and compile first.")

    if dtype_key is not None:
        # Nested structure: artifacts[model_name][dtype_key]
        dtype_artifacts = model_artifacts[model_name].get(dtype_key, {})
        if "onnx_path" not in dtype_artifacts:
            pytest.skip(f"ONNX path not available for {model_name} [{dtype_key}]. Run export and compile first.")
        if "qpc_path" not in dtype_artifacts:
            pytest.skip(f"QPC path not available for {model_name} [{dtype_key}]. Run export and compile first.")
    else:
        # Flat structure
        if "onnx_path" not in model_artifacts[model_name]:
            pytest.skip(f"ONNX path not available for {model_name}. Run export and compile first.")
        if "qpc_path" not in model_artifacts[model_name]:
            pytest.skip(f"QPC path not available for {model_name}. Run export and compile first.")

    return compile_params, generate_params


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


# ---------------------------------------------------------------------------
# Golden output helpers
# ---------------------------------------------------------------------------

GOLDEN_OUTPUTS_DIR = Path(__file__).resolve().parent / "golden_outputs"


def _make_config_digest(params: dict) -> str:
    """Return an 8-char MD5 digest of the config dict for use in golden keys."""
    s = json.dumps(params, sort_keys=True)
    import hashlib
    return hashlib.md5(s.encode()).hexdigest()[:8]


def make_golden_key(dtype: str, config_params: dict, extra_tags: dict = None) -> str:
    """Build a golden output key encoding dtype, config params and optional tags.

    Example: 'fp32_ctx32_mean_seqlen32_a1b2c3d4'
    """
    parts = [dtype.replace("torch.", "").replace("float", "fp")]
    if extra_tags:
        for k, v in sorted(extra_tags.items()):
            val = str(v).replace("torch.", "").replace("float", "fp")
            parts.append(f"{k}{val}")
    digest = _make_config_digest({**config_params, **(extra_tags or {})})
    parts.append(digest)
    return "_".join(parts)


def _load_golden_file(category: str) -> dict:
    """Load the golden output JSON file for a category. Returns empty dict if not found."""
    GOLDEN_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLDEN_OUTPUTS_DIR / f"{category}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {category: {}}


def _save_golden_file(category: str, data: dict):
    """Atomically save the golden output JSON file for a category."""
    GOLDEN_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLDEN_OUTPUTS_DIR / f"{category}.json"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _make_json_serializable(obj):
    """Recursively convert numpy/torch types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(i) for i in obj]
    return obj


def run_or_load_golden(
    category: str,
    model_name: str,
    golden_key: str,
    run_pytorch_fn,
    config_fp: str = None,
) -> dict:
    """Load golden output if it exists, otherwise run PyTorch and save it.

    Args:
        category: golden file name e.g. 'audio_models', 'embedding_models'.
        model_name: HuggingFace model id.
        golden_key: config key built via make_golden_key().
        run_pytorch_fn: callable() → dict of PyTorch outputs.
        config_fp: path to the pipeline config file used for this run.

    Returns:
        golden output dict for this (model_name, golden_key).
    """
    import datetime

    data = _load_golden_file(category)
    category_data = data.setdefault(category, {})
    model_data = category_data.setdefault(model_name, {})

    if golden_key in model_data:
        print(f"\n[GOLDEN] Loaded existing output: {category}/{model_name}/{golden_key}")
        return model_data[golden_key]

    print(f"\n[GOLDEN] No entry found for {model_name} [{golden_key}]. Running PyTorch reference...")
    output = run_pytorch_fn()
    entry = _make_json_serializable(output)
    entry["config_fp"] = str(config_fp) if config_fp else None
    entry["timestamp"] = datetime.datetime.utcnow().isoformat()

    model_data[golden_key] = entry
    _save_golden_file(category, data)
    print(f"\n[GOLDEN] Saved to: {GOLDEN_OUTPUTS_DIR / category}.json")
    return entry


def compare_with_golden(qpc_output: dict, golden: dict, tolerance: float = 0.0) -> dict:
    """Compare QPC output dict against golden PyTorch output dict.

    - str values: exact match (transcription).
    - list/numeric values: element-wise with tolerance (embeddings, token ids).

    Returns dict with 'passed' bool and 'per_key' details.
    """
    # Skip metadata keys
    skip_keys = {"config_fp", "timestamp"}
    results = {}

    for key, golden_val in golden.items():
        if key in skip_keys:
            continue
        if key not in qpc_output:
            results[key] = {"passed": False, "details": f"Key '{key}' missing in QPC output"}
            continue

        qpc_val = qpc_output[key]

        print(f"\nComparing key '{key}': golden={golden_val} qpc={qpc_val}")
        print(f"Type: golden={type(golden_val)} qpc={type(qpc_val)}")

        if isinstance(golden_val, str):
            # Exact string comparison e.g. transcription
            passed = golden_val.strip() == str(qpc_val).strip()
            results[key] = {"passed": passed, "details": f"golden='{golden_val}' qpc='{qpc_val}'"}

        elif isinstance(golden_val, (list, int, float)):
            # Numeric comparison
            g = np.array(golden_val, dtype=float)
            q = np.array(qpc_val, dtype=float)
            if g.shape != q.shape:
                results[key] = {
                    "passed": False,
                    "details": f"Shape mismatch: golden={g.shape} qpc={q.shape}",
                }
            else:
                # Use MAD (mean absolute difference) matching CI test threshold convention
                mad = float(np.mean(np.abs(g - q)))
                passed = mad <= tolerance
                results[key] = {
                    "passed": passed,
                    "details": f"mad={mad:.6f} tolerance={tolerance}",
                }
        else:
            results[key] = {"passed": True, "details": "skipped (unsupported type)"}

    all_passed = all(v["passed"] for v in results.values())
    return {"passed": all_passed, "per_key": results}


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
    # Embedding Models
    "jinaai/jina-embeddings-v2-base-code",
}
