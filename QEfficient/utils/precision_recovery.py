# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transformers import AutoConfig

DEFAULT_ITERATIVE_MLP_CANDIDATES = (
    "sparse_moe_post_ops_fp32,"
    "shared_expert_matmul_fp32_keep_fp32,"
    "shared_expert_matmul_fp32,"
    "shared_expert_gate_matmul_fp32,"
    "experts_matmul_fp32,"
    "mlp_full_fp32"
)


@dataclass
class PrecisionRecoveryRequest:
    model_id: str
    prompt: str = "Tell me about yourself."
    max_input_tokens: int = 64
    start_layer: int = 22
    max_layers: int = 0
    boundary_cache_dir: str = "scripts/debug/artifacts/qwen3_5_moe_window23_cache"
    continuation_cache_dir: str = "scripts/debug/artifacts/qwen3_5_moe_full_depth_iterative_cache"
    no_continuation_cache: bool = False
    iterative_mlp_candidates: str = DEFAULT_ITERATIVE_MLP_CANDIDATES
    output_json: str = "scripts/debug/artifacts/qwen3_5_moe_full_depth_iterative_fp32_recovery.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_qwen35_moe_runner_module():
    script_path = _repo_root() / "scripts" / "debug" / "qwen3_5_moe_full_depth_iterative_fp32_recovery.py"
    spec = importlib.util.spec_from_file_location("qwen3_5_moe_full_depth_recovery", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load precision recovery runner module at {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def detect_precision_recovery_backend(model_id: str) -> str:
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    text_cfg = getattr(cfg, "text_config", cfg)
    model_type = str(getattr(text_cfg, "model_type", getattr(cfg, "model_type", "")))
    arch = [str(x) for x in getattr(cfg, "architectures", [])]
    if model_type == "qwen3_5_moe" or any("Qwen3_5Moe" in x for x in arch):
        return "qwen3_5_moe"
    raise ValueError(
        f"No precision-recovery backend registered for model_id={model_id!r} "
        f"(detected model_type={model_type!r}, architectures={arch})."
    )


def run_precision_recovery(request: PrecisionRecoveryRequest) -> dict[str, Any]:
    backend = detect_precision_recovery_backend(request.model_id)
    if backend != "qwen3_5_moe":
        raise ValueError(f"Unsupported precision-recovery backend: {backend}")

    module = _load_qwen35_moe_runner_module()
    args = argparse.Namespace(**asdict(request))
    return module.run_recovery(args)


def summarize_precision_recovery_result(result: dict[str, Any]) -> dict[str, Any]:
    layers = result.get("layers", [])
    promoted_layers = []
    max_hf_mad = None
    max_qeff_mad = None
    max_hf_layer = None
    max_qeff_layer = None

    for row in layers:
        layer_idx = row["layer_idx"]
        baseline = row.get("final_variant", row.get("baseline", {}))
        hf_mad = baseline.get("layer_out_mad_vs_hf_fp32", {}).get("hf_fp16")
        qeff_mad = baseline.get("layer_out_mad_vs_hf_fp32", {}).get("qeff_fp16")
        if hf_mad is not None and (max_hf_mad is None or hf_mad > max_hf_mad):
            max_hf_mad = hf_mad
            max_hf_layer = layer_idx
        if qeff_mad is not None and (max_qeff_mad is None or qeff_mad > max_qeff_mad):
            max_qeff_mad = qeff_mad
            max_qeff_layer = layer_idx
        selected_patches = row.get("selected_patches", [])
        if selected_patches:
            promoted_layers.append({"layer_idx": layer_idx, "selected_patches": selected_patches})

    final = result.get("final", {})
    predictions = final.get("predictions", {})
    return {
        "completed_boundary": result.get("resume", {}).get("completed_boundary"),
        "unresolved": result.get("unresolved") is not None,
        "promoted_layers": promoted_layers,
        "max_layer_out_mad_vs_hf_fp32": {
            "hf_fp16": {"layer_idx": max_hf_layer, "mad": max_hf_mad},
            "qeff_fp16": {"layer_idx": max_qeff_layer, "mad": max_qeff_mad},
        },
        "final_token_prediction": {
            "hf_fp32": predictions.get("hf_fp32"),
            "hf_fp16_recovered": predictions.get("hf_fp16_recovered"),
            "qeff_fp16_recovered": predictions.get("qeff_fp16_recovered"),
        },
        "final_logits_mad": {
            "hf_fp16_vs_hf_fp32": final.get("logits_hf_fp16_vs_hf_fp32", {}).get("mad"),
            "qeff_fp16_vs_hf_fp32": final.get("logits_qeff_fp16_vs_hf_fp32", {}).get("mad"),
            "qeff_fp16_vs_hf_fp16": final.get("logits_qeff_fp16_vs_hf_fp16", {}).get("mad"),
        },
    }
