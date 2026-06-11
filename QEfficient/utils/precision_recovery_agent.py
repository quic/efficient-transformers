# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from QEfficient.utils.layer_scale_checkpoint import (
    build_layer_scale_recipe_from_recovery_result,
    dump_layer_scale_recipe_yaml,
)
from QEfficient.utils.precision_recovery import (
    DEFAULT_ITERATIVE_MLP_CANDIDATES,
    PrecisionRecoveryRequest,
    run_precision_recovery,
    summarize_precision_recovery_result,
)

MODEL_ID_KEYS = ("model_id", "model_name", "hf_model_id", "repo_id")
DEFAULT_SCALE_CANDIDATES = "1.0,0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125,0.00390625,0.001953125"
DEFAULT_SCALE_CANDIDATE_SCHEDULES = (DEFAULT_SCALE_CANDIDATES,)
_MODEL_CARD_CANDIDATE_ATTRS = (
    "_pretrained_model_name_or_path",
    "pretrained_model_name_or_path",
    "pretrained_path",
    "_name_or_path",
    "name_or_path",
)


def parse_scale_candidate_schedules(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_SCALE_CANDIDATE_SCHEDULES
    values = tuple(part.strip() for part in raw.split(";") if part.strip())
    if not values:
        raise ValueError("scale candidate schedules must be a non-empty ';'-separated string.")
    return values


def _extract_model_id_from_mapping(mapping: dict[str, Any]) -> str | None:
    for key in MODEL_ID_KEYS:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_model_id_from_text(text: str) -> str | None:
    for line in text.splitlines():
        lower = line.lower().strip()
        for key in MODEL_ID_KEYS:
            prefix = f"{key}:"
            if lower.startswith(prefix):
                value = line.split(":", 1)[1].strip()
                if value:
                    return value

    match = re.search(r"\b([A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*)\b", text)
    if match:
        return match.group(1)
    return None


def resolve_model_id_from_card(model_card: str) -> str:
    path = Path(model_card).expanduser()
    if not path.is_file():
        return model_card

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object.")
        model_id = _extract_model_id_from_mapping(payload)
        if model_id is None:
            raise ValueError(f"Could not resolve model id from {path}. Expected one of keys: {MODEL_ID_KEYS}")
        return model_id

    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
        if isinstance(payload, dict):
            model_id = _extract_model_id_from_mapping(payload)
            if model_id is not None:
                return model_id
        raise ValueError(f"Could not resolve model id from {path}. Expected one of keys: {MODEL_ID_KEYS}")

    model_id = _extract_model_id_from_text(text)
    if model_id is None:
        raise ValueError(
            f"Could not resolve model id from model card text at {path}. "
            f"Expected one of keys {MODEL_ID_KEYS} or a '<org>/<name>' token."
        )
    return model_id


def _non_empty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value else None


def _append_model_card_candidate(candidates: list[str], value: Any) -> None:
    candidate = _non_empty_str(value)
    if candidate is None:
        return
    candidates.append(candidate)


def _collect_model_card_candidates_from_object(obj: Any, candidates: list[str]) -> None:
    if obj is None:
        return

    for attr_name in _MODEL_CARD_CANDIDATE_ATTRS:
        _append_model_card_candidate(candidates, getattr(obj, attr_name, None))

    hash_params = getattr(obj, "hash_params", None)
    if isinstance(hash_params, dict):
        _append_model_card_candidate(candidates, hash_params.get("pretrained_model_name_or_path"))

    config = getattr(obj, "config", None)
    if config is not None and config is not obj:
        for attr_name in ("_name_or_path", "name_or_path"):
            _append_model_card_candidate(candidates, getattr(config, attr_name, None))


def resolve_model_card_from_loaded_qeff_model(qeff_model: Any) -> str:
    candidates: list[str] = []
    _collect_model_card_candidates_from_object(qeff_model, candidates)
    _collect_model_card_candidates_from_object(getattr(qeff_model, "model", None), candidates)
    _collect_model_card_candidates_from_object(getattr(qeff_model, "lang_model", None), candidates)
    _collect_model_card_candidates_from_object(
        getattr(getattr(qeff_model, "lang_model", None), "model", None), candidates
    )
    _collect_model_card_candidates_from_object(getattr(qeff_model, "vision_model", None), candidates)
    _collect_model_card_candidates_from_object(
        getattr(getattr(qeff_model, "vision_model", None), "model", None), candidates
    )

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        candidate_path = Path(candidate).expanduser()
        if candidate_path.exists():
            return str(candidate_path.resolve())
        return candidate

    raise ValueError(
        "Could not resolve model card/model id from loaded QEff model. "
        "Pass model_card explicitly to run_precision_recovery_agent_from_loaded_qeff_model(...)."
    )


def run_precision_recovery_agent_from_loaded_qeff_model(
    qeff_model: Any,
    *,
    model_card: str | None = None,
    **request_kwargs: Any,
) -> dict[str, Any]:
    """
    Run the precision-recovery agent starting from a loaded QEff wrapper.

    Parameters
    ----------
    qeff_model:
        A loaded QEff auto-wrapper instance (for example, `QEFFAutoModelForCausalLM.from_pretrained(...)`).
    model_card:
        Optional explicit model card/model id override. When not provided, it is inferred from the loaded wrapper.
    **request_kwargs:
        Additional `PrecisionRecoveryAgentRequest` fields.
    """

    if "model_card" in request_kwargs:
        if model_card is not None:
            raise ValueError("Pass model_card either as an explicit argument or in request_kwargs, not both.")
        model_card = request_kwargs.pop("model_card")

    resolved_model_card = model_card or resolve_model_card_from_loaded_qeff_model(qeff_model)
    request = PrecisionRecoveryAgentRequest(model_card=resolved_model_card, **request_kwargs)
    return run_precision_recovery_agent(request)


def needs_scale_search(analysis_summary: dict[str, Any]) -> bool:
    promoted = analysis_summary.get("promoted_layers", [])
    for layer in promoted:
        for patch in layer.get("selected_patches", []):
            if "matmul_fp32" in patch or patch == "mlp_full_fp32":
                return True
    return False


@dataclass
class PrecisionRecoveryAgentRequest:
    model_card: str
    prompt: str = "Tell me about yourself."
    max_input_tokens: int = 64
    start_layer: int = 22
    max_layers: int = 0
    boundary_cache_dir: str = "scripts/debug/artifacts/qwen3_5_moe_window23_cache"
    analysis_continuation_cache_dir: str = "scripts/debug/artifacts/qwen3_5_moe_full_depth_iterative_cache"
    scale_continuation_cache_dir: str = "scripts/debug/artifacts/qwen3_5_moe_full_depth_mlp_scale_cache"
    no_continuation_cache: bool = False
    iterative_mlp_candidates: str = DEFAULT_ITERATIVE_MLP_CANDIDATES
    scale_candidate_schedules: tuple[str, ...] = DEFAULT_SCALE_CANDIDATE_SCHEDULES
    default_scale: float = 1.0
    output_dir: str = "scripts/debug/artifacts/precision_recovery_agent"
    analysis_output_json: str | None = None
    recipe_yaml: str | None = None
    report_json: str | None = None


class PrecisionRecoveryAgent:
    def __init__(self, request: PrecisionRecoveryAgentRequest):
        self.request = request
        self.model_id = resolve_model_id_from_card(request.model_card)
        self.output_dir = Path(request.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _analysis_output_path(self) -> Path:
        if self.request.analysis_output_json:
            return Path(self.request.analysis_output_json).expanduser().resolve()
        return self.output_dir / "analysis_fp32_recovery.json"

    def _report_path(self) -> Path:
        if self.request.report_json:
            return Path(self.request.report_json).expanduser().resolve()
        return self.output_dir / "agent_report.json"

    def _recipe_path(self) -> Path:
        if self.request.recipe_yaml:
            return Path(self.request.recipe_yaml).expanduser().resolve()
        return self.output_dir / "layer_scales_recipe.yaml"

    def _scale_run_output_path(self, run_idx: int) -> Path:
        return self.output_dir / f"scale_recovery_run_{run_idx:02d}.json"

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _run_analysis(self) -> dict[str, Any]:
        request = PrecisionRecoveryRequest(
            model_id=self.model_id,
            prompt=self.request.prompt,
            max_input_tokens=self.request.max_input_tokens,
            start_layer=self.request.start_layer,
            max_layers=self.request.max_layers,
            boundary_cache_dir=self.request.boundary_cache_dir,
            continuation_cache_dir=self.request.analysis_continuation_cache_dir,
            no_continuation_cache=self.request.no_continuation_cache,
            iterative_mlp_candidates=self.request.iterative_mlp_candidates,
            output_json=str(self._analysis_output_path()),
        )
        return run_precision_recovery(request)

    def _scale_recovery_script(self) -> Path:
        return (
            Path(__file__).resolve().parents[2] / "scripts" / "debug" / "qwen3_5_moe_full_depth_mlp_scale_recovery.py"
        )

    def _run_scale_search_once(self, schedule: str, output_json: Path) -> dict[str, Any]:
        cmd = [
            sys.executable,
            str(self._scale_recovery_script()),
            "--model-id",
            self.model_id,
            "--prompt",
            self.request.prompt,
            "--max-input-tokens",
            str(self.request.max_input_tokens),
            "--start-layer",
            str(self.request.start_layer),
            "--max-layers",
            str(self.request.max_layers),
            "--boundary-cache-dir",
            str(Path(self.request.boundary_cache_dir).expanduser().resolve()),
            "--continuation-cache-dir",
            str(Path(self.request.scale_continuation_cache_dir).expanduser().resolve()),
            "--scale-candidates",
            schedule,
            "--output-json",
            str(output_json),
        ]
        if self.request.no_continuation_cache:
            cmd.append("--no-continuation-cache")
        subprocess.run(cmd, check=True)
        with open(output_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Scale recovery output must be a JSON object: {output_json}")
        return payload

    def _run_scale_search_loop(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        runs = []
        last_result = None
        for idx, schedule in enumerate(self.request.scale_candidate_schedules, start=1):
            output_path = self._scale_run_output_path(idx)
            result = self._run_scale_search_once(schedule=schedule, output_json=output_path)
            unresolved = result.get("unresolved") is not None
            runs.append(
                {
                    "run_idx": idx,
                    "scale_candidates": schedule,
                    "output_json": str(output_path),
                    "resolved": not unresolved,
                }
            )
            last_result = result
            if not unresolved:
                break
        if last_result is None:
            raise RuntimeError("Scale search loop did not execute any runs.")
        return last_result, runs

    def _emit_recipe_yaml(self, scale_recovery_result: dict[str, Any]) -> Path:
        recipe = build_layer_scale_recipe_from_recovery_result(
            recovery_result=scale_recovery_result,
            model_id=self.model_id,
            default_scale=self.request.default_scale,
        )
        return dump_layer_scale_recipe_yaml(
            recipe=recipe,
            output_path=self._recipe_path(),
            include_expanded_specs=True,
            extra_fields={
                "Method": "agentic_precision_recovery",
                "Description": (
                    "Generated by PrecisionRecoveryAgent. Use with Qwen3.5-MoE post-KV metadata transform runtime."
                ),
            },
        )

    def _runtime_integration_status(self, recipe_yaml: Path | None) -> dict[str, bool]:
        from QEfficient.transformers.models.modeling_auto import (
            QEFFAutoModelForCausalLM,
            QEffCausalLMForTextImageToTextModel,
            _QEFFAutoModelForImageTextToTextSingleQPC,
        )
        from QEfficient.transformers.models.pytorch_transforms import (
            Qwen3_5MoeLayerScaleMetadataTransform,
        )

        return {
            "causal_lm_pipeline_has_transform": (
                Qwen3_5MoeLayerScaleMetadataTransform in QEFFAutoModelForCausalLM._pytorch_transforms
            ),
            "image_text_dual_pipeline_has_transform": (
                Qwen3_5MoeLayerScaleMetadataTransform in QEffCausalLMForTextImageToTextModel._pytorch_transforms
            ),
            "image_text_single_pipeline_has_transform": (
                Qwen3_5MoeLayerScaleMetadataTransform in _QEFFAutoModelForImageTextToTextSingleQPC._pytorch_transforms
            ),
            "recipe_yaml_emitted": recipe_yaml is not None and recipe_yaml.is_file(),
        }

    def run(self) -> dict[str, Any]:
        analysis_result = self._run_analysis()
        analysis_summary = summarize_precision_recovery_result(analysis_result)
        run_scale_search = needs_scale_search(analysis_summary)

        scale_recovery_result = None
        scale_search_runs = []
        recipe_yaml = None

        if run_scale_search:
            scale_recovery_result, scale_search_runs = self._run_scale_search_loop()
            if scale_recovery_result.get("unresolved") is None:
                recipe_yaml = self._emit_recipe_yaml(scale_recovery_result)

        report = {
            "model_card": self.request.model_card,
            "model_id": self.model_id,
            "analysis": {
                "output_json": str(self._analysis_output_path()),
                "summary": analysis_summary,
            },
            "scale_search": {
                "required": run_scale_search,
                "runs": scale_search_runs,
                "resolved": (scale_recovery_result is not None and scale_recovery_result.get("unresolved") is None),
            },
            "recipe_yaml": str(recipe_yaml) if recipe_yaml else None,
            "runtime_integration": self._runtime_integration_status(recipe_yaml),
        }

        report_path = self._report_path()
        self._write_json(report_path, report)
        report["report_json"] = str(report_path)
        return report


def run_precision_recovery_agent(request: PrecisionRecoveryAgentRequest) -> dict[str, Any]:
    return PrecisionRecoveryAgent(request).run()
