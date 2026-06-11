# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import mmap
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import yaml

from QEfficient.utils._utils import load_yaml

SUPPORTED_RUNTIME_EQUIVALENCE_MODES = {"checkpoint_scaled_mlp_and_residual_branch"}
DEFAULT_RUNTIME_EQUIVALENCE_MODE = "checkpoint_scaled_mlp_and_residual_branch"


@dataclass(frozen=True)
class TensorScalingRule:
    name: str
    tensor_template: str
    operation: str


@dataclass(frozen=True)
class LayerScaleRecipe:
    model_id: str | None
    default_scale: float
    layer_scales: Dict[int, float]
    rules: tuple[TensorScalingRule, ...]
    runtime_equivalence_mode: str = DEFAULT_RUNTIME_EQUIVALENCE_MODE
    runtime_equivalence: Dict[str, Any] | None = None
    config_scale_key: str = "qeff_layer_scales"
    config_default_key: str = "qeff_layer_scale_default"
    config_mode_key: str = "qeff_layer_scale_mode"
    config_mode_value: str = DEFAULT_RUNTIME_EQUIVALENCE_MODE


@dataclass(frozen=True)
class TensorScaleSpec:
    layer_idx: int
    tensor_key: str
    operation: str
    scale: float
    rule_name: str


_SUPPORTED_SCALING_OPS = {"scale_all", "scale_second_half_dim1"}

_DTYPE_CODE_TO_TORCH = {
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}


def _parse_layer_scale_map(raw: Any) -> Dict[int, float]:
    if raw is None:
        return {}
    out: Dict[int, float] = {}
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = []
        for row in raw:
            if not isinstance(row, dict):
                raise ValueError(f"Invalid layer scale entry: expected dict, got {type(row).__name__}")
            if "layer_idx" not in row or "scale" not in row:
                raise ValueError(f"Layer scale rows must include layer_idx and scale. Got: {row}")
            items.append((row["layer_idx"], row["scale"]))
    else:
        raise ValueError(f"Unsupported layer scale format: {type(raw).__name__}")

    for key, value in items:
        layer_idx = int(key)
        scale = float(value)
        if layer_idx < 0:
            raise ValueError(f"Layer index must be >= 0. Got {layer_idx}")
        if scale <= 0.0:
            raise ValueError(f"Scale must be > 0. Got layer={layer_idx} scale={scale}")
        out[layer_idx] = scale
    return out


def _parse_scaling_rules(raw_rules: Any) -> tuple[TensorScalingRule, ...]:
    if raw_rules is None:
        return (
            TensorScalingRule(
                name="experts_gate_up_proj_up_half",
                tensor_template="model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj",
                operation="scale_second_half_dim1",
            ),
            TensorScalingRule(
                name="shared_expert_up_proj",
                tensor_template="model.language_model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight",
                operation="scale_all",
            ),
        )

    if not isinstance(raw_rules, list):
        raise ValueError(f"TensorScalingRules must be a list. Got {type(raw_rules).__name__}")

    rules: list[TensorScalingRule] = []
    for idx, row in enumerate(raw_rules):
        if not isinstance(row, dict):
            raise ValueError(f"TensorScalingRules[{idx}] must be a dict. Got {type(row).__name__}")
        name = str(row.get("name", f"rule_{idx}"))
        tensor_template = row.get("tensor_template")
        operation = row.get("operation")
        if not tensor_template or not operation:
            raise ValueError(f"TensorScalingRules[{idx}] missing tensor_template/operation: {row}")
        operation = str(operation)
        if operation not in _SUPPORTED_SCALING_OPS:
            raise ValueError(
                f"TensorScalingRules[{idx}] has unsupported operation={operation!r}. "
                f"Supported={sorted(_SUPPORTED_SCALING_OPS)}"
            )
        rules.append(
            TensorScalingRule(
                name=name,
                tensor_template=str(tensor_template),
                operation=operation,
            )
        )
    return tuple(rules)


def _default_runtime_equivalence() -> Dict[str, Any]:
    return {
        "mode": DEFAULT_RUNTIME_EQUIVALENCE_MODE,
        "compensation_points": [
            {
                "name": "pre_input_inverse",
                "module": "QEffQwen3_5MoeDecoderLayer.forward",
                "location": "before input_layernorm",
                "equation": "hidden_in = hidden_in / scale(layer_idx - 1)",
            },
            {
                "name": "scaled_residual_add",
                "module": "QEffQwen3_5MoeDecoderLayer.forward",
                "location": "final residual + mlp add",
                "equation": "layer_out_scaled = scale(layer_idx) * residual + mlp_out_scaled",
            },
            {
                "name": "final_unscale_before_norm",
                "module": "QEffQwen3_5MoeTextModel.forward",
                "location": "before final norm on last layer window",
                "equation": "hidden_before_norm = hidden_before_norm / scale(last_layer_idx)",
            },
        ],
    }


def _parse_runtime_equivalence(
    raw_runtime_equivalence: Any,
    mode_override: str | None,
) -> tuple[str, Dict[str, Any]]:
    runtime_equivalence = _default_runtime_equivalence()

    if raw_runtime_equivalence is not None:
        if not isinstance(raw_runtime_equivalence, dict):
            raise ValueError(
                f"RuntimeEquivalence must be a dict when provided. Got {type(raw_runtime_equivalence).__name__}"
            )
        runtime_equivalence.update(raw_runtime_equivalence)

    runtime_mode = str(runtime_equivalence.get("mode", DEFAULT_RUNTIME_EQUIVALENCE_MODE))
    if mode_override is not None and mode_override != runtime_mode:
        raise ValueError(
            "ConfigMetadata mode_value must match RuntimeEquivalence.mode for mathematical-equivalence safety. "
            f"Got mode_value={mode_override!r}, RuntimeEquivalence.mode={runtime_mode!r}."
        )
    runtime_mode = str(mode_override if mode_override is not None else runtime_mode)
    if runtime_mode not in SUPPORTED_RUNTIME_EQUIVALENCE_MODES:
        raise ValueError(
            f"Unsupported RuntimeEquivalence mode={runtime_mode!r}. "
            f"Supported={sorted(SUPPORTED_RUNTIME_EQUIVALENCE_MODES)}"
        )
    runtime_equivalence["mode"] = runtime_mode
    return runtime_mode, runtime_equivalence


def load_layer_scale_recipe(recipe_path: str | Path) -> LayerScaleRecipe:
    recipe_path = Path(recipe_path).expanduser().resolve()
    raw = load_yaml(str(recipe_path))
    if not isinstance(raw, dict):
        raise ValueError(f"Recipe YAML must deserialize to a dict. Got {type(raw).__name__}")

    model_id = raw.get("ModelID") or raw.get("model_id")

    if "LayerScales" in raw:
        layer_scales_block = raw["LayerScales"] or {}
        default_scale = float(layer_scales_block.get("default_scale", 1.0))
        per_layer = _parse_layer_scale_map(layer_scales_block.get("per_layer", {}))
    elif "ScaleConfig" in raw:
        scale_cfg = raw["ScaleConfig"] or {}
        default_scale = float(scale_cfg.get("default_scale", 1.0))
        per_layer = _parse_layer_scale_map(scale_cfg.get("non_unit_layer_scales", []))
    else:
        default_scale = 1.0
        per_layer = _parse_layer_scale_map(raw.get("layer_scales", {}))

    if default_scale <= 0.0:
        raise ValueError(f"default_scale must be > 0. Got {default_scale}")

    rules = _parse_scaling_rules(raw.get("TensorScalingRules"))

    cfg_overrides = raw.get("ConfigMetadata", {}) or {}
    if not isinstance(cfg_overrides, dict):
        raise ValueError(f"ConfigMetadata must be a dict. Got {type(cfg_overrides).__name__}")

    mode_override = cfg_overrides.get("mode_value", None)
    runtime_mode, runtime_equivalence = _parse_runtime_equivalence(raw.get("RuntimeEquivalence"), mode_override)

    return LayerScaleRecipe(
        model_id=model_id,
        default_scale=default_scale,
        layer_scales=per_layer,
        rules=rules,
        runtime_equivalence_mode=runtime_mode,
        runtime_equivalence=runtime_equivalence,
        config_scale_key=str(cfg_overrides.get("layer_scale_key", "qeff_layer_scales")),
        config_default_key=str(cfg_overrides.get("default_scale_key", "qeff_layer_scale_default")),
        config_mode_key=str(cfg_overrides.get("mode_key", "qeff_layer_scale_mode")),
        config_mode_value=runtime_mode,
    )


def build_tensor_scale_specs(recipe: LayerScaleRecipe) -> list[TensorScaleSpec]:
    specs: list[TensorScaleSpec] = []
    for layer_idx, scale in sorted(recipe.layer_scales.items()):
        if scale == recipe.default_scale:
            continue
        for rule in recipe.rules:
            specs.append(
                TensorScaleSpec(
                    layer_idx=layer_idx,
                    tensor_key=rule.tensor_template.format(layer_idx=layer_idx),
                    operation=rule.operation,
                    scale=scale,
                    rule_name=rule.name,
                )
            )
    return specs


def serialize_layer_scale_recipe(recipe: LayerScaleRecipe, include_expanded_specs: bool = False) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "ModelID": recipe.model_id,
        "LayerScales": {
            "default_scale": float(recipe.default_scale),
            "per_layer": {int(k): float(v) for k, v in sorted(recipe.layer_scales.items())},
        },
        "TensorScalingRules": [
            {
                "name": rule.name,
                "tensor_template": rule.tensor_template,
                "operation": rule.operation,
            }
            for rule in recipe.rules
        ],
        "RuntimeEquivalence": recipe.runtime_equivalence
        if recipe.runtime_equivalence
        else _default_runtime_equivalence(),
        "ConfigMetadata": {
            "layer_scale_key": recipe.config_scale_key,
            "default_scale_key": recipe.config_default_key,
            "mode_key": recipe.config_mode_key,
            "mode_value": recipe.config_mode_value,
        },
    }

    if include_expanded_specs:
        data["ScaledTensorSpecs"] = [
            {
                "layer_idx": int(spec.layer_idx),
                "tensor_key": spec.tensor_key,
                "operation": spec.operation,
                "scale": float(spec.scale),
                "rule_name": spec.rule_name,
            }
            for spec in build_tensor_scale_specs(recipe)
        ]
    return data


def dump_layer_scale_recipe_yaml(
    recipe: LayerScaleRecipe,
    output_path: str | Path,
    include_expanded_specs: bool = True,
    extra_fields: Dict[str, Any] | None = None,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = serialize_layer_scale_recipe(recipe, include_expanded_specs=include_expanded_specs)
    if extra_fields:
        payload.update(extra_fields)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)
    return output_path


def build_layer_scale_recipe_from_recovery_result(
    *,
    recovery_result: Dict[str, Any],
    model_id: str | None = None,
    default_scale: float = 1.0,
    rules: tuple[TensorScalingRule, ...] | None = None,
) -> LayerScaleRecipe:
    if default_scale <= 0.0:
        raise ValueError(f"default_scale must be > 0. Got {default_scale}")

    unresolved = recovery_result.get("unresolved")
    if unresolved is not None:
        raise ValueError("Cannot build mathematically-equivalent scale recipe: recovery result is unresolved.")

    layer_rows = recovery_result.get("layers")
    if not isinstance(layer_rows, list):
        raise ValueError("Recovery result must contain a list under `layers`.")

    per_layer: Dict[int, float] = {}
    for row in layer_rows:
        if not isinstance(row, dict):
            continue
        if not row.get("resolved", False):
            continue
        layer_idx = int(row["layer_idx"])
        selected_scale = float(row.get("selected_scale", default_scale))
        if selected_scale <= 0.0:
            raise ValueError(f"Invalid selected_scale for layer {layer_idx}: {selected_scale}")
        if selected_scale == default_scale:
            continue

        final_variant = row.get("final_variant", {}) or {}
        finite_ratio = final_variant.get("layer_out_finite_ratio", {}) or {}
        hf_fp16_finite = float(finite_ratio.get("hf_fp16", 0.0))
        qeff_fp16_finite = float(finite_ratio.get("qeff_fp16", 0.0))
        if hf_fp16_finite < 1.0 or qeff_fp16_finite < 1.0:
            raise ValueError(
                f"Layer {layer_idx} selected_scale={selected_scale} is not fully finite "
                f"(hf_fp16={hf_fp16_finite}, qeff_fp16={qeff_fp16_finite})."
            )
        per_layer[layer_idx] = selected_scale

    resolved_model_id = model_id
    if resolved_model_id is None:
        resolved_model_id = recovery_result.get("config", {}).get("model_id")

    recipe_rules = rules if rules is not None else _parse_scaling_rules(None)
    runtime_mode, runtime_equivalence = _parse_runtime_equivalence(
        raw_runtime_equivalence=None,
        mode_override=DEFAULT_RUNTIME_EQUIVALENCE_MODE,
    )
    return LayerScaleRecipe(
        model_id=resolved_model_id,
        default_scale=float(default_scale),
        layer_scales=per_layer,
        rules=recipe_rules,
        runtime_equivalence_mode=runtime_mode,
        runtime_equivalence=runtime_equivalence,
        config_mode_value=runtime_mode,
    )


def build_layer_scale_recipe_from_recovery_json(
    *,
    recovery_json_path: str | Path,
    model_id: str | None = None,
    default_scale: float = 1.0,
    rules: tuple[TensorScalingRule, ...] | None = None,
) -> LayerScaleRecipe:
    recovery_json_path = Path(recovery_json_path).expanduser().resolve()
    with open(recovery_json_path, "r", encoding="utf-8") as f:
        recovery_result = json.load(f)
    if not isinstance(recovery_result, dict):
        raise ValueError(f"Recovery JSON must deserialize to a dict. Got {type(recovery_result).__name__}")
    return build_layer_scale_recipe_from_recovery_result(
        recovery_result=recovery_result,
        model_id=model_id,
        default_scale=default_scale,
        rules=rules,
    )


def _link_or_copy(src: Path, dst: Path, *, allow_hardlink: bool, allow_symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if allow_hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    if allow_symlink:
        try:
            os.symlink(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _collect_snapshot_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def _safetensor_header(path: Path) -> tuple[int, dict[str, Any]]:
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header_data = f.read(header_len)
    header = json.loads(header_data)
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensor header in {path}")
    return header_len, header


def _apply_scale_op(tensor_view: torch.Tensor, operation: str, scale: float) -> None:
    scalar = tensor_view.new_tensor(scale)
    if operation == "scale_all":
        tensor_view.mul_(scalar)
        return
    if operation == "scale_second_half_dim1":
        if tensor_view.ndim < 2:
            raise ValueError(f"scale_second_half_dim1 requires rank>=2. Got rank={tensor_view.ndim}")
        dim1 = int(tensor_view.shape[1])
        if dim1 % 2 != 0:
            raise ValueError(f"scale_second_half_dim1 requires even dim1. Got dim1={dim1}")
        half = dim1 // 2
        index = (slice(None), slice(half, None), *([slice(None)] * (tensor_view.ndim - 2)))
        tensor_view[index].mul_(scalar)
        return
    raise ValueError(f"Unsupported operation={operation!r}")


def _tensor_summary(tensor: torch.Tensor, max_sample: int = 1_000_000) -> dict[str, float]:
    flat = tensor.reshape(-1)
    if flat.numel() > max_sample:
        step = max(1, flat.numel() // max_sample)
        flat = flat[::step]
    as_float = flat.float()
    finite = torch.isfinite(as_float)
    finite_ratio = float(finite.float().mean().item())
    if finite.any():
        finite_vals = as_float[finite]
        min_v = float(finite_vals.min().item())
        max_v = float(finite_vals.max().item())
        abs_max = float(finite_vals.abs().max().item())
    else:
        min_v = float("nan")
        max_v = float("nan")
        abs_max = float("nan")
    return {"finite_ratio": finite_ratio, "min": min_v, "max": max_v, "abs_max": abs_max}


def _patch_safetensor_file(
    shard_path: Path,
    specs: list[TensorScaleSpec],
) -> list[dict[str, Any]]:
    header_len, header = _safetensor_header(shard_path)
    data_base = 8 + header_len
    audit_rows: list[dict[str, Any]] = []

    with open(shard_path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        try:
            for spec in specs:
                tensor_meta = header.get(spec.tensor_key)
                if not isinstance(tensor_meta, dict):
                    raise KeyError(f"Tensor key {spec.tensor_key!r} not found in shard {shard_path.name}")
                dtype_code = tensor_meta.get("dtype")
                if dtype_code not in _DTYPE_CODE_TO_TORCH:
                    raise ValueError(
                        f"Unsupported dtype={dtype_code!r} for tensor={spec.tensor_key}. "
                        f"Supported={sorted(_DTYPE_CODE_TO_TORCH)}"
                    )
                shape = tuple(int(x) for x in tensor_meta.get("shape", []))
                begin, end = tensor_meta.get("data_offsets", [None, None])
                if begin is None or end is None:
                    raise ValueError(f"Missing data_offsets for tensor={spec.tensor_key} in {shard_path.name}")
                abs_begin = data_base + int(begin)
                abs_end = data_base + int(end)
                view = memoryview(mm)[abs_begin:abs_end]
                tensor = torch.frombuffer(view, dtype=_DTYPE_CODE_TO_TORCH[dtype_code]).view(shape)
                before = _tensor_summary(tensor)
                _apply_scale_op(tensor, spec.operation, spec.scale)
                after = _tensor_summary(tensor)

                del tensor
                del view

                audit_rows.append(
                    {
                        "layer_idx": spec.layer_idx,
                        "tensor_key": spec.tensor_key,
                        "rule_name": spec.rule_name,
                        "operation": spec.operation,
                        "scale": spec.scale,
                        "shard": shard_path.name,
                        "dtype": dtype_code,
                        "shape": list(shape),
                        "before": before,
                        "after": after,
                    }
                )
        finally:
            mm.close()

    return audit_rows


def _inject_scale_metadata(config_path: Path, recipe: LayerScaleRecipe, recipe_path: Path) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    if not isinstance(config_data, dict):
        raise ValueError(f"{config_path} must contain a JSON object.")

    metadata_patch = _build_config_metadata_patch(recipe, recipe_path)

    config_data.update(metadata_patch)

    text_cfg = config_data.get("text_config")
    if isinstance(text_cfg, dict):
        text_cfg.update(metadata_patch)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, sort_keys=True)
        f.write("\n")


def _build_config_metadata_patch(recipe: LayerScaleRecipe, recipe_path: Path) -> dict[str, Any]:
    scale_map_str = {str(k): float(v) for k, v in sorted(recipe.layer_scales.items()) if v != recipe.default_scale}
    metadata_patch: dict[str, Any] = {
        "qeff_layer_scales": scale_map_str,
        "qeff_layer_scale_default": float(recipe.default_scale),
        "qeff_layer_scale_mode": recipe.config_mode_value,
        "qeff_layer_scale_recipe": str(recipe_path),
    }
    metadata_patch[recipe.config_scale_key] = scale_map_str
    metadata_patch[recipe.config_default_key] = float(recipe.default_scale)
    metadata_patch[recipe.config_mode_key] = recipe.config_mode_value
    return metadata_patch


def _inject_scale_metadata_to_loaded_model(model: torch.nn.Module, recipe: LayerScaleRecipe, recipe_path: Path) -> None:
    metadata_patch = _build_config_metadata_patch(recipe, recipe_path)
    config = getattr(model, "config", None)
    if config is None:
        return
    for key, value in metadata_patch.items():
        setattr(config, key, value)
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return
    for key, value in metadata_patch.items():
        setattr(text_config, key, value)


def _collect_named_model_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    tensor_map: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        tensor_map[name] = param
    for name, buffer in model.named_buffers():
        if name not in tensor_map:
            tensor_map[name] = buffer
    return tensor_map


def apply_layer_scale_recipe_to_loaded_model(
    *,
    model: torch.nn.Module,
    recipe_path: str | Path,
    strict: bool = True,
    inject_config_metadata: bool = True,
) -> dict[str, Any]:
    recipe_path = Path(recipe_path).expanduser().resolve()
    recipe = load_layer_scale_recipe(recipe_path)
    specs = build_tensor_scale_specs(recipe)
    tensor_map = _collect_named_model_tensors(model)

    missing_specs: list[str] = []
    matched: list[tuple[TensorScaleSpec, torch.Tensor]] = []
    for spec in specs:
        tensor = tensor_map.get(spec.tensor_key)
        if tensor is None:
            missing_specs.append(spec.tensor_key)
            continue
        matched.append((spec, tensor))

    if strict and missing_specs:
        raise KeyError(f"{len(missing_specs)} recipe tensor keys missing in model tensors. First few: {missing_specs[:8]}")

    audit_rows: list[dict[str, Any]] = []
    for spec, tensor in matched:
        before = _tensor_summary(tensor)
        with torch.no_grad():
            _apply_scale_op(tensor, spec.operation, spec.scale)
        after = _tensor_summary(tensor)
        audit_rows.append(
            {
                "layer_idx": int(spec.layer_idx),
                "tensor_key": spec.tensor_key,
                "rule_name": spec.rule_name,
                "operation": spec.operation,
                "scale": float(spec.scale),
                "dtype": str(tensor.dtype),
                "shape": [int(x) for x in tensor.shape],
                "before": before,
                "after": after,
            }
        )

    if inject_config_metadata:
        _inject_scale_metadata_to_loaded_model(model, recipe, recipe_path)

    return {
        "recipe_path": str(recipe_path),
        "model_id": recipe.model_id,
        "default_scale": float(recipe.default_scale),
        "layer_scales": {str(k): float(v) for k, v in sorted(recipe.layer_scales.items())},
        "scaled_tensor_count": len(audit_rows),
        "scaled_unique_keys": len({row["tensor_key"] for row in audit_rows}),
        "missing_recipe_keys": missing_specs,
        "entries": audit_rows,
    }


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise ValueError(f"output_dir exists and is not empty: {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)


def _load_safetensor_weight_map(index_path: Path) -> dict[str, str]:
    with open(index_path, "r", encoding="utf-8") as handle:
        index_data = json.load(handle)
    if not isinstance(index_data, dict) or "weight_map" not in index_data:
        raise ValueError(f"Invalid safetensor index format: {index_path}")
    weight_map = index_data["weight_map"]
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json: weight_map must be a dict")
    return {str(key): str(value) for key, value in weight_map.items()}


def _group_specs_by_shard(
    specs: list[TensorScaleSpec],
    weight_map: dict[str, str],
) -> tuple[dict[str, list[TensorScaleSpec]], list[str]]:
    missing_specs: list[str] = []
    shard_to_specs: dict[str, list[TensorScaleSpec]] = {}

    for spec in specs:
        shard_name = weight_map.get(spec.tensor_key)
        if shard_name is None:
            missing_specs.append(spec.tensor_key)
            continue
        shard_to_specs.setdefault(shard_name, []).append(spec)
    return shard_to_specs, missing_specs


def _copy_snapshot_with_patch_targets(
    source_dir: Path,
    output_dir: Path,
    patched_shards: set[str],
    *,
    allow_hardlink_unchanged: bool,
    allow_symlink_unchanged: bool,
) -> None:
    for src_file in _collect_snapshot_files(source_dir):
        rel_path = src_file.relative_to(source_dir)
        dst_file = output_dir / rel_path
        must_copy = src_file.name in patched_shards or src_file.name in {"config.json", "model.safetensors.index.json"}
        _link_or_copy(
            src_file,
            dst_file,
            allow_hardlink=allow_hardlink_unchanged and not must_copy,
            allow_symlink=allow_symlink_unchanged and not must_copy,
        )


def _patch_shards(
    output_dir: Path,
    shard_to_specs: dict[str, list[TensorScaleSpec]],
) -> list[dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    for shard_name, shard_specs in sorted(shard_to_specs.items()):
        shard_path = output_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing copied shard in output snapshot: {shard_path}")
        audit_rows.extend(_patch_safetensor_file(shard_path, shard_specs))
    return audit_rows


def _resolve_audit_json_path(output_dir: Path, audit_json_path: str | Path | None) -> Path:
    if audit_json_path is None:
        return output_dir / "qeff_layer_scale_audit.json"
    path = Path(audit_json_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _build_snapshot_scale_audit(
    *,
    source_dir: Path,
    output_dir: Path,
    recipe_path: Path,
    recipe: LayerScaleRecipe,
    patched_shards: set[str],
    missing_specs: list[str],
    audit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    touched_layers = sorted({int(row["layer_idx"]) for row in audit_rows})
    touched_keys = sorted({str(row["tensor_key"]) for row in audit_rows})
    return {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "recipe_path": str(recipe_path),
        "model_id": recipe.model_id,
        "default_scale": recipe.default_scale,
        "layer_scales": {str(k): float(v) for k, v in sorted(recipe.layer_scales.items())},
        "touched_layers": touched_layers,
        "patched_shards": sorted(patched_shards),
        "scaled_tensor_count": len(audit_rows),
        "scaled_unique_keys": len(touched_keys),
        "missing_recipe_keys": missing_specs,
        "entries": audit_rows,
    }


def apply_layer_scale_recipe_to_snapshot(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    recipe_path: str | Path,
    strict: bool = True,
    allow_hardlink_unchanged: bool = True,
    allow_symlink_unchanged: bool = True,
    inject_config_metadata: bool = True,
    audit_json_path: str | Path | None = None,
) -> dict[str, Any]:
    source_dir = Path(source_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    recipe_path = Path(recipe_path).expanduser().resolve()

    if not source_dir.is_dir():
        raise ValueError(f"source_dir does not exist: {source_dir}")

    index_path = source_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError(f"Missing model.safetensors.index.json under {source_dir}")

    _prepare_output_dir(output_dir)

    recipe = load_layer_scale_recipe(recipe_path)
    specs = build_tensor_scale_specs(recipe)
    weight_map = _load_safetensor_weight_map(index_path)
    shard_to_specs, missing_specs = _group_specs_by_shard(specs, weight_map)
    patched_shards = set(shard_to_specs)

    if strict and missing_specs:
        raise KeyError(f"{len(missing_specs)} recipe tensor keys missing in weight_map. First few: {missing_specs[:8]}")

    _copy_snapshot_with_patch_targets(
        source_dir,
        output_dir,
        patched_shards,
        allow_hardlink_unchanged=allow_hardlink_unchanged,
        allow_symlink_unchanged=allow_symlink_unchanged,
    )
    audit_rows = _patch_shards(output_dir, shard_to_specs)

    config_path = output_dir / "config.json"
    if inject_config_metadata and config_path.is_file():
        _inject_scale_metadata(config_path, recipe, recipe_path)

    audit_path = _resolve_audit_json_path(output_dir, audit_json_path)
    audit = _build_snapshot_scale_audit(
        source_dir=source_dir,
        output_dir=output_dir,
        recipe_path=recipe_path,
        recipe=recipe,
        patched_shards=patched_shards,
        missing_specs=missing_specs,
        audit_rows=audit_rows,
    )
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
        f.write("\n")

    return audit
