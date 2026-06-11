# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from transformers import AutoConfig

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.utils.layer_scale_checkpoint import apply_layer_scale_recipe_to_snapshot


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_recipe_path() -> Path:
    return (
        _repo_root()
        / "QEfficient"
        / "transformers"
        / "models"
        / "qwen3_5_moe"
        / "configs"
        / "layer_scales_qwen3_5_397b_mlp_equivalent.yaml"
    )


def _default_scaled_dir() -> Path:
    return _repo_root() / "scripts" / "debug" / "artifacts" / "qwen3_5_397b_scaled_snapshot"


def _default_export_dir() -> Path:
    return _repo_root() / "scripts" / "debug" / "artifacts" / "qwen3_5_397b_layerwise_decode_onnx"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scale checkpoint from YAML, load via QEFFAutoModelForImageTextToText, "
            "and export decode-only layerwise ONNX."
        )
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-397B-A17B")
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Optional local source snapshot. If not set, resolves --model-name from local cache.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow hub download when cache is missing.",
    )
    parser.add_argument("--recipe-yaml", default=str(_default_recipe_path()))
    parser.add_argument("--scaled-dir", default=str(_default_scaled_dir()))
    parser.add_argument("--export-dir", default=str(_default_export_dir()))
    parser.add_argument("--reuse-scaled-dir", action="store_true")
    parser.add_argument("--skip-scaling", action="store_true")
    parser.add_argument("--layerwise-window-size", type=int, default=1)
    parser.add_argument("--prefill-seq-len", type=int, default=1)
    parser.add_argument(
        "--kv-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use dual-QPC language decoder path when true.",
    )
    parser.add_argument("--use-onnx-subfunctions", action="store_true")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    return parser


def _resolve_source_dir(source_dir: str | None, model_name: str, allow_download: bool) -> Path:
    if source_dir:
        return Path(source_dir).expanduser().resolve()
    try:
        snapshot_path = snapshot_download(
            repo_id=model_name,
            allow_patterns=["*.safetensors", "*.json", "*.txt"],
            local_files_only=not allow_download,
        )
    except LocalEntryNotFoundError as exc:
        raise ValueError(
            f"Model snapshot for {model_name!r} not found in local cache. "
            "Pass --source-dir or rerun with --allow-download."
        ) from exc
    return Path(snapshot_path).resolve()


def _metadata_exists(snapshot_dir: Path) -> bool:
    config_path = snapshot_dir / "config.json"
    if not config_path.is_file():
        return False
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    scales = config.get("qeff_layer_scales")
    if isinstance(scales, dict) and scales:
        return True
    text_scales = (config.get("text_config") or {}).get("qeff_layer_scales")
    return isinstance(text_scales, dict) and bool(text_scales)


def _prepare_scaled_snapshot(
    *,
    source_dir: Path | None,
    scaled_dir: Path,
    recipe_yaml: Path,
    reuse_scaled_dir: bool,
    skip_scaling: bool,
) -> Path:
    if skip_scaling:
        if not scaled_dir.is_dir():
            raise ValueError(f"--skip-scaling set, but scaled snapshot does not exist: {scaled_dir}")
        return scaled_dir

    if scaled_dir.exists() and any(scaled_dir.iterdir()):
        if not reuse_scaled_dir:
            raise ValueError(
                f"Scaled dir exists and is non-empty: {scaled_dir}. "
                "Use --reuse-scaled-dir or choose a new --scaled-dir."
            )
        return scaled_dir

    if source_dir is None:
        raise ValueError("source_dir is required when --skip-scaling is not set.")

    scaled_dir.mkdir(parents=True, exist_ok=True)
    apply_layer_scale_recipe_to_snapshot(
        source_dir=source_dir,
        output_dir=scaled_dir,
        recipe_path=recipe_yaml,
        strict=True,
        allow_hardlink_unchanged=True,
        allow_symlink_unchanged=True,
        inject_config_metadata=True,
        audit_json_path=None,
    )
    return scaled_dir


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    return torch.float16 if dtype_name == "float16" else torch.float32


def _find_qeff_text_model(module):
    for child in module.modules():
        if hasattr(child, "_qeff_layer_scaling_enabled") and hasattr(child, "_qeff_layer_scales"):
            return child
    raise RuntimeError("Could not find QEff text model with runtime scaling metadata fields.")


def _print_runtime_scaling_state(model_wrapper) -> None:
    text_model = _find_qeff_text_model(model_wrapper.model)
    layer_scales = getattr(text_model, "_qeff_layer_scales", {})
    print(f"[runtime_scaling_enabled] {bool(getattr(text_model, '_qeff_layer_scaling_enabled', False))}")
    print(f"[runtime_scale_default] {float(getattr(text_model, '_qeff_layer_scale_default', 1.0))}")
    print(f"[runtime_scale_layer22] {layer_scales.get(22)}")


def main() -> None:
    args = _build_parser().parse_args()
    dtype = _resolve_torch_dtype(args.dtype)

    source_dir = None
    if not args.skip_scaling:
        source_dir = _resolve_source_dir(args.source_dir, args.model_name, args.allow_download)
    recipe_yaml = Path(args.recipe_yaml).expanduser().resolve()
    scaled_dir = Path(args.scaled_dir).expanduser().resolve()
    export_dir = Path(args.export_dir).expanduser().resolve()

    print(f"[source] {source_dir if source_dir else '<unused (skip-scaling)>'}")
    print(f"[recipe] {recipe_yaml}")
    print(f"[scaled] {scaled_dir}")
    print(f"[export_dir] {export_dir}")

    snapshot_dir = _prepare_scaled_snapshot(
        source_dir=source_dir,
        scaled_dir=scaled_dir,
        recipe_yaml=recipe_yaml,
        reuse_scaled_dir=args.reuse_scaled_dir,
        skip_scaling=args.skip_scaling,
    )

    if not _metadata_exists(snapshot_dir):
        raise RuntimeError(
            "Scaled snapshot is missing qeff_layer_scales metadata in config.json. "
            "Regenerate the snapshot from YAML or point to a valid scaled snapshot."
        )

    config = AutoConfig.from_pretrained(snapshot_dir, local_files_only=True)
    config.torch_dtype = dtype

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        str(snapshot_dir),
        config=config,
        local_files_only=True,
        attn_implementation="eager",
        kv_offload=args.kv_offload,
        low_cpu_mem_usage=True,
        dtype=dtype,
        layerwise=True,
    )

    _print_runtime_scaling_state(qeff_model)

    onnx_path = qeff_model.export(
        export_dir=str(export_dir),
        skip_vision=True,
        prefill_seq_len=args.prefill_seq_len,
        layerwise=True,
        layerwise_window_size=args.layerwise_window_size,
        use_onnx_subfunctions=args.use_onnx_subfunctions,
    )
    print(f"[layerwise_decode_onnx] {onnx_path}")


if __name__ == "__main__":
    main()
