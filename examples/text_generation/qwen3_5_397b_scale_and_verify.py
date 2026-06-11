# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

# Ensure local workspace imports resolve to this repo without requiring PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from QEfficient.utils.layer_scale_checkpoint import apply_layer_scale_recipe_to_snapshot  # noqa: E402


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def _default_compare_json() -> Path:
    return _repo_root() / "scripts" / "debug" / "artifacts" / "qwen3_5_397b_scaled_chunked_compare.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Qwen3.5-397B end-to-end precision recovery verification:\n"
            "1) apply layer-scale recipe to checkpoint\n"
            "2) run full-depth chunked compare (HF fp32 / HF fp16 / QEff fp16)\n"
            "3) validate final token parity"
        )
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-397B-A17B")
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Local source snapshot path. If omitted, resolves from local Hugging Face cache by --model-name.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow downloading from Hugging Face Hub when model snapshot is not present in local cache.",
    )
    parser.add_argument("--scaled-dir", default=str(_default_scaled_dir()))
    parser.add_argument(
        "--recipe-yaml",
        default=None,
        help="Scale recipe YAML used for offline checkpoint scaling. Optional when --skip-scaling is set.",
    )
    parser.add_argument("--audit-json", default=None)
    parser.add_argument("--reuse-scaled-dir", action="store_true")
    parser.add_argument("--skip-scaling", action="store_true")
    parser.add_argument(
        "--runtime-layer-scale-yaml",
        default=None,
        help="Optional runtime recipe YAML for model-layer scale metadata (env-driven).",
    )
    parser.add_argument(
        "--disable-runtime-layer-scaling",
        action="store_true",
        help="Force-disable runtime layer scaling even if config metadata exists.",
    )
    parser.add_argument("--prompt", default="Tell me about yourself.")
    parser.add_argument("--max-input-tokens", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--max-layers", type=int, default=0, help="0 means full depth")
    parser.add_argument("--compare-output-json", default=str(_default_compare_json()))
    parser.add_argument("--expected-token-id", type=int, default=198)
    parser.add_argument("--strict-token-check", action="store_true")
    return parser


def _resolve_source_dir(source_dir: str | None, model_name: str, allow_download: bool) -> Path:
    if source_dir:
        return Path(source_dir).expanduser().resolve()
    try:
        return Path(
            snapshot_download(
                repo_id=model_name,
                allow_patterns=["*.safetensors", "*.json", "*.txt"],
                local_files_only=not allow_download,
            )
        )
    except LocalEntryNotFoundError as exc:
        raise ValueError(
            f"Model snapshot for {model_name!r} not found in local cache. "
            "Pass --source-dir or rerun with --allow-download."
        ) from exc


def _has_layer_scale_metadata(model_dir: Path) -> bool:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return False
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        return False

    scales = config.get("qeff_layer_scales")
    if isinstance(scales, dict) and len(scales) > 0:
        return True

    text_cfg = config.get("text_config")
    if isinstance(text_cfg, dict):
        text_scales = text_cfg.get("qeff_layer_scales")
        if isinstance(text_scales, dict) and len(text_scales) > 0:
            return True

    return False


def _run_chunked_compare(
    *,
    model_path: Path,
    prompt: str,
    max_input_tokens: int,
    chunk_size: int,
    max_layers: int,
    output_json: Path,
    runtime_layer_scale_yaml: str | None,
    disable_runtime_layer_scaling: bool,
) -> None:
    script_path = _repo_root() / "scripts" / "debug" / "qwen3_5_moe_chunked_precision_compare.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--model-id",
        str(model_path),
        "--prompt",
        prompt,
        "--max-input-tokens",
        str(max_input_tokens),
        "--chunk-size",
        str(chunk_size),
        "--output-json",
        str(output_json),
    ]
    if max_layers > 0:
        cmd.extend(["--max-layers", str(max_layers)])

    env = dict(os.environ)
    if disable_runtime_layer_scaling:
        env["QEFF_QWEN3_5_MOE_ENABLE_LAYER_SCALING"] = "0"
        env.pop("QEFF_QWEN3_5_MOE_LAYER_SCALE_YAML", None)
    elif runtime_layer_scale_yaml:
        env["QEFF_QWEN3_5_MOE_ENABLE_LAYER_SCALING"] = "1"
        env["QEFF_QWEN3_5_MOE_LAYER_SCALE_YAML"] = runtime_layer_scale_yaml
    subprocess.run(cmd, check=True, env=env)


def _extract_final_tokens(compare_json_path: Path) -> dict[str, dict]:
    with open(compare_json_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    preds = result["final"]["predictions"]
    return {
        "hf_fp32": preds["hf_fp32"],
        "hf_fp16": preds["hf_fp16"],
        "qeff_fp16": preds["qeff_fp16"],
    }


def main() -> None:
    args = _build_parser().parse_args()

    source_dir: Path | None = None
    scaled_dir = Path(args.scaled_dir).expanduser().resolve()
    recipe_yaml = Path(args.recipe_yaml).expanduser().resolve() if args.recipe_yaml else _default_recipe_path()
    runtime_layer_scale_yaml = (
        str(Path(args.runtime_layer_scale_yaml).expanduser().resolve()) if args.runtime_layer_scale_yaml else None
    )
    compare_output_json = Path(args.compare_output_json).expanduser().resolve()

    if not args.skip_scaling:
        source_dir = _resolve_source_dir(args.source_dir, args.model_name, args.allow_download)
        print(f"[source] {source_dir}")
    else:
        print("[source] <unused (skip-scaling)>")
    print(f"[scaled] {scaled_dir}")
    print(f"[recipe] {recipe_yaml if not args.skip_scaling else '<unused (skip-scaling)>'}")
    if runtime_layer_scale_yaml:
        print(f"[runtime-layer-scale-yaml] {runtime_layer_scale_yaml}")
    if args.disable_runtime_layer_scaling:
        print("[runtime-layer-scaling] disabled")

    if not args.skip_scaling:
        if scaled_dir.exists() and any(scaled_dir.iterdir()):
            if not args.reuse_scaled_dir:
                raise ValueError(
                    f"Scaled dir already exists and is non-empty: {scaled_dir}. "
                    "Pass --reuse-scaled-dir or choose a new --scaled-dir."
                )
        else:
            scaled_dir.mkdir(parents=True, exist_ok=True)
            apply_layer_scale_recipe_to_snapshot(
                source_dir=source_dir,
                output_dir=scaled_dir,
                recipe_path=recipe_yaml,
                strict=True,
                allow_hardlink_unchanged=True,
                allow_symlink_unchanged=True,
                inject_config_metadata=True,
                audit_json_path=args.audit_json,
            )
    elif not scaled_dir.exists():
        raise ValueError(f"--skip-scaling was set, but scaled snapshot path does not exist: {scaled_dir}")

    if not args.disable_runtime_layer_scaling and runtime_layer_scale_yaml is None:
        if not _has_layer_scale_metadata(scaled_dir):
            raise ValueError(
                "Scaled snapshot is missing qeff_layer_scales metadata in config.json. "
                "This usually means the path points to an old/unscaled snapshot. "
                "Regenerate the scaled snapshot, choose a scaled snapshot directory with metadata, "
                "or pass --runtime-layer-scale-yaml explicitly."
            )

    compare_output_json.parent.mkdir(parents=True, exist_ok=True)
    _run_chunked_compare(
        model_path=scaled_dir,
        prompt=args.prompt,
        max_input_tokens=args.max_input_tokens,
        chunk_size=args.chunk_size,
        max_layers=args.max_layers,
        output_json=compare_output_json,
        runtime_layer_scale_yaml=runtime_layer_scale_yaml,
        disable_runtime_layer_scaling=args.disable_runtime_layer_scaling,
    )

    preds = _extract_final_tokens(compare_output_json)
    print(
        "[tokens] "
        f"hf_fp32={preds['hf_fp32']['pred_id']}({preds['hf_fp32']['pred_token']!r}) "
        f"hf_fp16={preds['hf_fp16']['pred_id']}({preds['hf_fp16']['pred_token']!r}) "
        f"qeff_fp16={preds['qeff_fp16']['pred_id']}({preds['qeff_fp16']['pred_token']!r})"
    )
    expected = int(args.expected_token_id)
    all_match = (
        int(preds["hf_fp32"]["pred_id"]) == expected
        and int(preds["hf_fp16"]["pred_id"]) == expected
        and int(preds["qeff_fp16"]["pred_id"]) == expected
    )
    print(f"[expected-token] id={expected} all_paths_match={all_match}")

    if args.strict_token_check and not all_match:
        raise RuntimeError(
            f"Token mismatch against expected id={expected}: "
            f"hf_fp32={preds['hf_fp32']['pred_id']} "
            f"hf_fp16={preds['hf_fp16']['pred_id']} "
            f"qeff_fp16={preds['qeff_fp16']['pred_id']}"
        )


if __name__ == "__main__":
    main()
