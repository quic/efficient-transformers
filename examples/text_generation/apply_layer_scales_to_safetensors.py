# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

# Ensure local workspace imports resolve to this repo without requiring PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from QEfficient.utils.layer_scale_checkpoint import apply_layer_scale_recipe_to_snapshot  # noqa: E402


def _default_recipe_path() -> str:
    return str(
        Path(__file__).resolve().parents[2]
        / "QEfficient"
        / "transformers"
        / "models"
        / "qwen3_5_moe"
        / "configs"
        / "layer_scales_qwen3_5_397b_mlp_equivalent.yaml"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply layer-scale recipe to safetensor snapshots and emit a scaled checkpoint clone.",
    )
    parser.add_argument("--output-dir", required=True, help="Destination directory for scaled snapshot")
    parser.add_argument(
        "--recipe-yaml",
        default=_default_recipe_path(),
        help="Layer scale recipe YAML",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Local source snapshot directory (if omitted, --model-name is used)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Hugging Face model id to resolve snapshot from local cache when --source-dir is not passed",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow downloading from Hugging Face Hub when model snapshot is not present in local cache.",
    )
    parser.add_argument(
        "--allow-missing-recipe-keys",
        action="store_true",
        help="Do not fail if some recipe keys are missing from checkpoint weight_map",
    )
    parser.add_argument(
        "--copy-unchanged",
        action="store_true",
        help="Copy unchanged files instead of creating hardlinks",
    )
    parser.add_argument(
        "--no-config-metadata",
        action="store_true",
        help="Do not inject qeff layer-scale metadata into output config.json",
    )
    parser.add_argument(
        "--audit-json",
        default=None,
        help="Optional custom audit JSON path (default: <output-dir>/qeff_layer_scale_audit.json)",
    )
    return parser


def _resolve_source_dir(source_dir: str | None, model_name: str | None, allow_download: bool) -> Path:
    if source_dir:
        return Path(source_dir).expanduser().resolve()
    if not model_name:
        raise ValueError("Pass either --source-dir or --model-name")
    try:
        snapshot_path = snapshot_download(
            repo_id=model_name,
            allow_patterns=["*.safetensors", "*.json"],
            local_files_only=not allow_download,
        )
    except LocalEntryNotFoundError as exc:
        raise ValueError(
            f"Model snapshot for {model_name!r} not found in local cache. "
            "Pass --source-dir or rerun with --allow-download."
        ) from exc
    return Path(snapshot_path).resolve()


def main() -> None:
    args = _build_parser().parse_args()
    source_dir = _resolve_source_dir(args.source_dir, args.model_name, args.allow_download)
    output_dir = Path(args.output_dir).expanduser().resolve()
    recipe_yaml = Path(args.recipe_yaml).expanduser().resolve()

    print(f"[source] {source_dir}")
    print(f"[output] {output_dir}")
    print(f"[recipe] {recipe_yaml}")

    audit = apply_layer_scale_recipe_to_snapshot(
        source_dir=source_dir,
        output_dir=output_dir,
        recipe_path=recipe_yaml,
        strict=not args.allow_missing_recipe_keys,
        allow_hardlink_unchanged=not args.copy_unchanged,
        allow_symlink_unchanged=not args.copy_unchanged,
        inject_config_metadata=not args.no_config_metadata,
        audit_json_path=args.audit_json,
    )

    print(
        "[summary] "
        f"layers={audit['touched_layers']} patched_shards={len(audit['patched_shards'])} "
        f"scaled_tensors={audit['scaled_tensor_count']}"
    )


if __name__ == "__main__":
    main()
