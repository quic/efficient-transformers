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

# Ensure local workspace imports resolve to this repo without requiring PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from QEfficient.utils.layer_scale_checkpoint import (  # noqa: E402
    build_layer_scale_recipe_from_recovery_json,
    dump_layer_scale_recipe_yaml,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a mathematically-equivalent layer-scale recipe YAML from MLP-scale recovery JSON output."
        )
    )
    parser.add_argument("--recovery-json", required=True, help="Path to recovery JSON (mlp-scale recovery output)")
    parser.add_argument("--output-yaml", required=True, help="Destination layer-scale recipe YAML")
    parser.add_argument("--model-name", default=None, help="Optional model id override")
    parser.add_argument("--default-scale", type=float, default=1.0, help="Default layer scale")
    parser.add_argument(
        "--method",
        default="branch_aware_mlp_scale_equivalent",
        help="Recipe method field for traceability",
    )
    parser.add_argument(
        "--description",
        default=(
            "Auto-generated from recovery JSON. This recipe is valid only for "
            "checkpoint_scaled_mlp_and_residual_branch runtime equivalence mode."
        ),
        help="Recipe description field",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    recipe = build_layer_scale_recipe_from_recovery_json(
        recovery_json_path=args.recovery_json,
        model_id=args.model_name,
        default_scale=args.default_scale,
    )
    output_yaml = dump_layer_scale_recipe_yaml(
        recipe,
        args.output_yaml,
        include_expanded_specs=True,
        extra_fields={
            "Method": args.method,
            "Description": args.description,
            "GeneratedFrom": {
                "recovery_json": str(Path(args.recovery_json).expanduser().resolve()),
            },
        },
    )

    non_unit_layers = sorted(k for k, v in recipe.layer_scales.items() if float(v) != float(recipe.default_scale))
    print(f"[write] recipe yaml: {output_yaml}")
    print(f"[summary] model={recipe.model_id} default_scale={recipe.default_scale} non_unit_layers={non_unit_layers}")


if __name__ == "__main__":
    main()
