# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure local workspace imports resolve to this repo without requiring PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from QEfficient.utils.precision_recovery import DEFAULT_ITERATIVE_MLP_CANDIDATES  # noqa: E402
from QEfficient.utils.precision_recovery_agent import (  # noqa: E402
    DEFAULT_SCALE_CANDIDATE_SCHEDULES,
    PrecisionRecoveryAgentRequest,
    parse_scale_candidate_schedules,
    run_precision_recovery_agent,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Agent-driven precision recovery: model card -> fp32/fp16 analysis -> "
            "conditional scale-search -> YAML recipe emission."
        )
    )
    parser.add_argument(
        "--model-card",
        required=True,
        help=(
            "Model card path (.json/.yaml/.md/.txt) or direct Hugging Face model id. "
            "When a card path is provided, model_id/model_name/hf_model_id/repo_id is resolved from it."
        ),
    )
    parser.add_argument("--prompt", default="Tell me about yourself.")
    parser.add_argument("--max-input-tokens", type=int, default=64)
    parser.add_argument("--start-layer", type=int, default=22)
    parser.add_argument("--max-layers", type=int, default=0, help="0 means full model depth")
    parser.add_argument(
        "--boundary-cache-dir",
        default="scripts/debug/artifacts/qwen3_5_moe_window23_cache",
    )
    parser.add_argument(
        "--analysis-continuation-cache-dir",
        default="scripts/debug/artifacts/qwen3_5_moe_full_depth_iterative_cache",
    )
    parser.add_argument(
        "--scale-continuation-cache-dir",
        default="scripts/debug/artifacts/qwen3_5_moe_full_depth_mlp_scale_cache",
    )
    parser.add_argument("--no-continuation-cache", action="store_true")
    parser.add_argument(
        "--iterative-mlp-candidates",
        default=DEFAULT_ITERATIVE_MLP_CANDIDATES,
        help="Comma-separated fp32 promotion candidates for analysis stage",
    )
    parser.add_argument(
        "--scale-candidate-schedules",
        default=";".join(DEFAULT_SCALE_CANDIDATE_SCHEDULES),
        help=(
            "Semicolon-separated list of scale-candidate schedules. Each schedule is a comma-separated list of floats."
        ),
    )
    parser.add_argument("--default-scale", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        default="scripts/debug/artifacts/precision_recovery_agent",
    )
    parser.add_argument("--analysis-output-json", default=None)
    parser.add_argument("--recipe-yaml", default=None)
    parser.add_argument("--report-json", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    schedules = parse_scale_candidate_schedules(args.scale_candidate_schedules)

    request = PrecisionRecoveryAgentRequest(
        model_card=args.model_card,
        prompt=args.prompt,
        max_input_tokens=args.max_input_tokens,
        start_layer=args.start_layer,
        max_layers=args.max_layers,
        boundary_cache_dir=args.boundary_cache_dir,
        analysis_continuation_cache_dir=args.analysis_continuation_cache_dir,
        scale_continuation_cache_dir=args.scale_continuation_cache_dir,
        no_continuation_cache=args.no_continuation_cache,
        iterative_mlp_candidates=args.iterative_mlp_candidates,
        scale_candidate_schedules=schedules,
        default_scale=args.default_scale,
        output_dir=args.output_dir,
        analysis_output_json=args.analysis_output_json,
        recipe_yaml=args.recipe_yaml,
        report_json=args.report_json,
    )

    report = run_precision_recovery_agent(request)
    print(f"[model_id] {report['model_id']}")
    print(f"[analysis_json] {report['analysis']['output_json']}")
    print(f"[scale_search_required] {report['scale_search']['required']}")
    print(f"[scale_search_resolved] {report['scale_search']['resolved']}")
    print(f"[recipe_yaml] {report['recipe_yaml']}")
    print(f"[report_json] {report['report_json']}")

    summary = report["analysis"]["summary"]
    print("[promoted_layers]", json.dumps(summary.get("promoted_layers", []), indent=2))


if __name__ == "__main__":
    main()
