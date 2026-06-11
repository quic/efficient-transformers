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

from QEfficient.utils.precision_recovery import (  # noqa: E402
    DEFAULT_ITERATIVE_MLP_CANDIDATES,
    PrecisionRecoveryRequest,
    run_precision_recovery,
    summarize_precision_recovery_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Layerwise fp16->fp32 iterative recovery analysis (MAD + token parity).",
    )
    parser.add_argument("--model-name", required=True, help="Hugging Face model id")
    parser.add_argument("--prompt", default="Tell me about yourself.")
    parser.add_argument("--max-input-tokens", type=int, default=64)
    parser.add_argument("--start-layer", type=int, default=22)
    parser.add_argument("--max-layers", type=int, default=0, help="0 means full model depth")
    parser.add_argument("--boundary-cache-dir", default="scripts/debug/artifacts/qwen3_5_moe_window23_cache")
    parser.add_argument(
        "--continuation-cache-dir",
        default="scripts/debug/artifacts/qwen3_5_moe_full_depth_iterative_cache",
    )
    parser.add_argument("--no-continuation-cache", action="store_true")
    parser.add_argument("--iterative-mlp-candidates", default=DEFAULT_ITERATIVE_MLP_CANDIDATES)
    parser.add_argument(
        "--output-json",
        default="scripts/debug/artifacts/qwen3_5_moe_full_depth_iterative_fp32_recovery_from_api.json",
    )
    parser.add_argument(
        "--summary-json",
        default="scripts/debug/artifacts/qwen3_5_moe_full_depth_iterative_fp32_recovery_from_api_summary.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    request = PrecisionRecoveryRequest(
        model_id=args.model_name,
        prompt=args.prompt,
        max_input_tokens=args.max_input_tokens,
        start_layer=args.start_layer,
        max_layers=args.max_layers,
        boundary_cache_dir=args.boundary_cache_dir,
        continuation_cache_dir=args.continuation_cache_dir,
        no_continuation_cache=args.no_continuation_cache,
        iterative_mlp_candidates=args.iterative_mlp_candidates,
        output_json=args.output_json,
    )

    result = run_precision_recovery(request)
    summary = summarize_precision_recovery_result(result)

    summary_path = Path(args.summary_json).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[write] summary json: {summary_path}")
    final_tokens = summary["final_token_prediction"]
    print(
        "[tokens] "
        f"hf_fp32={final_tokens['hf_fp32']['pred_id']}({final_tokens['hf_fp32']['pred_token']!r}) "
        f"hf_fp16_recovered={final_tokens['hf_fp16_recovered']['pred_id']}({final_tokens['hf_fp16_recovered']['pred_token']!r}) "
        f"qeff_fp16_recovered={final_tokens['qeff_fp16_recovered']['pred_id']}({final_tokens['qeff_fp16_recovered']['pred_token']!r})"
    )
    logits_mad = summary["final_logits_mad"]
    print(
        "[final logits mad] "
        f"hf_fp16_vs_hf_fp32={logits_mad['hf_fp16_vs_hf_fp32']:.6e} "
        f"qeff_fp16_vs_hf_fp32={logits_mad['qeff_fp16_vs_hf_fp32']:.6e} "
        f"qeff_fp16_vs_hf_fp16={logits_mad['qeff_fp16_vs_hf_fp16']:.6e}"
    )
    print(f"[promoted layers] {summary['promoted_layers']}")


if __name__ == "__main__":
    main()
