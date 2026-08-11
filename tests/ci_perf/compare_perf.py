#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
compare_perf.py — Compare current PR perf against the persistent baseline DB.

Usage:
    python tests/ci_perf/compare_perf.py \\
        --report-dir  $CI_PERF_REPORT_DIR \\
        --db-path     /ci_perf_db/baseline_db.json \\
        --stage       QAIC_LLM \\
        --hardware-key qeff_node_16 \\
        --thresholds  tests/ci_perf/thresholds.json \\
        [--update-baseline] \\
        [--output-csv /tmp/QAIC_LLM_comparison.csv]

Exit codes:
    0  — all models within tolerance (or no baseline yet)
    1  — at least one metric regressed
    2  — usage / file-not-found error
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from tests.ci_perf.perf_schema import GATED_METRICS, INFO_METRICS
from tests.ci_perf.storage import load_db, load_stage_report, update_baseline

CSV_COLUMNS = ["model_key", "metric", "baseline", "current", "pct_diff", "status", "note"]


# ---------------------------------------------------------------------------
# Threshold helpers
# ---------------------------------------------------------------------------


def load_thresholds(thresholds_path: Path) -> dict[str, Any]:
    if not thresholds_path.exists():
        return {"default": {m: {"percentage_tolerance": 5.0} for m in GATED_METRICS}}
    return json.loads(thresholds_path.read_text(encoding="utf-8"))


def get_threshold(thresholds: dict, model_key: str, metric: str) -> float:
    per_model = thresholds.get("per_model", {}).get(model_key, {})
    metric_override = per_model.get(metric, {})
    if "percentage_tolerance" in metric_override:
        return float(metric_override["percentage_tolerance"])
    default_metric = thresholds.get("default", {}).get(metric, {})
    return float(default_metric.get("percentage_tolerance", 5.0))


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_report(
    report: dict[str, Any],
    baseline_models: dict[str, Any],
    thresholds: dict[str, Any],
) -> tuple[list[dict], list[str]]:
    """
    Compare every model in *report* against *baseline_models*.

    Returns:
        rows     — list of dicts for CSV output.
        failures — list of human-readable regression messages.
    """
    rows: list[dict] = []
    failures: list[str] = []

    for model_key, current in sorted(report.get("models", {}).items()):
        baseline = baseline_models.get(model_key)

        if baseline is None:
            rows.append(
                {
                    "model_key": model_key,
                    "metric": "all",
                    "baseline": "N/A",
                    "current": "N/A",
                    "pct_diff": "N/A",
                    "status": "skipped",
                    "note": "not in baseline DB — first run",
                }
            )
            print(f"  SKIP  {model_key}  (not in baseline DB)")
            continue

        for metric, regression_direction in GATED_METRICS.items():
            curr_val = current.get(metric)
            base_val = baseline.get(metric)

            if curr_val is None or base_val is None:
                rows.append(
                    {
                        "model_key": model_key,
                        "metric": metric,
                        "baseline": base_val,
                        "current": curr_val,
                        "pct_diff": "N/A",
                        "status": "skipped",
                        "note": "null value — inference-only test",
                    }
                )
                continue

            if base_val == 0:
                rows.append(
                    {
                        "model_key": model_key,
                        "metric": metric,
                        "baseline": base_val,
                        "current": curr_val,
                        "pct_diff": "N/A",
                        "status": "skipped",
                        "note": "baseline is zero — cannot compute pct_diff",
                    }
                )
                continue

            pct_diff = (curr_val - base_val) / base_val * 100
            tol = get_threshold(thresholds, model_key, metric)

            is_regression = (regression_direction == "up" and pct_diff > tol) or (
                regression_direction == "down" and pct_diff < -tol
            )

            status = "FAILED" if is_regression else "passed"
            note = f"threshold ±{tol}%" if is_regression else ""

            rows.append(
                {
                    "model_key": model_key,
                    "metric": metric,
                    "baseline": round(base_val, 6),
                    "current": round(curr_val, 6),
                    "pct_diff": f"{pct_diff:+.2f}%",
                    "status": status,
                    "note": note,
                }
            )

            symbol = "FAIL" if is_regression else "pass"
            print(f"  {symbol}  {model_key} :: {metric}: {pct_diff:+.2f}%")

            if is_regression:
                failures.append(
                    f"{model_key} :: {metric}: {pct_diff:+.2f}% (threshold ±{tol}%)"
                )

    return rows, failures


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------


def write_csv(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare CI perf report against persistent baseline DB."
    )
    parser.add_argument("--report-dir", required=True, help="Directory containing per-stage perf_report.json files")
    parser.add_argument("--db-path", required=True, help="Path to baseline_db.json on the CI host")
    parser.add_argument("--stage", required=True, help="Stage name (e.g. QAIC_LLM)")
    parser.add_argument("--hardware-key", required=True, help="Hardware profile key (e.g. qeff_node_16)")
    parser.add_argument("--thresholds", default="tests/ci_perf/thresholds.json", help="Path to thresholds.json")
    parser.add_argument("--update-baseline", action="store_true", help="Update baseline DB with current report values")
    parser.add_argument("--output-csv", default=None, help="Optional path to write comparison CSV")
    args = parser.parse_args(argv)

    report_file = Path(args.report_dir) / args.stage / "perf_report.json"
    if not report_file.exists():
        print(f"ERROR: report file not found: {report_file}", file=sys.stderr)
        return 2

    report = json.loads(report_file.read_text(encoding="utf-8"))
    db_path = Path(args.db_path)
    db = load_db(db_path)
    thresholds = load_thresholds(Path(args.thresholds))

    baseline_models: dict = (
        db.get("hardware_profiles", {})
        .get(args.hardware_key, {})
        .get("stages", {})
        .get(args.stage, {})
        .get("models", {})
    )

    print(f"\n--- Perf Comparison: {args.stage} [{args.hardware_key}] ---")
    rows, failures = compare_report(report, baseline_models, thresholds)

    if args.output_csv:
        write_csv(Path(args.output_csv), rows)
        print(f"\nComparison CSV written to: {args.output_csv}")

    if failures:
        print(f"\n{len(failures)} regression(s) found:")
        for msg in failures:
            print(f"  ✗ {msg}")
    else:
        print("\nAll comparisons passed.")

    if args.update_baseline:
        update_baseline(
            db_path=db_path,
            hw_key=args.hardware_key,
            stage_name=args.stage,
            models_data=report.get("models", {}),
            build_tag=report.get("ci_build_tag", ""),
            git_sha=report.get("git_sha", ""),
        )
        print(f"Baseline DB updated: {db_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
