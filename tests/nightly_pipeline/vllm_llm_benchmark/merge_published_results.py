#!/usr/bin/env python3
"""
Generate a consolidated published CSV from all benchmark result CSVs.

This script scans a results directory for all *_results.csv files
(from different benchmark categories: default, ccl, blocking, disagg_pd, embedding, audio, vlm)
and generates a single consolidated published CSV with key fields for team distribution.

Usage:
    python3 merge_published_results.py --results-dir resultdir/ --output consolidated_published_results.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

PUBLISHED_FIELDS = [
    "model",
    "model_category",
    "config_name",
    "config_summary",
    "status",
    "export_compile_time_s",
    "prefill_mdp_export_compile_time_s",
    "prefill_export_compile_time_s",
    "decode_export_compile_time_s",
    "encode_export_compile_time_s",
    "mean_ttft_s",
    "mean_tpot_s",
    "mean_itl_s",
    "decode_TPS",
    "request_throughput_req_s",
    "vllm_qaic_branch",
    "qaic_disagg_branch",
    "qserve_branch",
    "qeff_branch",
    "qaic_sdk_version",
    "server_command",
    "client_command",
]


def _model_category_from_filename(filename: str) -> str | None:
    """Infer model category from a *_results.csv filename, e.g. 'vlm_results.csv' -> 'VLM'."""
    name = filename.lower()
    if "embedding" in name:
        return "Embedding"
    if "audio" in name:
        return "Audio"
    if "vlm" in name:
        return "VLM"
    return None


def _assign_model_category(row: dict, source_filename: str) -> None:
    pooling_method = row.get("pooling_method", "").lower()
    if pooling_method in ("mean", "avg", "cls", "max"):
        row["model_category"] = "Embedding"
        return
    row["model_category"] = _model_category_from_filename(source_filename) or "LLM"


def generate_published_csv(input_csv: Path, output_csv: Path) -> None:
    """Generate a simplified published CSV with only key fields for team distribution."""
    if not input_csv.exists():
        return
    with input_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return

    # Convert milliseconds to seconds for latency metrics and add model_category
    for row in rows:
        if row.get("mean_TTFT_ms"):
            row["mean_ttft_s"] = str(round(float(row["mean_TTFT_ms"]) / 1000, 4))
        if row.get("mean_TPOT_ms"):
            row["mean_tpot_s"] = str(round(float(row["mean_TPOT_ms"]) / 1000, 4))
        if row.get("mean_ITL_ms"):
            row["mean_itl_s"] = str(round(float(row["mean_ITL_ms"]) / 1000, 4))

        # Concatenate config_summary and mode_type
        config_summary = row.get("config_summary", "").strip()
        mode_type = row.get("mode_type", "").strip()
        if mode_type:
            row["config_summary"] = f"{config_summary} | {mode_type}" if config_summary else mode_type

        # Determine model category from the source filename (config_name alone doesn't
        # identify VLM/audio rows, since their effective config_name never contains those
        # words) and pooling_method
        _assign_model_category(row, input_csv.name)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PUBLISHED_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def merge_results_to_published_csv(results_dir: Path, output_csv: Path) -> int:
    """Merge all result CSVs in results_dir into a single consolidated published CSV."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"Error: results directory does not exist: {results_dir}")
        return 1

    all_rows = []
    result_csvs = sorted(results_dir.glob("*_results.csv"))
    # Exclude published CSVs if they exist
    result_csvs = [f for f in result_csvs if "_published" not in f.name]

    if not result_csvs:
        print(f"Warning: no result CSV files found in {results_dir}")
        return 1

    for result_csv in result_csvs:
        if not result_csv.exists():
            continue
        print(f"  Reading: {result_csv.name}")
        with result_csv.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            for row in rows:
                row["_source_filename"] = result_csv.name
            all_rows.extend(rows)
            print(f"    -> {len(rows)} rows")

    if not all_rows:
        print("Error: no data found in result CSV files")
        return 1

    # Convert milliseconds to seconds for latency metrics and add model_category
    for row in all_rows:
        if row.get("mean_TTFT_ms"):
            row["mean_ttft_s"] = str(round(float(row["mean_TTFT_ms"]) / 1000, 4))
        if row.get("mean_TPOT_ms"):
            row["mean_tpot_s"] = str(round(float(row["mean_TPOT_ms"]) / 1000, 4))
        if row.get("mean_ITL_ms"):
            row["mean_itl_s"] = str(round(float(row["mean_ITL_ms"]) / 1000, 4))

        # Concatenate config_summary and mode_type
        config_summary = row.get("config_summary", "").strip()
        mode_type = row.get("mode_type", "").strip()
        if mode_type:
            row["config_summary"] = f"{config_summary} | {mode_type}" if config_summary else mode_type

        # Determine model category from the source filename (config_name alone doesn't
        # identify VLM/audio rows, since their effective config_name never contains those
        # words) and pooling_method
        _assign_model_category(row, row.pop("_source_filename"))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PUBLISHED_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print()
    print(f"✓ Merged {len(result_csvs)} result CSV files")
    print(f"✓ Total rows: {len(all_rows)}")
    print(f"✓ Output: {output_csv}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a consolidated published CSV from all benchmark result CSVs."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing *_results.csv files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output consolidated published CSV path",
    )
    args = parser.parse_args()

    return merge_results_to_published_csv(Path(args.results_dir), Path(args.output))


if __name__ == "__main__":
    raise SystemExit(main())
