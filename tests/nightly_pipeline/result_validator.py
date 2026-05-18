# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CSV_COLUMNS = [
    "model_name",
    "status",
    "failure_reason",
    "export_time_before",
    "export_time_after",
    "export_time_pct_diff",
    "compile_time_before",
    "compile_time_after",
    "compile_time_pct_diff",
    "onnx_qpc_size_before",
    "onnx_qpc_size_after",
    "onnx_qpc_size_pct_diff",
    "prefill_before",
    "prefill_after",
    "prefill_pct_diff",
    "decode_before",
    "decode_after",
    "decode_pct_diff",
    "total_before",
    "total_after",
    "total_pct_diff",
    "total_time_before",
    "total_time_after",
    "total_time_pct_diff",
    "tokens_mad",
]

LOWER_IS_BETTER_METRICS = {
    "export_time_pct_diff",
    "compile_time_pct_diff",
    "onnx_qpc_size_pct_diff",
    "prefill_pct_diff",
    "total_time_pct_diff",
}

HIGHER_IS_BETTER_METRICS = {
    "decode_pct_diff",
    "total_pct_diff",
}

SIZE_UNITS = {
    "B": 1,
    "KB": 1024,
    "MB": 1024**2,
    "GB": 1024**3,
    "TB": 1024**4,
}


@dataclass(frozen=True)
class ValidationTolerances:
    percentage_tolerance: float = 5.0
    token_mad_tolerance: float = 1e-2


def load_json(filepath: Path) -> dict[str, Any]:
    with filepath.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_validation_tolerances(pipeline_configs: dict[str, Any], model_class: str) -> ValidationTolerances:
    validation_configs = pipeline_configs.get("validation_configs", {})
    default_config = validation_configs.get("default", {})
    model_class_configs = validation_configs.get("model_class_tolerances", {})
    class_config = model_class_configs.get(model_class, {})
    default_percentage_tolerance = default_config.get("percentage_tolerance", 5.0)
    default_token_mad_tolerance = default_config.get("token_mad_tolerance", 1e-2)

    return ValidationTolerances(
        percentage_tolerance=float(class_config.get("percentage_tolerance", default_percentage_tolerance)),
        token_mad_tolerance=float(class_config.get("token_mad_tolerance", default_token_mad_tolerance)),
    )


def validate_artifact_file(
    current_artifact_file: Path,
    previous_artifact_file: Path,
    output_csv_file: Path,
    tolerances: ValidationTolerances,
) -> list[dict[str, Any]]:
    rows = validate_artifacts(load_json(current_artifact_file), load_json(previous_artifact_file), tolerances)
    write_validation_csv(output_csv_file, rows)
    return rows


def validate_artifacts(
    current_artifacts: dict[str, Any],
    previous_artifacts: dict[str, Any],
    tolerances: ValidationTolerances,
) -> list[dict[str, Any]]:
    rows = []
    for model_name, current_payload in sorted(current_artifacts.items()):
        previous_payload = previous_artifacts.get(model_name)
        if previous_payload is None:
            rows.append(_missing_previous_model_row(model_name))
            continue
        rows.append(_validate_model(model_name, current_payload, previous_payload, tolerances))
    return rows


def write_validation_csv(output_csv_file: Path, rows: list[dict[str, Any]]) -> None:
    output_csv_file.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_csv_value(row.get(column, "N/A")) for column in CSV_COLUMNS})


def all_rows_passed(rows: list[dict[str, Any]]) -> bool:
    return all(row.get("status") == "passed" for row in rows)


def _validate_model(
    model_name: str,
    current_payload: dict[str, Any],
    previous_payload: dict[str, Any],
    tolerances: ValidationTolerances,
) -> dict[str, Any]:
    row = {column: "N/A" for column in CSV_COLUMNS}
    row["model_name"] = model_name

    _add_percentage_metric(row, "export_time", previous_payload.get("export_time"), current_payload.get("export_time"))
    _add_percentage_metric(
        row, "compile_time", previous_payload.get("compile_time"), current_payload.get("compile_time")
    )
    _add_percentage_metric(
        row, "onnx_qpc_size", _extract_total_size_bytes(previous_payload), _extract_total_size_bytes(current_payload)
    )

    previous_perf = previous_payload.get("perf_metrics", {}) or {}
    current_perf = current_payload.get("perf_metrics", {}) or {}
    _add_percentage_metric(row, "prefill", previous_perf.get("prefill_time"), current_perf.get("prefill_time"))
    _add_percentage_metric(row, "decode", previous_perf.get("decode_perf"), current_perf.get("decode_perf"))
    _add_percentage_metric(row, "total", previous_perf.get("total_perf"), current_perf.get("total_perf"))
    _add_percentage_metric(row, "total_time", previous_perf.get("total_time"), current_perf.get("total_time"))

    row["tokens_mad"] = _tokens_mad(previous_payload.get("generated_ids"), current_payload.get("generated_ids"))

    failures = _collect_failures(row, tolerances)
    row["status"] = "failed" if failures else "passed"
    row["failure_reason"] = "; ".join(failures) if failures else ""
    return row


def _missing_previous_model_row(model_name: str) -> dict[str, Any]:
    row = {column: "N/A" for column in CSV_COLUMNS}
    row.update(
        {
            "model_name": model_name,
            "status": "failed",
            "failure_reason": "Model not found in previous nightly results.",
        }
    )
    return row


def _add_percentage_metric(row: dict[str, Any], column_prefix: str, before: Any, after: Any) -> None:
    before_value = _to_float(before)
    after_value = _to_float(after)

    row[f"{column_prefix}_before"] = before_value if before_value is not None else "N/A"
    row[f"{column_prefix}_after"] = after_value if after_value is not None else "N/A"
    row[f"{column_prefix}_pct_diff"] = _percentage_difference(before_value, after_value)


def _percentage_difference(before: float | None, after: float | None) -> float | str:
    if before is None or after is None or before == 0:
        return "N/A"
    return ((after - before) / before) * 100


def _collect_failures(row: dict[str, Any], tolerances: ValidationTolerances) -> list[str]:
    failures = []
    percentage_tolerance = tolerances.percentage_tolerance

    for metric in sorted(LOWER_IS_BETTER_METRICS):
        pct_diff = row.get(metric)
        if isinstance(pct_diff, (int, float)) and pct_diff > percentage_tolerance:
            failures.append(f"{metric} regression {pct_diff:.2f}% exceeds {percentage_tolerance:.2f}% tolerance")

    for metric in sorted(HIGHER_IS_BETTER_METRICS):
        pct_diff = row.get(metric)
        if isinstance(pct_diff, (int, float)) and pct_diff < -percentage_tolerance:
            failures.append(f"{metric} regression {pct_diff:.2f}% exceeds {percentage_tolerance:.2f}% tolerance")

    tokens_mad = row.get("tokens_mad")
    if isinstance(tokens_mad, (int, float)) and tokens_mad > tolerances.token_mad_tolerance:
        failures.append(f"tokens_mad {tokens_mad:.6f} exceeds {tolerances.token_mad_tolerance:.6f} tolerance")

    return failures


def _extract_total_size_bytes(payload: dict[str, Any]) -> float | None:
    sizes = []
    for key, value in payload.items():
        if not _is_artifact_size_key(key):
            continue
        parsed_size = _parse_size_bytes(value)
        if parsed_size is not None:
            sizes.append(parsed_size)
    if not sizes:
        return None
    return float(sum(sizes))


def _is_artifact_size_key(key: str) -> bool:
    key_lower = key.lower()
    return key_lower == "size" or ("size" in key_lower and ("onnx" in key_lower or "qpc" in key_lower))


def _parse_size_bytes(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str):
        return None

    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?B)\s*", value, flags=re.IGNORECASE)
    if not match:
        return None

    amount = float(match.group(1))
    unit = match.group(2).upper()
    return amount * SIZE_UNITS[unit]


def _tokens_mad(previous_tokens: Any, current_tokens: Any) -> float | str:
    previous_flat = _flatten_numeric_tokens(previous_tokens)
    current_flat = _flatten_numeric_tokens(current_tokens)
    common_length = min(len(previous_flat), len(current_flat))
    if common_length == 0:
        return "N/A"

    total_difference = sum(abs(current_flat[index] - previous_flat[index]) for index in range(common_length))
    return total_difference / common_length


def _flatten_numeric_tokens(tokens: Any) -> list[float]:
    flattened = []
    if isinstance(tokens, bool):
        return flattened
    if isinstance(tokens, (int, float)):
        if math.isfinite(tokens):
            flattened.append(float(tokens))
        return flattened
    if isinstance(tokens, (list, tuple)):
        for item in tokens:
            flattened.extend(_flatten_numeric_tokens(item))
    return flattened


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _format_csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    return value
