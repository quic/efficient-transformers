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

COMMON_COLUMNS = [
    "model_name",
    "status",
    "failure_reason",
    "export_time_before",
    "export_time_after",
    "compile_time_before",
    "compile_time_after",
    "onnx_qpc_size_before",
    "onnx_qpc_size_after",
]

PERF_COLUMNS = [
    "prefill_time_before",
    "prefill_time_after",
    "prefill_time_pct_diff",
    "decode_perf_before",
    "decode_perf_after",
    "decode_perf_pct_diff",
    "total_perf_before",
    "total_perf_after",
    "total_time_before",
    "total_time_after",
]

PERF_VALIDATION_METRICS = {
    "prefill_time_pct_diff",
    "decode_perf_pct_diff",
}

SIZE_UNITS = {
    "B": 1,
    "KB": 1024,
    "MB": 1024**2,
    "GB": 1024**3,
    "TB": 1024**4,
}

FAMILY_SPECS = {
    "audio_embedding_model_configs": {
        "text_column": "transcription",
        "text_key": "transcription",
    },
    "audio_model_configs": {
        "text_column": "transcription",
        "text_key": "transcription",
        "mad_column": "generated_ids",
        "mad_key": "generated_ids",
        "mad_tolerance": "token_mad_tolerance",
        "include_perf": True,
    },
    "causal_pipeline_configs": {
        "text_column": "generated_text",
        "text_key": "generated_texts",
        "mad_column": "generated_ids",
        "mad_key": "generated_ids",
        "mad_tolerance": "token_mad_tolerance",
        "include_perf": True,
    },
    "image_text_to_text_model_configs": {
        "text_column": "generated_text",
        "text_key": "generated_text",
        "mad_column": "generated_ids",
        "mad_key": "generated_ids",
        "mad_tolerance": "token_mad_tolerance",
        "include_perf": True,
    },
    "embedding_model_configs": {
        "mad_column": "embedding",
        "mad_key": "embedding",
        "mad_tolerance": "embedding_mad_tolerance",
    },
    "sequence_model_configs": {
        "text_column": "prediction",
        "text_key": "Prediction",
        "compare_text": False,
        "mad_column": "generated_ids",
        "mad_key": "generated_ids",
        "mad_tolerance": "token_mad_tolerance",
    },
}


@dataclass(frozen=True)
class ValidationTolerances:
    percentage_tolerance: float = 5.0
    perf_delta_tolerance: float = 0.1
    token_mad_tolerance: float = 1e-2
    embedding_mad_tolerance: float = 1e-2


def load_json(filepath: Path) -> dict[str, Any]:
    with filepath.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_validation_tolerances(pipeline_configs: dict[str, Any], model_class: str) -> ValidationTolerances:
    validation_configs = pipeline_configs.get("validation_configs", {})
    default_config = validation_configs.get("default", {})
    model_class_configs = validation_configs.get("model_class_tolerances", {})
    class_config = model_class_configs.get(model_class, {})
    default_percentage_tolerance = default_config.get("percentage_tolerance", 5.0)
    default_perf_delta_tolerance = default_config.get("perf_delta_tolerance", 0.1)
    default_token_mad_tolerance = default_config.get("token_mad_tolerance", 1e-2)
    default_embedding_mad_tolerance = default_config.get("embedding_mad_tolerance", 1e-2)

    return ValidationTolerances(
        percentage_tolerance=float(class_config.get("percentage_tolerance", default_percentage_tolerance)),
        perf_delta_tolerance=float(class_config.get("perf_delta_tolerance", default_perf_delta_tolerance)),
        token_mad_tolerance=float(class_config.get("token_mad_tolerance", default_token_mad_tolerance)),
        embedding_mad_tolerance=float(class_config.get("embedding_mad_tolerance", default_embedding_mad_tolerance)),
    )


def validate_artifact_file(
    current_artifact_file: Path,
    previous_artifact_file: Path | None,
    output_csv_file: Path,
    model_class: str,
    tolerances: ValidationTolerances,
) -> list[dict[str, Any]]:
    previous_artifacts = load_json(previous_artifact_file) if previous_artifact_file is not None else {}
    rows = validate_artifacts(load_json(current_artifact_file), previous_artifacts, model_class, tolerances)
    write_validation_csv(output_csv_file, model_class, rows)
    return rows


def validate_artifacts(
    current_artifacts: dict[str, Any],
    previous_artifacts: dict[str, Any],
    model_class: str,
    tolerances: ValidationTolerances,
) -> list[dict[str, Any]]:
    rows = []
    for model_name, current_payload in sorted(current_artifacts.items()):
        previous_payload = previous_artifacts.get(model_name)
        if previous_payload is None:
            rows.append(_current_only_model_row(model_name, current_payload, model_class))
            continue
        rows.append(_validate_model(model_name, current_payload, previous_payload, model_class, tolerances))
    return rows


def write_validation_csv(output_csv_file: Path, model_class: str, rows: list[dict[str, Any]]) -> None:
    output_csv_file.parent.mkdir(parents=True, exist_ok=True)
    columns = get_csv_columns(model_class)
    with output_csv_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_csv_value(row.get(column, "N/A")) for column in columns})


def get_csv_columns(model_class: str) -> list[str]:
    spec = _get_family_spec(model_class)
    columns = list(COMMON_COLUMNS)

    text_column = spec.get("text_column")
    if text_column:
        columns.extend([f"{text_column}_before", f"{text_column}_after"])
        if spec.get("compare_text", True):
            columns.append(f"{text_column}_assertion")

    mad_column = spec.get("mad_column")
    if mad_column:
        if mad_column == "generated_ids":
            columns.append(f"{mad_column}_mad")
        else:
            columns.extend([f"{mad_column}_before", f"{mad_column}_after", f"{mad_column}_mad"])

    if spec.get("include_perf"):
        columns.extend(PERF_COLUMNS)

    return columns


def all_rows_passed(rows: list[dict[str, Any]]) -> bool:
    return all(row.get("status") == "passed" for row in rows)


def _validate_model(
    model_name: str,
    current_payload: dict[str, Any],
    previous_payload: dict[str, Any],
    model_class: str,
    tolerances: ValidationTolerances,
) -> dict[str, Any]:
    columns = get_csv_columns(model_class)
    spec = _get_family_spec(model_class)
    row = {column: "N/A" for column in columns}
    row["model_name"] = model_name

    _add_percentage_metric(row, "export_time", previous_payload.get("export_time"), current_payload.get("export_time"))
    _add_percentage_metric(
        row, "compile_time", previous_payload.get("compile_time"), current_payload.get("compile_time")
    )
    _add_size_metric(row, previous_payload, current_payload)

    if spec.get("include_perf"):
        _add_perf_metrics(row, previous_payload, current_payload)

    text_assertion_required = "mad_column" not in spec and spec.get("compare_text", True)
    _add_mad_comparison(row, spec, previous_payload, current_payload)

    if spec.get("text_column"):
        _add_text_values(row, spec, previous_payload, current_payload, text_assertion_required)

    failures = _collect_failures(row, spec, tolerances)
    row["status"] = "failed" if failures else "passed"
    row["failure_reason"] = "; ".join(failures) if failures else ""
    return row


def _current_only_model_row(model_name: str, current_payload: dict[str, Any], model_class: str) -> dict[str, Any]:
    spec = _get_family_spec(model_class)
    row = {column: "N/A" for column in get_csv_columns(model_class)}
    row["model_name"] = model_name

    _add_percentage_metric(row, "export_time", None, current_payload.get("export_time"))
    _add_percentage_metric(row, "compile_time", None, current_payload.get("compile_time"))
    _add_size_metric(row, None, current_payload)

    if spec.get("include_perf"):
        _add_perf_metrics(row, {}, current_payload)

    _add_mad_comparison(row, spec, {}, current_payload)

    if spec.get("text_column"):
        _add_text_values(row, spec, {}, current_payload, assertion_required=False)

    row["status"] = "passed"
    row["failure_reason"] = "Previous model artifact not found; comparison skipped."
    return row


def _add_percentage_metric(row: dict[str, Any], column_prefix: str, before: Any, after: Any) -> None:
    before_value = _to_float(before)
    after_value = _to_float(after)

    row[f"{column_prefix}_before"] = before_value if before_value is not None else "N/A"
    row[f"{column_prefix}_after"] = after_value if after_value is not None else "N/A"
    row[f"{column_prefix}_pct_diff"] = _percentage_difference(before_value, after_value)


def _add_value_metric(row: dict[str, Any], column_prefix: str, before: Any, after: Any) -> None:
    before_value = _to_float(before)
    after_value = _to_float(after)

    row[f"{column_prefix}_before"] = before_value if before_value is not None else "N/A"
    row[f"{column_prefix}_after"] = after_value if after_value is not None else "N/A"


def _add_size_metric(
    row: dict[str, Any], previous_payload: dict[str, Any] | None, current_payload: dict[str, Any]
) -> None:
    before_size = _extract_total_size_bytes(previous_payload or {})
    after_size = _extract_total_size_bytes(current_payload)

    row["onnx_qpc_size_before"] = _human_readable_size(before_size) if before_size is not None else "N/A"
    row["onnx_qpc_size_after"] = _human_readable_size(after_size) if after_size is not None else "N/A"
    row["onnx_qpc_size_pct_diff"] = _percentage_difference(before_size, after_size)


def _add_perf_metrics(row: dict[str, Any], previous_payload: dict[str, Any], current_payload: dict[str, Any]) -> None:
    previous_perf = previous_payload.get("perf_metrics", {}) or {}
    current_perf = current_payload.get("perf_metrics", {}) or {}
    _add_percentage_metric(row, "prefill_time", previous_perf.get("prefill_time"), current_perf.get("prefill_time"))
    _add_percentage_metric(row, "decode_perf", previous_perf.get("decode_perf"), current_perf.get("decode_perf"))
    _add_value_metric(row, "total_perf", previous_perf.get("total_perf"), current_perf.get("total_perf"))
    _add_value_metric(row, "total_time", previous_perf.get("total_time"), current_perf.get("total_time"))


def _add_text_values(
    row: dict[str, Any],
    spec: dict[str, Any],
    previous_payload: dict[str, Any],
    current_payload: dict[str, Any],
    assertion_required: bool,
) -> None:
    text_column = spec["text_column"]
    text_key = spec["text_key"]
    previous_text = previous_payload.get(text_key)
    current_text = current_payload.get(text_key)
    row[f"{text_column}_before"] = previous_text if previous_text is not None else "N/A"
    row[f"{text_column}_after"] = current_text if current_text is not None else "N/A"

    assertion_column = f"{text_column}_assertion"
    if assertion_column not in row:
        return
    if not assertion_required:
        row[assertion_column] = "not_applicable"
        return
    row[assertion_column] = "passed" if _values_equal(previous_text, current_text) else "failed"


def _add_mad_comparison(
    row: dict[str, Any],
    spec: dict[str, Any],
    previous_payload: dict[str, Any],
    current_payload: dict[str, Any],
) -> float | str:
    mad_column = spec.get("mad_column")
    if not mad_column:
        return "N/A"

    mad_key = spec["mad_key"]
    previous_value = previous_payload.get(mad_key)
    current_value = current_payload.get(mad_key)
    row[f"{mad_column}_before"] = previous_value if previous_value is not None else "N/A"
    row[f"{mad_column}_after"] = current_value if current_value is not None else "N/A"
    mad_value = _numeric_mad(previous_value, current_value)
    row[f"{mad_column}_mad"] = mad_value
    return mad_value


def _percentage_difference(before: float | None, after: float | None) -> float | str:
    if before is None or after is None or before == 0:
        return "N/A"
    return ((after - before) / before) * 100


def _collect_failures(row: dict[str, Any], spec: dict[str, Any], tolerances: ValidationTolerances) -> list[str]:
    failures = []

    for metric in sorted(PERF_VALIDATION_METRICS):
        _collect_perf_metric_failure(failures, row, metric, tolerances)

    _collect_mad_failures(failures, row, spec, tolerances)
    _collect_assertion_failures(failures, row, spec)
    return failures


def _collect_perf_metric_failure(
    failures: list[str], row: dict[str, Any], metric: str, tolerances: ValidationTolerances
) -> None:
    pct_diff = row.get(metric)
    if not isinstance(pct_diff, (int, float)):
        return

    metric_prefix = metric.removesuffix("_pct_diff")
    before_value = row.get(f"{metric_prefix}_before")
    after_value = row.get(f"{metric_prefix}_after")
    if not isinstance(before_value, (int, float)) or not isinstance(after_value, (int, float)):
        return

    delta = after_value - before_value
    if abs(pct_diff) <= tolerances.percentage_tolerance or abs(delta) < tolerances.perf_delta_tolerance:
        return

    failures.append(
        f"{metric} diff {abs(pct_diff):.2f}% and delta {abs(delta):.6f} exceed "
        f"{tolerances.percentage_tolerance:.2f}%/{tolerances.perf_delta_tolerance:.6f} tolerances"
    )


def _collect_mad_failures(
    failures: list[str], row: dict[str, Any], spec: dict[str, Any], tolerances: ValidationTolerances
) -> None:
    mad_column = spec.get("mad_column")
    if not mad_column:
        return

    mad_value = row.get(f"{mad_column}_mad")
    tolerance_name = spec["mad_tolerance"]
    tolerance_value = getattr(tolerances, tolerance_name)
    if isinstance(mad_value, (int, float)):
        if mad_value > tolerance_value:
            failures.append(f"{mad_column}_mad {mad_value:.6f} exceeds {tolerance_value:.6f} tolerance")
        return

    failures.append(f"{mad_column}_mad is unavailable")


def _collect_assertion_failures(failures: list[str], row: dict[str, Any], spec: dict[str, Any]) -> None:
    text_column = spec.get("text_column")
    if not text_column:
        return

    assertion_value = row.get(f"{text_column}_assertion")
    if assertion_value == "failed":
        failures.append(f"{text_column}_assertion failed")


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


def _human_readable_size(size_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024 or unit == "TB":
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def _numeric_mad(previous_value: Any, current_value: Any) -> float | str:
    previous_flat = _flatten_numeric_values(previous_value)
    current_flat = _flatten_numeric_values(current_value)
    common_length = min(len(previous_flat), len(current_flat))
    if common_length == 0:
        return "N/A"

    total_difference = sum(abs(current_flat[index] - previous_flat[index]) for index in range(common_length))
    return total_difference / common_length


def _flatten_numeric_values(value: Any) -> list[float]:
    flattened = []
    if isinstance(value, bool):
        return flattened
    if isinstance(value, (int, float)):
        if math.isfinite(value):
            flattened.append(float(value))
        return flattened
    if isinstance(value, (list, tuple)):
        for item in value:
            flattened.extend(_flatten_numeric_values(item))
    return flattened


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _values_equal(previous_value: Any, current_value: Any) -> bool:
    if previous_value is None or current_value is None:
        return False
    return _normalize_for_assertion(previous_value) == _normalize_for_assertion(current_value)


def _normalize_for_assertion(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value).strip()


def _format_csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    return value


def _get_family_spec(model_class: str) -> dict[str, Any]:
    if model_class not in FAMILY_SPECS:
        raise KeyError(f"Unknown nightly model class: {model_class}")
    return FAMILY_SPECS[model_class]
