#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Generate an email-safe HTML report for nightly pipeline validation results."""

import argparse
import csv
import datetime as dt
import html
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from .model_age_utils import MODEL_AGE_UNKNOWN
except ImportError:  # pragma: no cover - supports direct script execution in Jenkins.
    from model_age_utils import MODEL_AGE_UNKNOWN

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML is expected in Jenkins, optional for local report generation.
    yaml = None


MODEL_CLASS_LABELS = {
    "audio_embedding_model": "Audio Embedding Models",
    "audio_model": "Audio Models",
    "causal_model": "Causal LM Models",
    "embedding_model": "Embedding Models",
    "image_text_to_text_model": "Image Text-to-Text Models",
    "sequence_model": "Sequence Models",
}

VALIDATION_FILE_ORDER = [
    "causal_model_validation.csv",
    "image_text_to_text_model_validation.csv",
    "embedding_model_validation.csv",
    "audio_model_validation.csv",
    "audio_embedding_model_validation.csv",
    "sequence_model_validation.csv",
]

PIPELINE_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "pipeline_configs.json"

PIPELINE_CONFIG_ORDER = [
    ("causal_model", "causal_pipeline_configs"),
    ("image_text_to_text_model", "image_text_to_text_model_configs"),
    ("embedding_model", "embedding_model_configs"),
    ("audio_model", "audio_model_configs"),
    ("audio_embedding_model", "audio_embedding_model_configs"),
    ("sequence_model", "sequence_model_configs"),
]

REPORT_TITLE = "QEFF Nightly Report"
DEFAULT_REPO_URL = "https://github.com/quic/efficient-transformers"
STATUS_COLUMN_HEADER = "Status (output MAD, performance, and onnx/qpc size within configured tolerances)"

QAIC_VERSION_UTIL = "/opt/qti-aic/tools/qaic-version-util"
QAIC_APPS_XML = "/opt/qti-aic/versions/apps.xml"
QAIC_PLATFORM_XML = "/opt/qti-aic/versions/platform.xml"
QNN_SDK_ENV_VAR = "QNN_SDK_ROOT"
QNN_SDK_YAML = "sdk.yaml"
NIGHTLY_TEST_FAILURES_FILE = "nightly_test_failures.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", required=True, type=Path, help="Directory containing validation CSV files.")
    parser.add_argument("--output-html", required=True, type=Path, help="Path to write the HTML email body.")
    parser.add_argument("--output-json", type=Path, help="Optional path to write a machine-readable summary JSON.")
    parser.add_argument("--log-file", type=Path, help="Optional Jenkins console log file for metadata fallback.")
    parser.add_argument("--environment-json", type=Path, help="Optional environment metadata JSON captured by Jenkins.")
    parser.add_argument(
        "--output-environment-json", type=Path, help="Optional path to write runtime/SDK metadata JSON."
    )
    parser.add_argument("--build-start-epoch", type=float, help="Optional build start epoch seconds.")
    parser.add_argument("--build-end-epoch", type=float, help="Optional build end epoch seconds.")
    parser.add_argument("--build-status", help="Optional final build status override.")
    return parser.parse_args()


def read_text(path: Optional[Path]) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def load_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        return {"metadata_error": f"Unable to load {path}: {exc}"}


def run_command(command: List[str]) -> str:
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=10).strip()
    except Exception:
        return "N/A"


def extract_qaic_sdk_version(xml_path: str) -> str:
    if not os.path.exists(xml_path):
        return "N/A"
    try:
        root = ET.parse(xml_path).getroot()
        base_version = root.find(".//base_version")
        if base_version is not None and base_version.text:
            return base_version.text.strip()
    except Exception:
        return "N/A"
    return "N/A"


def extract_qaic_versions_from_util() -> Dict[str, str]:
    versions = {
        "qaic_platform_version": "N/A",
        "qaic_apps_version": "N/A",
        "qaic_factory_version": "N/A",
        "qaic_sdk_source": QAIC_VERSION_UTIL,
    }
    if not os.path.exists(QAIC_VERSION_UTIL):
        versions["qaic_sdk_source"] = "XML fallback"
        return versions

    try:
        output = subprocess.check_output([QAIC_VERSION_UTIL], stderr=subprocess.STDOUT, text=True, timeout=10)
    except Exception as exc:
        versions["qaic_sdk_source"] = f"XML fallback; {QAIC_VERSION_UTIL} failed: {exc}"
        return versions

    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip() or "N/A"
        if key == "platform":
            versions["qaic_platform_version"] = value
        elif key == "apps":
            versions["qaic_apps_version"] = value
        elif key == "factory":
            versions["qaic_factory_version"] = value

    return versions


def load_qnn_sdk_details() -> Dict[str, Any]:
    qnn_sdk_root = os.environ.get(QNN_SDK_ENV_VAR)
    if not qnn_sdk_root:
        return {"qnn_sdk_root": "N/A", "qnn_sdk_details": "N/A"}

    sdk_yaml_path = Path(qnn_sdk_root) / QNN_SDK_YAML
    details: Any = "N/A"
    if sdk_yaml_path.exists():
        try:
            if yaml is not None:
                with sdk_yaml_path.open("r", encoding="utf-8") as file:
                    details = yaml.safe_load(file) or {}
            else:
                details = sdk_yaml_path.read_text(encoding="utf-8", errors="replace")[:1000]
        except Exception as exc:
            details = f"Unable to parse {sdk_yaml_path}: {exc}"

    return {"qnn_sdk_root": qnn_sdk_root, "qnn_sdk_details": details}


def collect_runtime_environment() -> Dict[str, Any]:
    qnn_details = load_qnn_sdk_details()
    qaic_versions = extract_qaic_versions_from_util()
    if qaic_versions["qaic_apps_version"] == "N/A":
        qaic_versions["qaic_apps_version"] = extract_qaic_sdk_version(QAIC_APPS_XML)
    if qaic_versions["qaic_platform_version"] == "N/A":
        qaic_versions["qaic_platform_version"] = extract_qaic_sdk_version(QAIC_PLATFORM_XML)

    return {
        **qaic_versions,
        "qnn_sdk_root": qnn_details.get("qnn_sdk_root", "N/A"),
        "qnn_sdk_details": qnn_details.get("qnn_sdk_details", "N/A"),
        "python_version": sys.version.split()[0],
        "qefficient_version": run_command(
            [sys.executable, "-c", "import importlib.metadata as m; print(m.version('QEfficient'))"]
        ),
        "torch_version": run_command([sys.executable, "-c", "import torch; print(torch.__version__)"]),
        "transformers_version": run_command(
            [sys.executable, "-c", "import transformers; print(transformers.__version__)"]
        ),
    }


def csv_class_key(path: Path) -> str:
    suffix = "_validation.csv"
    name = path.name
    return name[: -len(suffix)] if name.endswith(suffix) else path.stem


def ordered_validation_files(artifacts_dir: Path) -> List[Path]:
    known = [artifacts_dir / filename for filename in VALIDATION_FILE_ORDER if (artifacts_dir / filename).exists()]
    known_names = {path.name for path in known}
    extra = sorted(path for path in artifacts_dir.glob("*_validation.csv") if path.name not in known_names)
    return known + extra


def load_validation_rows(artifacts_dir: Path) -> Dict[str, List[Dict[str, str]]]:
    class_rows: Dict[str, List[Dict[str, str]]] = {}
    for path in ordered_validation_files(artifacts_dir):
        with path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        class_rows[csv_class_key(path)] = rows
    merge_test_failure_rows(class_rows, artifacts_dir)
    return class_rows


def merge_test_failure_rows(class_rows: Dict[str, List[Dict[str, str]]], artifacts_dir: Path) -> None:
    failures = load_json(artifacts_dir / NIGHTLY_TEST_FAILURES_FILE)
    if not failures:
        return

    existing_models = {
        (class_key, row.get("model_name"))
        for class_key, rows in class_rows.items()
        for row in rows
        if row.get("model_name")
    }
    for row in failures.values():
        if not isinstance(row, dict):
            continue
        class_key = str(row.get("model_class") or "")
        model_name = str(row.get("model_name") or "")
        if not class_key or not model_name or (class_key, model_name) in existing_models:
            continue
        class_rows.setdefault(class_key, []).append(
            {
                "model_name": model_name,
                "model_age": str(row.get("model_age") or MODEL_AGE_UNKNOWN),
                "status": str(row.get("status") or "warning"),
                "failure_reason": str(row.get("failure_reason") or "pytest test failed"),
            }
        )


def is_current_only(row: Dict[str, str]) -> bool:
    reason = (row.get("failure_reason") or "").lower()
    if "previous model artifact not found" in reason or "comparison skipped" in reason:
        return True
    before_fields = [value for key, value in row.items() if key.endswith("_before")]
    return bool(before_fields) and all(str(value or "").strip().upper() in {"N/A", "NA", ""} for value in before_fields)


def summarize_rows(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    total = len(rows)
    passed = sum(1 for row in rows if (row.get("status") or "").lower() == "passed")
    warning = sum(1 for row in rows if (row.get("status") or "").lower() == "warning")
    failed = sum(1 for row in rows if (row.get("status") or "").lower() == "failed")
    current_only = sum(1 for row in rows if is_current_only(row))
    pass_rate = ((passed + warning) / total * 100.0) if total else 0.0
    return {
        "total": total,
        "passed": passed,
        "warning": warning,
        "failed": failed,
        "current_only": current_only,
        "pass_rate": pass_rate,
    }


def extract_first(patterns: Iterable[str], text: str) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            return match.group(1).strip()
    return "N/A"


def env_or_na(name: str) -> str:
    value = os.environ.get(name)
    return value if value else "N/A"


def format_epoch(epoch: Optional[float]) -> str:
    if epoch is None:
        return "N/A"
    try:
        return dt.datetime.fromtimestamp(epoch).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return "N/A"


def format_duration(start_epoch: Optional[float], end_epoch: Optional[float]) -> str:
    if start_epoch is None or end_epoch is None or end_epoch < start_epoch:
        return "N/A"
    total_seconds = int(end_epoch - start_epoch)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def parse_optional_epoch(value: Optional[str]) -> Optional[float]:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def derive_build_metadata(
    artifacts_dir: Path,
    log_text: str,
    environment: Dict[str, Any],
    start_epoch: Optional[float],
    end_epoch: Optional[float],
    build_status: str,
) -> Dict[str, Any]:
    branch_from_checkout = extract_first([r"Checking out Revision [0-9a-f]+ \(([^)]+)\)"], log_text)
    branch = env_or_na("BRANCH_NAME")
    if branch == "N/A" and branch_from_checkout != "N/A":
        branch = branch_from_checkout

    pr_number = env_or_na("PR_NUMBER")
    if pr_number == "N/A":
        pr_number = extract_first([r"Checking out PR #?(\d+)", r"Checking out PR (\d+)", r"origin/pr-(\d+)"], log_text)

    commit = env_or_na("GIT_COMMIT")
    if commit == "N/A":
        commit = extract_first(
            [
                r"Checking out Revision ([0-9a-f]{7,40})",
                r"Checked out commit:\s*\n([0-9a-f]{7,40})",
                r"git checkout -f ([0-9a-f]{7,40})",
            ],
            log_text,
        )

    return {
        "status": build_status,
        "job_name": env_or_na("JOB_NAME"),
        "build_number": env_or_na("BUILD_NUMBER"),
        "build_tag": env_or_na("BUILD_TAG"),
        "build_url": env_or_na("BUILD_URL"),
        "node_name": os.environ.get("NODE_NAME") or extract_first([r"Building remotely on\s+([^\n]+?)\s+\("], log_text),
        "branch": branch,
        "pr_number": pr_number,
        "commit_id": commit,
        "commit_message": extract_first([r"Commit message:\s*\"([^\"]+)\"", r"^[0-9a-f]{7,40}\s+(.+)$"], log_text),
        "trigger": extract_first([r"^(Started by .+)$"], log_text),
        "repo_url": os.environ.get("GIT_URL", DEFAULT_REPO_URL),
        "docker_image": os.environ.get("DOCKER_LATEST", environment.get("docker_image", "N/A")),
        "artifacts_dir": str(artifacts_dir),
        "previous_artifacts_dir": env_or_na("NIGHTLY_PIPELINE_PREVIOUS_ARTIFACTS_DIR"),
        "start_time": format_epoch(start_epoch),
        "end_time": format_epoch(end_epoch),
        "total_duration": format_duration(start_epoch, end_epoch),
    }


def short_value(value: Any, max_len: int = 180) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, sort_keys=True)
    text = str(value)
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def html_escape(value: Any) -> str:
    return html.escape(short_value(value), quote=True)


def normalize_repo_url(repo_url: Any) -> str:
    repo_url = short_value(repo_url).strip()
    if not repo_url or repo_url == "N/A":
        return DEFAULT_REPO_URL
    if repo_url.startswith("git@github.com:"):
        repo_url = "https://github.com/" + repo_url.split(":", 1)[1]
    if repo_url.endswith(".git"):
        repo_url = repo_url[:-4]
    return repo_url if repo_url.startswith(("http://", "https://")) else DEFAULT_REPO_URL


def html_link(url: Any, label: Any) -> str:
    normalized_url = normalize_repo_url(url)
    return (
        f'<a href="{html_escape(normalized_url)}" '
        'style="color:#2563eb;text-decoration:none;font-weight:bold;">'
        f"{html_escape(label)}</a>"
    )


def bold_text(value: Any) -> str:
    return f'<strong style="font-weight:bold;color:#0f172a;">{html_escape(value)}</strong>'


def status_badge(status: str) -> str:
    normalized = (status or "unknown").lower()
    display_status = normalized.replace("_", " ")
    color = "#6b7280"
    if normalized == "passed":
        color = "#15803d"
    elif normalized == "failed":
        color = "#b91c1c"
    elif normalized in {"partial", "unstable", "warning", "passed_with_warnings"}:
        color = "#a16207"

    return (
        '<table role="presentation" cellpadding="0" cellspacing="0" border="0" '
        'style="border-collapse:collapse;mso-table-lspace:0pt;mso-table-rspace:0pt;display:inline-table;">'
        f'<tr><td bgcolor="{color}" style="background-color:{color};color:#ffffff;font-family:Arial,Helvetica,sans-serif;'
        'font-size:12px;font-weight:bold;line-height:16px;padding:3px 8px;text-align:center;white-space:nowrap;">'
        f"{html_escape(display_status.upper())}</td></tr></table>"
    )


def table(headers: List[str], rows: List[List[Any]], row_classes: Optional[List[str]] = None) -> str:
    table_style = (
        "border-collapse:collapse;mso-table-lspace:0pt;mso-table-rspace:0pt;"
        "width:100%;font-family:Arial,Helvetica,sans-serif;font-size:13px;line-height:18px;"
    )
    th_style = (
        "background-color:#e2e8f0;color:#0f172a;font-family:Arial,Helvetica,sans-serif;"
        "font-size:13px;font-weight:bold;line-height:18px;text-align:left;padding:8px;"
        "border:1px solid #cbd5e1;vertical-align:top;"
    )
    td_style = (
        "color:#111827;font-family:Arial,Helvetica,sans-serif;font-size:13px;line-height:18px;"
        "padding:8px;border:1px solid #e5e7eb;vertical-align:top;word-wrap:break-word;"
    )

    header_html = "".join(
        f'<th bgcolor="#e2e8f0" style="{th_style}" scope="col">{html_escape(header)}</th>' for header in headers
    )
    body_parts = []
    for index, row in enumerate(rows):
        row_class = row_classes[index] if row_classes and index < len(row_classes) else ""
        background = "#ffffff"
        if row_class == "failed":
            background = "#fef2f2"
        elif row_class == "passed":
            background = "#f0fdf4"
        elif row_class == "warning":
            background = "#fffbeb"

        cells = "".join(
            f'<td bgcolor="{background}" style="background-color:{background};{td_style}">{cell}</td>' for cell in row
        )
        body_parts.append(f"<tr>{cells}</tr>")

    return (
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="{table_style}">'
        f"<thead><tr>{header_html}</tr></thead><tbody>{''.join(body_parts)}</tbody></table>"
    )


def compact_key_value_rows(rows: List[List[Any]]) -> List[List[Any]]:
    compact_rows = []
    for index in range(0, len(rows), 2):
        left = rows[index]
        right = rows[index + 1] if index + 1 < len(rows) else ["", ""]
        compact_rows.append([left[0], left[1], right[0], right[1]])
    return compact_rows


def section(title: str, body: str) -> str:
    return (
        '<tr><td style="padding:20px 24px;border-top:1px solid #e5e7eb;">'
        f'<h2 style="margin:0 0 12px 0;color:#0f172a;font-family:Arial,Helvetica,sans-serif;font-size:20px;'
        f'line-height:24px;font-weight:bold;">{html_escape(title)}</h2>{body}</td></tr>'
    )


def subsection(title: str, body: str) -> str:
    return (
        f'<h3 style="margin:18px 0 8px 0;color:#1e293b;font-family:Arial,Helvetica,sans-serif;font-size:16px;'
        f'line-height:22px;font-weight:bold;">{html_escape(title)}</h3>{body}'
    )


def metric_card(value: Any, label: str, color: str = "#111827") -> str:
    return (
        '<td width="25%" valign="top" style="padding:4px;">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        'style="border-collapse:collapse;mso-table-lspace:0pt;mso-table-rspace:0pt;">'
        '<tr><td bgcolor="#f8fafc" style="background-color:#f8fafc;border:1px solid #dbeafe;padding:14px;'
        'text-align:center;font-family:Arial,Helvetica,sans-serif;color:#111827;">'
        f'<div style="font-size:26px;line-height:32px;font-weight:bold;color:{color};">{html_escape(value)}</div>'
        f'<div style="font-size:13px;line-height:18px;color:#334155;">{html_escape(label)}</div>'
        "</td></tr></table></td>"
    )


def build_summary(class_rows: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    class_summaries = {class_key: summarize_rows(rows) for class_key, rows in class_rows.items()}
    total = sum(summary["total"] for summary in class_summaries.values())
    passed = sum(summary["passed"] for summary in class_summaries.values())
    warning = sum(summary["warning"] for summary in class_summaries.values())
    failed = sum(summary["failed"] for summary in class_summaries.values())
    current_only = sum(summary["current_only"] for summary in class_summaries.values())
    return {
        "total": total,
        "passed": passed,
        "warning": warning,
        "failed": failed,
        "current_only": current_only,
        "pass_rate": ((passed + warning) / total * 100.0) if total else 0.0,
        "model_classes": class_summaries,
    }


def overall_status(summary: Dict[str, Any], explicit_status: Optional[str]) -> str:
    if explicit_status:
        return explicit_status.lower()
    if summary["failed"]:
        return "failed"
    if summary.get("warning"):
        return "passed_with_warnings"
    if summary["total"] == 0:
        return "partial"
    return "passed"


def artifact_link(artifacts_dir: Path, class_key: str, suffix: str) -> str:
    path = artifacts_dir / f"{class_key}_{suffix}"
    return html_escape(path if path.exists() else "N/A")


def compact_config_value(value: Any) -> str:
    if value in (None, ""):
        return "N/A"
    if isinstance(value, dict):
        if not value:
            return "{}"
        return ", ".join(f"{key}={compact_config_value(config_value)}" for key, config_value in value.items())
    if isinstance(value, list):
        return ", ".join(compact_config_value(item) for item in value) if value else "[]"
    return str(value)


def build_pipeline_config_rows() -> List[List[str]]:
    pipeline_configs = load_json(PIPELINE_CONFIG_PATH)
    validation_configs = pipeline_configs.get("validation_configs", {})
    default_tolerances = validation_configs.get("default", {})
    model_class_tolerances = validation_configs.get("model_class_tolerances", {})
    rows = []

    for report_class_key, config_key in PIPELINE_CONFIG_ORDER:
        config_entries = pipeline_configs.get(config_key, [])
        config = config_entries[0] if config_entries else {}
        tolerances = {**default_tolerances, **model_class_tolerances.get(config_key, {})}
        rows.append(
            [
                html_escape(MODEL_CLASS_LABELS.get(report_class_key, report_class_key)),
                html_escape(compact_config_value(config.get("export_params", {}))),
                html_escape(compact_config_value(config.get("compile_params", {}))),
                html_escape(compact_config_value(config.get("generate_params", {}))),
                html_escape(compact_config_value(tolerances)),
            ]
        )

    return rows


def render_html(
    class_rows: Dict[str, List[Dict[str, str]]],
    summary: Dict[str, Any],
    metadata: Dict[str, Any],
    environment: Dict[str, Any],
    artifacts_dir: Path,
) -> str:
    generated_at = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    title_status = str(metadata.get("status", "unknown")).upper()
    repo_url = normalize_repo_url(metadata.get("repo_url"))

    build_sdk_rows = [
        ["Build Status", status_badge(str(metadata.get("status", "unknown")))],
        ["Repository", html_link(repo_url, "quic/efficient-transformers")],
        ["Job", html_escape(metadata.get("job_name"))],
        ["Build Number", html_escape(metadata.get("build_number"))],
        ["Build URL", html_escape(metadata.get("build_url"))],
        ["Node", html_escape(metadata.get("node_name"))],
        ["Branch", html_escape(metadata.get("branch"))],
        ["PR Number", html_escape(metadata.get("pr_number"))],
        ["Commit ID", html_escape(metadata.get("commit_id"))],
        ["Docker Image", html_escape(metadata.get("docker_image"))],
        [bold_text("QAIC Apps Version"), html_escape(environment.get("qaic_apps_version", "N/A"))],
        [bold_text("QAIC Platform Version"), html_escape(environment.get("qaic_platform_version", "N/A"))],
        ["QEfficient", html_escape(environment.get("qefficient_version", "N/A"))],
        ["Torch", html_escape(environment.get("torch_version", "N/A"))],
        ["Transformers", html_escape(environment.get("transformers_version", "N/A"))],
        ["Artifacts Dir", html_escape(metadata.get("artifacts_dir"))],
        ["Previous Artifacts Dir", html_escape(metadata.get("previous_artifacts_dir"))],
        ["Start Time", html_escape(metadata.get("start_time"))],
        ["End Time", html_escape(metadata.get("end_time"))],
        [bold_text("Total Duration"), html_escape(metadata.get("total_duration"))],
    ]

    summary_rows = []
    summary_classes = []
    for class_key, class_summary in summary["model_classes"].items():
        label = MODEL_CLASS_LABELS.get(class_key, class_key.replace("_", " ").title())
        summary_rows.append(
            [
                html_escape(label),
                html_escape(class_summary["total"]),
                html_escape(class_summary["passed"]),
                html_escape(class_summary["warning"]),
                html_escape(class_summary["failed"]),
                html_escape(class_summary["current_only"]),
                html_escape(f"{class_summary['pass_rate']:.1f}%"),
                artifact_link(artifacts_dir, class_key, "validation.csv"),
            ]
        )
        if class_summary["failed"]:
            summary_classes.append("failed")
        elif class_summary["warning"]:
            summary_classes.append("warning")
        else:
            summary_classes.append("passed")

    detail_sections = []
    for class_key, rows in class_rows.items():
        label = MODEL_CLASS_LABELS.get(class_key, class_key.replace("_", " ").title())
        detail_rows = []
        row_classes = []
        for row in rows:
            status = (row.get("status") or "unknown").lower()
            detail_rows.append(
                [
                    html_escape(row.get("model_name")),
                    html_escape(row.get("model_age") or "unknown"),
                    status_badge(status),
                    html_escape(row.get("failure_reason") or ""),
                ]
            )
            row_classes.append("failed" if status == "failed" else "passed" if status == "passed" else "warning")
        detail_sections.append(
            subsection(
                label,
                table(["Model Name", "Model Age", STATUS_COLUMN_HEADER, "Failure Reason"], detail_rows, row_classes),
            )
        )

    pass_rate_text = f"{summary['pass_rate']:.1f}%"
    metrics_html = (
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        'style="border-collapse:collapse;mso-table-lspace:0pt;mso-table-rspace:0pt;">'
        f"<tr>{metric_card(summary['total'], 'Total Models')}"
        f"{metric_card(summary['passed'], 'Passed', '#15803d')}"
        f"{metric_card(summary['warning'], 'Warnings', '#a16207')}"
        f"{metric_card(summary['failed'], 'Failed', '#b91c1c')}"
        f"{metric_card(pass_rate_text, 'Pass + Warning Rate')}</tr></table>"
    )
    detail_html = (
        "".join(detail_sections)
        if detail_sections
        else (
            '<p style="margin:0;color:#334155;font-family:Arial,Helvetica,sans-serif;font-size:13px;line-height:18px;">'
            "No validation CSV files found.</p>"
        )
    )
    sections_html = "".join(
        [
            section("Validation Metrics", metrics_html),
            section(
                "Build and SDK Details",
                table(["Field", "Value", "Field", "Value"], compact_key_value_rows(build_sdk_rows)),
            ),
            section(
                "Nightly Pipeline Configuration",
                table(
                    ["Model Class", "Export Params", "Compile Params", "Generate Params", "Validation Tolerances"],
                    build_pipeline_config_rows(),
                ),
            ),
            section("Model Class Details", detail_html),
            section(
                "Validation Summary",
                table(
                    ["Model Class", "Total", "Passed", "Warnings", "Failed", "Current-only", "Pass Rate", "CSV"],
                    summary_rows,
                    summary_classes,
                ),
            ),
        ]
    )

    return f"""<!doctype html>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html_escape(REPORT_TITLE)} - {html_escape(title_status)}</title>
</head>
<body bgcolor="#eef2ff" style="margin:0;padding:0;background-color:#eef2ff;color:#111827;font-family:Arial,Helvetica,sans-serif;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" bgcolor="#eef2ff" style="border-collapse:collapse;mso-table-lspace:0pt;mso-table-rspace:0pt;background-color:#eef2ff;">
<tr>
<td align="center" style="padding:20px 10px;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" bgcolor="#ffffff" style="border-collapse:collapse;mso-table-lspace:0pt;mso-table-rspace:0pt;background-color:#ffffff;border:1px solid #c7d2fe;max-width:1180px;">
<tr>
<td bgcolor="#0f172a" style="background-color:#0f172a;padding:24px 26px;font-family:Arial,Helvetica,sans-serif;color:#ffffff;border-bottom:4px solid #2563eb;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="border-collapse:collapse;mso-table-lspace:0pt;mso-table-rspace:0pt;">
<tr>
<td valign="top" style="font-family:Arial,Helvetica,sans-serif;color:#ffffff;">
<div style="margin:0 0 8px 0;color:#93c5fd;font-family:Arial,Helvetica,sans-serif;font-size:12px;line-height:16px;font-weight:bold;letter-spacing:0.08em;text-transform:uppercase;">Nightly Validation</div>
<h1 style="margin:0 0 8px 0;color:#ffffff;font-family:Arial,Helvetica,sans-serif;font-size:28px;line-height:34px;font-weight:bold;">{html_escape(REPORT_TITLE)}</h1>
<p style="margin:0;color:#e5e7eb;font-family:Arial,Helvetica,sans-serif;font-size:14px;line-height:20px;">{html_escape(metadata.get("job_name"))} #{html_escape(metadata.get("build_number"))} &bull; {html_escape(metadata.get("branch"))} &bull; {html_escape(metadata.get("total_duration"))}</p>
<p style="margin:6px 0 0 0;color:#bfdbfe;font-family:Arial,Helvetica,sans-serif;font-size:13px;line-height:18px;">Repository: <a href="{html_escape(repo_url)}" style="color:#bfdbfe;text-decoration:underline;">{html_escape(repo_url)}</a></p>
<p style="margin:4px 0 0 0;color:#cbd5e1;font-family:Arial,Helvetica,sans-serif;font-size:12px;line-height:18px;">Generated at {html_escape(generated_at)}</p>
</td>
<td align="right" valign="top" style="font-family:Arial,Helvetica,sans-serif;">{status_badge(str(metadata.get("status", "unknown")))}</td>
</tr>
</table>
</td>
</tr>
{sections_html}
</table>
</td>
</tr>
</table>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.output_environment_json:
        args.output_environment_json.parent.mkdir(parents=True, exist_ok=True)

    class_rows = load_validation_rows(artifacts_dir)
    summary = build_summary(class_rows)
    environment = collect_runtime_environment()
    environment.update(load_json(args.environment_json))
    if args.output_environment_json:
        args.output_environment_json.write_text(json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8")

    log_text = read_text(args.log_file)
    start_epoch = args.build_start_epoch or parse_optional_epoch(os.environ.get("BUILD_START_EPOCH"))
    end_epoch = (
        args.build_end_epoch or parse_optional_epoch(os.environ.get("BUILD_END_EPOCH")) or dt.datetime.now().timestamp()
    )
    status = overall_status(summary, args.build_status or os.environ.get("BUILD_RESULT"))
    metadata = derive_build_metadata(artifacts_dir, log_text, environment, start_epoch, end_epoch, status)

    html_report = render_html(class_rows, summary, metadata, environment, artifacts_dir)
    args.output_html.write_text(html_report, encoding="utf-8")

    if args.output_json:
        payload = {
            "metadata": metadata,
            "environment": environment,
            "summary": summary,
            "model_classes": class_rows,
        }
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote nightly HTML report: {args.output_html}")
    if args.output_json:
        print(f"Wrote nightly summary JSON: {args.output_json}")
    if args.output_environment_json:
        print(f"Wrote nightly environment JSON: {args.output_environment_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
