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
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

QAIC_VERSION_UTIL = "/opt/qti-aic/tools/qaic-version-util"
QAIC_APPS_XML = "/opt/qti-aic/versions/apps.xml"
QAIC_PLATFORM_XML = "/opt/qti-aic/versions/platform.xml"
QNN_SDK_ENV_VAR = "QNN_SDK_ROOT"
QNN_SDK_YAML = "sdk.yaml"


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
    return class_rows


def is_current_only(row: Dict[str, str]) -> bool:
    reason = (row.get("failure_reason") or "").lower()
    if "previous model artifact not found" in reason or "comparison skipped" in reason:
        return True
    before_fields = [value for key, value in row.items() if key.endswith("_before")]
    return bool(before_fields) and all((value or "").strip().upper() in {"N/A", "NA", ""} for value in before_fields)


def summarize_rows(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    total = len(rows)
    passed = sum(1 for row in rows if (row.get("status") or "").lower() == "passed")
    failed = sum(1 for row in rows if (row.get("status") or "").lower() == "failed")
    current_only = sum(1 for row in rows if is_current_only(row))
    pass_rate = (passed / total * 100.0) if total else 0.0
    return {
        "total": total,
        "passed": passed,
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
        "commit_id": commit,
        "commit_message": extract_first([r"Commit message:\s*\"([^\"]+)\"", r"^[0-9a-f]{7,40}\s+(.+)$"], log_text),
        "trigger": extract_first([r"^(Started by .+)$"], log_text),
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


def status_badge(status: str) -> str:
    normalized = (status or "unknown").lower()
    color = "#6b7280"
    if normalized == "passed":
        color = "#15803d"
    elif normalized == "failed":
        color = "#b91c1c"
    elif normalized in {"partial", "unstable"}:
        color = "#a16207"
    return f'<span style="display:inline-block;padding:3px 9px;border-radius:999px;background:{color};color:#fff;font-weight:700;font-size:12px;">{html_escape(status.upper())}</span>'


def table(headers: List[str], rows: List[List[Any]], row_classes: Optional[List[str]] = None) -> str:
    header_html = "".join(f"<th>{html_escape(header)}</th>" for header in headers)
    body_parts = []
    for index, row in enumerate(rows):
        row_class = row_classes[index] if row_classes and index < len(row_classes) else ""
        row_style = ""
        if row_class == "failed":
            row_style = ' style="background:#fef2f2;"'
        elif row_class == "passed":
            row_style = ' style="background:#f0fdf4;"'
        elif row_class == "warning":
            row_style = ' style="background:#fffbeb;"'
        body_parts.append("<tr{}>{}</tr>".format(row_style, "".join(f"<td>{cell}</td>" for cell in row)))
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(body_parts)}</tbody></table>"


def build_summary(class_rows: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    class_summaries = {class_key: summarize_rows(rows) for class_key, rows in class_rows.items()}
    total = sum(summary["total"] for summary in class_summaries.values())
    passed = sum(summary["passed"] for summary in class_summaries.values())
    failed = sum(summary["failed"] for summary in class_summaries.values())
    current_only = sum(summary["current_only"] for summary in class_summaries.values())
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "current_only": current_only,
        "pass_rate": (passed / total * 100.0) if total else 0.0,
        "model_classes": class_summaries,
    }


def overall_status(summary: Dict[str, Any], explicit_status: Optional[str]) -> str:
    if explicit_status:
        return explicit_status.lower()
    if summary["failed"]:
        return "failed"
    if summary["total"] == 0:
        return "partial"
    return "passed"


def artifact_link(artifacts_dir: Path, class_key: str, suffix: str) -> str:
    path = artifacts_dir / f"{class_key}_{suffix}"
    return html_escape(path if path.exists() else "N/A")


def render_html(
    class_rows: Dict[str, List[Dict[str, str]]],
    summary: Dict[str, Any],
    metadata: Dict[str, Any],
    environment: Dict[str, Any],
    artifacts_dir: Path,
) -> str:
    generated_at = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    title_status = str(metadata.get("status", "unknown")).upper()

    overview_rows = [
        ["Build Status", status_badge(str(metadata.get("status", "unknown")))],
        ["Job", html_escape(metadata.get("job_name"))],
        ["Build Number", html_escape(metadata.get("build_number"))],
        ["Build URL", html_escape(metadata.get("build_url"))],
        ["Node", html_escape(metadata.get("node_name"))],
        ["Trigger", html_escape(metadata.get("trigger"))],
        ["Branch", html_escape(metadata.get("branch"))],
        ["Commit ID", html_escape(metadata.get("commit_id"))],
        ["Commit Message", html_escape(metadata.get("commit_message"))],
        ["Docker Image", html_escape(metadata.get("docker_image"))],
        ["Artifacts Dir", html_escape(metadata.get("artifacts_dir"))],
        ["Previous Artifacts Dir", html_escape(metadata.get("previous_artifacts_dir"))],
        ["Start Time", html_escape(metadata.get("start_time"))],
        ["End Time", html_escape(metadata.get("end_time"))],
        ["Total Duration", html_escape(metadata.get("total_duration"))],
    ]

    sdk_rows = [
        ["QAIC Apps Version", html_escape(environment.get("qaic_apps_version", "N/A"))],
        ["QAIC Platform Version", html_escape(environment.get("qaic_platform_version", "N/A"))],
        ["QAIC Factory Version", html_escape(environment.get("qaic_factory_version", "N/A"))],
        ["QAIC SDK Source", html_escape(environment.get("qaic_sdk_source", "N/A"))],
        ["QNN SDK Root", html_escape(environment.get("qnn_sdk_root", "N/A"))],
        ["QNN SDK Details", html_escape(environment.get("qnn_sdk_details", "N/A"))],
        ["Python", html_escape(environment.get("python_version", "N/A"))],
        ["QEfficient", html_escape(environment.get("qefficient_version", "N/A"))],
        ["Torch", html_escape(environment.get("torch_version", "N/A"))],
        ["Transformers", html_escape(environment.get("transformers_version", "N/A"))],
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
                html_escape(class_summary["failed"]),
                html_escape(class_summary["current_only"]),
                html_escape(f"{class_summary['pass_rate']:.1f}%"),
                artifact_link(artifacts_dir, class_key, "validation.csv"),
            ]
        )
        summary_classes.append("failed" if class_summary["failed"] else "passed")

    failures: List[Tuple[str, Dict[str, str]]] = []
    perf_failures: List[Tuple[str, Dict[str, str]]] = []
    current_only_rows: List[Tuple[str, Dict[str, str]]] = []
    for class_key, rows in class_rows.items():
        label = MODEL_CLASS_LABELS.get(class_key, class_key)
        for row in rows:
            reason = row.get("failure_reason") or ""
            if (row.get("status") or "").lower() == "failed":
                failures.append((label, row))
                if "prefill_time_pct_diff" in reason or "decode_perf_pct_diff" in reason:
                    perf_failures.append((label, row))
            if is_current_only(row):
                current_only_rows.append((label, row))

    spotlight_rows = [
        [html_escape(model_class), html_escape(row.get("model_name")), html_escape(row.get("failure_reason"))]
        for model_class, row in failures[:5]
    ] or [["No validation failures", "", ""]]

    perf_rows = [
        [html_escape(model_class), html_escape(row.get("model_name")), html_escape(row.get("failure_reason"))]
        for model_class, row in perf_failures[:10]
    ] or [["No performance regression failures", "", ""]]

    current_only_table_rows = [
        [html_escape(model_class), html_escape(row.get("model_name")), html_escape(row.get("failure_reason"))]
        for model_class, row in current_only_rows[:20]
    ] or [["No current-only comparisons", "", ""]]

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
                    status_badge(status),
                    html_escape(row.get("failure_reason") or ""),
                ]
            )
            row_classes.append("failed" if status == "failed" else "passed" if status == "passed" else "warning")
        detail_sections.append(
            f"<h3>{html_escape(label)}</h3>{table(['Model Name', 'Status', 'Failure Reason'], detail_rows, row_classes)}"
        )

    css = """
    body { font-family: Arial, Helvetica, sans-serif; color: #111827; background: #f8fafc; margin: 0; padding: 20px; }
    .container { max-width: 1180px; margin: 0 auto; background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden; }
    .header { padding: 22px 26px; background: linear-gradient(135deg, #111827, #334155); color: #ffffff; }
    .header h1 { margin: 0 0 8px 0; font-size: 24px; }
    .header p { margin: 4px 0; color: #e5e7eb; }
    .section { padding: 20px 26px; border-top: 1px solid #e5e7eb; }
    .cards { display: table; width: 100%; border-spacing: 10px; margin-top: 12px; }
    .card { display: table-cell; padding: 14px; border-radius: 10px; background: #f8fafc; border: 1px solid #e5e7eb; text-align: center; }
    .card strong { display: block; font-size: 24px; margin-bottom: 4px; }
    table { border-collapse: collapse; width: 100%; margin: 10px 0 18px 0; font-size: 13px; }
    th { background: #e2e8f0; color: #0f172a; text-align: left; padding: 9px; border: 1px solid #cbd5e1; }
    td { padding: 8px; border: 1px solid #e5e7eb; vertical-align: top; }
    h2 { margin: 0 0 12px 0; font-size: 19px; color: #0f172a; }
    h3 { margin: 20px 0 8px 0; font-size: 16px; color: #1e293b; }
    .muted { color: #64748b; font-size: 12px; }
    """

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Nightly Pipeline Report - {html_escape(title_status)}</title>
<style>{css}</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>Nightly Pipeline Report {status_badge(str(metadata.get("status", "unknown")))}</h1>
    <p>{html_escape(metadata.get("job_name"))} #{html_escape(metadata.get("build_number"))} • {html_escape(metadata.get("branch"))} • {html_escape(metadata.get("total_duration"))}</p>
    <p class="muted">Generated at {html_escape(generated_at)}</p>
  </div>
  <div class="section">
    <div class="cards">
      <div class="card"><strong>{summary["total"]}</strong>Total Models</div>
      <div class="card"><strong style="color:#15803d;">{summary["passed"]}</strong>Passed</div>
      <div class="card"><strong style="color:#b91c1c;">{summary["failed"]}</strong>Failed</div>
      <div class="card"><strong>{summary["pass_rate"]:.1f}%</strong>Pass Rate</div>
    </div>
  </div>
  <div class="section"><h2>Build Details</h2>{table(["Field", "Value"], overview_rows)}</div>
  <div class="section"><h2>SDK and Runtime Details</h2>{table(["Field", "Value"], sdk_rows)}</div>
  <div class="section"><h2>Validation Summary</h2>{table(["Model Class", "Total", "Passed", "Failed", "Current-only", "Pass Rate", "CSV"], summary_rows, summary_classes)}</div>
  <div class="section"><h2>Failure Spotlight</h2>{table(["Model Class", "Model Name", "Failure Reason"], spotlight_rows, ["failed"] * len(spotlight_rows))}</div>
  <div class="section"><h2>Performance Regression Watch</h2>{table(["Model Class", "Model Name", "Failure Reason"], perf_rows)}</div>
  <div class="section"><h2>Current-only Comparisons</h2>{table(["Model Class", "Model Name", "Reason"], current_only_table_rows)}</div>
  <div class="section"><h2>Model Class Details</h2>{"".join(detail_sections) if detail_sections else "<p>No validation CSV files found.</p>"}</div>
</div>
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
