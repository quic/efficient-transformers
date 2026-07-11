# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
HTML + JSON report generator for the dynamo test suite.

Dark, dense, self-contained. Every drill-down is a native ``<details>`` so
the top of the page stays glanceable; charts are hand-drawn SVG so the
report renders inside any sandboxed CI artifact viewer without CDN access.
"""

from __future__ import annotations

import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DYNAMO_REPORT_COLUMNS: Tuple[str, ...] = (
    "End_To_End_E2E",
    "Export_Compile",
    "QAIC_Generate_Execute",
    "CPU_Parity",
    "HW_Parity_FP16",
    "CB_Dynamo_Subfn",
    "Subfunction_Coverage",
    "FP32_Coverage",
    "FP16_Coverage",
    "BF16_Coverage",
    "MXFP6_Coverage",
    "MXINT8_Coverage",
    "CCL_Coverage",
    "Blocking_KV_Coverage",
    "Sampler_Coverage",
    "Prefix_Caching_Coverage",
)

# Short header labels for the matrix (long names wrap and waste vertical space).
_COLUMN_SHORT: Dict[str, str] = {
    "End_To_End_E2E": "E2E",
    "Export_Compile": "Export / Compile",
    "QAIC_Generate_Execute": "QAIC Generate",
    "CPU_Parity": "CPU Parity (HF=QEff=ORT)",
    "HW_Parity_FP16": "HW Parity (ORT=QAIC FP16)",
    "CB_Dynamo_Subfn": "CB (Dynamo+Subfn)",
    "Subfunction_Coverage": "ONNX Subfunctions",
    "FP32_Coverage": "FP32 Export",
    "FP16_Coverage": "FP16 Compile",
    "BF16_Coverage": "BF16 Export",
    "MXFP6_Coverage": "MXFP6 Compile",
    "MXINT8_Coverage": "MXFP6+MXINT8 Compile",
    "CCL_Coverage": "CCL (Multi-Device)",
    "Blocking_KV_Coverage": "Blocking KV",
    "Sampler_Coverage": "On-Device Sampler",
    "Prefix_Caching_Coverage": "Prefix Caching",
}

_STATUS_PRIORITY = {
    "FAIL": 4,
    "PASS": 3,
    "XFAIL": 2,
    "SKIP": 1,
    "NOT_RUN": 0,
}

# Dark-theme palette: cell fill is a soft dark background of the base colour.
_STATUS_COLORS = {
    "PASS": "#3ddc84",
    "FAIL": "#ff5b6a",
    "SKIP": "#ffb03a",
    "XFAIL": "#b48bff",
    "NOT_RUN": "#495366",
}

_STATUS_GLYPH = {
    "PASS": "PASS",
    "FAIL": "FAIL",
    "SKIP": "SKIP",
    "XFAIL": "XFAIL",
    "NOT_RUN": "—",
}


def attach_dynamo_case(
    request,
    *,
    category: str,
    task: str,
    architecture: str,
    family: str,
    supported_model: Optional[str],
    coverage_columns: Iterable[str],
    notes: Optional[str] = None,
) -> None:
    request.node._dynamo_case = {
        "category": category,
        "task": task,
        "architecture": architecture,
        "family": family,
        "supported_model": supported_model or "",
        "coverage_columns": list(coverage_columns),
        "notes": notes or "",
    }


def merge_status(previous: str, current: str) -> str:
    return current if _STATUS_PRIORITY[current] >= _STATUS_PRIORITY[previous] else previous


# --- Aggregation -------------------------------------------------------------
def aggregate_results(results: List[Dict]) -> List[Dict]:
    rows: Dict[Tuple[str, str, str], Dict] = {}
    details: Dict[Tuple[str, str, str], List[Dict]] = defaultdict(list)

    for record in results:
        key = (record["architecture"], record["family"], record["supported_model"])
        row = rows.setdefault(
            key,
            {
                "Category": record["category"],
                "Task": "",
                "Architecture": record["architecture"],
                "Family": record["family"],
                "Supported_Model": record["supported_model"],
                **{column: "NOT_RUN" for column in DYNAMO_REPORT_COLUMNS},
            },
        )
        row["Category"] = record["category"]
        existing_tasks = [task for task in row["Task"].split(", ") if task]
        if record["task"] not in existing_tasks:
            existing_tasks.append(record["task"])
            row["Task"] = ", ".join(existing_tasks)

        outcome = record["outcome"]
        for column in record["coverage_columns"]:
            if column in DYNAMO_REPORT_COLUMNS:
                row[column] = merge_status(row[column], outcome)

        details[key].append(
            {
                "task": record["task"],
                "outcome": outcome,
                "reason": record.get("reason", ""),
                "notes": record.get("notes", ""),
                "coverage_columns": record["coverage_columns"],
                "nodeid": record.get("nodeid", ""),
                "duration": record.get("duration", 0.0),
            }
        )

    ordered = []
    for key in sorted(rows):
        row = rows[key]
        row["details"] = sorted(details[key], key=lambda item: (item["task"], item["outcome"], item["nodeid"]))
        ordered.append(row)
    return ordered


# --- JSON --------------------------------------------------------------------
def write_json_report(report_path: Path, rows: List[Dict]) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "columns": [
            "Category",
            "Task",
            "Architecture",
            "Family",
            "Supported_Model",
            *DYNAMO_REPORT_COLUMNS,
        ],
        "rows": rows,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# --- Summary helpers --------------------------------------------------------
def _status_counts(rows: List[Dict]) -> Dict[str, int]:
    counts = {status: 0 for status in _STATUS_COLORS}
    for row in rows:
        for column in DYNAMO_REPORT_COLUMNS:
            counts[row[column]] += 1
    return counts


def _task_counts(rows: List[Dict]) -> Dict[str, int]:
    counts = {status: 0 for status in _STATUS_COLORS}
    for row in rows:
        for item in row["details"]:
            counts[item["outcome"]] = counts.get(item["outcome"], 0) + 1
    return counts


def _column_health(rows: List[Dict]) -> List[Tuple[str, Dict[str, int]]]:
    out = []
    for column in DYNAMO_REPORT_COLUMNS:
        counts = {status: 0 for status in _STATUS_COLORS}
        for row in rows:
            counts[row[column]] += 1
        out.append((column, counts))
    return out


def _category_health(rows: List[Dict]) -> List[Tuple[str, Dict[str, int]]]:
    grouped: Dict[str, Dict[str, int]] = defaultdict(lambda: {status: 0 for status in _STATUS_COLORS})
    for row in rows:
        counts = grouped[row["Category"] or "uncategorised"]
        for column in DYNAMO_REPORT_COLUMNS:
            counts[row[column]] += 1
    return sorted(grouped.items())


# --- SVG chart builders ------------------------------------------------------
def _svg_donut(counts: Dict[str, int], *, title: str, size: int = 170) -> str:
    total = sum(counts.values()) or 1
    cx = cy = size / 2
    radius = size * 0.38
    stroke = size * 0.15
    circumference = 2 * 3.141592653589793 * radius
    segments = []
    offset = 0.0
    for status in ("PASS", "XFAIL", "SKIP", "NOT_RUN", "FAIL"):
        value = counts.get(status, 0)
        if value == 0:
            continue
        length = circumference * value / total
        segments.append(
            f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="{_STATUS_COLORS[status]}" '
            f'stroke-width="{stroke}" stroke-dasharray="{length:.3f} {circumference:.3f}" '
            f'stroke-dashoffset="{-offset:.3f}" transform="rotate(-90 {cx} {cy})" />'
        )
        offset += length

    pct_pass = 100.0 * counts.get("PASS", 0) / total
    return (
        f'<div class="donut">'
        f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" role="img" aria-label="{html.escape(title)}">'
        f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="#1f2733" stroke-width="{stroke}" />'
        f"{''.join(segments)}"
        f'<text x="{cx}" y="{cy - 2}" text-anchor="middle" font-size="{size * 0.19:.1f}" '
        f'font-weight="700" fill="#f4f5f7">{pct_pass:.0f}%</text>'
        f'<text x="{cx}" y="{cy + size * 0.11:.1f}" text-anchor="middle" font-size="{size * 0.075:.1f}" '
        f'fill="#98a0b0" letter-spacing="0.5">PASS</text>'
        f"</svg>"
        f'<div class="donut-title">{html.escape(title)}</div>'
        f"</div>"
    )


def _svg_stacked_bar(label: str, counts: Dict[str, int], *, width: int = 320, height: int = 14) -> str:
    total = sum(counts.values())
    if total == 0:
        segments_svg = ""
    else:
        x = 0.0
        parts = []
        for status in ("PASS", "XFAIL", "SKIP", "NOT_RUN", "FAIL"):
            value = counts.get(status, 0)
            if value == 0:
                continue
            w = width * value / total
            parts.append(
                f'<rect x="{x:.3f}" y="0" width="{w:.3f}" height="{height}" fill="{_STATUS_COLORS[status]}">'
                f"<title>{status}: {value}</title></rect>"
            )
            x += w
        segments_svg = "".join(parts)
    return (
        f'<div class="stacked">'
        f'<div class="stacked-label" title="{html.escape(label)}">{html.escape(label)}</div>'
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" preserveAspectRatio="none">'
        f"{segments_svg}</svg>"
        f'<div class="stacked-total">{total}</div>'
        f"</div>"
    )


# --- HTML sections ----------------------------------------------------------
def _kpi_strip(rows: List[Dict], task_counts: Dict[str, int]) -> str:
    total_arch = len(rows)
    pass_cells = sum(1 for row in rows for c in DYNAMO_REPORT_COLUMNS if row[c] == "PASS")
    fail_cells = sum(1 for row in rows for c in DYNAMO_REPORT_COLUMNS if row[c] == "FAIL")
    covered = pass_cells + fail_cells
    pass_rate = (100.0 * pass_cells / covered) if covered else 0.0
    total_tasks = sum(task_counts.values())
    task_fail = task_counts.get("FAIL", 0)
    task_skip = task_counts.get("SKIP", 0)

    def _kpi(label: str, value: str, tone: str = "") -> str:
        return (
            f'<div class="kpi {tone}">'
            f'<div class="kpi-value">{html.escape(value)}</div>'
            f'<div class="kpi-label">{html.escape(label)}</div>'
            f"</div>"
        )

    return (
        '<div class="kpi-strip">'
        + _kpi("Architectures", str(total_arch))
        + _kpi("Task Runs", str(total_tasks))
        + _kpi("Passed", str(task_counts.get("PASS", 0)), "pass")
        + _kpi("Failed", str(task_fail), "fail" if task_fail else "pass")
        + _kpi("Skipped", str(task_skip), "skip" if task_skip else "")
        + _kpi("Pass Rate", f"{pass_rate:.0f}%", "pass" if pass_rate >= 80 else "fail")
        + "</div>"
    )


def _matrix_table(rows: List[Dict]) -> str:
    headers = ["Arch", "Family", "Model"] + [_COLUMN_SHORT[c] for c in DYNAMO_REPORT_COLUMNS]
    header_html = "".join(f"<th>{html.escape(h)}</th>" for h in headers)

    body = []
    for row in rows:
        row_id = f"detail-{row['Architecture']}"
        cells = [
            f'<td class="arch"><a href="#{html.escape(row_id)}">{html.escape(row["Architecture"])}</a></td>',
            f'<td class="dim">{html.escape(row["Family"])}</td>',
            f'<td class="dim model" title="{html.escape(row["Supported_Model"] or "")}">'
            f"{html.escape(row['Supported_Model'] or '—')}</td>",
        ]
        for c in DYNAMO_REPORT_COLUMNS:
            status = row[c]
            cls = status.lower().replace("_", "-")
            cells.append(
                f'<td class="cell {cls}" title="{html.escape(c)}: {html.escape(status)}">'
                f'<span class="glyph">{html.escape(_STATUS_GLYPH[status])}</span></td>'
            )
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (
        '<div class="matrix-wrap"><table class="matrix"><thead><tr>'
        f"{header_html}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def _details_block(row: Dict) -> str:
    row_id = f"detail-{row['Architecture']}"
    task_rows = []
    for item in row["details"]:
        outcome = item["outcome"]
        pill_cls = outcome.lower().replace("_", "-")
        reason = (item.get("reason") or "").strip()
        notes = (item.get("notes") or "").strip()
        duration = item.get("duration") or 0.0
        # keep the reason readable: first line + short trailing hint
        short_reason = reason.splitlines()[0] if reason else ""
        if len(short_reason) > 220:
            short_reason = short_reason[:217] + "…"
        reason_html = f'<div class="reason">{html.escape(short_reason)}</div>' if short_reason else ""
        notes_html = f'<div class="notes">{html.escape(notes)}</div>' if notes else ""
        task_rows.append(
            f"<tr>"
            f'<td class="task"><code>{html.escape(item["task"])}</code></td>'
            f'<td class="cell {pill_cls}"><span class="pill {pill_cls}">{html.escape(outcome)}</span></td>'
            f'<td class="dur">{duration:.2f}s</td>'
            f'<td class="ctx">{reason_html}{notes_html}</td>'
            f"</tr>"
        )
    summary = f"{row['Architecture']} — {row['Family']}"
    if row["Supported_Model"]:
        summary += f" · {row['Supported_Model']}"
    return (
        f'<details id="{html.escape(row_id)}" class="drill">'
        f"<summary>{html.escape(summary)} "
        f'<span class="drill-count">{len(row["details"])} task run(s)</span></summary>'
        f'<table class="detail-table"><thead><tr>'
        f"<th>Task</th><th>Status</th><th>Duration</th><th>Reason / Notes</th></tr></thead>"
        f"<tbody>{''.join(task_rows)}</tbody></table>"
        f"</details>"
    )


def _legend_bar() -> str:
    items = []
    for status in ("PASS", "FAIL", "SKIP", "XFAIL", "NOT_RUN"):
        color = _STATUS_COLORS[status]
        items.append(
            f'<span class="legend-item"><span class="legend-dot" style="background:{color}"></span>'
            f"{html.escape(status)}</span>"
        )
    return f'<div class="legend">{"".join(items)}</div>'


# --- CSS ---------------------------------------------------------------------
_CSS = """
:root {
  --bg: #0e131b;
  --panel: #151b26;
  --panel-2: #1a2130;
  --border: #232c3d;
  --border-strong: #2c3648;
  --fg: #eef1f7;
  --fg-dim: #98a0b0;
  --fg-mute: #6d7688;
  --accent: #66aaff;
  --pass: #3ddc84;
  --fail: #ff5b6a;
  --skip: #ffb03a;
  --xfail: #b48bff;
  --not-run: #495366;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: "Inter", "IBM Plex Sans", -apple-system, BlinkMacSystemFont, sans-serif;
  background: var(--bg);
  color: var(--fg);
  font-size: 13px;
  line-height: 1.45;
  -webkit-font-smoothing: antialiased;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
header {
  padding: 20px 32px 16px;
  border-bottom: 1px solid var(--border);
  background: linear-gradient(180deg, #101725, #0e131b);
}
header h1 {
  margin: 0;
  font-size: 20px;
  font-weight: 600;
  letter-spacing: 0.2px;
}
header .subtitle { color: var(--fg-dim); font-size: 12px; margin-top: 3px; }
main { padding: 20px 32px 40px; max-width: 1500px; margin: 0 auto; }
section.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px 20px;
  margin-bottom: 16px;
}
section.panel h2 {
  margin: 0 0 12px;
  font-size: 13px;
  font-weight: 600;
  color: var(--fg-dim);
  letter-spacing: 0.6px;
  text-transform: uppercase;
}
.summary-row {
  display: grid;
  grid-template-columns: 1fr auto auto;
  gap: 20px;
  align-items: center;
}
.kpi-strip {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 10px;
  min-width: 0;
}
.kpi {
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px 12px;
}
.kpi-value {
  font-size: 20px;
  font-weight: 700;
  letter-spacing: -0.2px;
}
.kpi-label {
  color: var(--fg-mute);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  margin-top: 2px;
}
.kpi.pass  .kpi-value { color: var(--pass); }
.kpi.fail  .kpi-value { color: var(--fail); }
.kpi.skip  .kpi-value { color: var(--skip); }
.donut-row { display: flex; gap: 14px; }
.donut { text-align: center; }
.donut-title {
  margin-top: 4px;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  color: var(--fg-mute);
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  padding: 8px 12px;
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 6px;
  font-size: 11px;
  color: var(--fg-dim);
  margin-bottom: 16px;
}
.legend-item { display: inline-flex; align-items: center; gap: 6px; }
.legend-dot { display: inline-block; width: 10px; height: 10px; border-radius: 2px; }
.dual-panel {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
.stacked {
  display: grid;
  grid-template-columns: 140px 1fr 40px;
  align-items: center;
  gap: 10px;
  margin: 4px 0;
}
.stacked-label {
  font-size: 11px;
  color: var(--fg-dim);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.stacked-total {
  font-size: 10px;
  color: var(--fg-mute);
  text-align: right;
}
.stacked svg { border-radius: 3px; background: var(--panel-2); }
.matrix-wrap {
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--panel);
}
table.matrix {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 12px;
}
table.matrix th, table.matrix td {
  padding: 6px 8px;
  border-bottom: 1px solid var(--border);
  text-align: center;
  white-space: nowrap;
}
table.matrix thead th {
  position: sticky;
  top: 0;
  background: var(--panel-2);
  color: var(--fg-dim);
  font-weight: 600;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  border-bottom: 1px solid var(--border-strong);
  z-index: 2;
}
table.matrix td.arch {
  text-align: left;
  font-weight: 600;
  color: var(--fg);
}
table.matrix td.dim { text-align: left; color: var(--fg-dim); }
table.matrix td.model {
  max-width: 240px;
  overflow: hidden;
  text-overflow: ellipsis;
  font-family: "SF Mono", "Menlo", monospace;
  font-size: 11px;
}
table.matrix tbody tr:hover { background: rgba(102, 170, 255, 0.04); }
td.cell .glyph {
  display: inline-block;
  min-width: 44px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.4px;
  padding: 2px 6px;
  border-radius: 3px;
}
td.cell.pass    .glyph { background: rgba(61, 220, 132, 0.14); color: var(--pass); }
td.cell.fail    .glyph { background: rgba(255, 91, 106, 0.16); color: var(--fail); }
td.cell.skip    .glyph { background: rgba(255, 176, 58, 0.14); color: var(--skip); }
td.cell.xfail   .glyph { background: rgba(180, 139, 255, 0.16); color: var(--xfail); }
td.cell.not-run .glyph { background: rgba(73, 83, 102, 0.24); color: var(--fg-mute); }
details.drill {
  margin: 6px 0;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--panel);
}
details.drill[open] { background: var(--panel-2); }
details.drill > summary {
  cursor: pointer;
  padding: 8px 12px;
  font-weight: 600;
  color: var(--fg);
  list-style: none;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
details.drill > summary::-webkit-details-marker { display: none; }
details.drill > summary::before {
  content: "▸";
  color: var(--fg-mute);
  transition: transform 0.15s ease;
  margin-right: 8px;
  display: inline-block;
}
details.drill[open] > summary::before { transform: rotate(90deg); }
.drill-count { color: var(--fg-mute); font-weight: 500; font-size: 11px; }
table.detail-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  margin: 4px 0 8px;
}
table.detail-table th, table.detail-table td {
  padding: 6px 12px;
  border-top: 1px solid var(--border);
  text-align: left;
  vertical-align: top;
}
table.detail-table th {
  color: var(--fg-mute);
  font-weight: 500;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  border-top: 1px solid var(--border-strong);
}
table.detail-table td.task code {
  font-family: "SF Mono", "Menlo", monospace;
  font-size: 11px;
  color: var(--accent);
}
table.detail-table td.dur { color: var(--fg-mute); font-size: 11px; white-space: nowrap; }
table.detail-table .reason,
table.detail-table .notes {
  color: var(--fg-dim);
  font-size: 11px;
  margin-top: 2px;
  font-family: "SF Mono", "Menlo", monospace;
}
table.detail-table .notes { color: var(--fg-mute); font-style: italic; font-family: inherit; }
.pill {
  display: inline-block;
  padding: 1px 8px;
  border-radius: 10px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.4px;
}
.pill.pass    { background: rgba(61, 220, 132, 0.15); color: var(--pass); }
.pill.fail    { background: rgba(255, 91, 106, 0.16); color: var(--fail); }
.pill.skip    { background: rgba(255, 176, 58, 0.14); color: var(--skip); }
.pill.xfail   { background: rgba(180, 139, 255, 0.16); color: var(--xfail); }
.pill.not-run { background: rgba(73, 83, 102, 0.24); color: var(--fg-mute); }

.filter-bar { display: flex; gap: 8px; align-items: center; margin-bottom: 8px; }
.filter-bar input {
  flex: 1;
  background: var(--panel-2);
  color: var(--fg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 6px 10px;
  font-size: 12px;
}
"""


_JS = """
(function() {
  var input = document.getElementById('drill-filter');
  if (!input) return;
  var details = Array.prototype.slice.call(document.querySelectorAll('details.drill'));
  input.addEventListener('input', function() {
    var q = input.value.trim().toLowerCase();
    details.forEach(function(d) {
      var text = d.textContent.toLowerCase();
      var match = !q || text.indexOf(q) !== -1;
      d.style.display = match ? '' : 'none';
      if (match && q) d.open = true;
    });
  });
})();
"""


def write_html_report(report_path: Path, rows: List[Dict]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    cell_counts = _status_counts(rows)
    task_counts = _task_counts(rows)

    donut_cells = _svg_donut(cell_counts, title="Coverage cells")
    donut_tasks = _svg_donut(task_counts, title="Task runs")

    column_bars = "".join(_svg_stacked_bar(col, counts) for col, counts in _column_health(rows))
    category_bars = "".join(_svg_stacked_bar(cat, counts) for cat, counts in _category_health(rows))

    matrix_html = _matrix_table(rows)
    details_html = "".join(_details_block(row) for row in rows)

    kpi_html = _kpi_strip(rows, task_counts)

    body = f"""
<header>
  <h1>QEfficient · Dynamo Coverage</h1>
  <div class="subtitle">{html.escape(generated_at)} · {len(rows)} architectures · {sum(task_counts.values())} task runs</div>
</header>
<main>
  <section class="panel">
    <div class="summary-row">
      {kpi_html}
      <div class="donut-row">{donut_cells}{donut_tasks}</div>
    </div>
  </section>

  {_legend_bar()}

  <section class="panel">
    <h2>Coverage matrix</h2>
    {matrix_html}
  </section>

  <div class="dual-panel">
    <section class="panel">
      <h2>Feature column health</h2>
      {column_bars}
    </section>
    <section class="panel">
      <h2>Category health</h2>
      {category_bars}
    </section>
  </div>

  <section class="panel">
    <h2>Per-architecture drill-down</h2>
    <div class="filter-bar">
      <input id="drill-filter" type="search" placeholder="Filter architectures, tasks, or reasons…" />
    </div>
    {details_html}
  </section>
</main>
<script>{_JS}</script>
"""

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>QEff Dynamo Coverage</title>
  <style>{_CSS}</style>
</head>
<body>{body}</body>
</html>
"""
    report_path.write_text(document, encoding="utf-8")


__all__ = [
    "DYNAMO_REPORT_COLUMNS",
    "attach_dynamo_case",
    "merge_status",
    "aggregate_results",
    "write_html_report",
    "write_json_report",
]

_SEQUENCE: Sequence[str] = DYNAMO_REPORT_COLUMNS
