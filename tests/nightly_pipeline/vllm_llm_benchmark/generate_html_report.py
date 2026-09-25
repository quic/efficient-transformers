#!/usr/bin/env python3
"""
Generate an HTML report from the consolidated published CSV for email distribution.

This script reads the consolidated published CSV and generates a formatted HTML report
with environment info (branch details, SDK version) and test results table.

Usage:
    python3 generate_html_report.py --csv consolidated_published_results.csv --output report.html
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


def generate_html_report(csv_path: Path, output_path: Path, build_url: str = "N/A") -> int:
    """Generate an HTML report from the consolidated published CSV."""
    if not csv_path.exists():
        print(f"Error: CSV file does not exist: {csv_path}")
        return 1

    rows = []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("Error: no data found in CSV file")
        return 1

    # Extract environment info from first row
    env_info = {
        "vllm_qaic_branch": rows[0].get("vllm_qaic_branch", "N/A"),
        "qaic_disagg_branch": rows[0].get("qaic_disagg_branch", "N/A"),
        "qserve_branch": rows[0].get("qserve_branch", "N/A"),
        "qeff_branch": rows[0].get("qeff_branch", "N/A"),
        "qaic_sdk_version": rows[0].get("qaic_sdk_version", "N/A"),
        "build_url": build_url,
    }

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>vLLM QAIC Benchmark Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 100%;
            margin: 0 auto;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 28px;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section-title {{
            font-size: 20px;
            font-weight: 600;
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        .env-grid {{
            width: 100%;
            margin-bottom: 20px;
        }}
        .env-row {{
            display: block;
            margin-bottom: 15px;
        }}
        .env-card {{
            background-color: #f9f9f9;
            border-left: 4px solid #667eea;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 10px;
            display: inline-block;
            width: 48%;
            margin-right: 2%;
            vertical-align: top;
        }}
        .env-card:nth-child(odd) {{
            margin-right: 2%;
        }}
        .env-card:nth-child(even) {{
            margin-right: 0;
        }}
        .env-card-label {{
            font-size: 11px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
            font-weight: 600;
        }}
        .env-card-value {{
            font-size: 13px;
            font-weight: 500;
            color: #333;
            word-break: break-all;
            font-family: 'Courier New', monospace;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        thead {{
            background-color: #f0f0f0;
        }}
        th {{
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #333;
            border: 1px solid #ddd;
            font-size: 12px;
            white-space: nowrap;
        }}
        td {{
            padding: 12px;
            border: 1px solid #eee;
            font-size: 12px;
            word-wrap: break-word;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .status-success {{
            color: #27ae60;
            font-weight: 600;
        }}
        .status-failed {{
            color: #e74c3c;
            font-weight: 600;
        }}
        .model-name {{
            font-family: 'Courier New', monospace;
            font-size: 11px;
            word-break: break-word;
        }}
        .metric {{
            text-align: right;
            font-family: 'Courier New', monospace;
            font-size: 11px;
        }}
        .footer {{
            background-color: #f5f5f5;
            padding: 20px 30px;
            text-align: center;
            font-size: 12px;
            color: #666;
            border-top: 1px solid #eee;
        }}
        .summary-table {{
            width: 100%;
            margin-bottom: 20px;
        }}
        .summary-cell {{
            width: 33.33%;
            padding: 15px;
            background-color: #f0f7ff;
            border: 1px solid #b3d9ff;
            text-align: center;
            vertical-align: top;
        }}
        .summary-card-value {{
            font-size: 24px;
            font-weight: 700;
            color: #667eea;
        }}
        .summary-card-label {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>vLLM QAIC Benchmark Report</h1>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
        </div>

        <div class="content">
            <!-- Environment Info Section -->
            <div class="section">
                <div class="section-title">Environment Information</div>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="width: 50%; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #667eea; border-bottom: 1px solid #eee;">
                            <div style="font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 5px; font-weight: 600;">vLLM QAIC Branch</div>
                            <div style="font-size: 13px; font-weight: 500; color: #333; font-family: 'Courier New', monospace; word-break: break-all;">{env_info["vllm_qaic_branch"]}</div>
                        </td>
                        <td style="width: 50%; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #667eea; border-bottom: 1px solid #eee;">
                            <div style="font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 5px; font-weight: 600;">QAIC Disagg Branch</div>
                            <div style="font-size: 13px; font-weight: 500; color: #333; font-family: 'Courier New', monospace; word-break: break-all;">{env_info["qaic_disagg_branch"]}</div>
                        </td>
                    </tr>
                    <tr>
                        <td style="width: 50%; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #667eea; border-bottom: 1px solid #eee;">
                            <div style="font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 5px; font-weight: 600;">QServe Branch</div>
                            <div style="font-size: 13px; font-weight: 500; color: #333; font-family: 'Courier New', monospace; word-break: break-all;">{env_info["qserve_branch"]}</div>
                        </td>
                        <td style="width: 50%; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #667eea; border-bottom: 1px solid #eee;">
                            <div style="font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 5px; font-weight: 600;">QEff Branch</div>
                            <div style="font-size: 13px; font-weight: 500; color: #333; font-family: 'Courier New', monospace; word-break: break-all;">{env_info["qeff_branch"]}</div>
                        </td>
                    </tr>
                    <tr>
                        <td style="width: 50%; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #667eea; border-bottom: 1px solid #eee;">
                            <div style="font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 5px; font-weight: 600;">QAIC SDK Version</div>
                            <div style="font-size: 13px; font-weight: 500; color: #333; font-family: 'Courier New', monospace; word-break: break-all;">{env_info["qaic_sdk_version"]}</div>
                        </td>
                        <td style="width: 50%; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #667eea; border-bottom: 1px solid #eee;">
                            <div style="font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 5px; font-weight: 600;">Build URL</div>
                            <div style="font-size: 13px; font-weight: 500; color: #0066cc; font-family: 'Courier New', monospace; word-break: break-all;">
                                <a href="{env_info["build_url"]}" style="color: #0066cc; text-decoration: none;">{env_info["build_url"]}</a>
                            </div>
                        </td>
                    </tr>
                </table>
            </div>

            <!-- Test Results Summary -->
            <div class="section">
                <div class="section-title">Test Results Summary</div>
                <table class="summary-table">
                    <tr>
                        <td class="summary-cell">
                            <div class="summary-card-value">{len(rows)}</div>
                            <div class="summary-card-label">Total Tests</div>
                        </td>
                        <td class="summary-cell">
                            <div class="summary-card-value">{sum(1 for r in rows if r.get("status", "").lower() == "success")}</div>
                            <div class="summary-card-label">Passed</div>
                        </td>
                        <td class="summary-cell">
                            <div class="summary-card-value">{sum(1 for r in rows if r.get("status", "").lower() != "success")}</div>
                            <div class="summary-card-label">Failed</div>
                        </td>
                    </tr>
                </table>
            </div>

            <!-- Test Results Table -->
            <div class="section">
                <div class="section-title">Detailed Test Results</div>
                <table>
                    <thead>
                        <tr>
                            <th style="text-align: left;">Model</th>
                            <th style="text-align: left;">Category</th>
                            <th style="text-align: left;">Config</th>
                            <th style="text-align: left;">Summary</th>
                            <th style="text-align: left;">Status</th>
                            <th style="text-align: right;">Export/Compile (s)</th>
                            <th style="text-align: right;">Prefill MDP Export/Compile (s)</th>
                            <th style="text-align: right;">Prefill Export/Compile (s)</th>
                            <th style="text-align: right;">Decode Export/Compile (s)</th>
                            <th style="text-align: right;">Encode Export/Compile (s)</th>
                            <th style="text-align: right;">TTFT (s)</th>
                            <th style="text-align: right;">TPOT (s)</th>
                            <th style="text-align: right;">ITL (s)</th>
                            <th style="text-align: right;">Decode TPS</th>
                            <th style="text-align: right;">Throughput (req/s)</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    for row in rows:
        status = row.get("status", "N/A").lower()
        status_class = "status-success" if status == "success" else "status-failed"
        status_text = "✓ PASS" if status == "success" else "✗ FAIL"

        html_content += f"""                        <tr>
                            <td class="model-name" style="text-align: left;">{row.get("model", "N/A")}</td>
                            <td style="text-align: left;">{row.get("model_category", "N/A")}</td>
                            <td style="text-align: left;">{row.get("config_name", "N/A")}</td>
                            <td style="text-align: left;">{row.get("config_summary", "N/A")}</td>
                            <td class="{status_class}" style="text-align: left;">{status_text}</td>
                            <td class="metric">{row.get("export_compile_time_s", "N/A")}</td>
                            <td class="metric">{row.get("prefill_mdp_export_compile_time_s", "N/A")}</td>
                            <td class="metric">{row.get("prefill_export_compile_time_s", "N/A")}</td>
                            <td class="metric">{row.get("decode_export_compile_time_s", "N/A")}</td>
                            <td class="metric">{row.get("encode_export_compile_time_s", "N/A")}</td>
                            <td class="metric">{row.get("mean_ttft_s", "N/A")}</td>
                            <td class="metric">{row.get("mean_tpot_s", "N/A")}</td>
                            <td class="metric">{row.get("mean_itl_s", "N/A")}</td>
                            <td class="metric">{row.get("decode_TPS", "N/A")}</td>
                            <td class="metric">{row.get("request_throughput_req_s", "N/A")}</td>
                        </tr>
"""

    html_content += """                    </tbody>
                </table>
            </div>
        </div>

        <div class="footer">
            <p>This report was automatically generated from vLLM QAIC benchmark results.</p>
            <p>For questions or issues, please contact the QEfficient team.</p>
        </div>
    </div>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✓ HTML report generated: {output_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an HTML report from the consolidated published CSV.")
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to consolidated published CSV file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output HTML report path",
    )
    parser.add_argument(
        "--build-url",
        default="N/A",
        help="Jenkins build URL (optional)",
    )
    args = parser.parse_args()

    return generate_html_report(Path(args.csv), Path(args.output), args.build_url)


if __name__ == "__main__":
    raise SystemExit(main())
