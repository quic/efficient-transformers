# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Convert pr_report.md to styled HTML."""

import sys
from pathlib import Path
try:
    import markdown
except ImportError:
    print("Installing markdown library...", file=sys.stderr)
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "markdown"])
    import markdown


def convert_md_to_html(md_file, html_file):
    """Convert markdown to HTML with styling."""

    # Read markdown content
    with open(md_file, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Convert markdown to HTML
    html_body = markdown.markdown(md_content, extensions=["tables", "fenced_code"])

    # Create full HTML document with styling
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Open PR Dashboard - quic/efficient-transformers</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
                'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
            line-height: 1.6;
            color: #24292e;
            background-color: #f6f8fa;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        }}
        
        h1 {{
            color: #1a1a2e;
            border-bottom: 3px solid #0366d6;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 2em;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.9em;
            overflow-x: auto;
            display: block;
        }}
        
        thead {{
            background-color: #f6f8fa;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        
        th {{
            padding: 12px 8px;
            text-align: left;
            font-weight: 600;
            color: #24292e;
            border-bottom: 2px solid #d1d5da;
            white-space: nowrap;
        }}
        
        td {{
            padding: 10px 8px;
            border-bottom: 1px solid #e1e4e8;
            vertical-align: top;
        }}
        
        tbody tr:hover {{
            background-color: #f6f8fa;
        }}
        
        a {{
            color: #0366d6;
            text-decoration: none;
        }}
        
        a:hover {{
            text-decoration: underline;
        }}
        
        .summary-table {{
            width: auto;
            margin: 20px 0;
            display: table;
        }}
        
        .summary-table td {{
            padding: 8px 16px;
        }}
        
        .summary-table td:first-child {{
            font-weight: 600;
            color: #586069;
        }}
        
        .summary-table td:last-child {{
            color: #24292e;
        }}
        
        svg {{
            display: block;
            margin: 20px auto;
        }}
        
        /* Status badges */
        td:nth-child(5) {{ /* Draft column */
            text-align: center;
        }}
        
        td:nth-child(4) {{ /* Age column */
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
        
        /* CI status styling */
        td:last-child {{
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
        }}
        
        /* Make PR links bold */
        td:first-child a {{
            font-weight: 500;
        }}
        
        /* Responsive design */
        @media (max-width: 1200px) {{
            .container {{
                padding: 20px;
            }}
            
            table {{
                font-size: 0.85em;
            }}
            
            th, td {{
                padding: 8px 6px;
            }}
        }}
        
        @media print {{
            body {{
                background-color: white;
            }}
            
            .container {{
                box-shadow: none;
            }}
            
            table {{
                page-break-inside: auto;
            }}
            
            tr {{
                page-break-inside: avoid;
                page-break-after: auto;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
{html_body}
    </div>
</body>
</html>"""

    # Write HTML file
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_template)

    print(f"✓ Converted {md_file} to {html_file}")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    md_file = script_dir / "pr_report.md"
    html_file = script_dir / "pr_report.html"

    convert_md_to_html(md_file, html_file)
