#!/usr/bin/env python3
"""
Daily PR report generator.

Writes scripts/pr_report/pr_report.html with a styled HTML dashboard and
scripts/pr_report/github_mentions.txt with GitHub usernames for @mentions.
"""

import base64
import html
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend — must be set before importing pyplot
import matplotlib.pyplot as plt

API = "https://api.github.com"
ACCEPT = "application/vnd.github+json"

# ── GitHub API helpers ────────────────────────────────────────────────────────


def gh_request(path, token, params=None):
    """
    Make a single GitHub API request with up to 3 retries on rate-limit errors.
    Returns (parsed_json, headers).
    """
    url = API + path
    if params:
        url += "?" + urllib.parse.urlencode(params)

    req = urllib.request.Request(url)
    req.add_header("Accept", ACCEPT)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8")), resp.headers
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            # Retry on rate-limit (403 with rate-limit body, or 429)
            if e.code == 429 or (e.code == 403 and "rate limit" in body.lower()):
                wait = 60 * (attempt + 1)
                print(
                    f"Rate limited on {path} (attempt {attempt + 1}/3), waiting {wait}s …",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            print(f"HTTP {e.code} for {path}: {body[:300]}", file=sys.stderr)
            raise
        except urllib.error.URLError as e:
            print(f"URL error for {path}: {e.reason}", file=sys.stderr)
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            raise

    raise RuntimeError(f"GitHub API request failed after 3 retries: {path}")


def paginate(path, token, params=None):
    """
    Fetch all pages from a GitHub list endpoint.
    Uses the Link header (rel="next") for correct pagination — avoids the
    off-by-one bug of stopping when len(chunk) == 100.
    """
    page = 1
    out = []
    while True:
        p = dict(params or {})
        p.update({"per_page": 100, "page": page})
        chunk, headers = gh_request(path, token, p)
        if not chunk:
            break
        out.extend(chunk)
        # Stop only when GitHub says there is no next page
        if 'rel="next"' not in (headers.get("Link") or ""):
            break
        page += 1
    return out


def paginate_check_runs(path, token, params=None):
    """
    Paginate the check-runs endpoint, which wraps results in
    {"check_runs": [...], "total_count": N} instead of a plain list.
    """
    page = 1
    out = []
    while True:
        p = dict(params or {})
        p.update({"per_page": 100, "page": page})
        resp, headers = gh_request(path, token, p)
        chunk = resp.get("check_runs", [])
        out.extend(chunk)
        if 'rel="next"' not in (headers.get("Link") or ""):
            break
        page += 1
    return out


# ── Utility helpers ───────────────────────────────────────────────────────────


def parse_iso(dt):
    return datetime.fromisoformat(dt.replace("Z", "+00:00"))


def is_bot(username):
    """Filter out GitHub bot accounts (e.g. github-actions[bot], dependabot[bot])."""
    return "[bot]" in username


def summarize_reviews(reviews):
    """
    Keep the latest meaningful review state per human reviewer.
    Bot accounts are excluded.
    States: APPROVED, CHANGES_REQUESTED, COMMENTED, DISMISSED, PENDING
    """
    latest = {}
    for r in sorted(reviews, key=lambda x: x.get("submitted_at") or ""):
        user = (r.get("user") or {}).get("login", "unknown")
        if is_bot(user):
            continue
        state = r.get("state", "UNKNOWN")
        latest[user] = state

    approvers = sorted([u for u, s in latest.items() if s == "APPROVED"])
    changers = sorted([u for u, s in latest.items() if s == "CHANGES_REQUESTED"])
    commenters = sorted([u for u, s in latest.items() if s == "COMMENTED"])
    dismissed = sorted([u for u, s in latest.items() if s == "DISMISSED"])

    return {
        "approvers": approvers,
        "changes_requested": changers,
        "commenters": commenters,
        "dismissed": dismissed,
        "latest_map": latest,
    }


def determine_pending_with(pr, reviews, reviews_summary, requested_reviewers):
    """
    Determine who the PR is currently pending with, based on its state.

    Rules (in priority order):
    1. Draft → author (still being worked on)
    2. No reviews yet, reviewers assigned → requested reviewers
    3. No reviews yet, no reviewers assigned → author
    4. Changes requested AND no new commits since the review (unresolved) → author
    5. Changes requested AND author pushed new commits after the review (resolved) → reviewer(s) who requested changes
    6. All approved, no outstanding change requests → author (ready to merge)
    7. Only comments → requested reviewers if any, else author

    "Resolved" is detected by comparing the PR's current head SHA against the
    commit_id recorded on the last CHANGES_REQUESTED review for each reviewer.
    If head_sha != that commit_id, the author has pushed new commits since the
    review — meaning they have addressed the feedback.
    """
    author = (pr.get("user") or {}).get("login", "unknown")
    is_draft = pr.get("draft", False)
    head_sha = (pr.get("head") or {}).get("sha", "")

    # 1. Draft → author
    if is_draft:
        return author

    changes_requesters = reviews_summary["changes_requested"]
    approvers = reviews_summary["approvers"]

    # 2 & 3. No reviews yet
    if not changes_requesters and not approvers and not reviews_summary["commenters"]:
        if requested_reviewers:
            return ", ".join(requested_reviewers)
        return author

    # 4 & 5. Outstanding change requests
    if changes_requesters:
        # For each reviewer whose latest state is CHANGES_REQUESTED, find the
        # commit_id of their most recent CHANGES_REQUESTED review.
        last_cr_commit_per_reviewer = {}
        for r in sorted(reviews, key=lambda x: x.get("submitted_at") or ""):
            user = (r.get("user") or {}).get("login", "unknown")
            if is_bot(user):
                continue
            if r.get("state") == "CHANGES_REQUESTED":
                last_cr_commit_per_reviewer[user] = r.get("commit_id", "")

        # Split reviewers into "resolved" (new commits pushed) vs "unresolved"
        resolved_reviewers = []
        unresolved_reviewers = []
        for reviewer in changes_requesters:
            cr_commit = last_cr_commit_per_reviewer.get(reviewer, "")
            if cr_commit and head_sha and cr_commit != head_sha:
                resolved_reviewers.append(reviewer)
            else:
                unresolved_reviewers.append(reviewer)

        if unresolved_reviewers:
            # At least one reviewer's changes haven't been addressed yet → author
            return author
        else:
            # All change requests have new commits pushed after them → pending re-review
            return ", ".join(resolved_reviewers)

    # 6. All approved, no outstanding change requests → author (ready to merge)
    if approvers and not changes_requesters:
        return author

    # 7. Only comments → requested reviewers if any, else author
    if requested_reviewers:
        return ", ".join(requested_reviewers)
    return author


def classify_check_runs(check_runs):
    """
    Return a list of (name, state) tuples for each check run.
    state is one of: PASS, FAIL, PENDING, or the raw conclusion uppercased.
    """
    if not check_runs:
        return []

    results = []
    for cr in sorted(check_runs, key=lambda x: x.get("name", "")):
        name = cr.get("name", "unknown")
        status = cr.get("status")
        conclusion = cr.get("conclusion")

        if status != "completed" or conclusion is None:
            state = "PENDING"
        elif conclusion in ("failure", "cancelled", "timed_out", "action_required", "stale"):
            state = "FAIL"
        elif conclusion in ("success", "neutral", "skipped"):
            state = "PASS"
        else:
            state = conclusion.upper()

        results.append((name, state))

    return results


# ── Fetch recently closed/merged PRs ─────────────────────────────────────────


def fetch_recent_closed_prs(owner, repo, token, days=10):
    """
    Fetch PRs that were closed (merged or not) within the last `days` days.
    Uses the GitHub search API to find PRs closed in the date range.
    Returns a list of PR dicts (lightweight — only fields available in list endpoint).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out = []
    page = 1
    while True:
        params = {
            "state": "closed",
            "sort": "updated",
            "direction": "desc",
            "per_page": 100,
            "page": page,
        }
        chunk, headers = gh_request(f"/repos/{owner}/{repo}/pulls", token, params)
        if not chunk:
            break

        stop_after_page = False
        for pr in chunk:
            closed_at_str = pr.get("closed_at")
            if closed_at_str and parse_iso(closed_at_str) >= cutoff:
                out.append(pr)

        # If the oldest `updated_at` in this page is before the cutoff, no
        # further pages can contain recently-closed PRs.
        if chunk:
            oldest_updated = min(
                parse_iso(pr["updated_at"])
                for pr in chunk
                if pr.get("updated_at")
            )
            if oldest_updated < cutoff:
                stop_after_page = True

        if stop_after_page or 'rel="next"' not in (headers.get("Link") or ""):
            break
        page += 1

    return out


# ── Trend chart helper ────────────────────────────────────────────────────────


def generate_trend_chart_png(pulls_open, pulls_closed, now, days=10, png_path=None):
    """
    Generate two stacked line charts:
      - Top subplot:    PRs Opened — Last N Days  (blue, area fill)
      - Bottom subplot: PRs Merged — Last N Days  (green, area fill)

    Each data point is annotated with its count.
    Saves the PNG to png_path if provided.
    Returns an HTML string with the chart embedded as a base64 data URI.
    """
    # Build the date range: oldest → today
    dates = [(now - timedelta(days=i)).date() for i in range(days - 1, -1, -1)]
    date_set = set(dates)

    opened_counts = {d: 0 for d in dates}
    merged_counts = {d: 0 for d in dates}

    # Opened: count from ALL PRs (open + recently closed) by created_at
    for pr in list(pulls_open) + list(pulls_closed):
        created_str = pr.get("created_at")
        if created_str:
            d = parse_iso(created_str).date()
            if d in date_set:
                opened_counts[d] += 1

    # Merged: from closed PRs only
    for pr in pulls_closed:
        merged_at_str = pr.get("merged_at")
        if merged_at_str:
            d = parse_iso(merged_at_str).date()
            if d in date_set:
                merged_counts[d] += 1

    x_labels = [d.strftime("%b %d") for d in dates]
    opened_vals = [opened_counts[d] for d in dates]
    merged_vals = [merged_counts[d] for d in dates]

    # ── Shared style helpers ──────────────────────────────────────────────────
    def _style_ax(ax, title, vals, line_color, fill_color):
        """Draw a single styled subplot with area fill and data labels."""
        x_idx = range(len(x_labels))
        ax.plot(x_idx, vals, marker="o", color=line_color, linewidth=2,
                markersize=5, zorder=3)
        ax.fill_between(x_idx, vals, alpha=0.15, color=fill_color)

        # Data labels above each point
        for xi, v in enumerate(vals):
            ax.annotate(
                str(v),
                xy=(xi, v),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#1a1a2e",
            )

        ax.set_title(title, fontsize=11, fontweight="bold", color="#1a1a2e",
                     loc="left", pad=10)
        ax.set_xticks(list(x_idx))
        ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(bottom=0, top=max(max(vals) * 1.35, 1))
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.set_facecolor("#f8f9fb")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig, (ax_open, ax_merge) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=False
    )
    fig.suptitle("PR Activity Trend", fontsize=14, fontweight="bold",
                 color="#1a1a2e", y=0.98)

    _style_ax(ax_open,  f"PRs Opened — Last {days} Days",
              opened_vals, "#0366d6", "#0366d6")
    _style_ax(ax_merge, f"PRs Merged — Last {days} Days",
              merged_vals, "#2ecc71", "#2ecc71")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # ── Save PNG file ─────────────────────────────────────────────────────────
    if png_path is not None:
        fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
        print(f"Saved trend chart PNG to {png_path}", file=sys.stderr)

    # ── Encode as base64 for inline embedding ─────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return (
        f'<div class="trend-charts">\n'
        f'<img src="data:image/png;base64,{img_b64}" '
        f'alt="PR Activity Trend — Last {days} Days" '
        f'style="max-width:100%;height:auto;">\n'
        f'</div>\n'
    )


# ── Pie chart helper ──────────────────────────────────────────────────────────


def generate_pie_chart_png(author_counts, png_path=None):
    """
    Generate a pie chart as a PNG using matplotlib.

    Saves the PNG to png_path (a Path or str) if provided.
    Returns an HTML string: a <div> wrapping an <img> with the chart
    embedded as a base64 data URI — supported by all major email clients
    (Gmail, Outlook web, Apple Mail, etc.), unlike inline SVG which is
    stripped by email clients for security reasons.
    """
    if not author_counts:
        return ""

    # Sort by count descending so the largest slice starts at the top
    items = sorted(author_counts.items(), key=lambda x: -x[1])
    labels = [author for author, _ in items]
    sizes = [count for _, count in items]
    total = sum(sizes)

    # 15-colour palette; cycles if there are more authors
    colors = [
        "#4a90d9", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
        "#1abc9c", "#e67e22", "#3498db", "#e91e63", "#00bcd4",
        "#ff5722", "#607d8b", "#795548", "#9c27b0", "#4caf50",
    ]
    chart_colors = [colors[i % len(colors)] for i in range(len(items))]

    fig, ax = plt.subplots(figsize=(13, 7))

    wedges, _, autotexts = ax.pie(
        sizes,
        colors=chart_colors,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        pctdistance=0.78,
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("white")
        at.set_fontweight("bold")

    ax.set_title(
        f"PR Distribution by Author (Total: {total})",
        fontsize=14,
        fontweight="bold",
        pad=20,
        color="#1a1a2e",
    )

    legend_labels = [
        f"{author}  {count} PR{'s' if count != 1 else ''}  ({count / total * 100:.1f}%)"
        for author, count in items
    ]
    ax.legend(
        wedges,
        legend_labels,
        title="Author",
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        fontsize=9,
        title_fontsize=10,
    )

    plt.tight_layout()

    # ── Save PNG file (used by Jenkins archiveArtifacts) ─────────────────────
    if png_path is not None:
        fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
        print(f"Saved pie chart PNG to {png_path}", file=sys.stderr)

    # ── Encode as base64 for inline embedding in HTML ────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return (
        f'<div class="chart-container">\n'
        f'<img src="data:image/png;base64,{img_b64}" '
        f'alt="PR Distribution by Author" '
        f'style="max-width:100%;height:auto;">\n'
        f'</div>\n'
    )


# ── HTML rendering helpers ────────────────────────────────────────────────────


def ci_badge(name, state):
    """Return an HTML badge <span> for a single CI check run."""
    colors = {
        "PASS":    ("#1a7f37", "#dafbe1"),   # dark-green text, light-green bg
        "FAIL":    ("#cf222e", "#ffebe9"),   # red text, light-red bg
        "PENDING": ("#9a6700", "#fff8c5"),   # amber text, light-yellow bg
    }
    text_color, bg_color = colors.get(state, ("#24292e", "#f6f8fa"))
    safe_name = html.escape(name)
    safe_state = html.escape(state)
    return (
        f'<span class="badge" style="color:{text_color};background:{bg_color};">'
        f'{safe_name}: {safe_state}</span>'
    )


def review_badge(label, users, text_color, bg_color):
    """Return an HTML badge group for a review category."""
    if not users:
        return ""
    safe_label = html.escape(label)
    safe_users = html.escape(", ".join(users))
    return (
        f'<div class="review-group">'
        f'<span class="badge" style="color:{text_color};background:{bg_color};">'
        f'{safe_label}</span> '
        f'<span class="review-users">{safe_users}</span>'
        f'</div>'
    )


def build_html(
    repo_full,
    date_str,
    total_open,
    pie_chart_html,
    trend_chart_html,
    rows_html,
    draft_count=0,
    opened_7d=0,
    merged_7d=0,
    closed_7d=0,
):
    """Assemble the complete HTML document."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Open PR Dashboard — {html.escape(repo_full)}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                   Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif;
      line-height: 1.6;
      color: #24292e;
      background: #f6f8fa;
      padding: 24px;
    }}

    .container {{
      max-width: 1500px;
      margin: 0 auto;
      background: #fff;
      padding: 32px 36px;
      border-radius: 10px;
      box-shadow: 0 2px 8px rgba(0,0,0,.12);
    }}

    h1 {{
      font-size: 1.75em;
      color: #1a1a2e;
      border-bottom: 3px solid #0366d6;
      padding-bottom: 10px;
      margin-bottom: 20px;
    }}

    /* ── Summary stat cards ── */
    .stat-strip {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 28px;
    }}
    .stat-card {{
      background: #f6f8fa;
      border: 1px solid #e1e4e8;
      border-radius: 8px;
      padding: 10px 18px;
      display: flex;
      flex-direction: column;
      min-width: 140px;
    }}
    .stat-label {{
      font-size: 0.78em;
      font-weight: 600;
      color: #586069;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 4px;
    }}
    .stat-value {{
      font-size: 1.15em;
      font-weight: 700;
      color: #1a1a2e;
    }}

    /* ── Charts layout: pie left, trend right ── */
    .charts-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 28px;
      align-items: flex-start;
      margin: 24px 0;
    }}
    .chart-left {{
      flex: 1 1 520px;
      min-width: 0;
    }}
    .chart-right {{
      flex: 2 1 640px;
      min-width: 0;
    }}
    .chart-container,
    .trend-charts {{
      overflow-x: auto;
    }}

    /* ── PR table ── */
    .pr-table-wrapper {{
      overflow-x: auto;
      margin-top: 24px;
    }}

    table.pr-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.875em;
    }}

    table.pr-table thead {{
      background: #f6f8fa;
      position: sticky;
      top: 0;
      z-index: 10;
    }}

    table.pr-table th {{
      padding: 10px 10px;
      text-align: left;
      font-weight: 600;
      color: #24292e;
      border-bottom: 2px solid #d1d5da;
      white-space: nowrap;
    }}

    table.pr-table th.num {{ text-align: right; }}

    table.pr-table td {{
      padding: 9px 10px;
      border-bottom: 1px solid #e1e4e8;
      vertical-align: top;
    }}

    table.pr-table tbody tr:hover {{ background: #f6f8fa; }}

    td.age {{ text-align: right; font-variant-numeric: tabular-nums; }}

    td.draft-yes {{ color: #9a6700; font-weight: 600; }}
    td.draft-no  {{ color: #57606a; }}

    a {{ color: #0366d6; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}

    .pr-link {{ font-weight: 600; }}

    /* ── Badges ── */
    .badge {{
      display: inline-block;
      padding: 1px 7px;
      border-radius: 12px;
      font-size: 0.78em;
      font-weight: 600;
      white-space: nowrap;
      margin: 1px 2px 1px 0;
    }}

    /* ── CI checks cell ── */
    td.ci-cell {{
      font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
      font-size: 0.8em;
    }}
    td.ci-cell .badge {{ font-family: inherit; }}

    /* ── Review summary cell ── */
    .review-group {{
      margin-bottom: 3px;
    }}
    .review-users {{
      font-size: 0.9em;
      color: #57606a;
    }}

    /* ── No-data row ── */
    .no-data {{
      text-align: center;
      color: #57606a;
      padding: 24px;
      font-style: italic;
    }}

    /* ── Footer ── */
    .footer {{
      margin-top: 32px;
      font-size: 0.8em;
      color: #8b949e;
      text-align: right;
    }}

    @media (max-width: 900px) {{
      .container {{ padding: 16px; }}
      table.pr-table {{ font-size: 0.8em; }}
      table.pr-table th, table.pr-table td {{ padding: 7px 6px; }}
    }}

    @media print {{
      body {{ background: #fff; }}
      .container {{ box-shadow: none; }}
      table.pr-table {{ page-break-inside: auto; }}
      tr {{ page-break-inside: avoid; }}
    }}
  </style>
</head>
<body>
<div class="container">

  <h1>Open PR Dashboard — {html.escape(repo_full)}</h1>

  <!-- Stat cards — table layout for email-client compatibility (flex is stripped) -->
  <table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom:24px;">
    <tr>
      <td style="padding:4px;">
        <table cellpadding="10" cellspacing="0" border="0" style="background:#f6f8fa;border:1px solid #e1e4e8;border-radius:8px;min-width:130px;">
          <tr><td><span class="stat-label">&#128197; Report Date</span><br><span class="stat-value">{html.escape(date_str)}</span></td></tr>
        </table>
      </td>
      <td style="padding:4px;">
        <table cellpadding="10" cellspacing="0" border="0" style="background:#f6f8fa;border:1px solid #e1e4e8;border-radius:8px;min-width:130px;">
          <tr><td><span class="stat-label">&#128230; Repository</span><br><span class="stat-value">{html.escape(repo_full)}</span></td></tr>
        </table>
      </td>
      <td style="padding:4px;">
        <table cellpadding="10" cellspacing="0" border="0" style="background:#f6f8fa;border:1px solid #e1e4e8;border-radius:8px;min-width:130px;">
          <tr><td><span class="stat-label">&#128275; Open PRs</span><br><span class="stat-value">{total_open}</span></td></tr>
        </table>
      </td>
      <td style="padding:4px;">
        <table cellpadding="10" cellspacing="0" border="0" style="background:#f6f8fa;border:1px solid #e1e4e8;border-radius:8px;min-width:130px;">
          <tr><td><span class="stat-label">&#128221; Draft PRs</span><br><span class="stat-value">{draft_count}</span></td></tr>
        </table>
      </td>
      <td style="padding:4px;">
        <table cellpadding="10" cellspacing="0" border="0" style="background:#f6f8fa;border:1px solid #e1e4e8;border-radius:8px;min-width:130px;">
          <tr><td><span class="stat-label">&#128640; Opened (7 days)</span><br><span class="stat-value">{opened_7d}</span></td></tr>
        </table>
      </td>
      <td style="padding:4px;">
        <table cellpadding="10" cellspacing="0" border="0" style="background:#f6f8fa;border:1px solid #e1e4e8;border-radius:8px;min-width:130px;">
          <tr><td><span class="stat-label">&#9989; Merged (7 days)</span><br><span class="stat-value">{merged_7d}</span></td></tr>
        </table>
      </td>
      <td style="padding:4px;">
        <table cellpadding="10" cellspacing="0" border="0" style="background:#f6f8fa;border:1px solid #e1e4e8;border-radius:8px;min-width:130px;">
          <tr><td><span class="stat-label">&#128683; Closed (7 days)</span><br><span class="stat-value">{closed_7d}</span></td></tr>
        </table>
      </td>
    </tr>
  </table>

  <!-- Charts row — table layout for email-client compatibility (flex is stripped) -->
  <table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin:24px 0;">
    <tr>
      <td width="40%" valign="top" style="padding-right:14px;">{pie_chart_html}</td>
      <td width="60%" valign="top">{trend_chart_html}</td>
    </tr>
  </table>

  <div class="pr-table-wrapper">
    <table class="pr-table">
      <thead>
        <tr>
          <th>PR</th>
          <th>Author</th>
          <th>Assignee</th>
          <th class="num">Age (days)</th>
          <th>Draft</th>
          <th>Labels</th>
          <th>Reviewers</th>
          <th>Pending With</th>
          <th>Review Summary</th>
          <th>CI Checks</th>
        </tr>
      </thead>
      <tbody>
        {rows_html if rows_html else '<tr><td colspan="10" class="no-data">No open pull requests.</td></tr>'}
      </tbody>
    </table>
  </div>

  <div class="footer">Generated by pr_dashboard.py · {html.escape(date_str)}</div>

</div>
</body>
</html>"""


# ── Email/Username mapping ───────────────────────────────────────────────────


def load_github_usernames():
    """
    Load email-to-GitHub-username mapping from email_map.json.
    Returns a list of GitHub usernames to @mention in the issue.
    """
    script_dir = Path(__file__).parent
    email_map_file = os.environ.get("EMAIL_MAP_FILE", script_dir / "email_map.json")

    try:
        with open(email_map_file, "r") as f:
            email_map = json.load(f)

        # Handle both list format (old) and dict format (new)
        if isinstance(email_map, list):
            # Old format: just emails, no usernames available
            print(
                "Warning: email_map.json is in old format (list). "
                "Please update to dict format: {email: username}",
                file=sys.stderr,
            )
            return []
        elif isinstance(email_map, dict):
            # New format: {email: username}
            usernames = [username for username in email_map.values() if username]
            return usernames
        else:
            print(
                f"Warning: email_map.json has unexpected format: {type(email_map)}",
                file=sys.stderr,
            )
            return []
    except FileNotFoundError:
        print(
            f"Warning: {email_map_file} not found. No @mentions will be added.",
            file=sys.stderr,
        )
        return []
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse {email_map_file}: {e}", file=sys.stderr)
        return []


def write_mentions_file(usernames):
    """
    Write GitHub usernames to a file for the workflow to consume.
    """
    script_dir = Path(__file__).parent
    mentions_file = script_dir / "github_mentions.txt"

    try:
        with open(mentions_file, "w") as f:
            for username in usernames:
                f.write(f"@{username}\n")
        print(f"Wrote {len(usernames)} username(s) to {mentions_file}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Failed to write mentions file: {e}", file=sys.stderr)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Missing GITHUB_TOKEN", file=sys.stderr)
        sys.exit(1)

    repo_full = os.environ.get("GITHUB_REPOSITORY")  # owner/repo
    if not repo_full or "/" not in repo_full:
        print("Missing/invalid GITHUB_REPOSITORY", file=sys.stderr)
        sys.exit(1)

    owner, repo = repo_full.split("/", 1)
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%B %d, %Y  %H:%M UTC")

    # 1) Fetch all open PRs (correctly paginated via Link header)
    pulls = paginate(f"/repos/{owner}/{repo}/pulls", token, params={"state": "open"})
    total_open = len(pulls)

    print(f"Fetched {total_open} open PR(s) for {repo_full}", file=sys.stderr)

    script_dir = Path(__file__).parent

    # -- Fetch recently closed/merged PRs (needed for trend chart + stat cards)
    print("Fetching recently closed/merged PRs …", file=sys.stderr)
    pulls_closed = fetch_recent_closed_prs(owner, repo, token, days=10)
    print(f"Fetched {len(pulls_closed)} recently closed PR(s)", file=sys.stderr)

    # -- Compute stat-card values ---------------------------------------------
    cutoff_7d = now - timedelta(days=7)

    draft_count = sum(1 for pr in pulls if pr.get("draft", False))

    # Opened in last 7 days: from both open PRs and recently closed PRs
    opened_7d = sum(
        1 for pr in list(pulls) + list(pulls_closed)
        if pr.get("created_at") and parse_iso(pr["created_at"]) >= cutoff_7d
    )
    # Deduplicate (a PR could appear in both lists if it was opened and closed
    # within the last 7 days AND is still in the open list — shouldn't happen,
    # but guard with a set of PR numbers)
    seen_opened: set = set()
    opened_7d = 0
    for pr in list(pulls) + list(pulls_closed):
        num = pr.get("number")
        if num in seen_opened:
            continue
        seen_opened.add(num)
        if pr.get("created_at") and parse_iso(pr["created_at"]) >= cutoff_7d:
            opened_7d += 1

    merged_7d = sum(
        1 for pr in pulls_closed
        if pr.get("merged_at") and parse_iso(pr["merged_at"]) >= cutoff_7d
    )
    closed_7d = sum(
        1 for pr in pulls_closed
        if not pr.get("merged_at")
        and pr.get("closed_at")
        and parse_iso(pr["closed_at"]) >= cutoff_7d
    )

    # -- Pie chart (author distribution) — collected in first pass ------------
    author_counts: dict = {}
    for pr in pulls:
        author = (pr.get("user") or {}).get("login", "unknown")
        if not is_bot(author):
            author_counts[author] = author_counts.get(author, 0) + 1

    # Generate PNG chart — saved as pie_chart.png AND embedded as base64 <img>
    png_file = script_dir / "pie_chart.png"
    pie_chart_html = generate_pie_chart_png(author_counts, png_path=png_file)

    # -- Trend chart (opened / merged / closed per day) -----------------------
    trend_png_file = script_dir / "trend_chart.png"
    trend_chart_html = generate_trend_chart_png(
        pulls_open=pulls,
        pulls_closed=pulls_closed,
        now=now,
        days=10,
        png_path=trend_png_file,
    )

    # -- Build PR table rows --------------------------------------------------
    row_parts = []

    for pr in pulls:
        number = pr["number"]
        title = pr.get("title", "")
        url = pr.get("html_url", "")
        author = (pr.get("user") or {}).get("login", "unknown")
        is_draft = pr.get("draft", False)
        created_at = parse_iso(pr["created_at"])
        age_days = (now - created_at).days
        head_sha = (pr.get("head") or {}).get("sha")

        # Assignees (already in PR payload — no extra API call)
        assignees = [u["login"] for u in pr.get("assignees") or [] if not is_bot(u["login"])]
        assignee_str = html.escape(", ".join(assignees)) if assignees else "—"

        # Labels (already in PR payload — no extra API call)
        labels = [lbl["name"] for lbl in pr.get("labels") or []]
        labels_html = (
            " ".join(
                f'<span class="badge" style="color:#24292e;background:#e1e4e8;">'
                f'{html.escape(lbl)}</span>'
                for lbl in labels
            )
            if labels
            else "—"
        )

        # 2) Requested reviewers
        rr, _ = gh_request(f"/repos/{owner}/{repo}/pulls/{number}/requested_reviewers", token)
        users = [u["login"] for u in rr.get("users", []) if not is_bot(u["login"])]
        teams = [t["name"] for t in rr.get("teams", [])]
        requested_reviewers = users + [f"team:{t}" for t in teams]
        reviewers_str = html.escape(", ".join(requested_reviewers)) if requested_reviewers else "—"

        # 3) Reviews submitted (paginated, bots excluded)
        reviews = paginate(f"/repos/{owner}/{repo}/pulls/{number}/reviews", token)
        rs = summarize_reviews(reviews)

        review_html_parts = []
        review_html_parts.append(
            review_badge("Changes Requested", rs["changes_requested"], "#cf222e", "#ffebe9")
        )
        review_html_parts.append(
            review_badge("Approved", rs["approvers"], "#1a7f37", "#dafbe1")
        )
        review_html_parts.append(
            review_badge("Commented", rs["commenters"], "#0550ae", "#ddf4ff")
        )
        review_html_parts.append(
            review_badge("Dismissed", rs["dismissed"], "#57606a", "#f6f8fa")
        )
        review_html_parts = [p for p in review_html_parts if p]
        review_summary_html = (
            "\n".join(review_html_parts) if review_html_parts else '<em>No reviews yet</em>'
        )

        # Pending With — smart assignment based on PR state
        pending_with_str = html.escape(
            determine_pending_with(pr, reviews, rs, requested_reviewers)
        )

        # 4) Individual CI check runs — fully paginated
        ci_html = '<span style="color:#57606a;">UNKNOWN</span>'
        if head_sha:
            check_runs = paginate_check_runs(
                f"/repos/{owner}/{repo}/commits/{head_sha}/check-runs",
                token,
                params={"filter": "latest"},
            )
            classified = classify_check_runs(check_runs)
            if classified:
                ci_html = " ".join(ci_badge(name, state) for name, state in classified)
            else:
                ci_html = '<span style="color:#57606a;">NONE</span>'

        # Draft styling
        draft_class = "draft-yes" if is_draft else "draft-no"
        draft_text = "Yes" if is_draft else "No"

        row_parts.append(
            f"<tr>"
            f'<td><a class="pr-link" href="{html.escape(url)}">#{number}</a>'
            f" {html.escape(title)}</td>"
            f"<td>{html.escape(author)}</td>"
            f"<td>{assignee_str}</td>"
            f'<td class="age">{age_days}</td>'
            f'<td class="{draft_class}">{draft_text}</td>'
            f"<td>{labels_html}</td>"
            f"<td>{reviewers_str}</td>"
            f"<td>{pending_with_str}</td>"
            f"<td>{review_summary_html}</td>"
            f'<td class="ci-cell">{ci_html}</td>'
            f"</tr>"
        )

        print(f"  Processed PR #{number}", file=sys.stderr)

    rows_html = "\n        ".join(row_parts)

    # -- Write HTML file ------------------------------------------------------
    html_file = script_dir / "pr_report.html"

    html_content = build_html(
        repo_full=repo_full,
        date_str=date_str,
        total_open=total_open,
        pie_chart_html=pie_chart_html,
        trend_chart_html=trend_chart_html,
        rows_html=rows_html,
        draft_count=draft_count,
        opened_7d=opened_7d,
        merged_7d=merged_7d,
        closed_7d=closed_7d,
    )

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Wrote HTML report to {html_file}", file=sys.stderr)

    # -- Write mentions file --------------------------------------------------
    usernames = load_github_usernames()
    write_mentions_file(usernames)


if __name__ == "__main__":
    main()
