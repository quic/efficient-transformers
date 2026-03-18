# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

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


# ── Daily PR trend data ───────────────────────────────────────────────────────


def fetch_daily_pr_counts(owner, repo, token, days=10):
    """
    Return three ordered lists for the last `days` calendar days (UTC):
      opened_by_day       — (date_label, count) for PRs opened each day
      merged_by_day       — (date_label, count) for PRs merged each day
      closed_unmerged_by_day — (date_label, count) for PRs closed without merge

    Uses the GitHub Search API (/search/issues) — one request per metric per
    day (30 requests total for 10 days).
    """
    today = datetime.now(timezone.utc).date()
    opened_by_day = []
    merged_by_day = []
    closed_unmerged_by_day = []

    for offset in range(days - 1, -1, -1):  # oldest → newest
        day = today - timedelta(days=offset)
        day_str = day.strftime("%Y-%m-%d")
        label = day.strftime("%b %d")

        # PRs opened on this day
        q_opened = f"repo:{owner}/{repo} is:pr created:{day_str}"
        try:
            resp, _ = gh_request("/search/issues", token, {"q": q_opened, "per_page": 1})
            opened_by_day.append((label, resp.get("total_count", 0)))
        except Exception as exc:
            print(f"  Search (opened {day_str}) failed: {exc}", file=sys.stderr)
            opened_by_day.append((label, 0))
        time.sleep(0.5)  # stay well under 30 req/min

        # PRs merged on this day
        q_merged = f"repo:{owner}/{repo} is:pr is:merged merged:{day_str}"
        try:
            resp, _ = gh_request("/search/issues", token, {"q": q_merged, "per_page": 1})
            merged_by_day.append((label, resp.get("total_count", 0)))
        except Exception as exc:
            print(f"  Search (merged {day_str}) failed: {exc}", file=sys.stderr)
            merged_by_day.append((label, 0))
        time.sleep(0.5)

        # PRs closed WITHOUT merging on this day
        q_closed = f"repo:{owner}/{repo} is:pr is:unmerged is:closed closed:{day_str}"
        try:
            resp, _ = gh_request("/search/issues", token, {"q": q_closed, "per_page": 1})
            closed_unmerged_by_day.append((label, resp.get("total_count", 0)))
        except Exception as exc:
            print(f"  Search (closed {day_str}) failed: {exc}", file=sys.stderr)
            closed_unmerged_by_day.append((label, 0))
        time.sleep(0.5)

    return opened_by_day, merged_by_day, closed_unmerged_by_day


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


# ── Pie chart helper ──────────────────────────────────────────────────────────


# ── Chart top-margin constant — keeps pie title level with trend suptitle ────
_CHART_TOP = 0.91  # fraction of figure height used for content (title sits above)


def generate_pie_chart_img(author_counts):
    """
    Generate a pie chart showing PR distribution by author.

    Uses matplotlib to render a PNG and returns an HTML string containing
    an <img> tag with the chart embedded as a base64 data URI.  This format
    is universally supported by email clients, unlike inline SVG which is
    stripped by Gmail, Outlook, and most corporate mail clients.

    Falls back to an empty string if matplotlib is not available.
    """
    if not author_counts:
        return ""

    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend — no display required
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Warning: matplotlib not installed; pie chart will be omitted from the report.",
            file=sys.stderr,
        )
        return ""

    # Sort by count descending so the largest slice starts at the top
    items = sorted(author_counts.items(), key=lambda x: -x[1])
    sizes = [c for _, c in items]
    total = sum(sizes)

    # Tableau-10 extended palette — industry-standard professional colours
    colors = [
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#59A14F",
        "#EDC948",
        "#B07AA1",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AC",
        "#86BCB6",
        "#FFBE7D",
        "#8CD17D",
        "#B6992D",
        "#D37295",
    ]
    slice_colors = [colors[i % len(colors)] for i in range(len(items))]

    # Determine legend columns: 1 col for ≤10 authors, 2 cols otherwise
    ncols = 2 if len(items) > 10 else 1

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    fig.subplots_adjust(top=_CHART_TOP)

    wedges, _ = ax.pie(
        sizes,
        colors=slice_colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.58, "edgecolor": "white", "linewidth": 2},
    )

    # Centre label showing total
    ax.text(
        0,
        0,
        f"{total}\nPRs",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#1a1a2e",
        linespacing=1.4,
    )

    # Legend placed below the donut in 2 columns — eliminates side whitespace
    legend_labels = [
        f"{author}  ·  {count} PR{'s' if count != 1 else ''}  ({count / total * 100:.1f}%)" for author, count in items
    ]
    leg = ax.legend(
        wedges,
        legend_labels,
        title="Author",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=ncols,
        fontsize=10,
        title_fontsize=11,
        frameon=True,
        framealpha=1.0,
        edgecolor="#e1e4e8",
        handlelength=1.2,
        handleheight=1.2,
        borderpad=0.8,
        labelspacing=0.5,
        columnspacing=1.2,
    )
    leg.get_title().set_fontweight("bold")
    leg.get_title().set_color("#1a1a2e")

    ax.set_title(
        "PR Distribution by Author",
        fontsize=14,
        fontweight="bold",
        color="#1a1a2e",
        pad=14,
    )

    fig.tight_layout(rect=[0, 0, 1, _CHART_TOP])

    # Render to an in-memory PNG buffer and base64-encode it
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")

    return (
        '<div class="chart-container">\n'
        f'  <img src="data:image/png;base64,{b64}" '
        f'alt="PR Distribution by Author" '
        f'style="max-width:100%;height:auto;">\n'
        "</div>\n"
    )


# ── Trend line-chart helper ───────────────────────────────────────────────────


def generate_trend_charts_img(opened_by_day, merged_by_day, closed_unmerged_by_day=None):
    """
    Render three stacked line graphs (Opened / Merged / Closed-without-merge
    over the last 10 days) as a single base64-encoded PNG <img> tag.

    Email-safe: no SVG, no JavaScript — just a plain <img>.
    Returns an empty string if matplotlib is unavailable or data is empty.
    """
    if not opened_by_day and not merged_by_day:
        return ""
    if closed_unmerged_by_day is None:
        closed_unmerged_by_day = []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print(
            "Warning: matplotlib not installed; trend charts will be omitted.",
            file=sys.stderr,
        )
        return ""

    labels_o = [d for d, _ in opened_by_day]
    counts_o = [c for _, c in opened_by_day]
    labels_m = [d for d, _ in merged_by_day]
    counts_m = [c for _, c in merged_by_day]

    # Two graphs stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    fig.patch.set_facecolor("#ffffff")

    def _style_ax(ax, x_labels, y_values, line_color, title):
        xs = list(range(len(x_labels)))
        y_max = max(y_values) if any(v > 0 for v in y_values) else 1
        ax.fill_between(xs, y_values, alpha=0.15, color=line_color)
        ax.plot(
            xs,
            y_values,
            color=line_color,
            linewidth=2.5,
            marker="o",
            markersize=7,
            markerfacecolor=line_color,
            markeredgecolor="white",
            markeredgewidth=2,
            zorder=3,
        )
        for xi, yi in zip(xs, y_values):
            ax.annotate(
                str(yi),
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
                fontweight="bold",
                color="#333333",
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(x_labels, fontsize=9.5, rotation=0, ha="center")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
        ax.tick_params(axis="y", labelsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold", color="#1a1a2e", pad=10, loc="left")
        ax.set_facecolor("#f9fafb")
        ax.grid(axis="y", color="#e1e4e8", linewidth=0.9, linestyle="--", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#d1d5da")
        ax.spines["bottom"].set_color("#d1d5da")
        ax.set_ylim(bottom=0, top=y_max * 1.3 + 1)

    _style_ax(ax1, labels_o, counts_o, "#4E79A7", "PRs Opened — Last 10 Days")
    _style_ax(ax2, labels_m, counts_m, "#59A14F", "PRs Merged — Last 10 Days")

    fig.suptitle("PR Activity Trend", fontsize=13, fontweight="bold", color="#1a1a2e", y=_CHART_TOP + 0.01)
    fig.tight_layout(rect=[0, 0, 1, _CHART_TOP], pad=2.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")

    return (
        '<div class="trend-charts">\n'
        f'  <img src="data:image/png;base64,{b64}" '
        f'alt="PR Trend — Last 10 Days" '
        f'style="max-width:100%;height:auto;">\n'
        "</div>\n"
    )


# ── HTML rendering helpers ────────────────────────────────────────────────────


def ci_badge(name, state):
    """Return an HTML badge <span> for a single CI check run."""
    colors = {
        "PASS": ("#1a7f37", "#dafbe1"),  # dark-green text, light-green bg
        "FAIL": ("#cf222e", "#ffebe9"),  # red text, light-red bg
        "PENDING": ("#9a6700", "#fff8c5"),  # amber text, light-yellow bg
    }
    text_color, bg_color = colors.get(state, ("#24292e", "#f6f8fa"))
    safe_name = html.escape(name)
    safe_state = html.escape(state)
    return f'<span class="badge" style="color:{text_color};background:{bg_color};">{safe_name}: {safe_state}</span>'


def review_badge(label, users, text_color, bg_color):
    """Return an HTML badge group for a review category."""
    if not users:
        return ""
    safe_label = html.escape(label)
    safe_users = html.escape(", ".join(users))
    return (
        f'<div class="review-group">'
        f'<span class="badge" style="color:{text_color};background:{bg_color};">'
        f"{safe_label}</span> "
        f'<span class="review-users">{safe_users}</span>'
        f"</div>"
    )


def build_html(
    repo_full,
    date_str,
    total_open,
    draft_count,
    opened_last_7,
    merged_last_7,
    closed_last_7,
    pie_svg,
    trend_img,
    rows_html,
):
    """Assemble the complete HTML document.

    Uses table-based layout for the stat strip and charts row so the email
    renders correctly in clients (Gmail, Outlook, etc.) that strip CSS
    flexbox / grid.  Inline styles are applied to every layout-critical
    element so the report looks right even when the <style> block is ignored.
    """
    # Stat cards — built as <td> cells inside a single-row table so they sit
    # side-by-side in every email client (flexbox is not supported in email).
    stats = [
        ("📅 Report Date", date_str),
        ("📦 Repository", repo_full),
        ("🔓 Open PRs", str(total_open)),
        ("📝 Draft PRs", str(draft_count)),
        ("🚀 Opened (7 days)", str(opened_last_7)),
        ("✅ Merged (7 days)", str(merged_last_7)),
        ("🚫 Closed (7 days)", str(closed_last_7)),
    ]
    stat_cells_html = "\n      ".join(
        f'<td style="padding:0 6px 0 0;vertical-align:top;">'
        f'<div style="background:#f6f8fa;border:1px solid #e1e4e8;border-radius:8px;'
        f'padding:10px 18px;min-width:130px;">'
        f'<div style="font-size:0.78em;font-weight:600;color:#586069;text-transform:uppercase;'
        f'letter-spacing:0.04em;margin-bottom:4px;">{html.escape(label)}</div>'
        f'<div style="font-size:1.15em;font-weight:700;color:#1a1a2e;">{html.escape(value)}</div>'
        f"</div></td>"
        for label, value in stats
    )

    # Charts row — two-column table so pie and trend sit side-by-side.
    # Falls back gracefully to stacked layout on narrow screens.
    charts_row_html = (
        '<table width="100%" cellpadding="0" cellspacing="0" border="0" '
        'style="margin:24px 0;border-collapse:collapse;">\n'
        "  <tr>\n"
        '    <td width="40%" style="vertical-align:top;padding-right:14px;">\n'
        f"      {pie_svg}\n"
        "    </td>\n"
        '    <td width="60%" style="vertical-align:top;">\n'
        f"      {trend_img}\n"
        "    </td>\n"
        "  </tr>\n"
        "</table>"
    )

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

  <!-- Stat strip: table-based so it renders correctly in email clients
       that do not support CSS flexbox. -->
  <table cellpadding="0" cellspacing="0" border="0"
         style="border-collapse:collapse;margin-bottom:28px;">
    <tr>
      {stat_cells_html}
    </tr>
  </table>

  <!-- Charts: two-column table layout (email-safe, no flexbox). -->
  {charts_row_html}

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
                "Warning: email_map.json is in old format (list). Please update to dict format: {email: username}",
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

    # -- Pie chart (author distribution) — collected in first pass ------------
    author_counts: dict = {}
    for pr in pulls:
        author = (pr.get("user") or {}).get("login", "unknown")
        if not is_bot(author):
            author_counts[author] = author_counts.get(author, 0) + 1

    draft_count = sum(1 for pr in pulls if pr.get("draft", False))
    pie_svg = generate_pie_chart_img(author_counts)

    # -- Daily trend data (PRs opened / merged per day, last 10 days) ---------
    print("Fetching daily PR trend data …", file=sys.stderr)
    opened_by_day, merged_by_day, closed_unmerged_by_day = fetch_daily_pr_counts(owner, repo, token)
    trend_img = generate_trend_charts_img(opened_by_day, merged_by_day, closed_unmerged_by_day)

    # Last-7-days totals (sum of the last 7 entries in the 10-day lists)
    opened_last_7 = sum(c for _, c in opened_by_day[-7:])
    merged_last_7 = sum(c for _, c in merged_by_day[-7:])
    closed_last_7 = sum(c for _, c in closed_unmerged_by_day[-7:])

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
                f'<span class="badge" style="color:#24292e;background:#e1e4e8;">{html.escape(lbl)}</span>'
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
        review_html_parts.append(review_badge("Changes Requested", rs["changes_requested"], "#cf222e", "#ffebe9"))
        review_html_parts.append(review_badge("Approved", rs["approvers"], "#1a7f37", "#dafbe1"))
        review_html_parts.append(review_badge("Commented", rs["commenters"], "#0550ae", "#ddf4ff"))
        review_html_parts.append(review_badge("Dismissed", rs["dismissed"], "#57606a", "#f6f8fa"))
        review_html_parts = [p for p in review_html_parts if p]
        review_summary_html = "\n".join(review_html_parts) if review_html_parts else "<em>No reviews yet</em>"

        # Pending With — smart assignment based on PR state
        pending_with_str = html.escape(determine_pending_with(pr, reviews, rs, requested_reviewers))

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
    script_dir = Path(__file__).parent
    html_file = script_dir / "pr_report.html"

    html_content = build_html(
        repo_full,
        date_str,
        total_open,
        draft_count,
        opened_last_7,
        merged_last_7,
        closed_last_7,
        pie_svg,
        trend_img,
        rows_html,
    )

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Wrote HTML report to {html_file}", file=sys.stderr)

    # -- Write mentions file --------------------------------------------------
    usernames = load_github_usernames()
    write_mentions_file(usernames)


if __name__ == "__main__":
    main()
