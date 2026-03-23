#!/usr/bin/env python3
"""
Daily PR report generator.

Writes scripts/pr_report/pr_report.html with a styled HTML dashboard and
scripts/pr_report/github_mentions.txt with GitHub usernames for @mentions.
"""

import html
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
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


def generate_pie_chart_png(author_counts, png_path=None):
    """
    Generate a self-contained inline SVG pie chart showing PR distribution
    by author.  Returns an HTML string (a <div> wrapping an <svg>).
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

    cx, cy, r = 190, 190, 160  # pie centre and radius
    legend_x = cx * 2 + 30  # legend column starts here
    row_h = 22  # legend row height
    svg_w = legend_x + 260  # total SVG width
    svg_h = max(cy * 2, len(items) * row_h + 50)  # total SVG height

    # ── Build slice paths ────────────────────────────────────────────────────
    paths_svg = ""
    legend_svg = ""
    start_angle = -math.pi / 2  # begin at 12 o'clock

    for i, (author, count) in enumerate(items):
        angle = 2 * math.pi * count / total
        end_angle = start_angle + angle

        x1 = cx + r * math.cos(start_angle)
        y1 = cy + r * math.sin(start_angle)
        x2 = cx + r * math.cos(end_angle)
        y2 = cy + r * math.sin(end_angle)

        large_arc = 1 if angle > math.pi else 0
        color = colors[i % len(colors)]
        pct = count / total * 100

        # SVG arc path: move to centre → line to arc start → arc → close
        path = f"M {cx},{cy} L {x1:.2f},{y1:.2f} A {r},{r} 0 {large_arc},1 {x2:.2f},{y2:.2f} Z"
        paths_svg += (
            f'  <path d="{path}" fill="{color}" '
            f'stroke="white" stroke-width="2">\n'
            f"    <title>{html.escape(author)}: {count} PR{'s' if count != 1 else ''} ({pct:.1f}%)</title>\n"
            f"  </path>\n"
        )

        # Legend row
        ly = 40 + i * row_h
        legend_svg += (
            f'  <rect x="{legend_x}" y="{ly}" width="14" height="14" '
            f'fill="{color}" rx="2"/>\n'
            f'  <text x="{legend_x + 20}" y="{ly + 11}" '
            f'font-size="12" font-family="Arial, sans-serif" fill="#333">'
            f"{html.escape(author)}  {count} PR{'s' if count != 1 else ''}  ({pct:.1f}%)"
            f"</text>\n"
        )

        start_angle = end_angle

    # ── Assemble SVG ─────────────────────────────────────────────────────────
    svg = (
        f'<div class="chart-container">\n'
        f'<svg width="{svg_w}" height="{svg_h}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family:Arial,sans-serif;">\n'
        # Chart title
        f'  <text x="{cx}" y="20" text-anchor="middle" '
        f'font-size="14" font-weight="bold" fill="#1a1a2e">'
        f"PR Distribution by Author (Total: {total})</text>\n"
        # Slices
        + paths_svg
        # Legend header
        + f'  <text x="{legend_x}" y="22" font-size="13" '
        f'font-weight="bold" fill="#1a1a2e">Author</text>\n'
        # Legend rows
         + legend_svg + "</svg>\n</div>\n"
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


def build_html(repo_full, date_str, total_open, pie_svg, rows_html):
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

    /* ── Summary table ── */
    .summary-table {{
      border-collapse: collapse;
      margin-bottom: 24px;
    }}
    .summary-table td {{
      padding: 6px 16px 6px 0;
      font-size: 0.95em;
    }}
    .summary-table td:first-child {{
      font-weight: 600;
      color: #586069;
      padding-right: 12px;
    }}

    /* ── Pie chart ── */
    .chart-container {{
      margin: 24px 0;
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

  <table class="summary-table">
    <tr>
      <td>Report Date</td>
      <td>{html.escape(date_str)}</td>
    </tr>
    <tr>
      <td>Open PRs</td>
      <td><strong>{total_open}</strong></td>
    </tr>
  </table>

  {pie_svg}

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

    pie_svg = generate_pie_chart_svg(author_counts)

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

    html_content = build_html(repo_full, date_str, total_open, pie_svg, rows_html)

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Wrote HTML report to {html_file}", file=sys.stderr)

    # -- Write mentions file --------------------------------------------------
    usernames = load_github_usernames()
    write_mentions_file(usernames)


if __name__ == "__main__":
    main()
