#!/usr/bin/env python3
"""
Daily PR report generator.

Outputs a Markdown table to stdout and writes
scripts/git_workflow/recipients.txt with resolved email addresses.
"""

import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

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


def format_check_runs(check_runs):
    """
    Return each individual check run name and its status.
    Format: "job-name: PASS / job-name2: FAIL / ..."
    """
    if not check_runs:
        return "NONE"

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

        results.append(f"{name}: {state}")

    return " / ".join(results)


# ── Pie chart helper ──────────────────────────────────────────────────────────


def generate_pie_chart_svg(author_counts):
    """
    Generate a self-contained inline SVG pie chart showing PR distribution
    by author.  Returns an HTML string (a <div> wrapping an <svg>) that can
    be embedded directly in Markdown — the markdown library passes raw HTML
    blocks through unchanged.
    """
    if not author_counts:
        return ""

    # Sort by count descending so the largest slice starts at the top
    items = sorted(author_counts.items(), key=lambda x: -x[1])
    total = sum(v for _, v in items)

    # 15-colour palette; cycles if there are more authors
    colors = [
        "#4a90d9",
        "#e74c3c",
        "#2ecc71",
        "#f39c12",
        "#9b59b6",
        "#1abc9c",
        "#e67e22",
        "#3498db",
        "#e91e63",
        "#00bcd4",
        "#ff5722",
        "#607d8b",
        "#795548",
        "#9c27b0",
        "#4caf50",
    ]

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
            f"    <title>{author}: {count} PR{'s' if count != 1 else ''} ({pct:.1f}%)</title>\n"
            f"  </path>\n"
        )

        # Legend row
        ly = 40 + i * row_h
        legend_svg += (
            f'  <rect x="{legend_x}" y="{ly}" width="14" height="14" '
            f'fill="{color}" rx="2"/>\n'
            f'  <text x="{legend_x + 20}" y="{ly + 11}" '
            f'font-size="12" font-family="Arial, sans-serif" fill="#333">'
            f"{author}  {count} PR{'s' if count != 1 else ''}  ({pct:.1f}%)"
            f"</text>\n"
        )

        start_angle = end_angle

    # ── Assemble SVG ─────────────────────────────────────────────────────────
    svg = (
        f'<div style="margin:24px 0;">\n'
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
    return svg


# ── Email list helper ─────────────────────────────────────────────────────────


def load_email_list(path):
    """
    Load email_map.json — a plain JSON array of email addresses.
    Returns a list of strings.
    """
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"Warning: {path} should be a JSON array of email addresses.", file=sys.stderr)
            return []
        return [e for e in data if isinstance(e, str) and e.strip()]
    except FileNotFoundError:
        print(f"Warning: email list not found at {path}", file=sys.stderr)
        return []


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

    # Load recipient email list (path configurable via EMAIL_MAP_FILE env var)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_map = os.path.join(script_dir, "email_map.json")
    email_map_path = os.environ.get("EMAIL_MAP_FILE", default_map)
    recipients = load_email_list(email_map_path)

    # 1) Fetch all open PRs (correctly paginated via Link header)
    pulls = paginate(f"/repos/{owner}/{repo}/pulls", token, params={"state": "open"})
    total_open = len(pulls)

    # -- Header ---------------------------------------------------------------
    print(f"# Open PR Dashboard — {owner}/{repo}")
    print()
    print("| | |")
    print("|---|---|")
    print(f"| Report Date | {date_str} |")
    print(f"| Open PRs | **{total_open}** |")
    print()

    # -- Pie chart (author distribution) — collected in first pass ------------
    author_counts: dict = {}
    for pr in pulls:
        author = (pr.get("user") or {}).get("login", "unknown")
        if not is_bot(author):
            author_counts[author] = author_counts.get(author, 0) + 1

    print(generate_pie_chart_svg(author_counts))

    # -- Table ----------------------------------------------------------------
    print(
        "| PR | Author | Assignee | Age (days) | Draft | Labels | Reviewers | Pending With | Review Summary | CI Checks |"
    )
    print("|---|---|---|---:|:---:|---|---|---|---|---|")

    for pr in pulls:
        number = pr["number"]
        title = pr.get("title", "").replace("|", "\\|")
        url = pr.get("html_url", "")
        author = (pr.get("user") or {}).get("login", "unknown")
        draft = "Yes" if pr.get("draft") else "No"
        created_at = parse_iso(pr["created_at"])
        age_days = (now - created_at).days
        head_sha = (pr.get("head") or {}).get("sha")

        # Assignees (already in PR payload — no extra API call)
        assignees = [u["login"] for u in pr.get("assignees") or [] if not is_bot(u["login"])]
        assignee_str = ", ".join(assignees) if assignees else "—"

        # Labels (already in PR payload — no extra API call)
        labels = [lbl["name"].replace("|", "\\|") for lbl in pr.get("labels") or []]
        labels_str = ", ".join(labels) if labels else "—"

        # 2) Requested reviewers
        rr, _ = gh_request(f"/repos/{owner}/{repo}/pulls/{number}/requested_reviewers", token)
        users = [u["login"] for u in rr.get("users", []) if not is_bot(u["login"])]
        teams = [t["name"] for t in rr.get("teams", [])]
        requested_reviewers = users + [f"team:{t}" for t in teams]
        reviewers_str = ", ".join(requested_reviewers) if requested_reviewers else "—"

        # 3) Reviews submitted (paginated, bots excluded)
        reviews = paginate(f"/repos/{owner}/{repo}/pulls/{number}/reviews", token)
        rs = summarize_reviews(reviews)
        parts = []
        if rs["changes_requested"]:
            parts.append("Changes Requested: " + ", ".join(rs["changes_requested"]))
        if rs["approvers"]:
            parts.append("Approved: " + ", ".join(rs["approvers"]))
        if rs["commenters"]:
            parts.append("Commented: " + ", ".join(rs["commenters"]))
        if rs["dismissed"]:
            parts.append("Dismissed: " + ", ".join(rs["dismissed"]))
        if not parts:
            parts.append("No reviews yet")
        review_summary = " / ".join(parts)

        # Pending With — smart assignment based on PR state
        pending_with_str = determine_pending_with(pr, reviews, rs, requested_reviewers)

        # 4) Individual CI check runs — fully paginated
        ci_str = "UNKNOWN"
        if head_sha:
            check_runs = paginate_check_runs(
                f"/repos/{owner}/{repo}/commits/{head_sha}/check-runs",
                token,
                params={"filter": "latest"},
            )
            ci_str = format_check_runs(check_runs)

        pr_label = f"[#{number}]({url}) {title}"
        print(
            f"| {pr_label} | {author} | {assignee_str} | {age_days} | {draft} | {labels_str} | {reviewers_str} | {pending_with_str} | {review_summary} | {ci_str} |"
        )

    # -- Write recipients.txt -------------------------------------------------
    recipients_path = os.path.join(script_dir, "recipients.txt")
    with open(recipients_path, "w") as f:
        f.write(", ".join(recipients))

    print(f"recipients written to {recipients_path} ({len(recipients)} addresses)", file=sys.stderr)


if __name__ == "__main__":
    main()
