#!/usr/bin/env python3
"""
Daily PR report generator.

Outputs a Markdown table to stdout and writes
scripts/git_workflow/recipients.txt with resolved email addresses.
"""

import json
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

    # -- Table ----------------------------------------------------------------
    print(
        "| PR | Author | Assignee | Age (days) | Draft | Labels | Pending With (requested reviewers) | Review Summary | CI Checks |"
    )
    print("|---|---|---|---:|:---:|---|---|---|---|")

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

        # 2) Requested reviewers (who it's pending with)
        rr, _ = gh_request(f"/repos/{owner}/{repo}/pulls/{number}/requested_reviewers", token)
        users = [u["login"] for u in rr.get("users", []) if not is_bot(u["login"])]
        teams = [t["name"] for t in rr.get("teams", [])]
        pending_with = users + [f"team:{t}" for t in teams]
        pending_with_str = ", ".join(pending_with) if pending_with else "—"

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
            f"| {pr_label} | {author} | {assignee_str} | {age_days} | {draft} | {labels_str} | {pending_with_str} | {review_summary} | {ci_str} |"
        )

    # -- Write recipients.txt -------------------------------------------------
    recipients_path = os.path.join(script_dir, "recipients.txt")
    with open(recipients_path, "w") as f:
        f.write(", ".join(recipients))

    print(f"recipients written to {recipients_path} ({len(recipients)} addresses)", file=sys.stderr)


if __name__ == "__main__":
    main()
