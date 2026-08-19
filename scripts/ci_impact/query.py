# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Read-only repository queries exposed to the external CI selector."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

MAX_OUTPUT_BYTES = 250_000
MAX_READ_LINES = 500
MAX_RESULTS = 500
GIT_TIMEOUT_SECONDS = 30
SHA_RE = re.compile(r"^[0-9a-f]{40,64}$")


class QueryError(RuntimeError):
    """Raised when a repository query is invalid or cannot be completed."""


def _inside_repo(repo: Path, path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(repo)
    except (OSError, ValueError) as error:
        raise QueryError(f"{label} must be an existing path inside the repository") from error
    return resolved


def _repo_path(value: str, *, allow_root: bool = False) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise QueryError(f"invalid repository path: {value!r}")
    normalized = path.as_posix()
    if normalized in {"", "."}:
        if allow_root:
            return "."
        raise QueryError("repository path must not be empty")
    return normalized


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QueryError(f"cannot read {label}: {error}") from error
    if not isinstance(payload, dict):
        raise QueryError(f"{label} must contain a JSON object")
    return payload


class QueryContext:
    def __init__(self) -> None:
        try:
            repo_value = os.environ["QEFF_CI_QUERY_REPO"]
            plan_value = os.environ["QEFF_CI_QUERY_PLAN"]
            catalog_value = os.environ["QEFF_CI_QUERY_CATALOG"]
        except KeyError as error:
            raise QueryError(f"missing query environment variable: {error.args[0]}") from error

        self.repo = Path(repo_value).resolve(strict=True)
        if not (self.repo / ".git").exists():
            raise QueryError("query repository is not a Git checkout")
        self.plan_path = _inside_repo(self.repo, Path(plan_value), "deterministic plan")
        self.catalog_path = _inside_repo(self.repo, Path(catalog_value), "pytest catalog")
        self.plan = _load_json(self.plan_path, "deterministic plan")
        self.catalog = _load_json(self.catalog_path, "pytest catalog")

        self.base = self.plan.get("base")
        self.head = self.plan.get("head")
        if not isinstance(self.base, str) or not SHA_RE.fullmatch(self.base):
            raise QueryError("deterministic plan has an invalid base revision")
        if not isinstance(self.head, str) or not SHA_RE.fullmatch(self.head):
            raise QueryError("deterministic plan has an invalid head revision")

    def revision(self, name: str) -> str:
        return self.base if name == "base" else self.head

    def git(self, *arguments: str, allow_no_matches: bool = False) -> str:
        command = ["git", "--no-pager", "-c", "color.ui=false", *arguments]
        try:
            process = subprocess.run(
                command,
                cwd=self.repo,
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
                timeout=GIT_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise QueryError(f"Git query failed: {error}") from error
        if process.returncode != 0 and not (allow_no_matches and process.returncode == 1):
            details = process.stderr.strip()[-2000:] or f"exit code {process.returncode}"
            raise QueryError(f"Git query failed: {details}")
        return process.stdout


def _changes(context: QueryContext, _args: argparse.Namespace) -> dict[str, object]:
    output = context.git("diff", "--name-status", "--find-renames", f"{context.base}...{context.head}")
    changes = []
    for line in output.splitlines():
        fields = line.split("\t")
        status = fields[0]
        if status.startswith("R") and len(fields) == 3:
            changes.append({"status": "R", "old_path": fields[1], "path": fields[2]})
        elif len(fields) == 2:
            changes.append({"status": status[0], "path": fields[1]})
        else:
            raise QueryError(f"unrecognized Git change record: {line!r}")
    return {"base": context.base, "head": context.head, "changes": changes}


def _diff(context: QueryContext, args: argparse.Namespace) -> dict[str, object]:
    path = _repo_path(args.path)
    output = context.git(
        "diff",
        f"--unified={args.context}",
        "--no-ext-diff",
        "--no-textconv",
        f"{context.base}...{context.head}",
        "--",
        path,
    )
    return {"path": path, "diff": output}


def _read(context: QueryContext, args: argparse.Namespace) -> dict[str, object]:
    path = _repo_path(args.path)
    if args.end < args.start or args.end - args.start + 1 > MAX_READ_LINES:
        raise QueryError(f"read range must contain between 1 and {MAX_READ_LINES} lines")
    revision = context.revision(args.revision)
    output = context.git("show", f"{revision}:{path}")
    lines = output.splitlines()
    selected = lines[args.start - 1 : args.end]
    return {
        "revision": args.revision,
        "path": path,
        "start": args.start,
        "end": args.start + len(selected) - 1,
        "lines": selected,
    }


def _search(context: QueryContext, args: argparse.Namespace) -> dict[str, object]:
    if not args.pattern or len(args.pattern) > 200:
        raise QueryError("search pattern must contain between 1 and 200 characters")
    prefix = _repo_path(args.prefix, allow_root=True)
    revision = context.revision(args.revision)
    command = ["grep", "-n", "-I", "-F", "-e", args.pattern, revision]
    if prefix != ".":
        command.extend(["--", prefix])
    output = context.git(*command, allow_no_matches=True)
    matches = output.splitlines()[: args.limit]
    return {
        "revision": args.revision,
        "pattern": args.pattern,
        "prefix": prefix,
        "matches": matches,
        "truncated": len(output.splitlines()) > len(matches),
    }


def _plan(context: QueryContext, args: argparse.Namespace) -> dict[str, object]:
    if args.stage is None:
        stages = context.plan.get("stages", {})
        if not isinstance(stages, dict):
            raise QueryError("deterministic plan has invalid stages")
        summary = {}
        for stage, stage_plan in stages.items():
            if not isinstance(stage_plan, dict):
                raise QueryError(f"deterministic plan stage {stage!r} is invalid")
            nodeids = stage_plan.get("nodeids", [])
            summary[stage] = {
                "enabled": stage_plan.get("enabled"),
                "selected_test_count": len(nodeids) if isinstance(nodeids, list) else None,
            }
        return {
            "mode": context.plan.get("mode"),
            "reasons": context.plan.get("reasons"),
            "unresolved": context.plan.get("unresolved"),
            "stages": summary,
        }

    stages = context.plan.get("stages")
    if not isinstance(stages, dict) or args.stage not in stages:
        raise QueryError(f"unknown deterministic plan stage: {args.stage}")
    return {"stage": args.stage, "plan": stages[args.stage]}


def _catalog_tests(context: QueryContext) -> list[dict[str, object]]:
    tests = context.catalog.get("tests")
    if not isinstance(tests, list):
        raise QueryError("pytest catalog has invalid tests")
    if not all(isinstance(test, dict) for test in tests):
        raise QueryError("pytest catalog test entries must be objects")
    return tests


def _tests(context: QueryContext, args: argparse.Namespace) -> dict[str, object]:
    query = args.query.casefold()
    path_prefix = _repo_path(args.path_prefix, allow_root=True)
    matches = []
    total_matches = 0
    for test in _catalog_tests(context):
        nodeid = test.get("nodeid")
        stages = test.get("stages")
        if not isinstance(nodeid, str) or not isinstance(stages, list):
            raise QueryError("pytest catalog test entry is invalid")
        if query and query not in nodeid.casefold():
            continue
        if args.stage and args.stage not in stages:
            continue
        if path_prefix != "." and not nodeid.startswith(path_prefix):
            continue
        total_matches += 1
        if len(matches) < args.limit:
            matches.append({"nodeid": nodeid, "stages": stages})
    return {"matches": matches, "total_matches": total_matches, "truncated": total_matches > len(matches)}


def _test(context: QueryContext, args: argparse.Namespace) -> dict[str, object]:
    for test in _catalog_tests(context):
        if test.get("nodeid") == args.nodeid:
            return {"test": test}
    raise QueryError(f"nodeid is not present in the pytest catalog: {args.nodeid}")


def _help(_context: QueryContext, _args: argparse.Namespace) -> dict[str, object]:
    return {
        "operations": {
            "changes": "list changed paths and statuses",
            "diff": "show one changed or dependency path diff",
            "read": "read bounded committed source lines",
            "search": "search committed repository text",
            "plan": "inspect deterministic coverage",
            "tests": "search eligible pytest nodeids",
            "test": "inspect one exact pytest catalog entry",
        }
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    help_parser = subparsers.add_parser("help", help="describe allowed query operations")
    help_parser.set_defaults(handler=_help)

    changes = subparsers.add_parser("changes", help="list changed paths")
    changes.set_defaults(handler=_changes)

    diff = subparsers.add_parser("diff", help="show the diff for one path")
    diff.add_argument("--path", required=True)
    diff.add_argument("--context", type=int, choices=range(21), default=3)
    diff.set_defaults(handler=_diff)

    read = subparsers.add_parser("read", help="read committed source lines")
    read.add_argument("--path", required=True)
    read.add_argument("--revision", choices=("base", "head"), default="head")
    read.add_argument("--start", type=int, default=1)
    read.add_argument("--end", type=int, default=200)
    read.set_defaults(handler=_read)

    search = subparsers.add_parser("search", help="search committed repository text")
    search.add_argument("--pattern", required=True)
    search.add_argument("--revision", choices=("base", "head"), default="head")
    search.add_argument("--prefix", default=".")
    search.add_argument("--limit", type=int, choices=range(1, MAX_RESULTS + 1), default=100)
    search.set_defaults(handler=_search)

    plan = subparsers.add_parser("plan", help="read deterministic coverage")
    plan.add_argument("--stage")
    plan.set_defaults(handler=_plan)

    tests = subparsers.add_parser("tests", help="search eligible pytest nodeids")
    tests.add_argument("--query", default="")
    tests.add_argument("--stage")
    tests.add_argument("--path-prefix", default=".")
    tests.add_argument("--limit", type=int, choices=range(1, MAX_RESULTS + 1), default=100)
    tests.set_defaults(handler=_tests)

    test = subparsers.add_parser("test", help="read one exact pytest catalog entry")
    test.add_argument("--nodeid", required=True)
    test.set_defaults(handler=_test)
    return parser


def _emit(payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise QueryError("query output is too large; narrow the path, range, search, or test filter")
    sys.stdout.buffer.write(encoded + b"\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        context = QueryContext()
        _emit(args.handler(context, args))
    except QueryError as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
