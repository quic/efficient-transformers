# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Command-line interface for generating regression impact plans."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .core import STAGES, ImpactPlan, build_plan, write_plan
from .llm import (
    LLMSelection,
    LLMStageError,
    expand_plan_with_catalog,
    load_catalog,
    merge_selection,
    select_tests,
    write_llm_artifact,
)


def _skipped_llm_selection() -> LLMSelection:
    return LLMSelection(
        run_full_ci=True,
        tests=(),
        unnecessary_tests=(),
        reason="deterministic impact plan already requires full CI; LLM selection skipped",
        response_id="skipped",
        model="skipped",
        attempts=0,
        context_incomplete=False,
    )


def _full_plan(repo: Path, reason: str) -> ImpactPlan:
    head = "unknown"
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return ImpactPlan(
        mode="full",
        base=head,
        head=head,
        changed_files=[],
        reasons=[reason],
        unresolved=[],
        stages={
            stage: {
                "enabled": True,
                "nodeids": [],
                "changed_nodeids": [],
                "profile_override_nodeids": [],
            }
            for stage in STAGES
        },
    )


def _plan(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    try:
        plan = build_plan(repo, args.base, args.head, force_full=args.force_full)
    except Exception as error:  # noqa: BLE001 - selector failures must never remove test coverage.
        plan = _full_plan(repo, f"selector generation failed closed: {type(error).__name__}: {error}")
    if args.force_reason and plan.mode == "full":
        plan.reasons = [args.force_reason]
    write_plan(plan, args.deterministic_output)
    try:
        catalog = load_catalog(args.catalog, plan.head)
        plan = expand_plan_with_catalog(plan, catalog)
        write_plan(plan, args.deterministic_output)
        if plan.mode == "full":
            selection = _skipped_llm_selection()
            plan.llm = selection.to_dict()
            write_llm_artifact(selection, args.llm_output)
            write_plan(plan, args.output)
            print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
            return 0
        selection = select_tests(
            repo,
            plan,
            catalog,
            catalog_path=args.catalog,
            deterministic_plan_path=args.deterministic_output,
        )
        write_llm_artifact(selection, args.llm_output)
        merged = merge_selection(plan, selection, catalog)
    except LLMStageError as error:
        args.llm_output.parent.mkdir(parents=True, exist_ok=True)
        args.llm_output.write_text(
            json.dumps({"error": str(error), "status": "failed"}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    write_plan(merged, args.output)
    print(json.dumps(merged.to_dict(), indent=2, sort_keys=True))
    return 0


def _escalate(args: argparse.Namespace) -> int:
    payload = json.loads(args.plan.read_text(encoding="utf-8"))
    payload["mode"] = "full"
    payload.setdefault("reasons", []).append(args.reason)
    for stage_plan in payload["stages"].values():
        stage_plan["enabled"] = True
    temporary = args.plan.with_name(f".{args.plan.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.plan)
    return 0


def _validate(args: argparse.Namespace) -> int:
    payload = json.loads(args.plan.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "mode",
        "base",
        "head",
        "changed_files",
        "reasons",
        "unresolved",
        "stages",
        "llm",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"impact plan is missing keys: {', '.join(missing)}")
    if payload["mode"] not in {"full", "selective", "install_only", "no_tests"}:
        raise ValueError(f"invalid impact mode: {payload['mode']!r}")
    if not isinstance(payload["llm"], dict):
        raise TypeError("merged impact plan has no LLM decision")
    if set(payload["stages"]) != set(STAGES):
        raise ValueError("impact plan stage set does not match the regression pipeline")
    for stage, stage_plan in payload["stages"].items():
        if not isinstance(stage_plan.get("enabled"), bool):
            raise TypeError(f"stage {stage!r} has no boolean enabled flag")
        for field in ("nodeids", "changed_nodeids", "profile_override_nodeids"):
            if not isinstance(stage_plan.get(field), list) or not all(
                isinstance(item, str) for item in stage_plan[field]
            ):
                raise ValueError(f"stage {stage!r} has an invalid {field}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="compare two revisions and write an impact plan")
    plan_parser.add_argument("--repo", type=Path, default=Path.cwd())
    plan_parser.add_argument("--base", required=True)
    plan_parser.add_argument("--head", default="HEAD")
    plan_parser.add_argument("--output", type=Path, default=Path(".ci-impact-plan.json"))
    plan_parser.add_argument(
        "--deterministic-output",
        type=Path,
        default=Path(".ci-impact-deterministic-plan.json"),
    )
    plan_parser.add_argument("--llm-output", type=Path, default=Path(".ci-impact-llm.json"))
    plan_parser.add_argument("--catalog", type=Path, default=Path(".ci-impact-catalog.json"))
    plan_parser.add_argument("--force-full", action="store_true")
    plan_parser.add_argument("--force-reason")
    plan_parser.set_defaults(func=_plan)

    escalate_parser = subparsers.add_parser(
        "escalate-full",
        help="escalate a merged plan without losing LLM metadata",
    )
    escalate_parser.add_argument("plan", type=Path)
    escalate_parser.add_argument("--reason", required=True)
    escalate_parser.set_defaults(func=_escalate)

    validate_parser = subparsers.add_parser("validate", help="validate an impact plan schema")
    validate_parser.add_argument("plan", type=Path)
    validate_parser.set_defaults(func=_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, TypeError, ValueError, json.JSONDecodeError, LLMStageError) as error:
        print(f"ci-impact: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
