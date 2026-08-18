# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Codex PreToolUse policy for the unsandboxed CI impact selector."""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path

FORBIDDEN_SHELL_SYNTAX = re.compile(r"[\n\r;&|<>`$()\\]")
ALLOWED_OPERATIONS = {"help", "changes", "diff", "read", "search", "plan", "tests", "test"}
ALLOWED_COORDINATION_TOOLS = {"spawn_agent", "send_input", "wait_agent", "close_agent"}
COORDINATION_TOOL_PREFIX = re.compile(r"^multi_agent_v[12]")


def _decision(allowed: bool, reason: str) -> dict[str, object]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow" if allowed else "deny",
            "permissionDecisionReason": reason,
        }
    }


def evaluate(event: object) -> tuple[bool, str, str]:
    if not isinstance(event, dict):
        return False, "tool event must be a JSON object", ""
    tool_name = event.get("tool_name")
    tool_input = event.get("tool_input")
    normalized_tool_name = COORDINATION_TOOL_PREFIX.sub("", tool_name) if isinstance(tool_name, str) else tool_name
    if normalized_tool_name in ALLOWED_COORDINATION_TOOLS:
        return True, "approved CI impact subagent coordination", str(tool_name)
    if tool_name != "Bash":
        return False, f"tool {tool_name!r} is not allowed by the CI impact policy", ""
    if not isinstance(tool_input, dict) or not isinstance(tool_input.get("command"), str):
        return False, "Bash tool input must contain one command string", ""

    command = tool_input["command"].strip()
    if FORBIDDEN_SHELL_SYNTAX.search(command):
        return False, "shell composition, expansion, redirection, and substitution are not allowed", command
    try:
        arguments = shlex.split(command, posix=True)
        expected = shlex.split(os.environ["QEFF_CI_QUERY_COMMAND"], posix=True)
    except (KeyError, ValueError) as error:
        return False, f"invalid query policy configuration: {error}", command
    if arguments[: len(expected)] != expected:
        return False, "only the configured CI impact query command is allowed", command
    remaining = arguments[len(expected) :]
    if not remaining or remaining[0] not in ALLOWED_OPERATIONS:
        return False, "query command must use an allowed operation", command
    return True, "approved read-only CI impact query", command


def _audit(allowed: bool, reason: str, command: str) -> None:
    audit_path = os.environ.get("QEFF_CI_HOOK_AUDIT")
    if not audit_path:
        return
    record = {"allowed": allowed, "command": command, "reason": reason}
    with Path(audit_path).open("a", encoding="utf-8") as audit:
        audit.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")


def main() -> int:
    try:
        event = json.load(sys.stdin)
        allowed, reason, command = evaluate(event)
    except (OSError, json.JSONDecodeError) as error:
        allowed, reason, command = False, f"cannot parse tool event: {error}", ""
    _audit(allowed, reason, command)
    print(json.dumps(_decision(allowed, reason), separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
