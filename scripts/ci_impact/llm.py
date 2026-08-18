# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Provider-neutral LLM test-selection client and deterministic-plan merger."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import stat
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from .core import SCHEMA_VERSION, STAGES, ImpactPlan, TestCase

DEFAULT_MODEL = "azure::gpt-5.5"
DEFAULT_REASONING_EFFORT = "high"
MAX_PROMPT_BYTES = 400_000
MAX_RESPONSE_BYTES = 1_000_000
REQUEST_TIMEOUT_SECONDS = 90
EXTERNAL_TIMEOUT_SECONDS = 600
MAX_SUBAGENTS = 4
MAX_ATTEMPTS = 2
SYSTEM_PROMPT_PATH = Path(__file__).with_name("SYSTEM_PROMPT.md")
QUERY_TOOL_PATH = Path(__file__).with_name("query.py")
TOOL_POLICY_PATH = Path(__file__).with_name("tool_policy.py")
HOOK_AUDIT_NAME = ".ci-impact-qgenie-audit.jsonl"
LLM_REFINABLE_FULL_REASONS = (
    "global pytest behavior changed:",
    "source snapshot generation failed",
    "unsafe static analysis for ",
    "unclassified production/configuration changes",
    "unparsable model inventory:",
)


class LLMStageError(RuntimeError):
    """Raised when mandatory LLM selection cannot be completed safely."""


@dataclass(frozen=True)
class LLMSelection:
    run_full_ci: bool
    tests: tuple[str, ...]
    unnecessary_tests: tuple[dict[str, object], ...]
    reason: str
    response_id: str
    model: str
    attempts: int
    context_incomplete: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_catalog(path: Path, expected_head: str) -> dict[str, TestCase]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LLMStageError(f"cannot load pytest catalog {path}: {error}") from error
    if not isinstance(payload, dict):
        raise LLMStageError("pytest catalog must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise LLMStageError(f"unsupported pytest catalog schema: {payload.get('schema_version')!r}")
    if payload.get("head") != expected_head:
        raise LLMStageError("pytest catalog was collected for a different HEAD")
    raw_tests = payload.get("tests")
    if not isinstance(raw_tests, list) or not raw_tests:
        raise LLMStageError("pytest catalog must contain at least one test")

    catalog: dict[str, TestCase] = {}
    for raw_test in raw_tests:
        if not isinstance(raw_test, dict):
            raise LLMStageError("pytest catalog test entries must be JSON objects")
        nodeid = raw_test.get("nodeid")
        stages = raw_test.get("stages")
        if not isinstance(nodeid, str) or not nodeid:
            raise LLMStageError("pytest catalog entries require a non-empty nodeid")
        if nodeid in catalog:
            raise LLMStageError(f"pytest catalog contains duplicate nodeid: {nodeid}")
        if not isinstance(stages, list) or not stages or not all(isinstance(stage, str) for stage in stages):
            raise LLMStageError(f"pytest catalog entry {nodeid!r} has invalid stages")
        unknown = sorted(set(stages) - set(STAGES))
        if unknown:
            raise LLMStageError(f"pytest catalog entry {nodeid!r} has unknown stages: {', '.join(unknown)}")
        catalog[nodeid] = TestCase(
            nodeid=nodeid,
            symbol=nodeid,
            path=nodeid.split("::", 1)[0],
            stages=set(stages),
        )
    return catalog


def _matching_callspecs(prefix: str, stage: str, catalog: dict[str, TestCase]) -> list[str]:
    return [
        nodeid
        for nodeid, test in sorted(catalog.items())
        if stage in test.stages and (nodeid == prefix or nodeid.startswith(prefix + "["))
    ]


def _resolve_llm_nodeid(nodeid: str, catalog: dict[str, TestCase]) -> str | None:
    if nodeid in catalog:
        return nodeid
    if "::" in nodeid:
        return None

    path = Path(nodeid)
    parent = path.parent.as_posix()
    function_and_params = path.name
    matches = [
        candidate
        for candidate, test in catalog.items()
        if Path(test.path).parent.as_posix() == parent and candidate.split("::", 1)[1] == function_and_params
    ]
    return matches[0] if len(matches) == 1 else None


def _normalize_llm_nodeids(nodeids: list[str], catalog: dict[str, TestCase]) -> tuple[list[str], list[str]]:
    normalized: list[str] = []
    unknown: list[str] = []
    for nodeid in nodeids:
        resolved = _resolve_llm_nodeid(nodeid, catalog)
        if resolved is None:
            unknown.append(nodeid)
        else:
            normalized.append(resolved)
    return normalized, unknown


def expand_plan_with_catalog(plan: ImpactPlan, catalog: dict[str, TestCase]) -> ImpactPlan:
    expanded = ImpactPlan(**plan.to_dict())
    if expanded.mode == "full":
        return expanded

    for stage, stage_plan in expanded.stages.items():
        for field in ("nodeids", "changed_nodeids", "profile_override_nodeids"):
            nodeids = stage_plan[field]
            if not isinstance(nodeids, list):
                raise LLMStageError(f"stage {stage!r} has invalid {field}")
            matches: set[str] = set()
            missing = []
            for nodeid in nodeids:
                if not isinstance(nodeid, str):
                    raise LLMStageError(f"stage {stage!r} has non-string {field} entry")
                callspecs = _matching_callspecs(nodeid, stage, catalog)
                if callspecs:
                    matches.update(callspecs)
                else:
                    missing.append(nodeid)
            if missing:
                raise LLMStageError(
                    f"stage {stage!r} selected tests outside the pytest catalog: {', '.join(sorted(missing))}"
                )
            stage_plan[field] = sorted(matches)
        stage_plan["enabled"] = bool(stage_plan["nodeids"]) if expanded.mode == "selective" else stage_plan["enabled"]

    if expanded.mode == "selective" and not any(expanded.stages[stage]["nodeids"] for stage in STAGES):
        raise LLMStageError("selective plan has no tests after pytest catalog expansion")
    return expanded


def _git_diff(repo: Path, base: str, head: str) -> str:
    process = subprocess.run(
        ["git", "diff", "--unified=3", "--no-ext-diff", f"{base}...{head}"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise LLMStageError(f"could not generate LLM diff context: {process.stderr.strip()}")
    return process.stdout


def _catalog_payload(catalog: dict[str, TestCase]) -> list[str]:
    return sorted(catalog)


def _prompt(
    repo: Path,
    deterministic: ImpactPlan,
    catalog: dict[str, TestCase],
) -> tuple[str, bool]:
    diff = _git_diff(repo, deterministic.base, deterministic.head)
    context = {
        "changed_files": deterministic.changed_files,
        "deterministic_mode": deterministic.mode,
        "deterministic_reasons": deterministic.reasons,
        "deterministic_unresolved": deterministic.unresolved,
        "deterministic_stages": deterministic.stages,
        "eligible_tests": _catalog_payload(catalog),
        "diff": diff,
    }
    encoded = json.dumps(context, separators=(",", ":"), sort_keys=True)
    if len(encoded.encode("utf-8")) <= MAX_PROMPT_BYTES:
        return encoded, False

    # Never silently depend on truncated context. The caller escalates the
    # merged result to full CI while the LLM still receives a useful summary.
    summary = {
        "changed_files": deterministic.changed_files,
        "deterministic_mode": deterministic.mode,
        "deterministic_reasons": deterministic.reasons,
        "deterministic_unresolved": deterministic.unresolved,
        "eligible_test_count": len(catalog),
        "context_incomplete": True,
        "instruction": "Context exceeded the CI budget; request full CI.",
    }
    return json.dumps(summary, separators=(",", ":"), sort_keys=True), True


def _external_prompt(
    repo: Path,
    deterministic: ImpactPlan,
    catalog_path: Path,
    deterministic_plan_path: Path,
) -> str:
    for label, path in {
        "pytest catalog": catalog_path,
        "deterministic plan": deterministic_plan_path,
    }.items():
        try:
            path.resolve().relative_to(repo.resolve())
        except ValueError as error:
            raise LLMStageError(f"external LLM {label} must be inside the repository") from error
    context = {
        "base": deterministic.base,
        "head": deterministic.head,
        "library_root": "QEfficient",
        "instruction": (
            "Act as the coordinator. First inspect the changes and deterministic plan, then partition only the affected "
            "code into two to four independent subsystem investigations. Spawn one parallel subagent per affected "
            "subsystem; do not create one agent per Jenkins stage and do not spawn agents for unaffected areas. Give "
            "each subagent the base, head, query_tool, assigned files or subsystem, and require exact required nodeids "
            "plus confidence-scored unnecessary tests. Limit each subagent to 25 focused query calls, avoid overlapping "
            "file reads, and return as soon as its assigned impact is bounded. After all subagents finish, reconcile "
            "overlaps and validate the final nodeids through query_tool without repeating their exploration. The catalog "
            "is deliberately omitted from this initial context; query only the portions needed. Use only query_tool, "
            "allowed_operations, and allowed_coordination_tools."
        ),
    }
    return json.dumps(context, separators=(",", ":"), sort_keys=True)


def _decision_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "run_full_ci": {"type": "boolean"},
            "tests": {"type": "array", "items": {"type": "string"}},
            "unnecessary_tests": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "nodeid": {"type": "string"},
                        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                        "reason": {"type": "string"},
                    },
                    "required": ["nodeid", "confidence", "reason"],
                    "additionalProperties": False,
                },
            },
            "reason": {"type": "string"},
        },
        "required": ["run_full_ci", "tests", "unnecessary_tests", "reason"],
        "additionalProperties": False,
    }


def _system_prompt() -> str:
    try:
        prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise LLMStageError(f"could not read system prompt {SYSTEM_PROMPT_PATH}: {error}") from error
    if not prompt:
        raise LLMStageError(f"system prompt is empty: {SYSTEM_PROMPT_PATH}")
    return prompt


def _request_payload(model: str, context: str) -> bytes:
    payload = {
        "model": model,
        "reasoning": {"effort": DEFAULT_REASONING_EFFORT},
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": _system_prompt()}]},
            {"role": "user", "content": [{"type": "input_text", "text": context}]},
        ],
        "max_output_tokens": 8192,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "qeff_ci_test_selection",
                "strict": True,
                "schema": _decision_schema(),
            }
        },
    }
    return json.dumps(payload).encode("utf-8")


def _external_request_prompt(context: str) -> str:
    return (
        f"{_system_prompt()} Use only the query and subagent-coordination tools identified in the repository context. "
        "All other tool calls are forbidden. Return only the schema-conforming decision.\n\n"
        f"Repository context JSON:\n{context}"
    )


def _tool_event(command: str) -> bytes:
    return json.dumps({"tool_name": "Bash", "tool_input": {"command": command}}).encode("utf-8")


def _run_tool_policy(policy_path: Path, command: str, environment: dict[str, str]) -> dict[str, object]:
    process = subprocess.run(
        ["/usr/bin/python3", str(policy_path)],
        input=_tool_event(command),
        capture_output=True,
        check=False,
        env=environment,
        timeout=10,
    )
    if process.returncode != 0:
        raise LLMStageError(f"CI impact tool policy preflight failed with exit code {process.returncode}")
    try:
        output = json.loads(process.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LLMStageError("CI impact tool policy preflight returned invalid JSON") from error
    if not isinstance(output, dict):
        raise LLMStageError("CI impact tool policy preflight returned a non-object")
    return output


def _policy_decision(output: dict[str, object]) -> str | None:
    specific = output.get("hookSpecificOutput")
    if not isinstance(specific, dict):
        return None
    decision = specific.get("permissionDecision")
    return decision if isinstance(decision, str) else None


def _preflight_external_tools(query_command: str, policy_path: Path, environment: dict[str, str]) -> None:
    query_arguments = shlex.split(query_command)
    query = subprocess.run(
        [*query_arguments, "help"],
        capture_output=True,
        check=False,
        env=environment,
        timeout=10,
    )
    if query.returncode != 0:
        details = query.stderr.decode("utf-8", errors="replace").strip()[-1000:]
        raise LLMStageError(f"CI impact query tool preflight failed: {details or query.returncode}")

    allowed = _run_tool_policy(policy_path, f"{query_command} help", environment)
    denied = _run_tool_policy(policy_path, "git status", environment)
    if _policy_decision(allowed) != "allow" or _policy_decision(denied) != "deny":
        raise LLMStageError("CI impact tool policy preflight did not enforce the expected allowlist")


def _validate_hook_audit(path: Path) -> None:
    try:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        raise LLMStageError(f"cannot validate qgenie tool audit: {error}") from error
    if not records or not any(
        isinstance(record, dict)
        and record.get("allowed") is True
        and record.get("reason") == "approved read-only CI impact query"
        for record in records
    ):
        raise LLMStageError("qgenie did not complete any allowed CI impact repository query")


def _trusted_tool_paths(temporary: Path) -> tuple[Path, Path]:
    trusted_directory = os.environ.get("QEFF_CI_TRUSTED_TOOL_DIR")
    if trusted_directory:
        directory = Path(trusted_directory).resolve(strict=True)
        query_path = directory / "query.py"
        policy_path = directory / "tool_policy.py"
        for path in (query_path, policy_path):
            try:
                metadata = path.stat()
            except OSError as error:
                raise LLMStageError(f"cannot access trusted qgenie tool {path}: {error}") from error
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != 0 or metadata.st_mode & 0o022:
                raise LLMStageError(f"trusted qgenie tool must be a root-owned, non-writable regular file: {path}")
        return query_path, policy_path

    query_path = temporary / "ci_impact_query.py"
    policy_path = temporary / "ci_impact_tool_policy.py"
    shutil.copyfile(QUERY_TOOL_PATH, query_path)
    shutil.copyfile(TOOL_POLICY_PATH, policy_path)
    query_path.chmod(0o555)
    policy_path.chmod(0o555)
    return query_path, policy_path


def _run_external_selector(
    repo: Path,
    command: str,
    model: str,
    context: str,
    catalog_path: Path,
    plan_path: Path,
) -> str:
    arguments = shlex.split(command)
    if not arguments:
        raise LLMStageError("LLM_SELECTOR_COMMAND must contain an executable command")

    try:
        context_payload = json.loads(context)
    except (json.JSONDecodeError, TypeError) as error:
        raise LLMStageError("external LLM context is invalid") from error

    audit_path = repo / HOOK_AUDIT_NAME
    audit_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="qeff-llm-selector-") as temporary_directory:
        temporary = Path(temporary_directory)
        query_path, policy_path = _trusted_tool_paths(temporary)

        query_command = f"/usr/bin/python3 {query_path}"
        context_payload["query_tool"] = query_command
        context_payload["allowed_operations"] = [
            "help",
            "changes",
            "diff",
            "read",
            "search",
            "plan",
            "tests",
            "test",
        ]
        context_payload["allowed_coordination_tools"] = ["spawn_agent", "send_input", "wait_agent", "close_agent"]
        prompt = _external_request_prompt(json.dumps(context_payload, separators=(",", ":"), sort_keys=True))

        schema_path = temporary / "decision-schema.json"
        output_path = temporary / "decision.json"
        schema_path.write_text(json.dumps(_decision_schema()), encoding="utf-8")
        environment = os.environ.copy()
        environment.update(
            {
                "QEFF_CI_QUERY_REPO": str(repo.resolve()),
                "QEFF_CI_QUERY_PLAN": str(plan_path.resolve()),
                "QEFF_CI_QUERY_CATALOG": str(catalog_path.resolve()),
                "QEFF_CI_QUERY_COMMAND": query_command,
                "QEFF_CI_HOOK_AUDIT": str(audit_path.resolve()),
            }
        )
        _preflight_external_tools(query_command, policy_path, environment)
        audit_path.unlink(missing_ok=True)

        hook_command = f'/usr/bin/python3 "{policy_path}"'
        hook_config = '[{matcher="*", hooks=[{type="command", command=' + json.dumps(hook_command) + ", timeout=5}]}]"
        external_arguments = [
            *arguments,
            "--ephemeral",
            "--sandbox",
            "danger-full-access",
            "--config",
            'approval_policy="never"',
            "--config",
            f'model_reasoning_effort="{DEFAULT_REASONING_EFFORT}"',
            "--enable",
            "multi_agent",
            "--config",
            "agents.enabled=true",
            "--config",
            f"agents.max_concurrent_threads_per_session={MAX_SUBAGENTS}",
            "--config",
            f'agents.default_subagent_model="{model}"',
            "--config",
            f'agents.default_subagent_reasoning_effort="{DEFAULT_REASONING_EFFORT}"',
            "--enable",
            "hooks",
            "--dangerously-bypass-hook-trust",
            "--config",
            f"hooks.PreToolUse={hook_config}",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "--model",
            model,
            "-",
        ]
        try:
            process = subprocess.run(
                external_arguments,
                cwd=repo,
                input=prompt,
                capture_output=True,
                text=True,
                check=False,
                env=environment,
                timeout=EXTERNAL_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise LLMStageError(f"external LLM selector timed out after {EXTERNAL_TIMEOUT_SECONDS} seconds") from error
        except OSError as error:
            raise LLMStageError(f"could not start external LLM selector: {error}") from error
        if process.returncode != 0:
            details = process.stderr.strip()[-2000:] or "no error output"
            raise LLMStageError(f"external LLM selector exited with {process.returncode}: {details}")
        _validate_hook_audit(audit_path)
        if not output_path.is_file():
            raise LLMStageError("external LLM selector did not create its decision output")
        output = output_path.read_bytes()
        if len(output) > MAX_RESPONSE_BYTES:
            raise LLMStageError("external LLM selector output exceeded the 1 MB limit")
        try:
            return output.decode("utf-8")
        except UnicodeDecodeError as error:
            raise LLMStageError("external LLM selector output was not UTF-8") from error


def _output_text(response: dict[str, object]) -> str:
    if response.get("status") != "completed":
        raise LLMStageError(f"LLM response did not complete: {response.get('status')!r}")
    output_items = response.get("output")
    if not isinstance(output_items, list):
        raise LLMStageError("LLM response output must be a list")
    texts = []
    for output in output_items:
        if not isinstance(output, dict):
            continue
        content_items = output.get("content")
        if not isinstance(content_items, list):
            continue
        for content in content_items:
            if isinstance(content, dict) and content.get("type") == "output_text":
                texts.append(content.get("text", ""))
    text = "".join(item for item in texts if isinstance(item, str))
    if not text:
        raise LLMStageError("LLM response did not contain output text")
    return text


def _post(api_base: str, api_key: str, payload: bytes) -> tuple[dict[str, object], int]:
    endpoint = api_base.rstrip("/") + "/responses"
    last_error = "unknown LLM service failure"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                raw_response = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw_response) > MAX_RESPONSE_BYTES:
                    raise LLMStageError("LLM response exceeded the 1 MB limit")
                decoded = json.loads(raw_response.decode("utf-8"))
                if not isinstance(decoded, dict):
                    raise LLMStageError("LLM response must be a JSON object")
                return decoded, attempt
        except urllib.error.HTTPError as error:
            last_error = f"HTTP {error.code}"
            if error.code in {401, 403} or (error.code < 500 and error.code != 429):
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = f"{type(error).__name__}: {error}"
        if attempt < MAX_ATTEMPTS:
            time.sleep(2)
    raise LLMStageError(f"LLM request failed after {attempt} attempt(s): {last_error}")


def select_tests(
    repo: Path,
    deterministic: ImpactPlan,
    catalog: dict[str, TestCase],
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    model: str | None = None,
    catalog_path: Path | None = None,
    deterministic_plan_path: Path | None = None,
) -> LLMSelection:
    model = model or os.environ.get("LLM_CI_MODEL", DEFAULT_MODEL)
    selector_command = os.environ.get("LLM_SELECTOR_COMMAND")
    if selector_command:
        catalog_path = catalog_path or repo / ".ci-impact-catalog.json"
        deterministic_plan_path = deterministic_plan_path or repo / ".ci-impact-deterministic-plan.json"
        context = _external_prompt(repo, deterministic, catalog_path, deterministic_plan_path)
        context_incomplete = False
        output_text = _run_external_selector(
            repo,
            selector_command,
            model,
            context,
            catalog_path,
            deterministic_plan_path,
        )
        response_id = "external-cli"
        response_model = model
        attempts = 1
    else:
        context, context_incomplete = _prompt(repo, deterministic, catalog)
        api_key = api_key or os.environ.get("LLM_STAGE_KEY")
        if not api_key:
            raise LLMStageError("LLM_STAGE_KEY is required for mandatory LLM test selection")
        api_base = api_base or os.environ.get("LLM_API_BASE")
        if not api_base:
            raise LLMStageError("LLM_API_BASE is required for mandatory LLM test selection")
        response, attempts = _post(api_base, api_key, _request_payload(model, context))
        output_text = _output_text(response)
        response_id = str(response.get("id", ""))
        response_model = str(response.get("model", model))
    try:
        decision = json.loads(output_text)
    except json.JSONDecodeError as error:
        raise LLMStageError(f"LLM output was not valid JSON: {error}") from error

    if not isinstance(decision, dict):
        raise LLMStageError("LLM output must be a JSON object")
    if set(decision) != {"run_full_ci", "tests", "unnecessary_tests", "reason"}:
        raise LLMStageError("LLM output did not match the required fields")
    if not isinstance(decision["run_full_ci"], bool):
        raise LLMStageError("LLM run_full_ci must be a boolean")
    if not isinstance(decision["tests"], list) or not all(isinstance(item, str) for item in decision["tests"]):
        raise LLMStageError("LLM tests must be a list of nodeid strings")
    normalized_tests, unknown = _normalize_llm_nodeids(decision["tests"], catalog)
    if unknown:
        raise LLMStageError("LLM returned tests outside the allowlist: " + ", ".join(sorted(unknown)))
    if len(normalized_tests) != len(set(normalized_tests)):
        raise LLMStageError("LLM returned duplicate test nodeids")
    decision["tests"] = normalized_tests
    unnecessary_tests = decision["unnecessary_tests"]
    if not isinstance(unnecessary_tests, list) or not all(isinstance(item, dict) for item in unnecessary_tests):
        raise LLMStageError("LLM unnecessary_tests must be a list of test assessments")
    unnecessary_nodeids = []
    for assessment in unnecessary_tests:
        if set(assessment) != {"nodeid", "confidence", "reason"}:
            raise LLMStageError("LLM unnecessary test assessment did not match the required fields")
        nodeid = assessment["nodeid"]
        confidence = assessment["confidence"]
        assessment_reason = assessment["reason"]
        if not isinstance(nodeid, str):
            raise LLMStageError(f"LLM returned unnecessary test outside the allowlist: {nodeid!r}")
        normalized_nodeid = _resolve_llm_nodeid(nodeid, catalog)
        if normalized_nodeid is None:
            raise LLMStageError(f"LLM returned unnecessary test outside the allowlist: {nodeid!r}")
        assessment["nodeid"] = normalized_nodeid
        if isinstance(confidence, bool) or not isinstance(confidence, int) or not 0 <= confidence <= 100:
            raise LLMStageError("LLM unnecessary test confidence must be an integer from 0 to 100")
        if not isinstance(assessment_reason, str) or not assessment_reason.strip():
            raise LLMStageError("LLM unnecessary test reason must be a non-empty string")
        if len(assessment_reason) > 500:
            raise LLMStageError("LLM unnecessary test reason exceeds the 500-character limit")
        unnecessary_nodeids.append(normalized_nodeid)
    if len(unnecessary_nodeids) != len(set(unnecessary_nodeids)):
        raise LLMStageError("LLM returned duplicate unnecessary test nodeids")
    overlap = sorted(set(decision["tests"]) & set(unnecessary_nodeids))
    if overlap:
        raise LLMStageError("LLM marked tests as both required and unnecessary: " + ", ".join(overlap))
    if not isinstance(decision["reason"], str) or not decision["reason"].strip():
        raise LLMStageError("LLM reason must be a non-empty string")
    if len(decision["reason"]) > 2000:
        raise LLMStageError("LLM reason exceeds the 2000-character limit")
    if context_incomplete and not decision["run_full_ci"]:
        raise LLMStageError("LLM must request full CI when supplied context is incomplete")

    return LLMSelection(
        run_full_ci=decision["run_full_ci"],
        tests=tuple(decision["tests"]),
        unnecessary_tests=tuple(unnecessary_tests),
        reason=decision["reason"].strip(),
        response_id=response_id,
        model=response_model,
        attempts=attempts,
        context_incomplete=context_incomplete,
    )


def merge_selection(
    deterministic: ImpactPlan,
    selection: LLMSelection,
    catalog: dict[str, TestCase],
) -> ImpactPlan:
    plan = ImpactPlan(**deterministic.to_dict())
    plan.llm = selection.to_dict()
    if selection.run_full_ci or selection.context_incomplete:
        plan.mode = "full"
        plan.reasons.append(f"LLM requested full CI: {selection.reason}")
        for stage in STAGES:
            plan.stages[stage]["enabled"] = True
        return plan

    if (
        plan.mode == "full"
        and plan.reasons
        and all(reason.startswith(LLM_REFINABLE_FULL_REASONS) for reason in plan.reasons)
    ):
        for stage in STAGES:
            plan.stages[stage]["enabled"] = False
        plan.reasons.append("LLM bounded a deterministic static-analysis full-CI fallback")

    for nodeid in selection.tests:
        test = catalog[nodeid]
        for stage in test.stages:
            stage_plan = plan.stages[stage]
            stage_plan["enabled"] = True
            stage_plan["nodeids"] = sorted(set(stage_plan["nodeids"]) | {nodeid})
            stage_plan["profile_override_nodeids"] = sorted(set(stage_plan["profile_override_nodeids"]) | {nodeid})

    enabled = {stage for stage in STAGES if plan.stages[stage]["enabled"]}
    if enabled == set(STAGES):
        plan.mode = "full"
        plan.reasons.append("combined deterministic and LLM selection reaches every Jenkins stage")
    elif enabled:
        plan.mode = "selective"
        plan.reasons.append(f"LLM selected {len(selection.tests)} allowlisted test(s): {selection.reason}")
    elif plan.mode not in {"no_tests", "install_only"}:
        raise LLMStageError("deterministic and LLM selectors returned no tests for an impact-bearing change")
    if plan.mode == "selective" and not any(plan.stages[stage]["nodeids"] for stage in STAGES):
        raise LLMStageError("deterministic and LLM selectors returned no tests for an impact-bearing change")
    return plan


def write_llm_artifact(selection: LLMSelection, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(selection.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
