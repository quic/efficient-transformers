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
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from .core import SCHEMA_VERSION, STAGES, ImpactPlan, TestCase

DEFAULT_MODEL = "azure::gpt-5.5"
MAX_PROMPT_BYTES = 400_000
MAX_RESPONSE_BYTES = 1_000_000
REQUEST_TIMEOUT_SECONDS = 90
EXTERNAL_TIMEOUT_SECONDS = 300
MAX_ATTEMPTS = 2
SYSTEM_PROMPT_PATH = Path(__file__).with_name("SYSTEM_PROMPT.md")


class LLMStageError(RuntimeError):
    """Raised when mandatory LLM selection cannot be completed safely."""


@dataclass(frozen=True)
class LLMSelection:
    run_full_ci: bool
    tests: tuple[str, ...]
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


def _decision_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "run_full_ci": {"type": "boolean"},
            "tests": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
        "required": ["run_full_ci", "tests", "reason"],
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
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": _system_prompt()}]},
            {"role": "user", "content": [{"type": "input_text", "text": context}]},
        ],
        "max_output_tokens": 2048,
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


def _run_external_selector(repo: Path, command: str, model: str, context: str) -> str:
    arguments = shlex.split(command)
    if not arguments:
        raise LLMStageError("LLM_SELECTOR_COMMAND must contain an executable command")

    prompt = (
        f"{_system_prompt()} Do not execute commands or modify files. Return only the schema-conforming decision.\n\n"
        f"Repository context JSON:\n{context}"
    )
    with tempfile.TemporaryDirectory(prefix="qeff-llm-selector-") as temporary_directory:
        temporary = Path(temporary_directory)
        schema_path = temporary / "decision-schema.json"
        output_path = temporary / "decision.json"
        schema_path.write_text(json.dumps(_decision_schema()), encoding="utf-8")
        external_arguments = [
            *arguments,
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--config",
            'approval_policy="never"',
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
                timeout=EXTERNAL_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise LLMStageError(f"external LLM selector timed out after {EXTERNAL_TIMEOUT_SECONDS} seconds") from error
        except OSError as error:
            raise LLMStageError(f"could not start external LLM selector: {error}") from error
        if process.returncode != 0:
            details = process.stderr.strip()[-2000:] or "no error output"
            raise LLMStageError(f"external LLM selector exited with {process.returncode}: {details}")
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
) -> LLMSelection:
    model = model or os.environ.get("LLM_CI_MODEL", DEFAULT_MODEL)
    context, context_incomplete = _prompt(repo, deterministic, catalog)
    selector_command = os.environ.get("LLM_SELECTOR_COMMAND")
    if selector_command:
        output_text = _run_external_selector(repo, selector_command, model, context)
        response_id = "external-cli"
        response_model = model
        attempts = 1
    else:
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
    if set(decision) != {"run_full_ci", "tests", "reason"}:
        raise LLMStageError("LLM output did not match the required fields")
    if not isinstance(decision["run_full_ci"], bool):
        raise LLMStageError("LLM run_full_ci must be a boolean")
    if not isinstance(decision["tests"], list) or not all(isinstance(item, str) for item in decision["tests"]):
        raise LLMStageError("LLM tests must be a list of nodeid strings")
    if len(decision["tests"]) != len(set(decision["tests"])):
        raise LLMStageError("LLM returned duplicate test nodeids")
    unknown = sorted(set(decision["tests"]) - catalog.keys())
    if unknown:
        raise LLMStageError("LLM returned tests outside the allowlist: " + ", ".join(unknown))
    if not isinstance(decision["reason"], str) or not decision["reason"].strip():
        raise LLMStageError("LLM reason must be a non-empty string")
    if len(decision["reason"]) > 2000:
        raise LLMStageError("LLM reason exceeds the 2000-character limit")
    if context_incomplete and not decision["run_full_ci"]:
        raise LLMStageError("LLM must request full CI when supplied context is incomplete")

    return LLMSelection(
        run_full_ci=decision["run_full_ci"],
        tests=tuple(decision["tests"]),
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
