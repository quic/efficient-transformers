# QEfficient — Agent Instructions

This is the canonical guidance for AI agents (Codex, Claude, Cursor, Copilot, etc.)
working in the `efficient-transformers` (QEfficient) repository. The root
`AGENTS.md` and `CLAUDE.md` are symlinks to this file.

QEfficient adapts Hugging Face Transformers models to run efficiently on Qualcomm
Cloud AI 100 (QAIC) hardware via a PyTorch → ONNX export → compile → on-device
runtime pipeline. The guiding principle is **minimal divergence from upstream HF**
while preserving PyTorch ↔ ONNX ↔ on-device parity.

## Useful commands
- Install (editable): `pip install -e .`
- Install with test extras: `pip install -e .[test]` (adds `pytest`, `pytest-mock`, `pytest-xdist`)
- Install quality tools: `pip install -e .[quality]`
- Lint + format (required before a PR): `pre-commit run --all-files`
  - Ruff (lint + format), config in `pyproject.toml`: line-length 120, isort enabled, target py310.
  - Or directly: `ruff check --fix .` then `ruff format .`
- Run tests: `pytest -n auto <path-or-expr>`
  - Narrow first with `-k '<expr>'`, then widen once green.
  - Tests are tagged with markers (see `pyproject.toml`): `on_qaic`, `cli`, `finetune`,
    `multimodal`, `qnn`, `diffusion_models`, `regular`, `nightly`, `vllm`, `wan`, `flux`.
  - `on_qaic` / `qnn` tests require QAIC hardware and will not run on plain CPU hosts.

## Acceptance gate
- Several skills reference `tests/test_model_quickcheck.py` as the consolidated
  acceptance gate. If that file is not present in your checkout, fall back to the
  most relevant subsystem tests under `tests/` (for example `tests/transformers/`)
  and clearly state in the PR which tests you ran and why.
- Never claim parity from export/compile success alone: validate
  HF PyTorch → QEff PyTorch → ONNXRuntime where the skill calls for it.

## Repository map
- Model wrappers: `QEfficient/transformers/models/*/modeling_*.py`
- HF → QEff replacement wiring: `QEfficient/transformers/models/pytorch_transforms.py`
- Auto / export / runtime glue: `QEfficient/transformers/models/modeling_auto.py`
- Cache abstractions: `QEfficient/transformers/cache_utils.py`
- Base export logic: `QEfficient/base/modeling_qeff.py`
- Runtime input + cache helpers: `QEfficient/utils/generate_inputs.py`, `QEfficient/utils/run_utils.py`
- Examples: `examples/`
- Tests + fixtures: `tests/`, `tests/conftest.py`

## Conventions
- Keep changes surgical and consistent with surrounding code; prefer fixing root causes.
- Preserve HF → QEff PyTorch parity before treating ONNX/compile success as sufficient.
- Keep tuple-cache compatibility wherever shared export/runtime code expects the legacy layout.
- Prefer feature detection (`hasattr`, optional imports, safe `getattr`) over HF-version checks.
- Keep compatibility shims at boundaries (input/output normalization), not deep in model math.
- Follow SOLID principles when writing code. If a change must intentionally diverge
  from SOLID for compatibility, performance, export constraints, or minimal-risk
  integration, call that out to the user with the reason.
- Do not add new test files when an existing test (or the quickcheck gate) can carry the regression.
- Do not add license/copyright headers unless explicitly requested.

## Required user inputs
- Ask the user for the virtualenv path only when there is a real need to execute Python-dependent commands, such as `python`, `pip`, `pytest`, `ruff`, or `pre-commit`; otherwise do not ask.
- Ask the user for `HF_HUB_CACHE` before downloading or loading any model.
- Ask the user for `QEFF_HOME` before export/compile work that produces QEff artifacts.

## Contribution policy
- Follow the fork → branch → PR flow in `CONTRIBUTING.md`; sync from `upstream/main`.
- Run `pre-commit run --all-files` and the relevant tests before opening a PR.
- A human submitter must understand and be able to defend every AI-assisted change
  end-to-end, and must review every changed line and run the relevant tests.
- Disclose AI assistance and the exact test commands run in the PR description.
- Do not open low-value busywork PRs or duplicate existing open PRs/issues.

## Local agent setup
- Skills live under `skills_studio/skills/`. Each skill is a `SKILL.md` plus optional
  `references/` and `scripts/`.
- Local OpenAI Codex agents: run `make codex` after cloning to wire skills under `.agents/skills`.
- Local Claude Code agents: run `make claude` after cloning to wire skills under `.claude/skills`.
- Remove the generated links with `make clean-ai`. The `.agents/` and `.claude/`
  directories are gitignored; only `skills_studio/` is tracked.

## Available skills
Use a skill when the task matches its description. Open the `SKILL.md` for the full workflow.
- `qeff-model-onboarding` — Add support for a new Hugging Face model: classify the
  architecture, reuse the closest existing wrapper family, wire export/runtime/cache,
  and validate via the quickcheck gate.
  (`skills_studio/skills/qeff-model-onboarding/SKILL.md`)
- `transformers-mainline-rebase` — Rebase QEff wrappers onto a newer HF Transformers
  release with minimal divergence while preserving runtime/export parity.
  (`skills_studio/skills/transformers-mainline-rebase/SKILL.md`)
- `qeff-transform-authoring` — Add, modify, or review QEfficient transform code,
  including module mappers, external method mappers, mutators, bespoke transforms,
  registration, and transform tests.
  (`skills_studio/skills/qeff-transform-authoring/SKILL.md`)
