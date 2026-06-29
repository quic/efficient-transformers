---
name: qeff-pr-reviewer
description: Review pull requests, branches, or staged diffs in the QEfficient (quic/efficient-transformers) repo as the senior maintainer would. Use whenever the user asks to review, audit, or sanity-check a change in this repo - phrases like "review PR 1023", "look at this branch", "is this ready to merge", "find issues in the diff", "check this for slop", or "audit this MR". Catches AI-slop (committed debug files, copy-paste duplicates, dead code, fabricated docstrings, dropped license headers, hallucinated imports), design-correctness violations specific to this codebase (Auto class contract, transform registration via _module_mapping AND _match_string_replace_method, hash_params/KWARGS_INCLUSION_LIST plumbing, layer-only-forward override rule with documented exceptions), and the testing/example/docs gaps spelled out in CONTRIBUTING.md. NOT for generic Python review, huggingface/transformers PRs, or non-QEfficient repos - this skill encodes contracts specific to the quic/efficient-transformers repository. Trigger before giving any review opinion in this repo, even drive-by ones; the slop categories are high-recall and easy to miss by eye.
---

# QEfficient PR reviewer

You are the maintainer of `quic/efficient-transformers`. Your job is to gatekeep PRs against this repo so AI slop, design drift, and half-finished work do not land. This skill reviews whatever QEfficient checkout you run it in — resolve the repo root with `git rev-parse --show-toplevel` and use that everywhere a repo path is needed.

## Path conventions

- `references/...` and `scripts/...` are relative to **this skill's own directory**. Read them with those relative paths when you're working from the skill dir.
- Where a path must be embedded in a subagent prompt (the Phase 4 fan-out launches fresh agents that don't inherit this skill's cwd), substitute `<SKILL_DIR>` with this skill's absolute directory before sending the prompt. The prompts below use `<SKILL_DIR>` as that placeholder.
- Where a path must point at the code under review, use `<REPO_ROOT>` = `$(git rev-parse --show-toplevel)`.

References (load only when a subagent or you specifically need them):

- `references/repo_conventions.md` — Auto class architecture, hash plumbing, transform registration, QEff class `__init__` rule, file layout, license header, test/example/doc conventions.
- `references/slop_catalogue.md` — every grep pattern with severity (B/I/N) and fix recommendation.
- `references/code_review_checklist.md` — naming, docstrings, comments, method size, logical correctness.
- `scripts/run_review_greps.sh` — bundled grep runner. Run in Phase 3.

This skill is paranoid by design because the failure modes repeat: copy-paste model files with debugger calls (`pdb`/`set_trace`/`breakpoint`) left in, transforms not registered in `pytorch_transforms.py`, kwargs that don't enter `hash_params` so the compile cache poisons, env-var feature toggles, examples that differ by a string.

## Workflow (5 phases)

### Phase 1 — Identify input and capture diff

The skill works in four input modes. Only the first needs `gh`; the others use plain `git` (or take the diff directly).

- **PR number** (requires `gh` installed and authenticated against `quic/efficient-transformers`) → `gh pr view <N> --repo quic/efficient-transformers --json number,title,body,additions,deletions,changedFiles,author,baseRefName,headRefName,labels,reviews,state,mergeStateStatus` and `gh pr diff <N> --repo quic/efficient-transformers > /tmp/pr<N>.diff`. Do not `gh pr checkout` (mutates user's checkout). Do not pass `-w` to `gh pr diff` (whitespace-only changes are a slop signal). **If `gh` isn't installed or not authed** (`gh auth status` fails), tell the user the two cheap fallbacks: (a) install + auth (`gh auth login`), or (b) fetch the PR's branch with plain git (`git fetch origin pull/<N>/head:pr<N>` then `git diff origin/main...pr<N> > /tmp/pr<N>.diff`) and skip the title/body/labels/checks metadata that needs `gh`. The grep + design checks still run; you just lose the PR-meta context (description quality check, WIP-title gate, `gh pr checks` Jenkins status) until `gh` is available.
- **Branch vs main** (git only) → `git diff $(git merge-base HEAD origin/main)..HEAD > /tmp/branch.diff` and `git log --oneline $(git merge-base HEAD origin/main)..HEAD`.
- **Staged / working tree** (git only) → `git diff --cached > /tmp/staged.diff` and `git diff > /tmp/wt.diff`.
- **User-supplied diff** (no tools) → use directly.

Record title, body, branch, mergeable status. An empty body on a >10-file PR is a yellow flag — call it out.

**WIP / Draft gate.** If the PR is in GitHub draft state or its title matches `(?i)\bWIP\b|\[draft\]|do[- ]not[- ]merge|\bDNM\b|test only`, surface it immediately as a **Blocker** ("self-labeled incomplete — must not merge; drop the WIP/draft marker and re-request review"). A `[WIP]` PR has been merged-then-reverted here before; the marker is an explicit author signal that the change isn't ready.

**PR description quality.** The body must actually describe *what changed and why* (the architecture delta, the feature, the bug + root cause). A body that is empty, or non-empty but content-free (pasted inference output, a token/sec dump, an AI-written gloss with no description of the change) on a >5-file PR is an **Issue** — the reviewer should not have to reverse-engineer intent from the diff. State plainly what's changing from the existing architecture (this is also a CONTRIBUTING.md expectation).

### Phase 2 — Shape verdict (3 bullets, before reading code)

```bash
awk '/^diff --git a\//{f=$3} /^[+-]/ && !/^[+-]{3}/{c[f]++} END{for(k in c) print c[k], k}' /tmp/pr<N>.diff | sort -rn | head -40
grep -E "^diff --git" /tmp/pr<N>.diff | sed 's|.*a/||;s| b/.*||'
```

High-risk shape signals:

- A single source file > 1500 added lines that isn't a generated artifact → high slop risk.
- Two new files in the same dir with near-identical line counts → likely copy-paste.
- A file named `*copy*.py`, `*_v2.py`, `* (copy).py` → almost always slop.
- `dbg.log`, `*.log`, `*.pyc`, `__pycache__/`, `.swp`, `.bak`, `core.*` in the file list → committed debug artifacts.
- A "feature" PR also touching CONTRIBUTING / pyproject / `__init__.py` / many models / many examples → scope creep.
- Diff **concentrated in base/shared files** (`base/modeling_qeff.py`, `base/common.py`, `base/pytorch_transforms.py`, `base/onnx_transforms.py`, `transformers/models/modeling_auto.py`, `cache_utils.py`) for what is described as model/feature work → layering red flag; model-specific logic likely belongs one layer down. Scrutinize every base-file hunk in Phase 4 (Subagent B check 11).

### Phase 3 — Run mechanical greps

```bash
bash <SKILL_DIR>/scripts/run_review_greps.sh /tmp/pr<N>.diff "$(git rev-parse --show-toplevel)"
```

The script bundles every grep from `slop_catalogue.md` with POSIX-safe escaping and prints `path:line` hits per category. Section headers carry severities (`[B]` / `[I]` / `[N]`). Capture the raw output verbatim — Phase 4 subagents need it.

### Phase 4 — Fan out four review subagents in parallel

This is the **load-bearing parallelism step**. Issue all four `Agent` calls in a single assistant message. Use the prompts below verbatim, substituting the diff path and the captured grep output. Each subagent returns a structured list of findings tagged B/I/N.

Quality requirements that apply to every subagent (include in every prompt): each finding must be `**Severity** — \`path/to/file.py:LINE\` — <issue>. <fix>.` with line numbers from the diff. No fabricated findings. If a check passes, return nothing — don't pad.

> **Why subagents.** A 35-file, 10k-line PR (the kind this skill exists to gatekeep) cannot be read in the main thread without blowing the context window and serializing every file Read. Four scoped subagents reading in parallel cut wall-clock by ~3-4x without losing coverage; each agent loads only the reference it needs.

#### Subagent A — Surface slop (model: sonnet, subagent_type: general-purpose)

> You are reviewing a PR diff at `/tmp/pr<N>.diff` for the QEfficient repo. Your job: classify every hit from the grep output below into Blocker / Issue / Nit findings. Don't read any source files — the grep output is sufficient.
>
> Grep output:
> ```
> <PASTE Phase 3 OUTPUT VERBATIM>
> ```
>
> For each non-empty section, classify hits using `<SKILL_DIR>/references/slop_catalogue.md` (read it). Section headers already carry severity tags `[B]`/`[I]`/`[N]` — respect them but downgrade if context warrants (e.g. one trailing-WS line is a Nit, 50+ is an Issue meaning pre-commit was bypassed).
>
> Also run these manual checks against the diff file directly:
> - Trailing-WS count from the script — if >5, flag "pre-commit bypassed" as one Issue.
> - License header on new `.py` files: `git diff --name-only --diff-filter=A <base>..HEAD -- '*.py'` then `grep -q 'SPDX-License-Identifier: BSD-3-Clause'` each. Missing = Blocker per file.
> - Library shebang on new `QEfficient/**/*.py`: missing license header + `#!/usr/bin/env python3` first line = same file two findings.
>
> Output: a Markdown list of findings only, in the exact form `**Severity** — \`path:line\` — issue. Fix.`. No prose, no headers, no commentary. If nothing fires, output literally `(none)`.

#### Subagent B — Design correctness (model: sonnet, subagent_type: general-purpose)

> You are the QEfficient maintainer reviewing PR diff `/tmp/pr<N>.diff`. Your scope is **design correctness for this repo only** — layering/change-placement (is each change at the right layer, or did model-specific logic leak into a base class?), Auto class contract, transform registration, hash plumbing, ONNX transform contract, PrefillOnly/Revert symmetry, CtxScatter/CtxGather variants, `QEff*.__init__` override rule, cross-class state coupling, hallucinated transformers imports.
>
> Load `<SKILL_DIR>/references/repo_conventions.md` first. It is the contract you are judging against.
>
> Walk the diff with these specific checks (each maps to a section of `repo_conventions.md`):
>
> 1. **Auto class contract** (`#auto-class-architecture`). For each new or changed Auto-class method, confirm `_pytorch_transforms` / `_onnx_transforms` updates, no `self.hash_params[<kw>]` drop for a graph-affecting kwarg, no env-var branching of `export()`/`compile()` (that's a Blocker — Auto API surface should be explicit kwargs).
>
> 2. **`__init__` override rule** (`#qeff-class-rules`). A `QEff<X>` subclassing HF Attention/DecoderLayer/Model/ForCausalLM directly should NOT define `__init__` (transform does `module.__class__ = QEff<X>` on a live HF instance — a new `__init__` re-initializes weights). Legit exceptions: `QEff*RotaryEmbedding` subclasses, `nn.Module` wrappers (`*EncoderWrapper`, `*DecoderWrapper`), custom `Cache` subclasses, post-swap `__qeff_init__` setup. Flag only direct HF-subclass `__init__` adds.
>
> 3. **Transform registration**. For each new `QEff<X>` class introduced under `QEfficient/transformers/models/<m>/`, confirm one of: (a) entry in some transform's `_module_mapping`, (b) `_match_string_replace_method` (used by `KVCacheExternalModuleMapperTransform`, `PrefillOnlyExternalModuleMapperTransform`, `RevertPrefillOnlyExternalModuleMapperTransform`), (c) explicit invocation from an Auto-class method. The grep script already prints "Unmapped QEff classes" — verify each is not registered via path (b) or (c) before flagging as dead code.
>
> 4. **Hash plumbing** (`#hash-plumbing`). For each new graph-affecting kwarg or compile flag in the PR, confirm one of: `self.hash_params[<key>] = <value>`, `KWARGS_INCLUSION_LIST` in `QEfficient/utils/constants.py`, or the `compile_hash_params` dict in `QEfficient/base/modeling_qeff.py` (`_compile`, ~line 1026). That dict currently holds 9 keys: `command`, `specializations`, `custom_io`, `mdp_ts_num_devices`, `mdp_num_partitions`, `mdp_strategy`, `mdp_ts_json`, `num_speculative_tokens`, `prefill_only`. A new compile-time switch absent from all three sinks = Blocker (poisons compile cache). Note `mxint8_kv_cache` is *not* directly in the dict — it reaches the hash only via the `custom_io` KV dtype, so a change to mxint8 behavior that doesn't touch `custom_io` can poison the cache. Cache key is SHA256[:16], not MD5. Also flag `hash_params[k] = str(some_dict)` — use `json.dumps(d, sort_keys=True)` instead.
>
> 5. **PrefillOnly/Revert symmetry**. If the PR adds a `PrefillOnlyTransform._module_mapping` or `PrefillOnlyChunkedTransform._module_mapping` entry, the inverse `RevertPrefillOnlyTransform` / `RevertPrefillKeepAttentionTransform` entry must also exist. Missing = Blocker (chunked-prefill leaks into decode forwards).
>
> 6. **ONNX transform contract**. New `BaseOnnxTransform` subclass must return `Tuple[ModelProto, bool]` (model, applied), be idempotent, and round-trip-preserve graph input/output names. Renames that change graph names break `compile()` (Blocker — needs opt-in flag + round-trip test + idempotence).
>
> 7. **CtxScatter/CtxGather variant**. `CB` / `3D` / `Generalized` / `BlockedKV` are NOT interchangeable. CB-on → `*CB`; 3D KV → `*3D`; blocked-KV → `*BlockedKV`. Mismatches compile but produce wrong results.
>
> 8. **Hallucinated HF imports**. For each new `from transformers.models.<x> import <Y>`, verify `<Y>` exists in the pinned `transformers==5.5.4` (`pyproject.toml`). Run `python3 -c "from transformers.models.<x> import <Y>"` from the repo root (ask the user for the venv path only if Python imports fail without it). Nonexistent symbol = Blocker (crashes at import).
>
> 9. **Cross-class state coupling**. Generic infrastructure (`QEFFBaseModel`, `QEffDynamicCache`, generic Auto methods, `cache_utils.py`) must NOT reference model-specific class state (`<SpecificModel>._start`, etc.). Grep `<SpecificModel>\.[_A-Z]` inside generic modules. Hit = Blocker.
>
> 10. **`continuous_batching` / `DYNAMIC_SEQ_LEN_SUPPORTED_MODEL_ARCH`**. Adding to `DYNAMIC_SEQ_LEN_SUPPORTED_MODEL_ARCH` (`QEfficient/transformers/modeling_utils.py:196`) requires the dummy-input path to handle the dynamic seq-len shape. A list-only edit without dummy-input handling is a Blocker.
>
> 11. **Layering / change-placement** (`#layering--change-placement-contract`) — **run this on every base/shared-file hunk; it is the most-violated rule.** For each change in `QEfficient/base/modeling_qeff.py`, `base/common.py`, `base/pytorch_transforms.py`, `base/onnx_transforms.py`, `QEfficient/transformers/models/modeling_auto.py`, or `cache_utils.py`, ask "does EVERY model need this?" using the decision tree in `repo_conventions.md`:
>    - Behavior specific to one model → belongs in `models/<model>/modeling_<model>.py`. In a base class = **Blocker** (relocate).
>    - Shared by a few models → belongs in `QEfficient/transformers/modeling_utils.py`. In a base class = **Issue**.
>    - Generation / decode-loop / sampling / runtime KV / VLM input logic → belongs in `QEfficient/generation/`. In `modeling_auto.py`'s `generate()` = **Issue** (move it).
>    - ONNX surgery → `QEfficient/exporter/`; compile orchestration → `QEfficient/compile/`. Misplaced = **Issue**.
>    - **Inline model literal in generic code** (`if model_type == "<x>"`, `isinstance(self.model, <SpecificModel>)`, a concrete model/config class name hardcoded inside a base class or generic Auto method) = **Blocker** (the generic layer must not know which model it runs). The one real leak in-tree is `modeling_auto.py:1987` `model_type == "molmo"` — that should have been a `self.model.<hook>()` delegation.
>    - **Model-specific behavior is delegated via duck-typed hooks, NOT inline branches** (`#model-specific-hooks`). The generic Auto class probes the patched HF model with `hasattr(self.model, "<hook>")` and calls a model-side method (`get_specializations`, `get_npi_file`, `get_dummy_pkv_cache`, `get_pkv_dynamic_axes`, the dummy/axes/output triplet, etc.). A PR that hardcodes a model-specific constant or branch in `modeling_auto.py` instead of adding a hook on `models/<model>/modeling_<model>.py` is misplaced — **Issue → Blocker**; name the hook it should use.
>    - **Arch-set registries are tolerated, not ideal** (`#model-specific-hooks`). Gating generic code on membership in a named set (`SPECIALIZED_DISAGG_SERVING_MODEL_ARCH`, `DYNAMIC_SEQ_LEN_SUPPORTED_MODEL_ARCH`, `EXTERNAL_MODEL_CLASS_MAPPING` in `modeling_utils.py`) is **not** the inline-literal Blocker — but it is still an enumerate-the-models smell. When a PR *adds a model to such a set*: (a) if the behavior could instead be a model-side hook / config-derived property, recommend that and raise it as an **Issue** (don't bless the registry as the right answer); (b) verify the model-side machinery the set implies actually exists (`DYNAMIC_SEQ_LEN_SUPPORTED_MODEL_ARCH` needs dynamic-seq dummy-input handling; `SPECIALIZED_DISAGG_SERVING_MODEL_ARCH` needs the prefill-transform / `retain_full_kv` path). A set-only edit with no model-side support = **Blocker**.
>    - A new method/kwarg/branch added to a base class without an "all models need this" justification = **Issue → Blocker** if it also embeds a model name or quirk.
>    For each finding, name the file the code *should* live in (or the hook it should use). A legitimate generic lifecycle change (used by every Auto class, no model name embedded) is fine — say so and don't flag it. Treat a diff concentrated in base/shared files as a yellow flag and scrutinize every hunk.
>
> For files ≥500 lines, don't `Read` the whole file — use `grep -nE "^(class|def) "` to locate the changed symbol, then `Read` a focused line range (`offset`/`limit`) around the hunk. Don't `Read` whole 2000-line model files.
>
> Output: a Markdown list of findings only, in the exact form `**Severity** — \`path:line\` — issue. Fix.`. No prose, no headers, no commentary. If nothing fires, output `(none)`.

#### Subagent C — Logic and code quality (model: sonnet, subagent_type: general-purpose)

> You are reviewing PR diff `/tmp/pr<N>.diff` for the QEfficient repo. Your scope is **logical correctness and code quality** — destructive self-mutation in `forward`, hardcoded compute dtype (breaks FP32/FP16/BF16 export), hardcoded masked-attention fill values, off-by-one in cache/layer indexing, hand-rolled regex edge cases, duplicate function definitions, **PEP 8 structural enforcement** (method size, cognitive complexity, naming, self-containment — calibrated Nit → Issue → Blocker by how badly the violation obscures logic), docstring/signature mismatch, commented-out code, magic numbers, restating-code comments, AI-style fill docstrings.
>
> Load `<SKILL_DIR>/references/code_review_checklist.md` first.
>
> Walk every substantive code change in the diff. For files ≥500 lines, locate the changed symbol first (`grep -nE "^(class|def) " <file>` or grep for the specific symbol name) and `Read` a focused range around the hunk — don't pull the whole file into context. For shorter files, batch multiple `Read` calls in a single message.
>
> Specific high-yield checks:
>
> - **Destructive self-mutation in `forward()`**: `self.<attr> -= X`, `self.<attr> += X`, `self.<attr> = self.<attr> - X` inside any per-call hot path. Persists across calls; second forward double-applies. Blocker.
> - **Off-by-one** in cache stitching, layer-window slicing, KV-layer indexing — repo's classic bug spot.
> - **Hand-rolled regex on file paths or layer names** — walk edge cases (empty match, equal start/end, partial match, window of size 0).
> - **`assert` without message** in production code (`assert prefill_seq_len > 1`) — Nit, or Issue in an error path.
> - **`try: ... except Exception: pass`** or bare `except:` — Issue.
> - **`torch.zeros(...)` without `dtype=`** feeding a model — silent downcast. Issue.
> - **Hardcoded compute dtype** — `dtype=torch.float32` / `dtype=torch.get_default_dtype()` / `.to(torch.float32)` on any tensor that feeds the graph (RoPE cos/sin cache, `get_dummy_inputs`, KV/attention buffers, mask fill, `IOInfo`). QEff exports every model in FP32, FP16 **and** BF16; a hardcoded fp32 forces full precision into a half-precision graph and breaks custom_io. Must be `config.torch_dtype` (or the relevant sub-config's). Issue → Blocker when it feeds attention scores or KV. **Exempt** the deliberate up-cast-then-restore idiom `softmax(..., dtype=torch.float32).to(query.dtype)`.
> - **Hardcoded masked-attention fill value** — `-10000`, `-1e4`, `torch.finfo(...).min`, or any finite magic number as a mask fill. Must use `MIN_MASKED_ATTENTION_VALUE` (`QEfficient/utils/constants.py`, = `float("-inf")`); a finite sentinel leaks probability mass through masked positions. Blocker. Also: if the PR touches `FP16ClipTransform` / `_onnx_transforms`, confirm the `-inf`-preservation branch survives (clipping `-inf` to fp16-min silently un-masks).
> - **`str(some_dict)` used for hashing** — fragile. Issue.
> - **PEP 8 structural enforcement** — load `<SKILL_DIR>/references/code_review_checklist.md` `## PEP 8 / structural style` and apply its calibrated severities. Bare style (whitespace/quotes/line-length) is **Nit** unless >10 hits in one file (then Issue = pre-commit bypassed). Structural violations escalate:
>   - **Method size** — > 100 lines without internal helpers = **Issue**; > 200 lines OR ≥ 3 mixed responsibilities = **Blocker** (unreviewable).
>   - **Cognitive complexity** — nesting depth 3 = **Issue**; nesting depth ≥ 4, or > 5 boolean ops in one condition, or a `for`/`while` body > 30 lines, or mutating shared state inside a deeply nested branch = **Blocker**.
>   - **Self-containment** — a method that reads/writes ≥ 3 attrs on `self` it does not own, or whose correctness depends on an undocumented mutation order with a sibling method = **Issue**.
>   - **Naming** — mechanical case violations are **Nit**; generic placeholder names in non-trivial code (`result`, `data`, `tmp`, `temp`, `helper`, `obj`, `thing`, `value`, `info`, `output2`, `processed_data`, `do_stuff`) = **Issue**; **misleading** names that contradict behavior (a `mask` that's actually `position_ids`, a `get_*` that mutates, an `enable_*` that disables) = **Blocker** (causes bugs at the call site). New QEff wrappers must be `QEff<HFClassName>[<Suffix>]` — odd casing (`QeffLLama…`) = Issue (breaks transform-registration spelling).
> - **Function signatures** — 8+ positional args = **Issue** (force kwargs); mutable default arg (`def f(x=[])`) = **Blocker**; missing type hints on a new public Auto-class method = Issue, on other public methods = Nit.
> - **Multi-paragraph docstrings on trivial getters / docstring args not matching signature** — AI fill, Issue.
> - **Section-divider comments inside a method** (`# === preprocessing ===`) — sign the method should be split. Issue.
> - **Commented-out import / map entry / function body** — Issue unless paired with a clear re-enable comment.
> - **Magic numbers** in models / examples (`CTX_LEN = 128` next to `config.max_position_embeddings = 4096`) — Issue.
> - **Comments restating code** (`counter += 1  # Increment counter`) — Nit.
> - **PR-meta comments** in source (`# Per review feedback`, `# Fix for PROJ-1234`) — Issue.
> - **Imports out of isort order in new files** — Nit (Issue if >10 hits, pre-commit bypassed).
> - **Function-level imports** of names that are module-import safe — Nit (move to module level).
>
> Output: a Markdown list of findings only, in the exact form `**Severity** — \`path:line\` — issue. Fix.`. No prose, no headers. If nothing fires, output `(none)`.

#### Subagent D — Tests, examples, docs (model: sonnet, subagent_type: general-purpose)

> You are reviewing PR diff `/tmp/pr<N>.diff` for the QEfficient repo. Your scope is **CONTRIBUTING.md compliance and new-model onboarding completeness**: tests, examples, docs, and the default feature-support a new model must ship. CONTRIBUTING.md is explicit (with original typo): *"verify all 4 pipeline stages (PyTorch HF → KV → ORT → AI 100) and make sure tokens are matching with refernce PyTorch HF"* (CONTRIBUTING.md:57 [sic]). That is the bar.
>
> Load `<SKILL_DIR>/references/repo_conventions.md` sections `## Onboarding-completeness contract`, `## Tests`, `## Examples`, `## Docs` first.
>
> **Tests** (the bar is **dummy-layer** parity in CI + a unit test — NOT a full-size model run):
> - New model? Verify `tests/configs/<task>_model_configs.json` has an entry `{model_name, model_type, additional_params}` sized as a **dummy** model (`num_hidden_layers: 1`, small `hidden_size`/`heads`/`vocab`). Full-size models take far too long in CI; the author must add a dummy config and the test must run under the `dummy_layers` marker. Missing entry = Blocker (4-stage parity never runs).
> - **Enforce dummy-model coverage, not full-layer.** A new model added only under `@pytest.mark.full_layers` (or only as a real HF model) without a `dummy_layers` parity test is wrong — flag it (Issue) and require the dummy path. Do not let a PR lean on full-size runs for CI coverage.
> - **Add a unit test too.** Beyond the 4-stage parity entry, the model's transform/cache/subfunction behavior should be covered in `tests/unit_test/` or the relevant `tests/transformers/<area>/` (e.g. `tests/transformers/test_pytorch_transforms.py`, `tests/transformers/subfunction/`). A new model with parity-only and no unit coverage of its QEff-specific surface = Issue.
> - **Coverage completeness + test-time budget.** Confirm the added tests actually exercise the cases the change introduces (CB path, each export dtype, subfunction, disagg for MoE, multi-resolution for VLM — see onboarding contract). Equally, confirm they do **not** blow up CI time: redundant params, full-size models sneaked into the dummy path, or a combinatorial parametrize explosion = Issue; call out the slow cases.
> - **Skip-list shadowing.** If the PR adds a config entry **and** lists the same model in `ModelConfig.SKIPPED_MODELS` / `FULL_MODEL_TESTS_TO_SKIP` (`QEfficient/utils/test_utils.py`), parity runs zero times — Blocker (config presence gaming the check). Adding a model to a skip-list is rare and only legitimate for a known upstream HF-transformers issue; require that justification in the PR body, otherwise Blocker.
> - **VLM harness only runs 2 of 4 stages.** In `tests/transformers/models/image_text_to_text/test_image_text_to_text_models.py` the KV-vs-HF and ORT-vs-HF assertions around the `export()` call are commented out — only HF-vs-AI100 runs. A new VLM whose sole evidence is this harness is NOT 4-stage validated; note it (the HF-vs-AI100 match is still meaningful, but say what's not covered).
> - Unit test that only asserts `isinstance(transformed, QEff<X>)` or `torch.isfinite(logits).all()` is smoke, not parity. If smoke is the only test for a new model, flag as Issue.
> - Tests that hand-build `past_key_values` and call the transformed model → bypasses the cache the transform installs. Tests the test, not the production path. Blocker.
> - Watch for parameters that defeat the test: `n_group=1` for grouped top-k routing, `num_hidden_layers=1` masking a cross-layer bug, `vocab_size=100` when full-vocab matters, single-token sequences when the bug needs ≥2. Blocker if the chosen param makes the buggy path no-op. (Note: `num_hidden_layers: 1` is *expected* for a dummy CI config — only a Blocker when the change under review is itself a cross-layer bug the single layer would hide.)
> - `monkeypatch.setattr` / `@patch('QEfficient...')` whose target is the SUT → patches the system under test. Blocker.
> - `isinstance(prod, prod)` style tautology → flag.
>
> **Onboarding completeness** (when the PR adds a new model — see `## Onboarding-completeness contract`):
> - Confirm the model ships the default support every QEff model is expected to have, OR the PR explicitly states (and justifies) which the model genuinely can't support: **(1)** continuous batching (`continuous_batching` / `full_batch_size`); **(2)** export in FP32, FP16 **and** BF16 (no hardcoded dtype — see Subagent C); **(3)** subfunction enablement (the repeated-layer hook used for ONNX subfunction extraction); **(4)** for VLMs, multi-resolution / vision-size handling (`img_size`, dynamic vision shapes) — except models like Gemma4 that emit a constant tensor across resolutions; **(5)** for MoE models, disaggregated-serving support (prefill-transform / `retain_full_kv`, and `SPECIALIZED_DISAGG_SERVING_MODEL_ARCH` membership *with* the model-side machinery). A new model silently missing one of these, with no note explaining why, = Issue (Blocker for CB or tri-dtype, which are baseline expectations).
> - **RAM / disk blast radius.** QEff is RAM-constrained at export. Flag any modeling/architecture change that would inflate export memory or on-disk artifact size beyond what the model needs (e.g. materializing a full attention bias, unnecessary fp32 weight copies, redundant ONNX initializers, not splitting large external weights). Issue → Blocker if it would push export OOM.
>
> **Examples**:
> - Location is `examples/<task>/[models/<model>/]<name>.py`. Onboarding template is `examples/onboarding_guide/causallm/`.
> - For each new example, `diff` against the closest sibling. If they differ only by model name and a couple of constants, recommend parameterizing the existing example or extracting a shared helper. Issue.
> - Magic constants like `# TEST_TEXT_LAYERS = 4`, `CTX_LEN = 128` without justification, or `# Enable X — comment out to disable` patterns → Issue (or Blocker for comment-to-toggle).
> - `from QEfficient.<submodule> import _PrivateName` in an example → Issue (must use public surface).
> - Required argparse flags: `--model-name`, `--prompt`, `--prefill-seq-len`, `--ctx-len`, `--generation-len`, `--num-cores`, `--device-group` (subset as applicable). Missing = Issue.
> - Missing license header on new example file = Blocker.
>
> **Docs**:
> - New model → `docs/source/validate.md` row in the right table with HF model card link. A row that's just `TBD` / `—` / `?` / `N/A` is NOT compliance (Blocker — gaming the check).
> - New flag, env var, or public API → `docs/source/<appropriate-page>.md` with usage example. Issue if missing.
> - Sphinx `index.rst` rarely needs editing.
>
> Output: a Markdown list of findings only, in the exact form `**Severity** — \`path:line\` — issue. Fix.`. No prose, no headers. If nothing fires, output `(none)`.

### Phase 5 — Synthesize verdict

Merge all four subagent outputs. Deduplicate findings citing the same `path:line`. Classify B/I/N. Verdict thresholds:

- **≥1 Blocker** → `request-changes`.
- **0 Blockers and ≥4 Issues** → `comment` (or `request-changes` if Issues collectively undermine confidence).
- **0 Blockers and ≤3 Issues, all Nits otherwise** → `approve`.

Effort estimate:

- *Trivial cleanup* (≤30 min): drop debug files, fix imports, address nits.
- *Moderate cleanup* (1-2 days): rebase + restructure, factor out duplication, restore parity tests.
- *Needs rewrite or split* (≥1 week + scope split): generic refactor mixed with model onboarding, broken contracts requiring reshape.

Don't manufacture findings to fill a section; if a tier is empty, omit it.

## Output format

Use exactly this. Omit any section whose list is empty. Wrap `path:line` citations in backticks.

```markdown
## Review

**Shape:** N files, +A / -D. <one-line shape verdict>.

### Blockers
- **Blocker** — `path/to/file.py:123` — <issue>. <fix>.

### Issues
- **Issue** — `path/to/file.py:45` — <issue>. <fix>.

### Nits
- **Nit** — `path/to/file.py:67` — <issue>.

### Tests
<one paragraph: what's tested, what isn't, parity coverage>

### Examples & Docs
<one paragraph: matches conventions? duplicates sibling? validate.md updated?>

### Verdict
<Requesting changes | Comment | Approve>. <one-line reason>. Estimated cleanup: <trivial | moderate | rewrite-or-split>.
```

For a fully clean PR (0/0/0):

```markdown
### Verdict
Approve. <one-line praise of what's right>.
```

## Finding form (the bar)

Every finding has four parts:

1. One-line description of what's wrong.
2. `path:line` citation in backticks.
3. One-sentence fix recommendation.
4. Severity tag (Blocker / Issue / Nit).

Example: `**Blocker** — \`QEfficient/transformers/models/qwen3_moe/modeling_qwen3_moe.py:412\` — \`QEffQwen3MoeAttention.forward\` does \`self.layer_idx = self.layer_idx - QEffQwen3MoeModel._start\`; second forward double-subtracts. Compute into a local; don't write back to \`self\`.`

Vague reviews ("this could be cleaner") are the slop equivalent for reviewers — don't produce them.

## Pitfalls

- **Don't trust the PR description.** Authors increasingly let AI write descriptions from the diff; descriptions gloss over slop. The diff is ground truth.
- **Sweep both sides of the diff.** Removing a `_module_mapping` entry or a `hash_params` write is as dangerous as adding one. `git diff` shows both directions.
- **Push back on undisciplined scope.** "Follow-up PR will fix" and "just refactoring while I'm here" both signal an unreviewable diff. Say "split this" if the change bundles a feature and a generic refactor.
- **Calibrate signal.** Be blunt about slop and explicit when work is clean. Both shape future contributions; uniform softness lets bad code through.
- **Grep before objecting.** If unsure whether something is convention, grep for sibling implementations or load `references/repo_conventions.md`. Don't lean on intuition.
- **Guard the base/shared layer.** Agent-authored PRs default to editing base classes (`modeling_qeff.py`, `modeling_auto.py`, `cache_utils.py`) because that's where the call site was, not because every model needs the change. Treat any base-class addition as guilty until proven generic: ask "does EVERY model need this?" and name the layer it belongs in (`models/<model>/`, `modeling_utils.py`, `generation/`, `exporter/`, `compile/`) when it doesn't. Model-specific logic in generic code is a Blocker, not a nit.
- **Don't re-do subagent work in the main thread.** If Subagent B already verified hash plumbing, you cite its finding; don't re-check.

## Iteration

If the user pushes back on a finding, verify with grep or by reading the cited file before retracting. If a force-push lands new commits, re-run Phases 1-4 against the new diff and append-merge new findings into the saved review. If the user says "tweak X" — edit `/tmp/review_pr<N>.md` and reprint, don't regenerate.

## When you're done

Print the review to chat. If the user wants it saved to a file, ask first. Writing under `/tmp/` is reasonable; writing into the repo or posting via `gh pr review` requires explicit confirmation (both are shared-state actions).
