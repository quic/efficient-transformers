# Rishin Raj — Working Style Notes

Synthesized from ~22 Claude Code sessions on QEfficient (efficient-transformers), covering MoE export/RAM debugging, CI test speedup, model onboarding, PR review, and design/planning work.

## 1. Non-negotiables (things that trigger pushback)

- **No claim without measurement.** "are you 100% sure? run profiling and confirm." Assertions about RAM, perf, or correctness must be backed by numbers (peak RSS, ΔRAM tables, timing breakdowns, parameter-count vs. replication-factor math) — never a plausible-sounding narrative.
- **Partial fixes are not fixes.** If a target metric (e.g. "decode should be ≤4x params in fp16") still doesn't hold after a claimed fix, that's a rejected fix, not progress. Expect to be told to re-open the investigation with more rigor ("ultrathink, use multiple subagents, find out with 100% accuracy").
- **Don't generalize a root cause across models without checking each one.** Burned once claiming "other models are safe" from one bug signature; two counter-examples appeared immediately. Verify per-model, don't pattern-match.
- **No fp32 anywhere in the export path**, even implicitly — assumptions about default dtypes get corrected explicitly ("I don't even want the model to export with fp32").
- **Keep designs minimal — no toggles/env vars for things that should just work by default.** Rejects config sprawl on sight ("I don't want a toggle for that... it should be on by default").
- **Root-cause over workaround**, always. A proposed workaround that doesn't fit the actual constraint gets shut down immediately (e.g. "layerwise is not an option here").
- **Story/task breakdowns must stay coarse** — caps like "10 story points max," 1–2 SP per story (sized for AI-assisted dev), and written in plain human language, not AI-flavored prose.

## 2. What's already trusted / validated (don't second-guess these)

- Cheaper proxy measurements (e.g. an export-only RAM probe instead of a full disagg compile) are fine as long as they target the exact failure point.
- Multi-subagent "ultrathink" fan-out for hard root-cause work — his own idea, use it for anything perf/RAM/correctness-ambiguous.
- Structured `AskUserQuestion`-style choices with a recommended default, especially for scope/risk tradeoffs (RAM-heavy runs on a shared host, venv selection) — he engages with these rather than treating them as friction.
- Self-correction: if you retract an earlier claim after re-verifying, that's accepted without pushback — rigor in the correction matters more than not having erred.
- The `qeff-pr-reviewer` skill's fan-out review methodology (diff capture → shape verdict → mechanical greps → parallel subagent review → synthesis) is his own established process.

## 3. Technical conventions

- **Validation ladder:** never claim parity from export/compile success alone. Validate HF PyTorch → QEff PyTorch → (ONNXRuntime/AI100) as the skill/task calls for.
- **RAM invariant:** fp16 export should stay within ~4x total param count; any excess is a bug worth root-causing, not tuning around.
- **Design consistency gate:** all modeling files should follow the same architecture/pattern; a change is acceptable if it's aligned with existing design and motivated by making tests functional — not a rewrite.
- **Fast iteration:** use 4-layer / tiny-random or from_config models while developing a fix, but always reproduce/confirm at full scale (e.g. 30B) before calling it done — tiny-model results alone are not trusted for production-scale claims.
- **Failure triage discipline:** always separate genuine regressions from harness/environment bugs from flaky tests; report each category with reproduction steps, not a flat pass/fail.
- **New validation harnesses/scripts stay out of `tests/`** unless they are the actual regression gate — ad hoc validation lives in scratch dirs like `scripts/<name>_validation/`.
- **Git/commit hygiene:** signed-off commits (`git commit -s`), explicit `--author "Rishin Raj <rishinr@qti.qualcomm.com>"`, custom message per change. Hard rule: no Claude/AI/co-author attribution anywhere in commits or PR text (matches global CLAUDE.md).
- **PRs target mainline**, not the branch work started on — rebase before opening a PR if development happened on a release branch.
- Reviews and often pastes his own draft of a code change before asking whether it's safe/correct — he authors non-trivial parts himself and wants a check, not a first draft.

## 4. Communication preferences

- Terse, direct, typo-tolerant prompts ("mesure", "avaialble", "run run it") — parse intent, don't correct his typing.
- Wants dense numeric evidence over prose: parameter counts, ΔRAM stage tables, time breakdowns, replication factors.
- `/effort max` before big analytical/design/root-cause tasks — reserve deepest reasoning for planning and hard debugging, not routine execution.
- When a design has open forks, present them as an explicit numbered list of "decisions that change the shape — your call," not a single silently-chosen path.
- Comfortable with long structured markdown + tables for design/CI docs, but will interrupt once satisfied — don't pad past the point he's gotten what he needs.
- Runs sessions in `bypassPermissions` for exploration/build work, but PR/commit/push/`gh pr comment` actions still require explicit go-ahead every time — permissiveness on one axis doesn't imply it on another.

## 5. Workflow habits

- **Empirical loop:** hypothesize → build a minimal/synthetic or tiny-model probe → measure → compare against a known invariant/ratio → only then propose a fix → reproduce at full scale.
- **Consolidation instinct:** proactively asks to merge sprawling design MD files into one doc and clean up stale ones — resists artifact accumulation.
- **Review-before-commit loop for docs:** write → self-review/critique pass ("review it once again without any bias") → revise → only then finalize.
- **Breadth check after a fix:** habitually asks "what else could this affect" across the model family before declaring a bug closed.
- **Sequencing:** fix → run relevant unit + model-specific tests → rebase onto mainline → commit/push/PR, each gated on the previous step succeeding.
- Ends work sessions by stating explicitly what was/wasn't committed, rather than assuming the human will ask.

## 6. Domain / role context

- Deep, load-bearing knowledge of QEff internals: ONNX export subfunctions, `FP16ClipTransform`/`SplitTensorsTransform`, `KVCacheTransform` expert-remap pattern, MXFP6/MXINT8 quantization constraints, BMM operand-order sensitivity in MoE, disaggregated prefill/decode, layerwise export, `offload_pt_weights`. Explain findings at this altitude — he corrects the assistant's architecture claims with specifics, not the other way around.
- Recurring ownership areas: MoE model onboarding + export-RAM correctness (Mixtral, Qwen3(.5)-MoE, Qwen3-VL-MoE, gpt-oss, Gemma), CI/test infra speed and scaling (xdist card pinning, compile/execute split, per-PR gating design), and PR gatekeeping/maintainer review across the repo.
- Also fields real customer/support-style issues (device-ID bugs, video-support questions) — bridges triage and deep engineering.
- Treats JIRA tickets as symptom buckets that can hide multiple unrelated root causes — insists on splitting them apart rather than treating a shared symptom (e.g. "QPC bloat") as one bug.

---
*Compiled from session transcripts; see `/home/rishinr/.claude/projects/-home-rishinr-qeff-pr-review-efficient-transformers/memory/` for the standing project/feedback memory this complements.*
