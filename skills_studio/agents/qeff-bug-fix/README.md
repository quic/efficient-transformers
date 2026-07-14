# QEff Bug Fix Agent

Subagent that turns a bug report into a reproduced, minimally-fixed, tested
change in this repo — without committing, pushing, or opening a PR.

## When to use it

Trigger it when the user provides a JIRA ticket key (e.g. `PROJ-1234`), a JIRA
URL, a pasted bug report, or a reproducer for `quic/efficient-transformers`
and asks for a bug fix. Trigger phrases: "fix this JIRA", "fetch JIRA ...",
"bug fix for QEff", "reproduce and fix this issue".

## What it does

1. Ingests the bug report (Qualcomm JIRA DC via REST API, public tracker via
   `WebFetch`, or pasted text).
2. Builds a mental model of the affected code path and generates at least two
   hypothesis families from the symptom before picking one to pursue.
3. Writes a standalone reproduction script into `/home/rishinr/e2e_stack/`
   (modeled on `examples/`) so the failure can be re-run by the agent or the
   user.
4. Designs and applies a minimal fix, adds or updates a regression test, and
   iterates a bounded verify-loop (reproduce → fix → re-verify → check for
   collateral damage → lint) until the evidence ladder is green — or reports
   honestly which rungs remain unverified.

## What it does not do

- Does not run `git commit`, `git push`, or `gh pr create`, and will not run
  destructive git commands. It edits files only.
- Does not scope-creep beyond the reported bug; adjacent issues are surfaced
  as follow-ups, not fixed inline.

See `AGENT.md` for the full workflow, the symptom → hypothesis signature
table, the JIRA REST API setup, and the evidence-ladder verification
checklist.

## Wiring

Same convention as skills in `skills_studio/skills/`: `make claude` symlinks
`skills_studio/agents` into `.claude/agents/`. Run `make clean-ai` to remove
the generated link.
