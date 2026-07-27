# CI Test Report Generator

`generate_ci_report.py` turns the per-stage JUnit XML from a per-PR CI run into a single,
self-contained `ci_report.html` — a top summary for reviewers/maintainers plus a full
per-test drill-down for the PR owner. It is **pure Python standard library** (no pip installs),
so it runs in the CI container venv as-is.

Layout: KPI strip → **Scenario coverage matrix** (model × per-PR end-to-end scenario grid) →
**Feature coverage matrix** (model × feature grid, causal-LM only) → Stage summary →
By model / config → Failures → per-stage detail → Slowest. The matrices sit directly under
the KPIs so reviewers can see which per-PR scenarios and capabilities passed on which models
without scrolling; scenarios come first because one cell answers the whole
dtype+subfunction+CB+feature question, and the feature grid then breaks the same run down
into atomic capabilities.

## Generate the bundled sample

The `sample/` directory holds synthetic per-stage XML fixtures (not from a real run) that
exercise passes, failures with tracebacks, an error, an xfail, skips, and "Not Run" stages.

```bash
python3 scripts/ci_report/generate_ci_report.py \
    --xml-dir scripts/ci_report/sample \
    --output scripts/ci_report/sample/ci_report.html \
    --pr 1216 --commit ff0f20b98abcdef1234 \
    --branch CI_optimization_fork --profile dummy_layers_model \
    --build-url https://jenkins.example/job/qeff/42/
```

Open `scripts/ci_report/sample/ci_report.html` in a browser.

## Generate from a real run

After the Jenkins job has written its per-stage XML under `tests/`:

```bash
python3 scripts/ci_report/generate_ci_report.py \
    --xml-dir tests --output tests/ci_report.html \
    --pr "$CHANGE_ID" --commit "$GIT_COMMIT" \
    --branch "$BRANCH_NAME" --profile "$TEST_PROFILE"
```

All arguments are optional; when omitted they fall back to the Jenkins environment variables
shown above (`CHANGE_ID`, `GIT_COMMIT`, `BRANCH_NAME`, `TEST_PROFILE`, `BUILD_URL`). Run
`python3 scripts/ci_report/generate_ci_report.py --help` for the full list.

The generator always exits 0 so it never breaks a Jenkins `post { always { ... } }` block.

## Generate from a Jenkins console log

When you only have the Jenkins console log (`ci_logs.log`) — not the per-stage XML — pass
`--console-log`:

```bash
python3 scripts/ci_report/generate_ci_report.py \
    --console-log ci_logs.log \
    --output scripts/ci_report/sample/ci_report.html \
    --pr 1216 --commit 32ff7bc44a5dd4b6f3bdce5e86459dd32098d749 \
    --branch CI_optimization_fork --profile dummy_layers_model
```

Totals come from each stage's pytest summary line (authoritative). Per-test drill-down is
populated from `[gw#] STATUS ...` lines (xdist-verbose stages) and the `slowest N durations`
block; non-verbose stages surface a note listing how many tests were un-attributable.

## What it reads

It scans `--xml-dir` for `--glob` (default `tests_log*.xml`) and maps each per-stage file to a
CI stage (see `STAGE_MAP`). Stages whose XML is absent render as **"Not Run"** rather than
being assumed to pass. If only the merged `tests_log.xml` exists, it falls back to a single
stage with a "provenance lost" banner.
