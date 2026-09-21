# Reported Reproducer Config Tests

This suite captures reported QEfficient regression reproducers as pre-merge tests.
Developers should add the reproducer config first, fix the bug, and rerun the
same scenario before review.

## Run the suite

Run the reduced/default matrix:

```bash
pytest tests/reproducer_configs/test_reported_reproducer_configs.py -q -rs
```

Run one scenario by its `RegressionScenario.name`:

```bash
QEFF_REPRODUCER_SCENARIO=hf-transfer-import-order \
pytest tests/reproducer_configs/test_reported_reproducer_configs.py -q
```

Run multiple scenarios with comma-separated values:

```bash
QEFF_REPRODUCER_SCENARIO=hf-transfer-import-order,ccl-default-enabled,embedding-model-compile \
pytest tests/reproducer_configs/test_reported_reproducer_configs.py -q
```

List available scenario names without running them:

```bash
pytest tests/reproducer_configs/test_reported_reproducer_configs.py --collect-only -q
```

## Full-model reproducers

Reduced/tiny models are used by default. Scenarios that need the official model
card are gated and must be enabled explicitly:

```bash
QEFF_REPRODUCER_RUN_FULL_MODELS=1 \
QEFF_REPRODUCER_SCENARIO=<scenario-name> \
pytest tests/reproducer_configs/test_reported_reproducer_configs.py -q
```

Use official model cards instead of tiny reductions where both are present:

```bash
QEFF_REPRODUCER_USE_OFFICIAL_MODELS=1 QEFF_REPRODUCER_RUN_FULL_MODELS=1 \
QEFF_REPRODUCER_SCENARIO=<scenario-name> \
pytest tests/reproducer_configs/test_reported_reproducer_configs.py -q
```

## Runtime report

The suite writes a local Markdown verdict ledger by default at
`tests/reproducer_configs/reproducer_config_results.md`. The report header records
the Hugging Face cache path selected for the run, including fallback to `/tmp`
when the default cache parent is not writable. Override the report location when
needed:

```bash
QEFF_REPRODUCER_REPORT_MD=/tmp/qeff_reproducer_results.md \
QEFF_REPRODUCER_SCENARIO=<scenario-name> \
pytest tests/reproducer_configs/test_reported_reproducer_configs.py -q
```

`--collect-only` does not create or update the verdict report.

## Add a scenario

1. Add one `RegressionScenario` entry in `test_reported_reproducer_configs.py`.
2. Keep reporter options exact in `config`, `load_kwargs`, `export_kwargs`,
   `compile_kwargs`, or `inference_kwargs`.
3. Use a tiny or 2/4-layer model when it faithfully reproduces the issue.
4. Keep the official model card and gate it with `QEFF_REPRODUCER_RUN_FULL_MODELS=1`
   when a reduction is not faithful.
5. Run only the new scenario with `QEFF_REPRODUCER_SCENARIO=<scenario-name>`
   before running the broader matrix.
