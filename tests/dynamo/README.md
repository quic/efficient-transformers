# QEfficient Dynamo Test Suite

Production-level test coverage for the `use_dynamo=True` export path
(PR #1169). Every architecture that QEff wraps for CausalLM export is
tracked in the coverage matrix — failures surface as red cells, not silent gaps.

## Layout

```
tests/dynamo/
├── conftest.py            # markers, torch>=2.13 guard, xdist aggregation, report emit
├── causal_lm/             # CausalLM + VLM text-side dynamo lanes
│   ├── conftest.py
│   ├── model_registry.py  # single source of truth for every tracked architecture
│   ├── _helpers.py        # shared loading / api_runner / assertion helpers
│   ├── test_export.py           # dynamo + subfunctions export smoke
│   ├── test_compile.py          # FP16 QAIC compile
│   ├── test_precision.py        # MXFP6 / MXFP6+MXINT8 compile-only
│   ├── test_bf16_export.py      # BF16 export-only lane
│   ├── test_output_verification.py  # CPU parity: HF PT == QEff PT == ORT
│   ├── test_generate_execute.py     # HW parity: ORT == QAIC FP16
│   ├── test_continuous_batching.py  # dynamo + subfunctions + CB runtime
│   ├── test_blocking.py             # KV blocking (user-tiled) runtime
│   ├── test_prefix_caching.py       # prefix-caching export shape + QAIC runtime
│   ├── test_device_sampling.py      # on-device sampler vs argmax
│   └── test_multi_device.py         # CCL multi-device compile
├── utils/
│   ├── hash_manager.py       # collision-safe compile/export workdirs (blake2b keyed)
│   ├── hardware_manager.py   # cross-process 4-device pool (flock-based)
│   └── report_generator.py   # HTML + JSON report with SVG charts
└── reports/                  # placeholder — real output goes to test-results/dynamo/
```

## Environment knobs

| Variable | Purpose | Default |
|---|---|---|
| `HF_HUB_CACHE` | HuggingFace model cache root | `~/.cache/huggingface/hub` |
| `HF_HUB_ENABLE_HF_TRANSFER` | Faster HF downloads | unset |
| `QEFF_DYNAMO_REPORT_DIR` | Where to write HTML + JSON reports | `./test-results/dynamo` |
| `QEFF_DYNAMO_WORKDIR` | Root for collision-safe compile/export dirs | `./test-results/dynamo/work` |
| `QEFF_DYNAMO_LOCK_DIR` | Where device-pool file locks live | `./test-results/dynamo/locks` |
| `QEFF_DYNAMO_DEVICE_COUNT` | Number of QAIC devices the pool manages | `4` |
| `QEFF_DYNAMO_DEVICE_IDS` | Explicit CSV device id list (overrides count) | unset |

## Running

Everything (export + compile in parallel, runtime serialised):

```bash
pytest tests/dynamo -n 4 --dist=loadgroup -q
```

CPU-only lanes (no QAIC required):

```bash
pytest tests/dynamo -m "dynamo_export" -n auto
```

QAIC compile only:

```bash
pytest tests/dynamo -m "dynamo_compile"
```

QAIC runtime only:

```bash
pytest tests/dynamo -m "dynamo_runtime" -n 4 --dist=loadgroup
```

Single architecture:

```bash
pytest tests/dynamo -k "llama" -vv
```

## Reports

After any run the suite writes to `test-results/dynamo/`:

```
dynamo-report.html   # dark-theme visual dashboard — open in a browser
dynamo-report.json   # machine-readable for CI tooling
```

The HTML report contains a KPI strip, coverage-matrix donuts, per-column and
per-category stacked health bars, a full arch × feature matrix, and collapsible
per-architecture drill-downs with task outcome, duration, and failure reason.

## Coverage columns

| Column | What it measures |
|---|---|
| `End_To_End_E2E` | Full export → compile → generate pipeline passed |
| `Export_Compile` | Export or compile completed without error |
| `QAIC_Generate_Execute` | QAIC hardware generate completed |
| `CPU_Parity` | HF PT == QEff PT == ORT (hardware-free correctness floor) |
| `HW_Parity_FP16` | ORT == QAIC FP16 (on-device numerical parity) |
| `CB_Dynamo_Subfn` | Dynamo + subfunctions + CB compile + generate |
| `Subfunction_Coverage` | `use_onnx_subfunctions=True` emits ONNX local functions |
| `FP32_Coverage` | FP32 export/parity lane |
| `FP16_Coverage` | FP16 QAIC compile |
| `BF16_Coverage` | BF16 export-only (compile + generate deliberately out of scope) |
| `MXFP6_Coverage` | MXFP6 compile-only |
| `MXINT8_Coverage` | MXFP6 + MXINT8-KV-cache compile-only |
| `CCL_Coverage` | Multi-device (≥2) compile |
| `Blocking_KV_Coverage` | Dynamo + subfunctions + CB + `BlockingMode.KV` |
| `Sampler_Coverage` | On-device sampler == argmax (greedy) |
| `Prefix_Caching_Coverage` | Prefix-caching ONNX shape + QAIC runtime |

## Adding a new architecture

1. Find the smallest available HuggingFace model for that arch — see
   `tests/unit_test/models/test_model_quickcheck.py` for the canonical list.
2. Add a `DynamoModelSpec(...)` entry in `causal_lm/model_registry.py`.
3. Set feature-support flags (`subfunctions_supported`, `continuous_batching_supported`,
   `prefix_caching_supported`, `sampler_supported`, `blocking_kv_supported`) to match
   the wrapper's current capabilities.
4. Run `pytest tests/dynamo/causal_lm/test_export.py -k <arch> -vv` first, then widen.

## Parallelism model

* CPU-bound lanes (export, parity, BF16) run under `pytest-xdist -n auto`. Every
  export/compile writes into a unique blake2b-keyed directory from `hash_manager`
  so parallel workers never collide and cross-run cache reuse works.
* Runtime lanes carry `@pytest.mark.xdist_group(name="qaic-runtime")` so xdist
  places them all on one worker. Tests additionally acquire from `DevicePool`
  (flock-based) before touching a physical device — the correctness backstop for
  any external concurrency.
* Multi-device tests atomically acquire `num_devices` from the pool; partial holds
  are released on failure so no deadlock survives.
