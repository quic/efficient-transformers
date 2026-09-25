# vLLM LLM Benchmark Pipeline

This directory contains the vLLM QAIC benchmark runners used by
`scripts/JenkinsfileVllmLlmBenchmark`. There are four runner scripts, each
covering one model domain, sharing all CSV parsing, command building, and
subprocess orchestration from `vllm_benchmark_common.py`:

- `vllm_llm_benchmark.py` — LLM serving configs.
- `vllm_embedding_benchmark.py` — embedding-model serving configs.
- `vllm_audio_benchmark.py` — audio (whisper) serving configs.
- `vllm_vlm_benchmark.py` — vision-language model serving configs.

Each script defines its own `LATEST_MODELS` curated set; `SKIPPED_MODELS`
(the >70B hardware-capability exclusion) is shared and applied to all four.

The runners are CSV-driven. Each input CSV owns one serving mode and produces
one output CSV:

| Mode | Input CSV | Output CSV |
| --- | --- | --- |
| default CB + subfunction | `configs/llm_default_configs.csv` | `llm_default_results.csv` |
| CCL enabled | `configs/llm_ccl_configs.csv` | `llm_ccl_results.csv` |
| blocking | `configs/llm_blocking_configs.csv` | `llm_blocking_results.csv` |
| disagg PD | `configs/llm_disagg_pd_configs.csv` | `llm_disagg_pd_results.csv` |
| embedding | `configs/embedding_configs.csv` | `embedding_results.csv` |
| audio | `configs/audio_configs.csv` | `audio_results.csv` |
| VLM | `configs/vlm_configs.csv` | `vlm_results.csv` |

## Local Dry Run

Dry run validates CSV parsing, command generation, log paths, and output CSV
writing without starting vLLM or using QAIC devices:

```bash
python3 tests/nightly_pipeline/vllm_llm_benchmark/vllm_llm_benchmark.py \
  --config-name default \
  --input-csv tests/nightly_pipeline/vllm_llm_benchmark/configs/llm_default_configs.csv \
  --output-csv /tmp/vllm_llm_smoke/llm_default_results.csv \
  --results-dir /tmp/vllm_llm_smoke \
  --rows 1 \
  --dry-run
```

Row 1 in the default CSV is `openai-community/gpt2`, intended for a quick smoke
check before running all rows.

The embedding and audio runners work the same way:

```bash
python3 tests/nightly_pipeline/vllm_llm_benchmark/vllm_embedding_benchmark.py \
  --config-name embedding \
  --input-csv tests/nightly_pipeline/vllm_llm_benchmark/configs/embedding_configs.csv \
  --output-csv /tmp/vllm_embedding_smoke/embedding_results.csv \
  --results-dir /tmp/vllm_embedding_smoke \
  --rows 1 \
  --dry-run

python3 tests/nightly_pipeline/vllm_llm_benchmark/vllm_audio_benchmark.py \
  --config-name audio \
  --input-csv tests/nightly_pipeline/vllm_llm_benchmark/configs/audio_configs.csv \
  --output-csv /tmp/vllm_audio_smoke/audio_results.csv \
  --results-dir /tmp/vllm_audio_smoke \
  --rows 1 \
  --dry-run

python3 tests/nightly_pipeline/vllm_llm_benchmark/vllm_vlm_benchmark.py \
  --config-name vlm \
  --input-csv tests/nightly_pipeline/vllm_llm_benchmark/configs/vlm_configs.csv \
  --output-csv /tmp/vllm_vlm_smoke/vlm_results.csv \
  --results-dir /tmp/vllm_vlm_smoke \
  --rows 1 \
  --dry-run
```

## Consolidated Published CSV

After all benchmark categories complete, run `merge_published_results.py` to generate
a single consolidated published CSV with only the key fields for team distribution:

```bash
python3 tests/nightly_pipeline/vllm_llm_benchmark/merge_published_results.py \
  --results-dir vllm_llm_results/ \
  --output vllm_llm_results/consolidated_published_results.csv
```

This merges all `*_results.csv` files (from LLM, embedding, audio, VLM categories)
into one file with 22 key columns: model, model_category, config_name, config_summary,
status, export_compile_time_s, prefill_mdp_export_compile_time_s,
prefill_export_compile_time_s, decode_export_compile_time_s,
encode_export_compile_time_s, mean_ttft_s, mean_tpot_s, mean_itl_s, decode_TPS,
request_throughput_req_s, vllm_qaic_branch, qaic_disagg_branch, qserve_branch,
qeff_branch, qaic_sdk_version, server_command, client_command. Model categories
are auto-detected based on config_name. The five `*_export_compile_time_s` columns
are populated only for the fields relevant to that row's server type — e.g. a
non-disagg (`api_server`) row only fills `export_compile_time_s`, while a disagg
row fills the per-stage `prefill_mdp_export_compile_time_s`/
`prefill_export_compile_time_s`/`decode_export_compile_time_s`/
`encode_export_compile_time_s` fields instead.

## HTML Report Generation

After generating the consolidated CSV, create an HTML report for email distribution:

```bash
python3 tests/nightly_pipeline/vllm_llm_benchmark/generate_html_report.py \
  --csv vllm_llm_results/consolidated_published_results.csv \
  --output vllm_llm_results/benchmark_report.html
```

The HTML report includes:
- **Environment Information**: Branch details (vLLM QAIC, QAIC Disagg, QServe, QEff) and QAIC SDK version
- **Test Results Summary**: Total tests, passed, and failed counts
- **Detailed Test Results**: Table with model name, category, config, status, per-stage
  export/compile times (export/compile, prefill MDP, prefill, decode, encode), and
  performance metrics (TTFT/TPOT/ITL/decode TPS/throughput)

## Jenkins Flow

The Jenkins pipeline:

1. Creates or reuses a Python venv.
2. Clones `https://github.com/qualcomm/vllm-qaic.git`.
3. Runs `scripts/install.sh aot`, which installs QEfficient internally.
4. Clones and installs `qaic-disagg`.
5. Runs the selected LLM/embedding/audio/VLM config CSVs.
6. Runs `merge_published_results.py` to generate consolidated published CSV.
7. Runs `generate_html_report.py` to render `benchmark_report.html`.
8. Archives `vllm_llm_results/**/*.csv` and `vllm_llm_results/**/*.log`.
9. Emails `benchmark_report.html` (rendered inline) plus the consolidated CSV
   attachment to `EMAIL_RECIPIENTS`, if set — this fires from the pipeline's
   `post { always { ... } }` block regardless of whether the overall build
   passed or failed, as long as the HTML report file exists (the subject line
   includes the build status). If the report could not be generated at all,
   a plain-text fallback failure notification is sent instead.

Use `DRY_RUN=true` and `ROWS_DEFAULT=1` to verify the gpt2 command and output
CSV/log generation on a Jenkins agent without consuming QAIC runtime. The
embedding, audio, and VLM stages have their own `ROWS_EMBEDDING`/`ROWS_AUDIO`/
`ROWS_VLM` and `RUN_EMBEDDING`/`RUN_AUDIO`/`RUN_VLM` params for the same
purpose.

The VLM stage additionally exposes three `choice()` params that filter rows
within `configs/vlm_configs.csv` (rather than selecting a different CSV/stage
per combination):

- `VLM_DISAGG_MODE` (`ALL`/`ED`/`PD`/`EPD`) — filters by the `disagg_mode`
  column: encode-decode, prefill-decode (no vision/encode stage), or
  encode-prefill-decode.
- `VLM_SPECIALIZATION_MODE` (`ALL`/`single`/`multi`) — filters by the
  `specialization_mode` column: single vs multi resolution/image-count
  buckets on the client's `--random-mm-bucket-config`.
- `VLM_BLOCKING_MODE` (`ALL`/`blocking`/`non_blocking`) — filters by the
  `blocking_mode` column.

These map to the runner's own `--disagg-mode`/`--specialization-mode`/
`--blocking-mode` CLI flags (also `ALL` by default), which no-op for any CSV
row lacking the corresponding column — so the LLM/embedding/audio CSVs are
unaffected.

## HuggingFace Cache Configuration

The `HF_HOME` Jenkins parameter controls where HuggingFace model caches are stored:

- **Default (empty):** `${HOME}/.cache/huggingface` on the Jenkins agent
- **Custom value:** Any path, e.g., `/mnt/fast_storage/hf_cache`

This is useful when:
- The default cache location has insufficient space
- You want to use a faster storage device (NVMe, SSD)
- You want to share a pre-populated cache across multiple builds
- You need to isolate caches for different benchmark runs

The cache is bind-mounted into the Docker container at the same path, so models
downloaded during one run are available to subsequent runs without re-downloading.

## Device Group Configuration

Device groups specify which QAIC devices are allocated to each benchmark run. They can be configured in two ways:

### CSV-based Configuration (Default)

Each config CSV includes device group columns that specify which devices to use:

| Category | Single-Device | Disaggregated |
| --- | --- | --- |
| **LLM (default/CCL/blocking)** | `device_group` (col 7) | N/A |
| **LLM (disagg_pd)** | N/A | `prefill_device_group` (col 9), `decode_device_group` (col 10) |
| **Embedding** | `additional_config` JSON | N/A |
| **Audio** | `additional_config` JSON | N/A |
| **VLM** | N/A | `encode_device_group` (col 14), `prefill_device_group` (col 15), `decode_device_group` (col 16) |

Example values:
- Single device: `0` or `1`
- Multiple devices: `0,1,2,3` or `2,3`
- Empty: uses default or falls back to `additional_config`

### Jenkins Parameter Override

Jenkins parameters allow overriding device groups for an entire benchmark run without modifying CSVs:

| Parameter | Applies To | Example |
| --- | --- | --- |
| `DEVICE_GROUP_DEFAULT` | LLM default configs | `0,1,2,3` |
| `DEVICE_GROUP_CCL` | LLM CCL configs | `0,1` |
| `DEVICE_GROUP_BLOCKING` | LLM blocking configs | `2,3` |
| `DEVICE_GROUP_DISAGG_PD_PREFILL` | LLM disagg PD prefill stage | `0,1` |
| `DEVICE_GROUP_DISAGG_PD_DECODE` | LLM disagg PD decode stage | `2,3` |
| `DEVICE_GROUP_EMBEDDING` | Embedding configs | `0` |
| `DEVICE_GROUP_AUDIO` | Audio configs | `0` |
| `DEVICE_GROUP_VLM_ENCODE` | VLM encode stage | `0` |
| `DEVICE_GROUP_VLM_PREFILL` | VLM prefill stage | `1` |
| `DEVICE_GROUP_VLM_DECODE` | VLM decode stage | `2,3` |

When a Jenkins parameter is set (non-empty), it **overrides** the corresponding CSV column for all rows in that benchmark run. When empty (default), CSV values are used.

**Use case:** Test the same model configs on different device allocations without creating new CSV rows.

### Device Group Override Implementation

Device group overrides work in two ways:

**Method 1: Direct Column Override** (LLM default/CCL/blocking, VLM)
- Device group is in a dedicated CSV column
- Override directly replaces the column value
- Example: `device_group` column → override value

**Method 2: JSON Override** (Embedding, Audio)
- Device group is in `additional_config` JSON
- Override parses JSON, updates field, re-serializes
- Example: `{"device_group": [0], ...}` → `{"device_group": [1], ...}`

**Method 3: Separate Stage Overrides** (LLM disagg_pd, VLM)
- Separate device group columns for each stage
- Each stage can be overridden independently
- Example: `prefill_device_group` and `decode_device_group` overridden separately

## Dry-Run Testing

A comprehensive dry-run test suite validates device group overrides across all 7 categories:

```bash
# Run all tests (recommended)
python3 dry_run_test.py

# Or use bash script
bash DRY_RUN_TEST.sh
```

Each test:
- Runs with a single row from the config CSV
- Applies specified device group override
- Runs in dry-run mode (no actual server/client execution)
- Validates command generation
- Checks output CSV creation

### Test Coverage

| # | Category | Override | Test |
|---|----------|----------|------|
| 1 | LLM Default | `device_group="0,1"` | ✅ |
| 2 | LLM CCL | `device_group="0,1"` | ✅ |
| 3 | LLM Blocking | `device_group="2,3"` | ✅ |
| 4 | LLM Disagg PD | `prefill="0,1"`, `decode="2,3"` | ✅ |
| 5 | Embedding | `device_group="0"` (JSON) | ✅ |
| 6 | Audio | `device_group="0"` (JSON) | ✅ |
| 7 | VLM | `encode="0"`, `prefill="1"`, `decode="2,3"` | ✅ |

### Expected Output

```
==================================================
Dry-Run Test: All Categories with Device Group Overrides
==================================================

Test Results:
✓ llm_default
✓ llm_ccl
✓ llm_blocking
✓ llm_disagg_pd
✓ embedding
✓ audio
✓ vlm

Passed: 7/7
Failed: 0/7

✓ No errors found in logs
```

### Verifying Device Group Overrides

Check generated commands:
```bash
grep "Server command" /tmp/vllm_dry_run_test/llm_default_output.log
# Should include: --additional-config '{"device_group": [0, 1], ...}'
```

Check output CSV:
```bash
head -5 /tmp/vllm_dry_run_test/llm_default_results.csv
# Should have device_group column with overridden value
```

Check for errors:
```bash
grep -r "ERROR\|FAILED\|Traceback" /tmp/vllm_dry_run_test/*.log
# Should return nothing if all tests passed
```

### Manual Testing

Test a single category manually:

```bash
python3 tests/nightly_pipeline/vllm_llm_benchmark/vllm_llm_benchmark.py \
  --config-name default \
  --input-csv tests/nightly_pipeline/vllm_llm_benchmark/configs/llm_default_configs.csv \
  --output-csv /tmp/test_llm_default.csv \
  --rows 1 \
  --latest-models-only false \
  --results-dir /tmp \
  --base-dir . \
  --python-bin python3 \
  --vllm-qaic-dir . \
  --server-ready-timeout-s 3600 \
  --device-group-override "0,1" \
  --dry-run
```

Test CCL LLM with device group override:
```bash
python3 tests/nightly_pipeline/vllm_llm_benchmark/vllm_llm_benchmark.py \
  --config-name ccl \
  --input-csv tests/nightly_pipeline/vllm_llm_benchmark/configs/llm_ccl_configs.csv \
  --output-csv /tmp/test_llm_ccl.csv \
  --rows 1 \
  --latest-models-only false \
  --results-dir /tmp \
  --base-dir . \
  --python-bin python3 \
  --vllm-qaic-dir . \
  --server-ready-timeout-s 3600 \
  --device-group-override "0,1" \
  --dry-run
```

Test blocking LLM with device group override:
```bash
python3 tests/nightly_pipeline/vllm_llm_benchmark/vllm_llm_benchmark.py \
  --config-name blocking \
  --input-csv tests/nightly_pipeline/vllm_llm_benchmark/configs/llm_blocking_configs.csv \
  --output-csv /tmp/test_llm_blocking.csv \
  --rows 1 \
  --latest-models-only false \
  --results-dir /tmp \
  --base-dir . \
  --python-bin python3 \
  --vllm-qaic-dir . \
  --server-ready-timeout-s 3600 \
  --device-group-override "2,3" \
  --dry-run
```

Test disaggregated prefill-decode LLM with separate device group overrides:
```bash
python3 tests/nightly_pipeline/vllm_llm_benchmark/vllm_llm_benchmark.py \
  --config-name disagg_pd \
  --input-csv tests/nightly_pipeline/vllm_llm_benchmark/configs/llm_disagg_pd_configs.csv \
  --output-csv /tmp/test_llm_disagg_pd.csv \
  --rows 1 \
  --latest-models-only false \
  --results-dir /tmp \
  --base-dir . \
  --python-bin python3 \
  --vllm-qaic-dir . \
  --server-ready-timeout-s 3600 \
  --device-group-prefill-override "0,1" \
  --device-group-decode-override "2,3" \
  --dry-run
```

Test embedding with device group override:
```bash
python3 tests/nightly_pipeline/vllm_llm_benchmark/vllm_embedding_benchmark.py \
  --config-name embedding \
  --input-csv tests/nightly_pipeline/vllm_llm_benchmark/configs/embedding_configs.csv \
  --output-csv /tmp/test_embedding.csv \
  --rows 1 \
  --latest-models-only false \
  --results-dir /tmp \
  --base-dir . \
  --python-bin python3 \
  --vllm-qaic-dir . \
  --server-ready-timeout-s 3600 \
  --device-group-override "0" \
  --dry-run
```

Test audio with device group override:
```bash
python3 tests/nightly_pipeline/vllm_llm_benchmark/vllm_audio_benchmark.py \
  --config-name audio \
  --input-csv tests/nightly_pipeline/vllm_llm_benchmark/configs/audio_configs.csv \
  --output-csv /tmp/test_audio.csv \
  --rows 1 \
  --latest-models-only false \
  --results-dir /tmp \
  --base-dir . \
  --python-bin python3 \
  --vllm-qaic-dir . \
  --server-ready-timeout-s 3600 \
  --device-group-override "0" \
  --dry-run
```

Test VLM with separate device group overrides:
```bash
python3 tests/nightly_pipeline/vllm_llm_benchmark/vllm_vlm_benchmark.py \
  --config-name vlm \
  --input-csv tests/nightly_pipeline/vllm_llm_benchmark/configs/vlm_configs.csv \
  --output-csv /tmp/test_vlm.csv \
  --rows 1 \
  --latest-models-only false \
  --results-dir /tmp \
  --base-dir . \
  --python-bin python3 \
  --vllm-qaic-dir . \
  --server-ready-timeout-s 3600 \
  --device-group-encode-override "0" \
  --device-group-prefill-override "1" \
  --device-group-decode-override "2,3" \
  --dry-run
```

## CSV Notes

Common columns:

- `model`, `server_type`, `client_type`, `host`, `port`
- `PL`, `GL`, `CL`, `num_prompts`, `max_concurrency`
- `backend`, `endpoint`, `dataset_name`, `ignore_eos`
- `server_extra_args`, `client_extra_args` for mode-specific flags not yet
  modeled as first-class columns

For `server_type=api_server`, the runner builds
`python -m vllm.entrypoints.openai.api_server`. The `additional_config` JSON is
constructed from columns such as `device_group`, `use_onnx_subfunctions`,
`ccl_enabled`, `comp_ctx_lengths_prefill`, `comp_ctx_lengths_decode`, and the
blocking columns.

For `server_type=qaic_disagg`, the runner builds `python -m qaic_disagg` using
the PD columns such as `prefill_device_group`, `decode_device_group`,
`prefill_override_qaic_config`, and `decode_override_qaic_config`. VLM rows
additionally use the encode-stage columns (`encode_port`,
`encode_device_group`, `encode_max_num_seqs`, `encode_override_qaic_config`)
for EPD layouts; ED rows leave the prefill columns blank, and PD rows leave
the encode columns blank — `qaic_disagg`'s CLI flags for the omitted stage are
simply not passed.

The embedding CSV (`configs/embedding_configs.csv`) has four rows per model,
one per `pooling_method` (`mean`, `avg`, `cls`, `max`), passed through the
`additional_config` column's `override_qaic_config.pooling_method` key, along
with the qserve-only client columns `tokenizer` and `random_range_ratio`.

The audio CSV (`configs/audio_configs.csv`) has one row per whisper model and
uses the qserve-only client columns `dataset_path`, `hf_subset`, `hf_split`,
and `hf_output_len` for the `hf` dataset backend.

The VLM CSV (`configs/vlm_configs.csv`) has one row per (model, disagg_mode,
specialization_mode, blocking_mode) combination. Rows tagged
`*_verified` reproduce a user-supplied example command token-for-token
(modulo host/port/paths); rows tagged `*_inferred_unverified` are best-effort
constructions for combinations no example was supplied for, and should be
validated against real hardware output before being treated as a golden
reference. Multi-specialization rows set `random_mm_limit_mm_per_prompt`,
`random_mm_bucket_config` (a Python-tuple-keyed literal, not JSON — passed
through to `vllm bench serve --random-mm-bucket-config` verbatim), and the
server-side `limit_mm_per_prompt` JSON column. Blocking rows embed a nested
`qaic_config":{"enable_blocking":true,...}` block inside the relevant
`*_override_qaic_config` column, the same mechanism used by the existing LLM
blocking CSV.

## Model Filtering

After `--rows` selects which CSV rows are in play, two model-name filters are
applied, in order:

1. **`SKIPPED_MODELS`** (always enforced, no override): a hardcoded set of
   models too large to run in this pipeline (>70B real params) — currently
   `zai-org/GLM-4.5` (~355B, MoE) and `hpcai-tech/grok-1` (~314B).
2. **`LATEST_MODELS_ONLY`** (`--latest-models-only`, default `true`): when
   enabled, restricts runs to the curated `LATEST_MODELS` set defined in the
   runner script for the domain being run (`vllm_llm_benchmark.py`,
   `vllm_embedding_benchmark.py`, `vllm_audio_benchmark.py`, or
   `vllm_vlm_benchmark.py`). Set to `false` (Jenkins param `LATEST_MODELS_ONLY`)
   to run all non-skipped models in the CSV.

`zai-org/GLM-4.5` is in both LLM's `LATEST_MODELS` and `SKIPPED_MODELS` — the
skip-list always wins, so it is excluded regardless of the latest-only flag.

For the VLM runner specifically, three additional filters apply after the two
above: `--disagg-mode`, `--specialization-mode`, and `--blocking-mode` (each
default `ALL`), matching the `disagg_mode`/`specialization_mode`/
`blocking_mode` CSV columns. They no-op for rows/CSVs lacking the column, so
they have no effect on the LLM/embedding/audio runners.

Skipped rows are logged to stdout with the reason and do not appear in the
output CSV, matching how `enabled=false` rows are already excluded silently.


