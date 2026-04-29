# Nightly Pipeline

This directory contains the nightly validation pipeline for QEfficient model families.

For a presentation-focused version of the flow, see [PIPELINE_DIAGRAM.md](/home/abukhoye/efficient-transformers/tests/nightly_pipeline/PIPELINE_DIAGRAM.md).

The pipeline is designed around one rule:

`parallelize across models inside a phase, but keep the phases ordered`

- `export`: parallel
- `compile`: parallel
- `generate`: sequential

That gives us better nightly wall-clock time without mixing phase dependencies or corrupting shared artifacts.

## Goal

The nightly pipeline is intended to validate all supported model families with one consistent execution pattern:

1. Export each model and capture the generated ONNX path.
2. Compile each exported model and capture the generated QPC path.
3. Run generation or inference validation after all required artifacts are ready.

The same pattern can be applied to:

- `causal_lm`
- `image_text_to_text`
- `embedding_models`
- `audio_models`
- `audio_embedding`
- `sequence_models`

## Pipeline Diagram

```mermaid
flowchart TD
    A[validated_models.json<br/>model list by family]
    B[pipeline_configs.json<br/>phase parameters]

    A --> E
    B --> E

    subgraph P1[Phase 1: Export]
        direction LR
        E[pytest -n auto<br/>test_export.py]
        E1[Worker 1<br/>Model A export]
        E2[Worker 2<br/>Model B export]
        E3[Worker N<br/>Model N export]
        E --> E1
        E --> E2
        E --> E3
    end

    subgraph S[(Shared Artifacts Store)]
        D[Artifacts directory]
        F[model_artifacts.json]
        L[model_artifacts.json.lock]
    end

    E1 --> D
    E2 --> D
    E3 --> D
    E1 --> F
    E2 --> F
    E3 --> F
    E1 -. lock .-> L
    E2 -. lock .-> L
    E3 -. lock .-> L

    subgraph P2[Phase 2: Compile]
        direction LR
        C[pytest -n auto<br/>test_compile.py]
        C1[Worker 1<br/>Model A compile]
        C2[Worker 2<br/>Model B compile]
        C3[Worker N<br/>Model N compile]
        C --> C1
        C --> C2
        C --> C3
    end

    F --> C
    C1 --> F
    C2 --> F
    C3 --> F
    C1 --> D
    C2 --> D
    C3 --> D

    subgraph P3[Phase 3: Generate]
        direction TB
        G[pytest<br/>test_generate.py]
        G1[Model A generate]
        G2[Model B generate]
        G3[Model N generate]
        G --> G1 --> G2 --> G3
    end

    F --> G
    G3 --> R[Final metrics and outputs]
```

## Why This Shape

### Export in Parallel

Each model export is independent from the others, so pytest-xdist can reduce the total nightly time substantially.

Tradeoff:

- More workers reduce wall time.
- More workers also increase host memory pressure.

The worker count should therefore match the nightly machine budget, not just CPU count.

### Compile in Parallel

Compilation is also model-local and is a good candidate for parallel workers, especially when many models are already exported.

This phase benefits from:

- CPU parallelism
- overlapping compiler execution
- better overnight throughput

### Generate Sequentially

Generation is intentionally serialized because it is the most measurement-sensitive phase.

Running it sequentially gives:

- stable device ownership
- cleaner latency and throughput numbers
- easier failure diagnosis
- no contention between models for inference hardware

## Important Scheduling Rule

Do **not** run all three phases in a single `pytest -n ...` invocation.

Even though export and compile are parallel-safe, pytest-xdist does not understand cross-test phase dependencies. If all modules are scheduled together, a compile test can start before the corresponding export test has finished.

The correct orchestration is:

1. run all export tests
2. run all compile tests
3. run all generate tests

Parallelism happens **within** a phase, not **across** phases.

## Artifact Strategy

The nightly pipeline uses a shared artifact manifest and a deterministic artifact root.

### Shared Artifact Root

The artifact directory is resolved in this order:

1. `NIGHTLY_PIPELINE_ARTIFACTS_DIR`
2. `output_dir` from `configs/pipeline_configs.json`
3. fallback timestamped cache directory

This ensures all xdist workers write into the same nightly run directory.

### Shared Manifest

The file `model_artifacts.json` is the phase handoff contract between tests.

Typical entries look like:

```json
{
  "openai-community/gpt2": {
    "onnx_path": "...",
    "export_time": 12.3,
    "qpc_path": "...",
    "compile_time": 45.1
  }
}
```

### Race Condition Protection

Parallel workers can update the same manifest safely because `tests/nightly_pipeline/conftest.py` now does:

- file locking with `model_artifacts.json.lock`
- reload-latest-before-write
- merge-on-write by model key
- atomic `os.replace()` for the final file write

This prevents one worker from overwriting another worker's updates.

## Directory Layout

```text
tests/nightly_pipeline/
├── README.md
├── conftest.py
├── configs/
│   ├── pipeline_configs.json
│   └── validated_models.json
├── causal_lm/
│   ├── test_export.py
│   ├── test_compile.py
│   └── test_generate.py
├── image_text_to_text/
├── embedding_models/
├── audio_models/
├── audio_embedding/
└── sequence_models/
```

Current implementation is centered on `causal_lm`, and the same phase contract is intended to be extended to the other model families.

## Execution Flow

### Phase 1: Export

- input: model names from `validated_models.json`
- action: export model artifacts in parallel
- output: `onnx_path` and export metrics in `model_artifacts.json`

Example:

```bash
pytest -n auto tests/nightly_pipeline/causal_lm/test_export.py
```

### Phase 2: Compile

- input: `onnx_path` from the export phase
- action: compile exported models in parallel
- output: `qpc_path` and compile metrics in `model_artifacts.json`

Example:

```bash
pytest -n auto tests/nightly_pipeline/causal_lm/test_compile.py
```

### Phase 3: Generate

- input: `qpc_path` from the compile phase
- action: run generation sequentially
- output: generated outputs and performance metrics

Example:

```bash
pytest tests/nightly_pipeline/causal_lm/test_generate.py
```

## CI-Friendly Command Pattern

For a single nightly run:

```bash
export NIGHTLY_PIPELINE_ARTIFACTS_DIR="$PWD/Nightly_Pipeline/$BUILD_ID"

pytest -n auto tests/nightly_pipeline/causal_lm/test_export.py
pytest -n auto tests/nightly_pipeline/causal_lm/test_compile.py
pytest tests/nightly_pipeline/causal_lm/test_generate.py
```

For multiple model families, keep the same phase ordering once the corresponding
family-specific `test_export.py`, `test_compile.py`, and `test_generate.py` files exist:

```bash
# Export phase
pytest -n auto \
  tests/nightly_pipeline/causal_lm/test_export.py \
  tests/nightly_pipeline/embedding_models/test_export.py \
  tests/nightly_pipeline/audio_models/test_export.py

# Compile phase
pytest -n auto \
  tests/nightly_pipeline/causal_lm/test_compile.py \
  tests/nightly_pipeline/embedding_models/test_compile.py \
  tests/nightly_pipeline/audio_models/test_compile.py

# Generate phase
pytest \
  tests/nightly_pipeline/causal_lm/test_generate.py \
  tests/nightly_pipeline/embedding_models/test_generate.py \
  tests/nightly_pipeline/audio_models/test_generate.py
```

## Config Files

### `configs/validated_models.json`

Defines the nightly model inventory by family.

Use this file when:

- adding a new model to nightly coverage
- splitting coverage by model family
- removing unstable or deprecated models

### `configs/pipeline_configs.json`

Defines per-phase execution settings, such as:

- output directory
- export parameters
- compile parameters
- generation parameters

Use this file when:

- tuning the nightly hardware profile
- changing default prompt or context length
- updating compilation flags

## Design Summary

If you need a short explanation for the team, use this:

1. We parallelize where model jobs are independent: export and compile.
2. We serialize where phase ordering and runtime stability matter: generate.
3. We never let pytest schedule phases out of order.
4. We use one shared artifact manifest as the handoff contract.
5. We protect that manifest with lock + merge + atomic replace to avoid races.

## Extension Rule For New Families

When onboarding a new nightly family, keep the same contract:

1. add the model list to `validated_models.json`
2. add `test_export.py`
3. add `test_compile.py`
4. add `test_generate.py`
5. write phase outputs into `model_artifacts.json`

That keeps every model family aligned with the same nightly orchestration model.
