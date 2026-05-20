# Nightly Pipeline

This directory contains the nightly validation pipeline for QEfficient model families.


The pipeline is designed around one rule:

`parallelize across models inside a phase, but keep the phases ordered`

- `export and compile`: parallel
- `generate`: sequential

That gives us better nightly wall-clock time without mixing phase dependencies or corrupting shared artifacts.

## Goal

The nightly pipeline is intended to validate all supported model families with one consistent execution pattern:

1. Export each model and capture the generated ONNX path.
2. Compile each exported model and capture the generated QPC path.
3. Run generation or inference validation after all required artifacts are ready.

The same pattern can be applied to:

- `causal_lm_models`
- `image_text_to_text_models`
- `embedding_models`
- `audio_models`
- `audio_embedding_models`
- `sequence_models`


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
│   ├── test_export_compile.py
│   └── test_generate.py
├── image_text_to_text/
├── embedding_models/
├── audio_models/
├── audio_embedding/
└── sequence_models/
```

Current implementation is centered on `causal_lm`, and the same phase contract is intended to be extended to
the other model families.

## Execution Flow

### Phase 1: Export

- input: model names from `validated_models.json`
- action: export model artifacts in parallel and  compile exported models in parallel
- output: `onnx_path` and export metrics , `qpc_path` and compile metrics in `model_artifacts.json`

Example:

```bash
pytest -n auto tests/nightly_pipeline/causal_lm_models/test_export_compile.py
```


### Phase 2: Generate

- input: `onnx_path` and `qpc_path` from the compile phase
- action: run generation sequentially
- output: generated outputs and performance metrics

Example:

```bash
pytest tests/nightly_pipeline/causal_lm_models/test_generate.py
```

### Phase 3: Validate Results

- input: current artifact JSON files and previous nightly artifact JSON files
- action: compare timing, size, family-specific outputs, and performance metrics using configured tolerances
- output: one family-specific validation CSV per model family in the current artifact directory

The validator uses MAD when `generated_ids` or `embedding` is available, and falls back to exact text/value
assertions for families such as audio embedding and sequence classification.

Example:

```bash
export NIGHTLY_PIPELINE_PREVIOUS_ARTIFACTS_DIR="$PWD/Nightly_Pipeline/$PREVIOUS_BUILD_ID"
pytest tests/nightly_pipeline/test_result_validation.py
```

## CI-Friendly Command Pattern

For a single nightly run: Currently running as a Freestyle Project in Jenkins, but should be converted to a
Pipeline job. The command pattern is:

```bash
export NIGHTLY_PIPELINE_ARTIFACTS_DIR="$PWD/Nightly_Pipeline/$BUILD_ID"
export NIGHTLY_PIPELINE_PREVIOUS_ARTIFACTS_DIR="$PWD/Nightly_Pipeline/$PREVIOUS_BUILD_ID"

pytest -n auto tests/nightly_pipeline/causal_lm_models/test_export_compile.py
pytest tests/nightly_pipeline/causal_lm_models/test_generate.py
pytest tests/nightly_pipeline/test_result_validation.py
```

### Runtime Model Skips

Freestyle jobs can skip selected models without editing `validated_models.json` by passing comma-separated model names
through family-specific environment variables:

- `SKIP_CAUSAL_LM_MODELS`
- `SKIP_IMAGE_TEXT_MODELS`
- `SKIP_EMBEDDING_MODELS`
- `SKIP_AUDIO_MODELS`
- `SKIP_AUDIO_EMBEDDING_MODELS`
- `SKIP_SEQUENCE_MODELS`

Example:

```bash
export SKIP_CAUSAL_LM_MODELS="meta-llama/Llama-3.2-3B,hpcai-tech/grok-1,meta-llama/Llama-3.2-1B"
export SKIP_AUDIO_MODELS="openai/whisper-base"
```

When running inside Docker, pass these variables through `docker exec`:

```bash
sudo docker exec \
  -e SKIP_CAUSAL_LM_MODELS="${SKIP_CAUSAL_LM_MODELS:-}" \
  -e SKIP_IMAGE_TEXT_MODELS="${SKIP_IMAGE_TEXT_MODELS:-}" \
  -e SKIP_EMBEDDING_MODELS="${SKIP_EMBEDDING_MODELS:-}" \
  -e SKIP_AUDIO_MODELS="${SKIP_AUDIO_MODELS:-}" \
  -e SKIP_AUDIO_EMBEDDING_MODELS="${SKIP_AUDIO_EMBEDDING_MODELS:-}" \
  -e SKIP_SEQUENCE_MODELS="${SKIP_SEQUENCE_MODELS:-}" \
  "${BUILD_NAME}" bash -lc "pytest tests/nightly_pipeline/causal_lm_models/test_export_compile.py -n 4"
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
- validation tolerances

Use this file when:

- tuning the nightly hardware profile
- changing default prompt or context length
- updating compilation flags


## License
Check the LICENSE file in the repository root.
