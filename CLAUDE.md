# efficient-transformers CI — Dev Guide

## Environment

```bash
pyenv activate qeff.vbaddi      # before any test command
export HF_HUB_CACHE=/home/huggingface_hub   # shared model cache (CI host)
```

## CI Pipeline Architecture

**Branch:** `ci/fast-pr-pipeline-xdist-tiny-models`
**Target:** <40 min per PR (was 2h+)
**Measured baseline:** ~10–12 min parallel / ~30 min sequential (see `CI_OPTIMIZATION_RESULTS.md`)

### Pipeline layout (`scripts/Jenkinsfile`)

```
Install QEfficient  (~5 min, venv build in docker)
Phase 1 (parallel):
  CPU Export & Compile  (-n auto --dist worksteal, 30 min cap)
  CLI                   (serial, 30 min cap)
  QAIC LLM cards 0-3   (-n 4 --dist worksteal, 45 min cap)
Phase 2 (parallel):
  QAIC Feature cards 0,1  (QEFF_NUM_QAIC_CARDS=2, OFFSET=0, -n 2, 20 min)
  QAIC Diffusion cards 2,3 (QEFF_NUM_QAIC_CARDS=2, OFFSET=2, -n 2, 60 min)
Phase 3 (sequential):
  QAIC Multimodal cards 0-3  (-n 4, 30 min)
  QAIC Reranker              (serial, 15 min)
Finetune (optional, off by default)
```

Card scheduling: `QEFF_QAIC_CARD_OFFSET` + `QEFF_NUM_QAIC_CARDS` let two stages share 4 cards without collision. `conftest.py::_qaic_device_for_xdist_worker` pins each xdist worker to `offset + (idx % n_cards)`.

### Test profiles

| `QEFF_TEST_PROFILE` | Purpose | Tiny overrides |
|---|---|---|
| `dummy_layers_model` | per-PR fast lane | YES |
| `few_layers_model` | per-PR fast lane | YES |
| `full_layers_model` | nightly (real weights) | NO |

Per-PR: `export QEFF_TEST_PROFILE=dummy_layers_model`
Nightly: `export QEFF_TEST_PROFILE=full_layers_model`

### Tiny model remapping

- Config: `tests/configs/tiny_overrides.json`
- Logic: `tests/utils/tiny_overrides.py::resolve_model_id()`
- Hook: `conftest.py::_install_tiny_model_remap_if_active()` wraps `AutoConfig/AutoModel/AutoTokenizer.from_pretrained`, `load_hf_tokenizer`, `load_hf_processor`, `LoRA.load_adapter`

To add a new model:
1. If a public tiny-random exists → add `"real/model-id": "hf-internal-testing/tiny-..."` to `overrides` in `tiny_overrides.json`
2. If no tiny exists (large VLM, GPTQ, etc.) → add to `skip_no_tiny` list

### Key metrics (per stage, dummy_layers_model profile)

| Stage | Time |
|---|---|
| CPU non-qaic (-n auto) | ~2–4 min |
| QAIC LLM (-n 4) | ~7–14 min |
| QAIC Feature (-n 2) | ~3–5 min |
| QAIC Diffusion (-n 2) | ~7–10 min |
| QAIC Multimodal (-n 4) | ~3–7 min |
| QAIC Reranker (serial) | ~9 s |

## Known issues / intentional skips

- `gpt2`: not remapped — some tests (cache-state, LoRA dim match) need real weights
- `mistralai/Mistral-7B-v0.1`: not remapped — LoRA adapter dim must match base
- `openai/whisper-tiny`: already tiny, no remap needed
- `facebook/wav2vec2-base-960h`: pre-existing test failure, not remapped
- `TheBloke/Llama-2-7B-GPTQ` → `yujiepan/llama-3-tiny-random-gptq-w4`: `load_hf_causal_lm_model` in `QEfficient/utils/test_utils.py` already calls `_resolve_model_id()` internally, so the remap works even for the QUANTIZED_MODELS branch
- `Qwen/Qwen3-VL-Reranker-2B/8B`: in `skip_no_tiny` — no public tiny reranker variants
- `Snowflake/SwiftKV`, `allenai/Molmo-7B-D`: in `skip_no_tiny` — pre-existing, no tiny available

## Running tests locally

```bash
# CPU only (fast sanity check)
pytest tests -m '(not on_qaic) and (not finetune) and (not full_layers) and (not few_layers)' \
  --ignore tests/vllm --ignore tests/unit_test --ignore tests/nightly_pipeline \
  -n 4 --dist worksteal --durations=10

# QAIC LLM tests (4 cards)
QEFF_TEST_PROFILE=dummy_layers_model \
QEFF_NUM_QAIC_CARDS=4 QEFF_QAIC_CARD_OFFSET=0 \
pytest tests -m '(llm_model) and (not qnn) and (not full_layers) and (not few_layers)' \
  --ignore tests/vllm --ignore tests/unit_test --ignore tests/nightly_pipeline \
  -n 4 --dist worksteal --durations=10
```

## Repo layout (CI-relevant)

```
scripts/Jenkinsfile            — active CI pipeline (optimized)
tests/conftest.py              — xdist card-pinning + tiny-model remapping hooks
tests/configs/tiny_overrides.json — model_id → tiny sibling mapping
tests/utils/tiny_overrides.py  — resolve_model_id(), _tiny_lane_active()
tests/configs/causal_model_configs.json — all causal LM test parametrization
```
