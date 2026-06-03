# Validation Playbook

Use this file to choose the right validation target and test command after implementing support.

## Acceptance Gate
- Primary gate: `tests/test_model_quickcheck.py`
- Treat this as the default single-test entrypoint for onboarding work unless the model category is clearly outside its scope.

## Coverage Tiers

### Runtime parity
- Use when the repo already has a stable CPU parity path for the category.
- Expect agreement across:
  - upstream HF PyTorch
  - transformed QEff PyTorch
  - ORT export/runtime
- Typical targets:
  - decoder-only causal LM families already covered in `CAUSAL_RUNTIME_MODEL_IDS`
  - text side of some VLMs when the full VLM runtime path is not yet stable
  - embeddings and CTC families with existing helper paths

### Export smoke
- Use when export is supported but stable CPU parity is not yet part of the consolidated path.
- Typical targets:
  - many VLM full-model exports
  - Whisper export
  - AWQ export
  - newly onboarded families that still need a future runtime parity lane

## Test Editing Strategy
- Prefer extending existing dictionaries or parametrized cases in `tests/test_model_quickcheck.py`.
- Keep the new model aligned with the existing test tier.
- If the new model needs a special environment flag, set it only around the relevant test path and keep the default quickcheck lane clean.

## Tiny Model Selection Order
1. Exact-architecture tiny model on Hugging Face Hub.
2. Vendor or community tiny checkpoint that preserves the same feature path.
3. Config-derived tiny model created from upstream config semantics.

## Commands

### Full quickcheck
```bash
python -m pytest -q tests/test_model_quickcheck.py -n auto
```

### Targeted causal LM slice
```bash
python -m pytest -q tests/test_model_quickcheck.py -n auto -k "causal_lm_cpu_runtime_parity_with_api_runner and <model_type>"
```

### Targeted VLM slice
```bash
python -m pytest -q tests/test_model_quickcheck.py -n auto -k "vlm"
```

### Targeted embedding, audio, or Whisper slices
```bash
python -m pytest -q tests/test_model_quickcheck.py -n auto -k "embedding or audio or whisper"
```

## Export and Runtime Checks
- When export behavior changes, confirm the emitted ONNX retains retained-state outputs where expected.
- When cache behavior changes, validate both tuple-based and object-based cache paths if the repo still uses both boundaries.
- When compile-specific shapes differ from PyTorch runtime shapes, update `QEfficient/transformers/models/modeling_auto.py` and the runtime input helpers together.

## Special Cases
- For GLM-style MLA or absorption paths, keep the validation model on the exact path that exercises compressed cache semantics.
- For chunking, sliding-window, or hybrid cache changes, rerun the relevant model-family subset and then the full quickcheck if shared helpers were touched.
- For AWQ or GPTQ changes, validate quantizer replacement and export smoke in the same turn.

## Claiming Success
- Do not claim support from export alone when a stable runtime parity lane exists for that category.
- Do not stop at a targeted green run if you touched shared cache, export, or modeling-auto code.
