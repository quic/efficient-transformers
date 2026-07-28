# Golden outputs

Committed reference token streams for the causal-LM parity tests. They let the Jenkins
**QAIC LLM** stage run only the on-device (QAIC) leg and compare its output against a
stored HuggingFace PyTorch reference, instead of re-running the three CPU legs
(`hf`, `qeff_hf` / torch-KV, `ORT` / ONNXRuntime) on every build.

## Layout

A single committed file holds every family, model and variant:

```
golden_outputs/goldens.json
```

The file is a `family → model → variant → record` nesting:

```json
{
  "causal_lm": {
    "gpt2": {
      "nocb_float16_pl8_cl32_gl24_<config_fp>_<digest>": {
        "pytorch_hf_tokens": [[...]],
        "config_fp": "…",
        "gen_len": 24,
        "timestamp": "…"
      }
    }
  }
}
```

- `<family>` — model family, e.g. `causal_lm`. New families are added as new top-level
  keys in the same file, so the mechanism scales without adding files.
- The model key is the raw HuggingFace model id.

## How variants are keyed

The HF token stream is a pure function of the model, so a variant key folds in exactly
the inputs that change it:

- continuous-batching flag, dtype, `prompt_len`, `ctx_len`, `generation_len`,
  `full_batch_size`, the prompts,
- and a `config_fp` fingerprint of the effective config (model_type, layer / head / dim /
  vocab / intermediate-size counts) — this is what distinguishes the `dummy_layers`,
  `few_layers` and `full_layers` scopes.

It is **independent of `qaic_config`** (blocking, CCL, speculative decoding, KV-head
replication), so one golden is reused across every QAIC variant of the same model.

## Regenerating

```bash
# Recompute and overwrite all goldens touched by the selected tests:
QEFF_REGENERATE_GOLDEN=1 pytest tests/transformers/models/causal_lm_models -k dummy

# Run the CPU reference legs live (hf + qeff_hf + ORT) instead of the golden path,
# e.g. to re-validate PyTorch -> ONNX parity:
QEFF_RUN_CPU_REFERENCES=1 pytest tests/transformers/models/causal_lm_models -k dummy
```

Missing goldens are generated automatically on first run and merged into `goldens.json`;
commit the resulting file so subsequent CI builds reuse them.
