# Golden outputs

Committed reference token streams for the causal-LM and VLM parity tests. They let the
Jenkins **QAIC LLM** / **QAIC Multimodal** stages run only the on-device (QAIC) leg and
compare its output against a stored HuggingFace PyTorch reference, instead of re-running
the CPU legs (`hf`, and for causal-LM also `qeff_hf` / torch-KV, `ORT` / ONNXRuntime) on
every build.

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
  },
  "image_text_to_text": {
    "llava-hf/llava-1.5-7b-hf": {
      "float32_gl10_<config_fp>_<digest>": {
        "pytorch_hf_tokens": [...],
        "config_fp": "…",
        "gen_len": 10,
        "timestamp": "…"
      }
    }
  }
}
```

- `<family>` — model family, e.g. `causal_lm` or `image_text_to_text`. New families are
  added as new top-level keys in the same file, so the mechanism scales without adding
  files.
- The model key is the raw HuggingFace model id.

## How variants are keyed

The HF token stream is a pure function of the model, so a variant key folds in exactly
the inputs that change it, and is **independent of `qaic_config`** (blocking, CCL,
speculative decoding, KV-head replication) and, for VLMs, `kv_offload` too -- those only
steer the QEff / on-device leg, so one golden is reused across every QAIC variant of the
same model.

- **causal_lm**: continuous-batching flag, dtype, `prompt_len`, `ctx_len`,
  `generation_len`, `full_batch_size`, the prompts, and a `config_fp` fingerprint of the
  effective config (model_type, layer / head / dim / vocab / intermediate-size counts) --
  this is what distinguishes the `dummy_layers`, `few_layers` and `full_layers` scopes.
- **image_text_to_text**: dtype, the fixed prompt/image-url pair from
  `image_text_model_configs.json`, `generation_len`, and a `config_fp` that hashes the
  full effective config (text + vision sub-configs) rather than a fixed attribute list --
  VLM architectures vary too much per family (e.g. Qwen3-VL's `deepstack_visual_indexes`
  vs Gemma3's `layer_types`) for a maintained whitelist.

## Regenerating

```bash
# Recompute and overwrite all goldens touched by the selected tests:
QEFF_REGENERATE_GOLDEN=1 pytest tests/transformers/models/causal_lm_models -k dummy
QEFF_REGENERATE_GOLDEN=1 pytest tests/transformers/models/image_text_to_text -k dummy

# Run the CPU reference legs live (hf + qeff_hf + ORT) instead of the golden path,
# e.g. to re-validate PyTorch -> ONNX parity (causal-LM only):
QEFF_RUN_CPU_REFERENCES=1 pytest tests/transformers/models/causal_lm_models -k dummy
```

`goldens.json` is only ever written under `QEFF_REGENERATE_GOLDEN=1`; commit the resulting
file so subsequent CI builds reuse it. A plain run that misses a golden still computes the
HF reference live -- so the parity assert stays meaningful -- but warns instead of writing,
keeping a committed file from being mutated as a side effect of an ordinary test run.
