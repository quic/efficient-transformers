# Gemma4 Vision-Language Examples

Runnable examples for Gemma4 vision-language models on Qualcomm AI 100/200 hardware
using QEfficient.

---

## Examples at a glance

| Script | Mode | Description |
|--------|------|-------------|
| `gemma4_example.py` | Regular batching | Standard single-request inference; text-only or image+text |
| `gemma4_cb.py` | Continuous batching | Multiple concurrent requests via CB scheduler |
| `gemma4_diss.py` | Dissected (split QPC) | Separate prefill-only and decode-only QPCs, raw session management |
| `gemma4_utils.py` | — | Shared helpers: chat template, prompt builders, compile-kwarg helpers |

---

## Script details

### `gemma4_example.py` — Standard inference

Single-request inference with `kv_offload=True` (dual QPC: vision encoder + language decoder compiled separately).

**Key knobs:**

| Variable | Default | Effect |
|----------|---------|--------|
| `MODEL_ID` | `google/gemma-4-E2B-it` | HF model ID |
| `SKIP_VISION` | `False` | `False` → text-only prompt; `True` → image+text prompt |
| `BS` | `1` | Batch size |
| `PREFILL_SEQ_LEN` | `128` | Base prefill length (auto-adjusted to prompt) |
| `CTX_LEN` | `2048` | Max context length |
| `GENERATION_LEN` | `1920` | Max tokens to generate |
| `NODE_PRECISION_INFO` | `True` | Auto-generate NPI file for mixed precision |
| `NUM_CORES` / `NUM_DEVICES` | `16` / `4` | Compiler target |

The script auto-computes effective `PREFILL_SEQ_LEN` and `CTX_LEN` from the actual prompt length via `effective_lens()`.

```bash
python examples/image_text_to_text/models/gemma_vision/gemma4/gemma4_example.py
```

---

### `gemma4_cb.py` — Continuous batching

Processes multiple prompts concurrently using `continuous_batching=True`. The CB runtime
fills all `FULL_BATCH_SIZE` slots and streams completions as slots free up.

**Key knobs:**

| Variable | Default | Effect |
|----------|---------|--------|
| `MODEL_ID` | `google/gemma-4-E2B-it` | HF model ID |
| `BATCH_SIZE` | `1` | Per-slot prefill batch size |
| `FULL_BATCH_SIZE` | `4` | Total concurrent CB slots |
| `PREFILL_SEQ_LEN` | `256` | Must be ≥ longest tokenised vision prompt |
| `CTX_LEN` | `2048` | Max context length |
| `GENERATION_LEN` | `100` | Max tokens per response |
| `IMAGE_URLS` / `PROMPTS` | (4 entries) | Input images and text prompts |

```bash
python examples/image_text_to_text/models/gemma_vision/gemma4/gemma4_cb.py
```

---

### `gemma4_diss.py` — Dissected (split QPC) flow

Compiles three separate QPCs and manages inference sessions manually:

1. **Vision encoder** — `skip_lang=True`
2. **Prefill-only language QPC** — `prefill_only=True`, `retain_full_kv=True`, `enable_chunking=True`
3. **Decode-only language QPC** — `prefill_only=False`, `skip_vision=True`

Input is padded to a multiple of `PREFILL_SEQ_LEN` and chunked manually. The KV cache is
retained in the prefill QPC and passed as inputs to the decode QPC. `mm_token_type_ids`
is passed per-chunk during prefill but is not required at decode time.

**Key knobs:**

| Variable | Default | Effect |
|----------|---------|--------|
| `model_id` | `google/gemma-4-26B-A4B-it` | HF model ID |
| `skip_vision` | `False` | Skip vision encoder (text-only run) |
| `PREFILL_SEQ_LEN` | `296` | Chunk size; input padded to multiple of this |
| `CTX_LEN` | `4096` | Max context length |
| `generation_len` | `200` | Max tokens to generate |

```bash
python examples/image_text_to_text/models/gemma_vision/gemma4/gemma4_diss.py
```

---

## Shared utilities (`gemma4_utils.py`)

| Helper | Purpose |
|--------|---------|
| `build_messages()` | Build HF-style chat messages (text-only or image+text) |
| `build_compile_kwargs()` | Translate `compiler_kwargs` dict to `qeff_model.compile()` kwargs |
| `effective_lens()` | Compute effective `PREFILL_SEQ_LEN` / `CTX_LEN` from prompt length |
| `normalize_generated_ids()` | Trim padding from generated ID tensors |
| `remove_fp16clip_transform_if_disabled()` | Remove FP16 clip transform when not needed |
| `CHAT_TEMPLATE` | Fallback Jinja chat template for Gemma4 |

---

## Notes

- First run will export to ONNX and compile — this can take several minutes.
- Ensure `CTX_LEN ≥ PREFILL_SEQ_LEN + GENERATION_LEN`.
- For fast end-to-end validation, uncomment the `_apply_reduced_layer_config()` block
  inside the script to shrink layer counts (CPU-RAM friendly, output will be gibberish).
- If model download fails, verify your Hugging Face authentication and that you have
  accepted the Gemma4 model licence at `https://huggingface.co/google/gemma-4-E2B-it`.
