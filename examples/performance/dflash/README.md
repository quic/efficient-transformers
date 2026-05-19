# DFlash SPD Examples

Two entry points wrap the SPD compile + run pipeline.

#### basic_inference.py
Basic DFlash usage with dense language models.

**Supported Models:**
- Llama3.1-8B-Instruct
- Qwen3-4B
- Qwen3-8B

## Single prompt

```bash
python basic_inference.py --model_name Qwen3-4B \
    --prompt "Explain speculative decoding in two sentences."
```

## Benchmark (dataset)

```bash
python benchmark.py --model_name Qwen3-4B --dataset humaneval
```

## `--model_name`

Accepts either the short key or the full HF repo path (case-insensitive):

```
Qwen3-4B
Qwen/Qwen3-4B
qwen3-4b
```

Run either script with `--help` to see the full supported list.

## Skipping compile (reuse QPCs)

Pass either or both. Whichever side is supplied skips its compile step; the
other side still compiles.

```bash
python basic_inference.py --model_name Qwen3-4B \
    --tlm_qpc /path/to/tlm/qpc \
    --dlm_qpc /path/to/dlm/qpc \
    --prompt "Hello"
```

## Common flags

| Flag | Default | Notes |
|---|---|---|
| `--tlm_devices` | `0 1 2 3` | TLM device IDs |
| `--dlm_devices` | `0 1 2 3` | DLM device IDs |
| `--tlm_cores` / `--dlm_cores` | `8` | per-side core count |
| `--ctx_len` | `4096` | |
| `--prefill_seq_len` | `128` | |
| `--generation_len` | `1024` (benchmark) / `256` (single) | |
| `--noise_embed_path` | `noise_embedding/<model_name>_noise_embeds.npy` | override if needed |
| `--hf_token` | `$HF_TOKEN` | required for gated repos |
| `--tlm_hf_path` | from `MODEL_MAP` | required when the map entry has `None` |

`benchmark.py` only:

| Flag | Default |
|---|---|
| `--dataset` | `humaneval` (also: `gsm8k`, `math500`) |
| `--num_samples` | `0` (= all) |
| `--iteration` | `300` |
| `--output_dir` | `./results-<model_name>` |

`basic_inference.py` only:

| Flag | Default |
|---|---|
| `--prompt` | *(required)* |
| `--category` | `""` (math / coding / reasoning / …) |

## Adding a new model

Edit `MODEL_MAP` in `benchmark.py`:

```python
"<short-name>": ("<tlm-hf-repo or None>", "<dlm-hf-repo>"),
```

`basic_inference.py` reuses the same map automatically.
