# DFlash SPD Examples

Entry points wrap the SPD (speculative-decoding) compile + run pipeline for both
**text-only** language models and **vision-language (VLM, multimodal)** models. Each side
has a model-name *front-end* (compiles the QPCs, or reuses pre-built ones) and a
lower-level *runner* that takes compiled QPC paths.

| | front-end (`--model_name`, builds QPCs) | runner (QPC paths) |
|---|---|---|
| text — single prompt | `basic_inference.py` | `dflash_spd_single_prompt.py` |
| text — dataset benchmark | `benchmark.py` | `dflash_spd_benchmark.py` |
| vision — single prompt/image | `basic_inference_vision.py` | `dflash_spd_vision_single_prompt.py` |
| vision — dataset benchmark | `benchmark_vision.py` | `dflash_spd_vision_benchmark.py` |

---

## Text (dense language models)

**Supported models:** Llama-3.1-8B-Instruct, Qwen3-4B, Qwen3-8B (see `--help` for the full list).

### Single prompt

```bash
python basic_inference.py --model_name Qwen3-4B \
    --prompt "Explain speculative decoding in two sentences."
```

### Benchmark (dataset)

```bash
python benchmark.py --model_name Qwen3-4B --dataset humaneval
```

---

## Vision-language models (VLM, multimodal)

**Supported models:** `gemma-4-31B-it` (more VLMs are added to `MODEL_MAP` over time —
run any script with `--help` for the current list). VLM entries have no default TLM path,
so pass `--tlm_hf_path` (e.g. `google/gemma-4-31B-it` for `gemma-4-31B-it`).

The vision path compiles **three** QPCs: the language decoder (TLM), the **vision
encoder** (`pixel_values -> vision_embeds`), and the DFlash draft (DLM). Because the
TLM and DLM fill their own cards, the vision encoder usually needs its **own** devices.

> Set `QEFF_HOME` to a filesystem with free space before compiling — the VLM TLM QPC
> is large (tens of GB):
> ```bash
> export QEFF_HOME=/local/mnt/workspace/<user>/qeff_home
> ```

### Single prompt (text through the VLM)

```bash
python basic_inference_vision.py --model_name gemma-4-31B-it --tlm_hf_path google/gemma-4-31B-it \
    --tlm_devices 40,41,42,43 --dlm_devices 44,45,46,47 --vision_devices 48,49,50,51 \
    --prompt "Tell me about the Taj Mahal."
```

### Single image + text prompt

```bash
python basic_inference_vision.py --model_name gemma-4-31B-it --tlm_hf_path google/gemma-4-31B-it \
    --tlm_devices 40,41,42,43 --dlm_devices 44,45,46,47 --vision_devices 48,49,50,51 \
    --image \
    --image_prompt "Describe this image in detail."
```

### Benchmark — MathVision dataset

```bash
# full testmini (~304 samples)
python benchmark_vision.py --model_name gemma-4-31B-it --tlm_hf_path google/gemma-4-31B-it \
    --tlm_devices 40,41,42,43 --dlm_devices 44,45,46,47 --vision_devices 48,49,50,51 \
    --split testmini --num_samples 0

# full test split (~3040 samples)
python benchmark_vision.py --model_name gemma-4-31B-it --tlm_hf_path google/gemma-4-31B-it \
    --tlm_devices 40,41,42,43 --dlm_devices 44,45,46,47 --vision_devices 48,49,50,51 \
    --split test --num_samples 200
```

The dataset (`MathLLMs/MathVision`) must be cached or downloadable; point
`HF_DATASETS_CACHE` at a disk with space if needed. Results are written to
`--output_dir` (per-sample + summary CSVs) alongside a printed avg/min/max table.

### Benchmark — text dataset (language part only)

To benchmark the VLM's **language decoder only** on a text dataset (humaneval / gsm8k /
math500), pass `--dataset`. This runs the language decoder with **no image**
(`vision_embeds` zero-bound, vision encoder not loaded), so you get language-only
throughput / acceptance-rate numbers:

```bash
python benchmark_vision.py --model_name gemma-4-31B-it --tlm_hf_path google/gemma-4-31B-it \
    --tlm_devices 40,41,42,43 --dlm_devices 44,45,46,47 \
    --dataset humaneval --num_samples 20
```

This is the validated language-only path (the vision-capable language QPC fed a text
prompt with zeroed `vision_embeds`); it does **not** need a separate `SKIP_VISION=True`
build. The runner `dflash_spd_vision_text_benchmark.py` can also be invoked directly with
pre-built `--tlm_qpc`/`--dlm_qpc` (no `--vision_qpc` needed).


---

## `--model_name`

Accepts either the short key or the full HF repo path (case-insensitive):

```
Qwen3-4B          Qwen/Qwen3-4B          qwen3-4b
gemma-4-31B-it
```

Run any script with `--help` to see the full supported list.

## Skipping compile (reuse QPCs)

Whichever QPC side you supply skips its compile step; the rest still compiles.

```bash
# text
python basic_inference.py --model_name Qwen3-4B \
    --tlm_qpc /path/to/tlm/qpc --dlm_qpc /path/to/dlm/qpc --prompt "Hello"

# vision (--tlm_qpc + --vision_qpc must be given together to skip the VLM build)
python basic_inference_vision.py --model_name gemma-4-31B-it \
    --tlm_qpc /path/to/lang/qpc --vision_qpc /path/to/vision/qpc --dlm_qpc /path/to/dlm/qpc \
    --tlm_devices 40,41,42,43 --dlm_devices 44,45,46,47 --vision_devices 48,49,50,51 \
    --prompt "Hello"
```

## Common flags

| Flag | Default | Notes |
|---|---|---|
| `--tlm_devices` | `0,1,2,3` | TLM device IDs |
| `--dlm_devices` | `0,1,2,3` | DLM device IDs |
| `--tlm_cores` / `--dlm_cores` | `8` | per-side core count |
| `--ctx_len` | `4096` (text) / `2048` (vision) | |
| `--prefill_seq_len` | `128` | |
| `--generation_len` | `1024` (benchmark) / `256` (single) | |
| `--hf_token` | `$HF_TOKEN` | required for gated repos |
| `--tlm_hf_path` | from `MODEL_MAP` | required when the map entry has `None` (VLM entries) |

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

Vision scripts (`basic_inference_vision.py`, `benchmark_vision.py`) add:

| Flag | Default | Notes |
|---|---|---|
| `--vision_devices` | `0,1,2,3` | vision-encoder device IDs (usually a separate group) |
| `--image` | off | (single-prompt) run an image+text prompt instead of text |
| `--image_url` | — | (single-prompt) image URL for `--image` |
| `--image_prompt` | — | (single-prompt) prompt text for `--image` |
| `--split` | `testmini` | (benchmark) MathVision split: `testmini` (~304) or `test` (~3040) |
| `--num_samples` | `0` (= all) | (benchmark) |

## Adding a new model

Edit `MODEL_MAP` in `utils.py`:

```python
"<short-name>": ("<tlm-hf-repo or None>", "<dlm-hf-repo>"),
```

All four entry points reuse the same map automatically.
