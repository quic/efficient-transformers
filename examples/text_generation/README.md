# Text generation

One script, one Auto class, one three-step API. The canonical entry point is
[`basic_inference.py`](basic_inference.py); every feature below is a flag on
that same script.

```
QEFFAutoModelForCausalLM.from_pretrained(...)   # load HF weights + attach qaic_config
              |
              v
              .compile(...)                     # export -> ONNX -> QPC (or use --onnx-path)
              |
              v
              .generate(tokenizer, prompts)     # runtime inference
```

**Supported models.** `QEFFAutoModelForCausalLM` covers dense decoders
(Llama, Qwen, Mistral, Gemma, Phi, Falcon, Granite, ...) and MoE decoders
(Mixtral, Qwen-MoE, GPT-OSS). For the authoritative list see
[`docs/source/validate.md`](../../docs/source/validate.md#text-only-language-models).

**Authentication.** Gated repos need `HF_TOKEN` in the environment. Fast
downloads: `HF_HUB_ENABLE_HF_TRANSFER=1`.

**Getting help.** `python basic_inference.py --help` shows the common flags;
`python basic_inference.py --help-advanced` reveals everything including
CI-only knobs.

---

## Recipes

Each recipe below sets only the flags it needs. Everything else falls back to
`basic_inference.py`'s defaults.

### Hello world (dense, single prompt)

```bash
python examples/text_generation/basic_inference.py \
    --model-name Qwen/Qwen2-1.5B-Instruct \
    --prompt "Hello, how are you?"
```

### Continuous batching (dynamic multi-request)

```bash
python examples/text_generation/basic_inference.py \
    --model-name meta-llama/Llama-3.1-8B \
    --continuous-batching --full-batch-size 4 \
    --prompt "Hello" "Hi there" "Good morning" "How are you"
```

`--continuous-batching` flips the ``from_pretrained`` flag; `--full-batch-size`
is the CB slot count and is required whenever continuous batching is on. Old
`--prompts "A|B|C"` pipe-form still works for backward compatibility.

### MoE with expert-blocked chunked prefill

```bash
python examples/text_generation/basic_inference.py \
    --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --use-onnx-subfunctions \
    --enable-chunking --stage prefill \
    --moe-prefill-packed-chunk-size 256
```

`--use-onnx-subfunctions` keeps the ONNX blob small; `--enable-chunking`
enables the expert-blocked prefill path; `--stage prefill` compiles only the
prefill QPC (pair with a `--stage decode` compile for a disaggregated deploy).

### GGUF (quantized weights)

```bash
pip install gguf
python examples/text_generation/basic_inference.py \
    --model-name MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF \
    --gguf-file Mistral-7B-Instruct-v0.3.fp16.gguf \
    --prompt "How are you?"
```

### Blocked attention (long context)

```bash
python examples/text_generation/basic_inference.py \
    --model-name meta-llama/Llama-3.2-1B \
    --prefill-seq-len 1 --ctx-len 131072 \
    --generation-len 64000 \
    --num-devices 8 \
    --mxfp6-matmul --mxint8-kv-cache --use-onnx-subfunctions \
    --enable-blocking --blocking-mode kv --num-kv-blocks 16 --skip-kv \
    --user-tiled
```

`--enable-blocking` opens the `qaic_config["enable_blocking"]` surface;
`--blocking-mode` picks the tile axes (`kv`, `q`, `h`, `b`, `qkv`, `hqkv`);
per-axis block counts (`--num-kv-blocks`, `--num-q-blocks`,
`--num-batch-blocks`, `--head-block-size`) tune the tile shape.

### Disaggregated serve (prefill + decode as separate QPCs)

Compile prefill and decode QPCs into distinct directories:

```bash
python examples/text_generation/basic_inference.py \
    --model-name meta-llama/Llama-3.1-8B \
    --stage prefill --enable-chunking --compile-dir /tmp/prefill_qpc

python examples/text_generation/basic_inference.py \
    --model-name meta-llama/Llama-3.1-8B \
    --stage decode --retain-full-kv --compile-dir /tmp/decode_qpc
```

vLLM-style chunked-context (CCL) lists are `--ccl-prefill` / `--ccl-decode`.

### Multi-device (MDP)

```bash
python examples/text_generation/basic_inference.py \
    --model-name meta-llama/Llama-3.1-8B \
    --num-devices 4 --device-group [0,1,2,3] \
    --num-cores 16 --mxfp6-matmul --aic-enable-depth-first --mos 1
```

`--num-devices` is authoritative; if omitted it falls back to
`len(--device-group)`, else 1.

### Speculative decoding (TLM side)

```bash
python examples/text_generation/basic_inference.py \
    --model-name meta-llama/Llama-3.1-8B \
    --speculative-model-type target --num-speculative-tokens 3
```

### On-device sampler

```bash
python examples/text_generation/basic_inference.py \
    --model-name Qwen/Qwen2-1.5B-Instruct \
    --include-sampler --max-top-k-ids 512 --return-pdfs
```

---

## `advanced/`

Two workloads sit outside the main script because they can't ride the standard
Auto pipeline; see [`advanced/README.md`](advanced/README.md) for details:

- **`kimik2_mla_absorption.py`** — Kimi-K2 MLA with hand-rolled prefill/decode.
- **`glm4_kv_head_surgery.py`** — GLM-4-MoE with live KV-head weight
  replication before compile.

---

## CLI shortcut: `QEfficient.cloud.infer`

For an all-in-one export -> compile -> execute driven purely from the shell
(useful in CI and quick smoke tests), the packaged CLI still works:

```bash
python -m QEfficient.cloud.infer \
    --model_name meta-llama/Llama-3.1-8B \
    --batch_size 1 --prompt_len 128 --ctx_len 512 \
    --num_cores 16 --device_group [0] \
    --prompt "Write a short story about AI" \
    --mxfp6 --mxint8_kv_cache --mos 1 --aic_enable_depth_first
```

Reference: [`QEfficient.cloud.infer`](https://quic.github.io/efficient-transformers/source/cli_api.html#qefficient-cloud-infer).

## Further reading

- [Quick Start](https://quic.github.io/efficient-transformers/source/quick_start.html)
- [Features Enablement](https://quic.github.io/efficient-transformers/source/features_enablement.html)
- [QEff Auto Classes](https://quic.github.io/efficient-transformers/source/qeff_autoclasses.html)
- [Validated Models](https://quic.github.io/efficient-transformers/source/validate.html)

## Cache locations

Exports and QPCs default to `~/.cache/qeff_cache`. Override with
`QEFF_HOME` (primary) or `XDG_CACHE_HOME`.
