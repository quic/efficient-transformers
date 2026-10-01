# Text generation

The canonical entry point is
[`basic_inference.py`](basic_inference.py); every feature below is a flag on
that same script. Standard inference uses this three-step API:

```
QEFFAutoModelForCausalLM.from_pretrained(...)   # load HF weights + attach qaic_config
              |
              v
              .compile(...)                     # export -> ONNX -> QPC (or use --onnx-path)
              |
              v
              .generate(tokenizer, prompts)     # runtime inference
```

Disaggregated serving uses separate prefill/decode QPCs and a DMA KV-handoff
loop. Text DFlash uses the existing target/draft runner through `--dflash`.
The replaced per-feature scripts are preserved in the local
[`archive/`](archive/README.md) directory.

**Supported models.** `QEFFAutoModelForCausalLM` covers dense decoders
(Llama, Qwen, Mistral, Gemma, Phi, Falcon, Granite, ...) and MoE decoders
(Mixtral, Qwen-MoE, GPT-OSS). For the authoritative list see
[`docs/source/validate.md`](../../docs/source/validate.md#text-only-language-models).

**Authentication.** Gated repos need `HF_TOKEN` in the environment. Fast
downloads: `HF_HUB_ENABLE_HF_TRANSFER=1`.

**Getting help.** `python basic_inference.py --help` shows the common flags;
`python basic_inference.py --help-advanced` reveals everything including
CI-only knobs.

For reduced-model checks, `--num-hidden-layers N` is an explicit alias for
`--num-hidden-layers-override N`. Positive values override the layer count;
omitting it or using a non-positive value preserves the checkpoint's configuration.

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

### Compiler and runner artifacts

```bash
python examples/text_generation/basic_inference.py \
    --model-name Qwen/Qwen2-1.5B-Instruct --prompt "Hello" \
    --artifacts --compile-dir /path/to/artifacts
```

`--artifacts` exports the model and writes compiler inputs plus a first-prefill
runner bundle. It prints both locations and executes neither the compiler nor
the runtime. The runner bundle represents one prefill invocation, not a full
generation loop. Add `--compile-only` to omit runner inputs. Continuous batching
also supports this mode.

For standard inference, `--write-io` writes the same first-prefill runner inputs
beside the compiled QPC before running generation. It uses the artifact writer;
`write_io` is not a supported runtime API keyword.

| Mode | Effect |
| --- | --- |
| `--dry-run --print-resolved` | Validate flags and print configuration; no model loading |
| `--artifacts` | Export and write compiler/runner inputs |
| `--artifacts --compile-only` | Export and write compiler inputs only |
| `--compile-only` | Export and execute the compiler; no inference |
| No mode flag | Export, compile, and generate |

Disaggregated, paged-attention, and decode-only artifacts require
`--compile-only`: their runner inputs are not supported by this example's
artifact path. QNN, DFlash, and MDP `intersection` partitioning cannot be
combined with `--artifacts`. Use MDP `onnx` partitioning for compiler bundles.

To replay a dense bundle, put the SDK's `exec/` directory on `PATH` (normally
`export PATH=/opt/qti-aic/exec:$PATH`), run `bash <compile-directory>/qaic-compile.sh`, then
from `<compile-directory>/io` run:

```bash
/opt/qti-aic/exec/qaic-runner -t ../qpc \
    --aic-batch-json-input aic_batch_io.json -n 1 --dev-list 0
```

Replay executes the compiler and requires an available QAIC device for the
runner. `--dynamo` restores the standard example's Dynamo export option; it
cannot be combined with disaggregated or layerwise execution.
It requires the [Dynamo export environment](../dynamo/causal_lm/README.md);
the standard runtime dependencies alone are not sufficient.

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

`--enable-blocking` makes the script forward `--blocking-mode` in `qaic_config`;
the mode picks the tile axes (`kv`, `q`, `h`, `b`, `qkv`, `hqkv`);
per-axis block counts (`--num-kv-blocks`, `--num-q-blocks`,
`--num-batch-blocks`, `--head-block-size`) tune the tile shape.

Disaggregated prefill and decode may select different modes. For example,
GPT-OSS head-parallel blocking uses
`--prefill-blocking-mode prefill_online --decode-blocking-mode kv_headpar`
with `--num-kv-blocks 2 --num-q-blocks 2`; `--headpar-split` optionally
overrides its default split.

### Paged attention

```bash
python examples/text_generation/basic_inference.py \
    --model-name meta-llama/Llama-3.2-1B \
    --enable-blocking --blocking-mode kv_paged --num-kv-blocks 2 \
    --prefill-seq-len 32 --ctx-len 128 --generation-len 16 --device-group 0
```

Supported modes are `kv_paged`, `qkv_paged` (also set `--num-q-blocks`), and
`hqkv_paged` (also set `--num-q-blocks` and `--head-block-size`). Keep
`--enable-blocking` explicit. Add `--continuous-batching --full-batch-size 2`
to exercise slot-based inference. The DMA disaggregated runner does not supply
block tables; combining paged attention with it requires `--compile-only`.

### Qwen3.5 gated-delta chunk size

Use `--gdn-chunk-size 64` with a supported Qwen3.5 **text** checkpoint
(`Qwen3_5TextConfig` or `Qwen3_5MoeTextConfig`). The value is passed through
`qaic_config` during loading and compilation, including both disaggregated
stages. It must be positive and may exceed the prefill sequence length. Omitting
the flag preserves the model's default. Multimodal Qwen3.5 checkpoints continue
to use the image-text-to-text examples.

### Disaggregated serving with pipeline-parallel prefill and DMA KV handoff

The canonical script can compile both QPCs and run continuous batching while
prefill writes KV directly into the host buffers consumed by decode. This path
requires a text architecture with full-KV disaggregated support, such as
GPT-OSS, Qwen3-MoE, GLM4-MoE, or Kimi-K2.

```bash
python examples/text_generation/basic_inference.py \
    --model-name Qwen/Qwen3-30B-A3B \
    --disaggregated --full-batch-size 4 \
    --prefill-seq-len 256 --ctx-len 512 --generation-len 200 \
    --prefill-num-devices 8 --decode-num-devices 4 \
    --mdp-num-partitions 4 --mdp-strategy onnx \
    --moe-expert-parallel-chunk-size 128 \
    --decode-aic-enable-depth-first --prefill-user-tiled \
    --mxfp6-matmul --mxint8-kv-cache --use-onnx-subfunctions \
    --prompt "Explain quantum computing." "What is the capital of France?"
```

Use `--prefill-device-group` and `--decode-device-group` when the two runtime
clusters need explicit device IDs. For GPT-OSS-120B, the stage-specific
`--prefill-node-precision-info` and `--decode-node-precision-info` flags allow
different NPI files. The older compile-only flow remains available through
`--stage prefill` and `--stage decode`; vLLM-style CCL lists are
`--ccl-prefill` / `--ccl-decode`.

### Multi-device tensor slicing

```bash
python examples/text_generation/basic_inference.py \
    --model-name meta-llama/Llama-3.1-8B \
    --num-devices 4 --device-group [0,1,2,3] \
    --num-cores 16 --mxfp6-matmul --aic-enable-depth-first --mos 1
```

`--num-devices` is authoritative; if omitted it falls back to
`len(--device-group)`, else 1.

For pipeline-parallel prefill, use `--stage prefill --mdp-num-partitions N` or
the complete `--disaggregated` recipe above. The prefill device count must be
divisible by the number of MDP partitions.

### Hardware validation matrix

Run the text-generation validation matrix with:

```bash
bash scripts/test_text_generation_matrix.sh
```

The default suite covers FP32 and FP16 inference, BF16 compilation, ONNX
subfunctions, CCL, continuous batching, tensor slicing, blocking modes, MDP,
artifact generation/replay, paged attention, DFlash, and GPT-OSS disaggregated
serving. Set `GDN_MODEL` to a Qwen3.5 text checkpoint to include GDN runtime
validation. Its default device assignment uses devices
0-3 for prefill and 4-5 for decode, so adjust the device-group variables in the
script when running on a different topology.

Each case has a separate log and artifact directory. The runner continues after
a failure and writes `summary.md` and `summary.tsv` at the end. Use
`SUITE=smoke` for the short baseline, `CASE_FILTER='ccl|mdp'` to select cases,
or `DRY_RUN=1` to validate the commands without loading models or compiling.
Dry-run rows are marked `DRY_RUN`; they do not establish export or runtime
parity. Use `CASE_FILTER='dense_artifacts'` for bundle creation and both replay
steps, or `CASE_FILTER='dflash'` for compilation, inference, and QPC reuse.
DFlash defaults to target device 0 and draft device 1; override
`DFLASH_TLM_DEVICE_GROUP` and `DFLASH_DLM_DEVICE_GROUP` for your hardware.
`QEFF_HOME` is honored, defaulting to `/home/rishinr/tmpdir/qeff_artifacts`.

Before hardware runs, run the example contracts and DFlash CPU tests:

```bash
pytest -n 2 tests/unit_test/models/test_prefill_decode_kv_handoff.py \
    tests/transformers/spd/test_dflash_spd_inference.py -m 'not on_qaic'
```

Use the existing causal-model blocking tests to compare HF PyTorch, QEff
PyTorch, ONNXRuntime, and QAIC outputs for the paged recipes. Use the causal
model tests' Qwen3.5 text configurations for GDN parity. Compilation and replay
success alone are not numerical-parity checks; retain logs and report any
hardware or checkpoint gaps.

### Speculative decoding (TLM side)

```bash
python examples/text_generation/basic_inference.py \
    --model-name meta-llama/Llama-3.1-8B \
    --speculative-model-type target --num-speculative-tokens 3
```

### Text DFlash (target and draft)

```bash
python examples/text_generation/basic_inference.py \
    --dflash --model-name Qwen/Qwen3-4B --prompt "Explain speculative decoding." \
    --ctx-len 4096 --prefill-seq-len 128 --generation-len 32 --iteration 32 \
    --tlm-devices 0 --dlm-devices 1 --tlm-cores 8 --dlm-cores 8 \
    --compile-dir /path/to/dflash
```

Supported targets are Qwen3-4B, Qwen3-8B, and Llama-3.1-8B-Instruct; short names
and full repository IDs use the existing DFlash target/draft mapping. The
standalone [DFlash example](../performance/dflash/README.md) remains available
with its original CLI and shares the same runner.

Use `--tlm-qpc` and/or `--dlm-qpc` to reuse either compiled model independently;
new compilations use `tlm/` and `dlm/` beneath `--compile-dir`. Target and draft
devices default to `--device-group`, or `[0]` when omitted; each uses 8 cores
by default. `--compile-only` stops before runtime setup. `--tlm-hf-path`
overrides the target repository; `--format-prompt --category math` enables the
existing category template. Authentication uses `HF_TOKEN`.

DFlash accepts exactly one prompt and batch size 1. Its helper uses the existing
FP32 load, MXFP6 weights, and MXINT8 KV recipe. Other explicitly supplied
compile/runtime controls are rejected rather than ignored. Artifacts,
disaggregated serving, blocking, sampler, and legacy speculative controls
cannot be combined with `--dflash`.

The canonical defaults remain `ctx-len=128`, `prefill-seq-len=32`, and
`iteration=1`; for DFlash, iteration is the maximum number of speculative
iterations. An omitted generation length uses the context length. Use the
complete recipe above for a longer run. The standalone DFlash CLI retains its
own defaults.

### On-device sampler

```bash
python examples/text_generation/basic_inference.py \
    --model-name Qwen/Qwen2-1.5B-Instruct \
    --include-sampler --max-top-k-ids 512 --return-pdfs
```

---

## Archived advanced examples

Two historical workloads sit outside the main script because they cannot use
the standard Auto pipeline; see [`archive/advanced/README.md`](archive/advanced/README.md)
for details:

- **`archive/advanced/kimik2_mla_absorption.py`** — Kimi-K2 MLA with hand-rolled prefill/decode.
- **`archive/advanced/glm4_kv_head_surgery.py`** — GLM-4-MoE with live KV-head weight
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
