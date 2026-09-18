# What is QEfficient?

**QEfficient** (published as the `efficient-transformers` library) is a toolkit
that adapts Hugging Face (HF) Transformers models so they run efficiently on
**Qualcomm Cloud AI 100 / AI200 accelerators** — referred to throughout the
codebase as **QAIC** hardware.

In one sentence:

> You hand QEfficient a Hugging Face model card (or a local checkpoint), and it
> gives you back a model that has been optimized, exported, compiled, and made
> ready to run on Qualcomm accelerator cards — through the same familiar
> `from_pretrained` / `compile` / `generate` shape of API you already know.

## The mental model

Think of QEfficient as a **thin, faithful adapter layer** between two worlds:

- **The Hugging Face world** you author and train in — PyTorch modules, tokenizers,
  model cards on the Hub.
- **The Qualcomm Cloud AI world** you deploy in — an ahead-of-time-compiled,
  fixed-shape accelerator runtime.

Its guiding principle (stated in the repository's contributor instructions) is
**minimal divergence from upstream Hugging Face** while preserving numerical
parity across three stages: *PyTorch → ONNX → on-device*. QEfficient does not
fork Transformers or reimplement models from scratch. Instead it swaps in
small, surgical reimplementations of the handful of blocks that don't map
cleanly onto the accelerator (attention, KV cache, normalization, RoPE), and
leaves everything else as upstream HF.

This is the positioning the project summarizes as:

> **Train anywhere, infer on Qualcomm Cloud AI — with a developer-centric toolchain.**

You train (or download) a model in the standard ecosystem; QEfficient handles the
transformation to an efficient, device-ready form.

## What you give it, and what you get back

**Input:** a model identifier — either a Hugging Face model card
(e.g. `"meta-llama/Llama-3.2-1B"`) or a path to a locally downloaded model.

**Output:** a compiled **QPC** ("Qaic Program Container") — the device-ready
binary — plus the glue needed to run inference on it. The intermediate ONNX
graph and the QPC are cached on disk so repeated runs are fast.

The public entry points (exported from `QEfficient/__init__.py`) mirror the HF
`AutoModel` family, one class per task:

| QEfficient class | Task |
| --- | --- |
| `QEFFAutoModelForCausalLM` | Text generation (decoder LLMs) |
| `QEFFAutoModelForImageTextToText` | Vision-language models (VLMs) |
| `QEFFAutoModelForSpeechSeq2Seq` | Speech-to-text (e.g. Whisper) |
| `QEFFAutoModelForCTC` | CTC speech models (e.g. Wav2Vec2) |
| `QEFFAutoModel` | Embeddings / encoder models |
| `QEFFAutoModelForSequenceClassification` | Classification (e.g. guard models) |
| `QEffAutoPeftModelForCausalLM` | PEFT / LoRA adapters on top of a base LLM |

Diffusion pipelines (`QEffFluxPipeline`, `QEffWanPipeline`,
`QEffWanImageToVideoPipeline`) are also exported when the optional `diffusers`
dependency is present.

A typical session looks exactly like Hugging Face, with one extra `compile`
step for the accelerator:

```python
from QEfficient import QEFFAutoModelForCausalLM
from transformers import AutoTokenizer

model = QEFFAutoModelForCausalLM.from_pretrained("gpt2")  # load + transform
model.compile(num_cores=16)                               # export + compile to QPC
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.generate(prompts=["My name is"], tokenizer=tokenizer)  # run on device
```

See the [Quick Start](../source/quick_start.md) for the full walkthrough,
including the equivalent one-command CLI (`python -m QEfficient.cloud.infer …`).

## What the library covers

QEfficient is not limited to plain text LLMs. The breadth of supported model
families includes:

- **Text generation** — Llama, Mistral/Mixtral, Qwen, Gemma, Phi, Granite,
  GPT-2/GPT-J/GPT-OSS, Falcon, DeepSeek, MoE models, and more.
- **Vision-language (multimodal)** — LLaVA, Llama-4, Gemma-3/4, Qwen-VL,
  Mistral-3, Kimi, and others.
- **Speech** — Whisper (seq2seq) and Wav2Vec2 (CTC).
- **Embeddings** — BERT/BGE, RoBERTa, MPNet, Jina, and rerankers.
- **Diffusion** — FLUX (image) and WAN (video) pipelines.
- **Finetuning on QAIC** — an on-device training path with PEFT/LoRA support.

The authoritative, continuously-updated lists live in
[Validated Models](../source/validate.md) and
[Supported Features](../source/supported_features.rst).

---

**Where this lives in the code**

- Public API surface: `QEfficient/__init__.py`
- Auto classes: `QEfficient/transformers/models/modeling_auto.py`
- Contributor principles: `CLAUDE.md` (repo root)

**Next:** [Why QEfficient?](02-why-qefficient.md) — the problem it solves.
