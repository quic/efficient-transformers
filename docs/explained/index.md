# Understanding QEfficient

Welcome. This folder is a guided, read-it-top-to-bottom introduction to
**QEfficient** (the `efficient-transformers` library). It explains *what the
library is, why it exists, how it works internally, and how it compares to
GPU-serving stacks like vLLM and NVIDIA TensorRT-LLM*.

If you just want to run a model, start with the
[Quick Start](../source/quick_start.md) instead. These pages are for people who
want to **understand the system** — new engineers, contributors, and anyone
trying to form a correct mental model of the pipeline before diving into code.

## Who these docs are for

- You are comfortable with Python and PyTorch.
- You know roughly what a transformer / LLM inference loop looks like
  (prefill, decode, KV cache).
- You do **not** need to know anything about Qualcomm Cloud AI hardware, ONNX
  export internals, or ahead-of-time compilation — we explain those here.

## Recommended reading order

1. **[What is QEfficient?](01-what-is-qefficient.md)** — the purpose and the
   mental model: what you give it and what you get back.
2. **[Why QEfficient?](02-why-qefficient.md)** — the problem it solves, and why
   you cannot just run stock Hugging Face `generate()` on the accelerator.
3. **[Architecture](03-architecture.md)** — the end-to-end pipeline:
   PyTorch → ONNX → QPC → on-device runtime, stage by stage. *This is the core
   document.*
4. **[Key concepts](04-key-concepts.md)** — the two ideas that make it work:
   model-block reimplementation and the static gather/scatter KV cache.
5. **[Optimization techniques](06-optimization-techniques.md)** — a catalog of
   the concrete techniques QEfficient uses to run models efficiently (graph &
   numeric, memory/KV-cache, decode-throughput, and compute-distribution), each
   explained in a little detail.
6. **[Comparison with vLLM and TensorRT-LLM](05-comparison.md)** — how
   QEfficient's design philosophy differs from GPU serving and GPU engine
   builders.

## Where to go next

- Hands-on usage and CLI: [Quick Start](../source/quick_start.md),
  [CLI API](../source/cli_api.md)
- Python API classes: [QEfficient Auto Classes](../source/qeff_autoclasses.md)
- Turning features on (continuous batching, speculative decoding, QNN, …):
  [Features Enablement](../source/features_enablement.md)
- What models and features are validated:
  [Validated Models](../source/validate.md),
  [Supported Features](../source/supported_features.rst)
- Finetuning on Qualcomm hardware: [Finetune](../source/finetune.md)

> These explainer pages are plain Markdown and are meant to be read on GitHub.
> They intentionally avoid line-number citations (which drift as code changes)
> and instead point you at the **modules** where each mechanism lives, so you
> can open the file and read the current implementation.
