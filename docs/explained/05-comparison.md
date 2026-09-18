# Comparison with vLLM and TensorRT-LLM

New readers often ask: *how is QEfficient different from vLLM or NVIDIA
TensorRT-LLM?* All three run transformer inference efficiently, but they sit at
different points in the design space. This doc compares them **conceptually and
architecturally** — by design philosophy and where each fits — not by benchmark
numbers.

> **Disclaimer.** vLLM and TensorRT-LLM are large, fast-moving external projects.
> The descriptions below are about their *architectural approach*, not an audit
> of their current feature sets, which change frequently. When in doubt, consult
> each project's own documentation.

## The axis that actually matters: how the graph gets to hardware

The cleanest way to place these three systems is by **when and how the model is
turned into something the hardware runs**, and **what hardware that is**.

| | Target hardware | Execution model | Graph preparation |
| --- | --- | --- | --- |
| **vLLM** | GPUs (primarily NVIDIA; multiple backends) | Eager / just-in-time, kernel-based | Runs the model with optimized kernels at request time |
| **TensorRT-LLM** | NVIDIA GPUs | Ahead-of-time compiled engine | Builds a TensorRT engine before serving |
| **QEfficient** | Qualcomm Cloud AI 100 / AI200 (QAIC) | Ahead-of-time compiled program (QPC) | Exports to ONNX, then compiles a QPC before serving |

Two of these (TensorRT-LLM and QEfficient) are **AOT compilers**; one (vLLM) is
primarily a **runtime**. And two of these target **NVIDIA GPUs**; QEfficient
targets **Qualcomm accelerators**. That framing explains most of the differences.

## vLLM — a GPU serving runtime

vLLM's center of gravity is **serving throughput on GPUs**. Its signature
contributions are at the *runtime/scheduling* layer:

- **PagedAttention** — a virtual-memory-like KV cache that stores attention keys
  and values in non-contiguous "pages," reducing fragmentation and enabling high
  batch occupancy.
- **A sophisticated request scheduler** — continuous batching, preemption, and
  prefix sharing to keep the GPU saturated across many concurrent requests.

vLLM largely executes the model **eagerly** with highly optimized GPU kernels;
there is no separate ahead-of-time "compile the whole model to a binary" step in
the way TensorRT-LLM or QEfficient have.

**How QEfficient differs.** QEfficient is not fundamentally a serving/scheduling
framework — it is the piece that gets a model *compiled and running correctly and
efficiently on Qualcomm hardware*. It does implement continuous batching (with
on-device cache-slot recycling), but the higher-level serving layer that
competes with vLLM's scheduler lives **outside this repository's core** (there is
a separate vLLM integration for Qualcomm hardware). Where vLLM optimizes a
runtime around a fixed set of GPU kernels, QEfficient's hard problem is the
*ahead-of-time translation* of an arbitrary HF model into a static, fp16-safe,
compilable graph for a different class of accelerator.

## TensorRT-LLM — the closest analogue

TensorRT-LLM is conceptually the **nearest neighbor** to QEfficient: it is also
an **ahead-of-time compiler** that turns an LLM into an optimized engine which is
then served. Both share the AOT philosophy — do the expensive graph-level work
once, up front, and produce a device-specific artifact.

The differences are in the details:

- **Target hardware.** TensorRT-LLM builds **TensorRT engines for NVIDIA GPUs**.
  QEfficient compiles **QPCs for Qualcomm Cloud AI 100 / AI200**. This is the
  fundamental divide — they are not substitutes; you pick based on the hardware
  you deploy on.
- **Interchange / build path.** QEfficient uses **ONNX as the interchange
  format** and then the Qualcomm `qaic-compile` (or QNN) toolchain. TensorRT-LLM
  uses NVIDIA's own model-definition and TensorRT builder path.
- **Relationship to Hugging Face.** QEfficient's explicit design goal is
  **minimal divergence from upstream HF** — it keeps small per-model wrappers and
  swaps only the class pointers of the blocks that must change (see
  [Key concepts](04-key-concepts.md)), so onboarding a new model reuses the
  closest existing wrapper. Model support in engine-builder ecosystems is
  typically organized around the builder's own model definitions.

## When would you use QEfficient?

The decision is mostly about hardware and ecosystem:

- **Use QEfficient** when you are deploying on **Qualcomm Cloud AI 100 / AI200**
  accelerators and want to bring standard Hugging Face models (LLMs, VLMs,
  speech, embeddings, diffusion) onto that hardware with minimal fuss and good
  parity to the original model.
- **Use TensorRT-LLM** when you are on **NVIDIA GPUs** and want a compiled-engine
  approach on that stack.
- **Use vLLM** when you want a **GPU serving runtime** with a mature scheduler
  and high-throughput batching (and note there is a vLLM path for Qualcomm
  hardware that can sit on top of QEfficient-style compilation).

These are not mutually exclusive layers in every case — a production system might
use QEfficient to compile for Qualcomm hardware and a serving framework on top to
schedule requests. The takeaway is that **QEfficient's distinctive role is the
faithful, ahead-of-time adaptation of HF models to Qualcomm accelerators**, which
is a different job from "be the fastest GPU serving runtime."

---

**Where this lives in the code**

- AOT compile path (the artifact QEfficient produces): `QEfficient/compile/`,
  `QEfficient/base/modeling_qeff.py` (`_compile`)
- Continuous batching (QEfficient's in-repo batching):
  `QEfficient/generation/text_generation_inference.py`
- The HF-divergence philosophy in practice:
  `QEfficient/transformers/models/pytorch_transforms.py`

**Back to:** [Understanding QEfficient index](index.md)
