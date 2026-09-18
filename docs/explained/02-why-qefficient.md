# Why QEfficient?

If Hugging Face already runs the model in PyTorch, why do we need a whole
library to run it on a Qualcomm accelerator? The short answer: **the way stock
Hugging Face inference works does not map onto the accelerator's execution
model.** QEfficient exists to bridge that gap without forcing you to abandon the
Hugging Face ecosystem.

## The core mismatch: eager + dynamic vs compiled + static

Hugging Face inference is **eager and dynamic**:

- The Python code runs op-by-op every step. Control flow (`if`, loops, ragged
  shapes) is decided at runtime.
- The **KV cache grows**: at each decode step, HF's `DynamicCache` concatenates
  the new key/value vectors onto the existing tensors, so the cache tensor gets
  one token longer every step. Its shape changes constantly.

Qualcomm Cloud AI accelerators are **ahead-of-time (AOT) compiled and
static-shape**:

- A graph compiler (`qaic-compile`) turns the whole model into a fixed program
  *before* any inference runs. There is no Python interpreter on the device.
- Every tensor shape must be **known and fixed at compile time**. A cache that
  grows by one token per step is not expressible — the compiler needs a buffer
  of a single, fixed size.
- Dynamic Python control flow has to be resolved into a static graph.

So you cannot simply move an HF model onto the card. Something has to translate
the dynamic, growing, eager computation into a static, fixed-shape, compiled
one — while keeping the *numbers* the same. That translation is QEfficient's
whole job.

## The obstacles QEfficient removes

Concretely, the library provides:

1. **On-device retention of intermediate state (the KV cache).**
   Instead of a growing cache, QEfficient pre-allocates a **fixed-size** KV
   buffer and writes each new token into the correct slot (scatter) and reads
   back the valid window (gather). Crucially, the compiled graph keeps this
   buffer **resident on the card** across steps, so the cache never has to be
   copied back and forth to the host. (Details in
   [Key concepts](04-key-concepts.md).)

2. **Lower-precision execution with overflow/underflow safety.**
   The accelerator runs in fp16 (and offers MXFP6 weight compression and MXINT8
   KV-cache compression). fp16 has a much smaller range than fp32/bf16, so naive
   conversion produces `inf`/`nan` and destabilizes the compiled graph.
   QEfficient inserts export-time clamping, saturating residual adds, and finite
   mask-fill constants so the lower-precision graph stays numerically stable.

3. **Mathematically-equivalent op replacement.**
   Some operations either aren't supported on the backend or compile poorly as a
   decomposed graph. QEfficient replaces them with equivalent forms the compiler
   handles well — for example a single fused `CustomRMSNorm` op instead of the
   pow/mean/rsqrt/mul decomposition, and boolean causal masks instead of large
   additive float masks.

4. **A one-command path from model card to running inference.**
   `from_pretrained → compile → generate` (or a single `QEfficient.cloud.infer`
   CLI call) downloads, transforms, exports to ONNX, compiles to a QPC, and runs
   — caching each artifact so repeat runs skip the expensive stages.

## Why "minimal divergence from Hugging Face" is a design goal

QEfficient could have hard-forked Transformers and rewritten each model. It
deliberately doesn't. The transform mechanism (see
[Architecture](03-architecture.md)) swaps only the **class pointer** of the few
blocks that need to change, reusing the original weights and leaving the rest of
the model untouched. This choice pays off in three ways:

- **Maintainability** — when upstream HF changes a model, the divergence surface
  QEfficient has to re-check is small and localized.
- **Parity** — because most of the model is unchanged upstream code, it is much
  easier to prove the QEfficient version produces the same numbers as the
  original (validated PyTorch → ONNX → on-device).
- **Fast model onboarding** — adding a new architecture usually means reusing the
  closest existing wrapper family and overriding only attention/cache/norm,
  rather than writing a model from scratch.

This is why the codebase is organized as small per-model wrappers
(`QEfficient/transformers/models/<arch>/modeling_<arch>.py`) plus a registry
that maps upstream classes to them, rather than as a monolithic reimplementation.

---

**Where this lives in the code**

- Guiding principle & conventions: `CLAUDE.md` (repo root),
  `docs/source/introduction.md`
- Transform registry that keeps divergence minimal:
  `QEfficient/transformers/models/pytorch_transforms.py`
- Lower-precision / overflow handling examples:
  `QEfficient/customop/rms_norm.py`, and per-model export-time clamping such as
  `QEfficient/transformers/models/gemma4/modeling_gemma4.py`

**Next:** [Architecture](03-architecture.md) — the end-to-end pipeline.
