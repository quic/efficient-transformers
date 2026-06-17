---
name: transformers-mainline-rebase
description: Rebase downstream QEff wrappers onto Hugging Face Transformers mainline with minimal divergence, preserving runtime/export parity and validating against tests/test_model_quickcheck.py.
---

# Transformers Mainline Rebase

## Purpose
Use this skill to rebase QEff from an existing HF baseline to a newer HF release while keeping:
- wrapper behavior stable
- PyTorch parity stable
- ONNX export/runtime stable
- local divergence minimal

For this validated migration:
- baseline: `transformers==4.55.0`
- target: `transformers==5.3.0`

This file is also a template for future rebases.

## Rebase Strategy Pattern
Use this repeatable structure for any HF mainline jump:
- do one broad compatibility alignment pass first (deps + wrappers + export/runtime glue)
- follow with focused cleanup passes (remove temporary compatibility hacks)
- isolate cache abstraction hardening into dedicated patches
- finish with deterministic test-hygiene improvements

This sequencing keeps diffs understandable, makes regressions easier to bisect, and prevents local hacks from becoming permanent.

## Source Of Truth
- Ultimate validation gate: `tests/test_model_quickcheck.py`
- Do not claim success unless full quickcheck passes in the user-provided environment.

## Environment Rule
- Environment is user-provided.
- Use the user’s activation/exec method and user-provided test command.
- For this repo, a known-good example is:
  - `pyenv activate qeff.mainline`
- If the user does not specify a command, default to:
  - `python -m pytest -q tests/test_model_quickcheck.py -n auto`

## Rebase Workflow (Always)
1. Align dependency stack first, then install.
2. Run full quickcheck once to capture failure baseline.
3. Fix failures in focused groups (small diffs, re-test quickly).
4. Re-run full quickcheck.
5. Report exact results and remaining gaps (if any).

## Turnkey Operator Flow (Low-Touch Handoff)
Use this exact sequence for a smooth first-time rebase execution:

1. Environment bootstrap
- activate the user-provided environment
- install/update dependencies from `pyproject.toml` (plus optional test extras)

2. Hygiene cleanup
- clear stale temp/export/cache artifacts before any test run
- ensure cleanup is single-run under xdist environments

3. Baseline gate
- run full quickcheck once:
  - `python -m pytest -q tests/test_model_quickcheck.py -n auto`
- record failures by bucket:
  - causal helper/mask drift
  - cache API/object drift
  - ONNX export signature drift
  - AWQ/quantizer drift
  - Mixtral/GPT-OSS MoE drift
  - Gemma3/VLM multimodal drift
  - Whisper/audio drift

4. Deterministic fix loop
- pick one bucket only
- implement minimal compatibility patch
- run only the relevant targeted quickcheck subset
- repeat until bucket is green, then move to next bucket

5. Final acceptance
- rerun full quickcheck command
- require full green before claiming successful migration

6. Handoff output
- include:
  - exact commands used
  - final pass/fail counts
  - runtime duration
  - any temporary shims left and their removal conditions

## Blocker Escalation Rules
If a rebase engineer is blocked, follow this order:
1. verify dependency pins and optional imports first
2. verify cache tuple/object conversion boundaries
3. verify wrapper signature drift (`head_mask`, rotary args, causal helpers)
4. verify export-path differences (runtime path may still be correct)
5. isolate model-family exceptions (Mixtral, GPT-OSS, Gemma3, Whisper, AWQ)

If still blocked after these checks:
- keep the change scoped
- document failing test IDs and exact traceback
- avoid broad refactors until the failing bucket is isolated

## Rebase Workflow (Detailed Execution Contract)
1. Dependency and API surface lock
- update dependency pins together (`transformers`, `huggingface-hub`, `diffusers`, `peft`, export/runtime libs)
- ensure no mixed-era cache/quantizer imports remain

2. Baseline failure snapshot
- run full quickcheck once immediately after dependency bump
- capture failing test IDs and map each to one drift bucket (cache, attention/mask, export signature, quantizer, MoE, multimodal)

3. Fix by bucket with strict scope
- keep each patch focused to one drift bucket
- preserve upstream semantics first, then add minimal compatibility shim only where needed
- avoid model-specific hacks until shared path is exhausted

4. Deterministic reruns
- run targeted `-k` subsets after each bucket
- always clean stale artifacts before full rerun
- run full quickcheck with the exact user-provided command as final gate

5. Report with evidence
- include commands, pass/fail counts, duration, and unresolved risk (if any)
- do not claim parity unless HF PyTorch -> QEff PyTorch -> ORT checks are green in quickcheck

## Practical Tricks That Keep Rebases Intact
- Keep compatibility shims at boundaries, not deep inside model math:
  - input normalization (tuple cache -> cache objects)
  - output normalization (cache objects -> legacy tuples for export/tracing)
- Prefer feature detection over version checks:
  - use `hasattr(...)`, optional imports, and safe `getattr(..., default)` guards
  - avoid hardcoding behavior to exact HF versions where possible
- Default to upstream semantics first:
  - patch wrappers only when parity/export requires it
  - if a temporary local hack is added to unblock, schedule its removal once stable path is validated
- Make cache code resilient:
  - preserve both legacy tuple and object cache interoperability
  - tolerate optional/extra args in cache methods likely to drift upstream
  - keep encoder-decoder and decoder-only paths explicit
- Keep ONNX export deterministic:
  - prefer one export-call pattern by default and only branch when architecture truly requires it
  - keep `past_key_values` naming and shape handling stable for both tuple/object forms
- Control flaky failures early:
  - clean stale temp artifacts before full/regression reruns
  - avoid per-worker destructive cleanup races in xdist
- Preserve API behavior externally:
  - wrappers should continue to accept previous public inputs and return expected output structure whenever feasible
  - if behavior must change, document exact impact and add targeted tests
- Use micro-triage loops:
  - run smallest useful `-k` scope after each bucket fix
  - only run full quickcheck after targeted scopes are green

## Robustness Rules (Non-Negotiable)
- Do not rely on private or fragile HF internals when a stable public abstraction exists.
- Do not merge rebase work with unrelated refactors.
- Do not claim parity based only on export success; runtime parity and ORT parity must both be validated.
- Do not keep compatibility hacks that are no longer required after upstream-aligned fixes.
- Do not break existing public wrapper call signatures unless explicitly approved.

## Dependency Alignment (4.55.0 -> 5.3.0)
Use this known-good set unless user overrides:
- `transformers==5.3.0`
- `diffusers==0.37.0`
- `huggingface-hub==1.7.1`
- `peft==0.18.1`
- `hf_transfer==0.1.9`
- `datasets==2.20.0`
- `fsspec==2023.6.0`
- `sentencepiece==0.2.0`
- `onnx==1.18.0`
- `onnxruntime==1.22`
- `onnxscript==0.2.5`
- `numpy==1.26.4`
- `protobuf==6.31.0`

## Must-Do Cleanup Before Export/Parity Tests
Clean stale artifacts before focused/full quickcheck:
- `/tmp/*qeff*`, `/tmp/*onnx*`, `/tmp/*qnn*`
- `~/.cache/qeff_models` (or active `QEFF_HOME`)
- ensure cleanup is done once per session when running xdist (avoid parallel worker races)

If shell globbing fails, use `find` cleanup.

## 4.55.0 -> 5.3.0 Required Fixes

### 1) Causal wrapper helper drift
Symptom:
- missing helper methods like `get_head_mask`, `_update_causal_mask`

Fix:
- remove dependency on removed HF helpers
- use explicit head-mask fallback: `head_mask = [None] * num_layers` when needed
- keep causal mask creation centralized and consistent
- if temporary mask alignment was introduced to unblock parity, remove it once upstream-compatible path is restored
- when `attention_mask` is absent, always build an explicit causal mask in wrapper `Model.forward` (do not rely on model-internal implicit causality only)
- resolve causal-mask `target_length` defensively for preallocated tuple caches:
  - start from `past_seen_tokens`
  - if `<=0`, fallback to `past_key_values.layers[0].keys.shape[-2]` when available
  - if still `<=0`, fallback to `int(position_ids.max()) + 1`
  - final fallback: current input sequence length
- for RoPE wrappers, ensure rotary cache length is at least max requested position:
  - `kv_seq_len = max(kv_seq_len, int(position_ids.max()) + 1)`
  - this avoids out-of-bounds rotary indexing and avoids silent clamp-induced parity drift

### 2) Cache API drift
Symptom:
- failures around `get_seq_length(...)`
- tuple/cache object incompatibility

Fix:
- keep `QEffDynamicCache` tuple interop and direct Cache API compatibility:
  - `from_legacy_cache()`
  - `to_legacy_cache()`
- provide `__getitem__` / `__iter__` compatibility where older call paths index cache tuples
- `QEffDynamicCache.get_seq_length` accepts extra args safely
- in `QEffEncoderDecoderCache.from_legacy_cache`, use positional ctor:
  - `cls(QEffDynamicCache(), QEffDynamicCache())`
- implement `QEffEncoderDecoderCache.to_legacy_cache()` for tuple-returning call sites
- when decoupling from HF internals, prefer inheriting from `Cache` with local layer class behavior instead of relying on unstable upstream `DynamicLayer` details

### 3) Runtime cache conversion (`run_utils.py`)
Fix:
- tuple -> `QEffDynamicCache` for decoder-only models
- tuple -> HF `EncoderDecoderCache(...)` for encoder-decoder models when upstream expects native object behavior
- keep model-specific exceptions where required:
  - `gpt_oss`
  - Gemma3 family (prefer HF `DynamicCache` and preserve text-side generation fallback behavior)

### 4) Encoder export `use_cache` collisions (BERT-like)
Symptom:
- `... got multiple values for argument 'use_cache'`

Fix:
- do not force cache for encoder-only export path
- keep exporter on keyword path by default:
  - `torch.onnx.export(..., args=(), kwargs=example_inputs, ...)`
- avoid mixing positional dict arguments for decoder-like models unless strictly required

### 5) Whisper drift
Symptoms:
- invalid/zero mask shape failures
- encoder layer arg drift (`layer_head_mask`)
- encoder-decoder cache constructor drift

Fix:
- tolerate invalid/zero masks (drop instead of hard fail)
- remove stale `layer_head_mask` forwarding if upstream encoder layer no longer accepts it
- ensure encoder-decoder cache legacy conversion and legacy return path both work

### 6) AWQ drift
Symptoms:
- enum/config attr drift
- optional field missing (e.g. `do_fuse`)

Fix:
- tolerate backend/format enum renames
- provide both `update_dtype` and `update_torch_dtype`
- guard optional attrs via `getattr(...)`
- keep post-load hook no-op
- keep matmul-nbits meta-safe guard for `.item()` usage

### 7) Mixtral v5 sparse-MoE drift
Symptoms:
- gate output shape/API drift
- ONNX failure from `aten::histc`

Fix:
- support both gate styles:
  - v5 tuple return (`router_logits, routing_weights, selected_experts`)
  - legacy logits path (`softmax + topk`)
- for v5 experts runtime path, preserve upstream semantics
- for ONNX export, use:
  - `transformers.integrations.moe.batched_mm_experts_forward(...)`
- keep legacy expert-container fallback

### 8) GPT-OSS v5 MoE/projection drift
Symptom:
- wrapper expects split gate/up projections while upstream uses fused `gate_up_proj`

Fix:
- initialize split compatibility tensors from fused tensors in `QEffGptOssExperts.__qeff_init__`
- keep existing MLP math paths unchanged
- retain runtime cache exception behavior for `gpt_oss`

## MoE Rebase Playbook (High Priority)
Use this section first when quickcheck failures are concentrated in Mixtral/GPT-OSS.

### A) MoE Invariants To Preserve
- router behavior:
  - preserve top-k routing semantics and expert selection ordering
  - keep routing normalization logic unchanged unless upstream behavior changed
- expert execution:
  - preserve expert parameter layout and math ordering
  - avoid hidden dtype/device casts that change logits distribution
- output contract:
  - wrappers should continue returning expected tuple/object shapes for downstream paths
  - keep cache return format compatible with runtime and export consumers
- export/runtime parity:
  - PyTorch runtime path and ONNX export path may differ internally, but token/logit parity expectations must remain stable

### B) Mixtral-Specific Guidance
- Gate API drift handling:
  - support both styles safely:
    - gate returns logits only
    - gate returns tuple including precomputed routing fields
  - do not assume one fixed return type across versions
- Expert container drift handling:
  - support callable aggregate experts and indexable legacy expert lists
  - preserve legacy fallback path if aggregate experts are unavailable
- ONNX-safe experts path:
  - during export, avoid operators known to break ORT pipelines in MoE routing
  - prefer upstream-provided batched expert helpers that are export-friendly
- Decoder integration:
  - tolerate decoder-layer attribute drift (`block_sparse_moe` vs alternative names)
  - keep residual/addition semantics identical to upstream

### C) GPT-OSS-Specific Guidance
- Projection layout drift:
  - handle fused projection layouts without rewriting downstream math
  - if downstream expects split tensors, initialize compatibility aliases from fused tensors
- MLP behavior:
  - keep existing MLP compute paths stable once compatibility tensors are created
  - do not re-order activation/projection math unless parity proves necessity
- Cache behavior:
  - preserve GPT-OSS-specific cache handling exceptions in runtime glue if generic conversion causes regressions
  - keep cache interface explicit at wrapper boundaries to avoid hidden conversion bugs

### D) MoE Failure Signatures -> Fast Actions
- symptom: shape mismatch in router/expert outputs
  - action: inspect gate return contract and normalize to one internal representation
- symptom: ORT export fails on MoE graph ops
  - action: switch/export-time path to export-safe expert forwarding helper
- symptom: PyTorch parity passes but ORT parity drifts
  - action: diff dtype conversions and expert-path branching between runtime/export code
- symptom: only GPT-OSS fails after cache changes
  - action: re-check GPT-OSS cache exception path before touching model math
- symptom: only Mixtral fails after wrapper updates
  - action: verify fallback logic for both expert container styles and decoder-layer attr names

### E) MoE Regression-Prevention Checks
- always run focused MoE triage first:
  - mixtral:
    - `python -m pytest -q tests/test_model_quickcheck.py -n auto -k 'causal_lm_cpu_runtime_parity_with_api_runner and mixtral'`
  - gpt_oss:
    - `python -m pytest -q tests/test_model_quickcheck.py -n auto -k 'causal_lm_cpu_runtime_parity_with_api_runner and gpt_oss'`
- after MoE-specific fixes, run full causal parity bucket before full quickcheck:
  - `python -m pytest -q tests/test_model_quickcheck.py -n auto -k 'causal_lm_cpu_runtime_parity_with_api_runner'`

### F) MoE Anti-Patterns (Avoid)
- hardcoding one router return shape for all HF versions
- deleting legacy fallback paths before proving aggregate-expert path is universal
- mixing cache-conversion refactors with MoE math rewrites in one patch
- accepting export-only success without runtime parity confirmation
- introducing model-specific hacks without documenting removal conditions

### 9) Gemma3 drift
Symptoms:
- rope config field drift (`rope_theta`)
- sliding cache/runtime mismatch
- multimodal wrapper aliasing issues (`language_model`)
- multimodal output object shape handling
- dtype mismatches in multimodal export cache inputs

Fix:
- rope fallback:
  - prefer `rope_theta`
  - fallback to `rope_global_base_freq`
- preserve `QEffSlidingWindowCache` behavior with logical token tracking
- keep/restore `language_model` alias in wrapper init
- handle `pooler_output` when image features return model outputs
- use model-dtype dummy KV tensors for multimodal export inputs
- bridge cache objects to legacy tuples at export boundaries when ONNX tracing requires tensor/tuple outputs

## Recommended Triage Commands
- all causal runtime parity:
  - `-k 'causal_lm_cpu_runtime_parity_with_api_runner'`
- mixtral only:
  - `-k 'causal_lm_cpu_runtime_parity_with_api_runner and mixtral'`
- gpt_oss only:
  - `-k 'causal_lm_cpu_runtime_parity_with_api_runner and gpt_oss'`
- whisper:
  - `-k 'whisper_export_smoke'`
- awq:
  - `-k 'awq_export_smoke'`
- text embedding export:
  - `-k 'text_embedding_cpu_parity_and_export'`
- gemma3/vlm:
  - `-k 'vlm_text_side_runtime_parity_and_full_export or vlm_export_smoke_additional_models'`

## File-Level Audit Checklist (Use Every Upgrade)
- `pyproject.toml`:
  - dependency pin alignment
  - `pytest-xdist` present in test extras if quickcheck uses `-n`
- `QEfficient/transformers/cache_utils.py`:
  - no hard dependency on removed HF cache-layer internals
  - tuple/object interop + `get_seq_length` compatibility
- `QEfficient/utils/run_utils.py` and `QEfficient/utils/generate_inputs.py`:
  - tuple/object cache normalization for runtime loops
  - no shape regressions on iterative token stepping
- `QEfficient/base/modeling_qeff.py`:
  - ONNX export call style consistent and deterministic
  - `past_key_values` input name construction works for tuple/cache objects
- model wrappers (`QEfficient/transformers/models/**`):
  - helper drift (`get_head_mask`, `_update_causal_mask`, rotary signatures)
  - model-family exceptions (Mixtral, GPT-OSS, Gemma3, Whisper, AWQ)
  - for MoE models, verify both runtime and export code paths preserve router/expert semantics
- tests:
  - `tests/test_model_quickcheck.py` remains deterministic and cleans stale artifacts
  - `tests/conftest.py` summary table still maps coverage categories correctly

## Final Acceptance
Required:
- full `tests/test_model_quickcheck.py` passes in user-provided env
- no unresolved regressions in HF PyTorch -> QEff PyTorch -> ONNXRuntime chain

## Final Report Format
Return:
1. Rebase Plan (what changed)
2. Compatibility Notes (why)
3. Validation Results (exact commands + pass/fail counts + duration)
4. Remaining Risks (only if tests not fully green)

## Style Constraints
- keep diffs minimal and production-safe
- prefer upstream semantics first
- keep local patches only where parity/export requires them
- never revert unrelated local changes
