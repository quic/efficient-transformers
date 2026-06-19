# MoE Transform Cleanup Plan

## Goal

Refactor MoE optimization so model files contain only minimal model-specific adapter facts, while transform code owns class/method replacement, weight canonicalization, export-time flavour selection, and cache-key-affecting settings.

The desired end state is:

- No MoE weight transformation lives in model `__qeff_init__` methods.
- No duplicated MoE forward implementations live in model files when the shared MoE flavours can serve them.
- Existing and future MoE models plug into a single source of truth through adapter specs/builders.
- `OptimizedMoETransform` remains the public compatibility entry point but is internally split into mapping, mutation, and export configuration steps.

## Current Problems

- `ModuleMappingTransform` and `ExternalModuleMapperTransform` call `__qeff_init__`, so class/method mapping is currently mixed with state mutation.
- Several MoE wrappers use `__qeff_init__` to split, transpose, alias, or cache expert weights.
- Several MoE wrappers still carry local `forward` methods that differ mostly in routing/profile glue, not in the core MoE execution.
- `OptimizedMoETransform` already does export-time flavour selection, but it also depends on per-model `build_moe_weights()` / `__qeff_init__` side effects.
- MoE prefill blocking knobs are configured in more than one place, making hash-param and export-shape behaviour harder to reason about.

## Design Principles

- Keep mapping state-preserving: class swaps and method binding must not reshape, clone, split, or register expert weights.
- Keep mutation explicit: any tensor/layout conversion must happen in a `ModuleMutatorTransform`-style transform with focused tests.
- Keep export policy explicit: prefill/decode flavour selection, blocking knobs, and hash params belong in the export-time MoE transform.
- Keep the existing public API stable: callers should continue to use `OptimizedMoETransform.apply(...)` while the internals are decomposed.
- Prefer two or three small transforms over a broad `MutateAndMapTransform`; add a combined base only if atomic map+mutate ordering cannot be expressed safely.

## Target Architecture

### 1. Shared MoE Contract

Create a small adapter contract under `QEfficient/transformers/moe/` that captures the model-specific pieces:

- `match`: HF/QEff class or class-name matcher.
- `weight_builder`: function that returns canonical `MoEWeights` from the original module/expert container.
- `router`: function/method that returns selected experts, routing weights, and optional router logits.
- `profile`: function that returns `MoEProfile` for activation/bias behaviour.
- `return_router_logits`: whether forward returns `(hidden_states, router_logits)`.
- `supports_blocking` / `supports_static_chunks`: export capability flags.

Model files may keep tiny adapter helpers when they need access to upstream private fields, but the shared MoE package should own the canonical forward flow and weight layouts.

### 2. Transform Split

Keep `OptimizedMoETransform` as a facade that calls these narrower transforms in order:

1. `OptimizedMoEMapperTransform`
   - Performs MoE class/method mapping only.
   - Binds shared methods such as `forward`, `get_moe_weights`, `route`, or `moe_profile` where adapter specs can provide them.
   - Does not call weight-building code or mutate parameters/buffers.

2. `OptimizedMoEWeightsTransform`
   - Performs canonical expert-weight conversion.
   - Owns fused gate/up splitting, transposes, bias splitting, dequantized expert conversion handoff, and any backward-compatible aliases that must remain temporarily.
   - Is idempotent and marks modules once weights have been canonicalized.

3. `OptimizedMoEExportConfigTransform`
   - Resolves `decode_bmm`, `simple_loop`, or `expert_blocked` from `prefill_only`, `enable_chunking`, `qaic_config`, and model capabilities.
   - Sets `_moe_flavour`, expert-blocking attributes, static packed-chunk counts, and `hash_params`.
   - Becomes the single owner of MoE prefill blocking export settings.

If mapping and mutation must occur in a single recursive pass, add `ModuleMapAndMutatorTransform` in `QEfficient/base/pytorch_transforms.py` with tests first. Otherwise, keep the split transforms independent for clarity.

## Migration Phases

### Phase 0: Audit and Classify

- Inventory every MoE wrapper and classify it by expert layout, routing semantics, bias support, and blocking support:
  - `qwen3_moe`, `qwen3_vl_moe`, `qwen3_5_moe`
  - `glm4_moe`
  - `mixtral_moe`, `granitemoe`
  - `gpt_oss`
  - `gemma4`, `llama4`, `deepseek_v3`, `grok_1` if they still use bespoke MoE paths
- For each `__qeff_init__`, label whether it is:
  - pure metadata setup,
  - class/method compatibility setup,
  - weight/layout mutation,
  - or temporary test/backward-compatibility aliasing.
- For each MoE `forward`, identify what is truly model-specific: routing, profile, output tuple shape, or unsupported mode.

### Phase 1: Establish Shared Adapter Specs

- Add adapter specs/builders in the shared MoE package instead of adding more logic to model files.
- Move reusable fused-weight canonicalization into `moe/weights.py` or a dedicated `moe/adapters.py` module.
- Encode per-model differences declaratively where possible:
  - fused vs separate gate/up/down projections,
  - fused split dimension and interleaving,
  - transpose requirements,
  - bias handling,
  - activation/profile function,
  - router output convention.
- Keep model-file helpers only when they are thin accessors over upstream model internals.

### Phase 2: Implement the Weight Mutator

- Add `OptimizedMoEWeightsTransform` as the only place that canonicalizes MoE expert weights.
- Make repeated application safe by checking a marker such as `module.moe_weights is not None` or a private `_qeff_moe_weights_ready` flag.
- Preserve device, dtype, training/eval mode, parameter registration, and existing quantization-mutator ordering.
- Remove weight-splitting and alias-parameter creation from MoE `__qeff_init__` methods after each model is covered by the mutator.
- Keep temporary compatibility aliases only when an existing runtime/test still needs them, and track each alias for later removal.

### Phase 3: Implement the Mapper/Method Layer

- Add `OptimizedMoEMapperTransform` for MoE-specific mapping, rather than hiding MoE setup in generic `KVCacheTransform` behaviour.
- Bind or inherit a single shared MoE `forward` from `QEffMoEBlockMixin` wherever routing/profile adapters are sufficient.
- Replace chunked subclass-only forward differences with `_moe_flavour` selection where possible.
- Keep bespoke forwards only for models that have real semantic differences; document the reason next to the adapter registration.
- Remove MoE `__qeff_init__` calls from mapper paths once they no longer perform required setup.

### Phase 4: Consolidate Export-Time Configuration

- Reduce `OptimizedMoETransform.apply(...)` to orchestration:
  1. ensure MoE methods are mapped,
  2. ensure weights are canonicalized,
  3. configure export flavour and hash params.
- Move duplicate prefill-blocking setup from helper paths into `OptimizedMoEExportConfigTransform`, or make those helpers call the same implementation.
- Ensure `hash_params` changes whenever the MoE export graph can change:
  - flavour,
  - `num_nsp`,
  - packed chunk size,
  - packed chunk count,
  - any future layout-specific option.
- Verify non-MoE models remain no-op and return `transformed=False`.

### Phase 5: Remove Legacy Hooks

- Remove MoE weight mutation from `__qeff_init__` methods.
- Remove manual test setup that calls `experts.__qeff_init__()` or block `__qeff_init__()` for MoE weight readiness.
- Remove redundant per-model MoE forwards that now dispatch through the shared MoE mixin.
- Keep non-MoE `__qeff_init__` methods out of scope unless they are directly coupled to MoE mapping/mutation.

## Suggested File Touch Points

- `QEfficient/base/pytorch_transforms.py`
  - Only touch if a reusable map+mutate base is truly needed.
- `QEfficient/transformers/models/pytorch_transforms.py`
  - Add the split MoE transforms and keep the compatibility facade.
- `QEfficient/transformers/moe/`
  - Add adapter specs and shared mapper/mutator helpers.
- `QEfficient/transformers/models/*/modeling_*.py`
  - Remove MoE weight mutations and duplicated forwards once covered by shared transforms.
- `QEfficient/transformers/models/modeling_auto.py`
  - Keep export calls stable, but route MoE prefill/export setup through the new facade.
- `tests/base/test_pytorch_transforms.py`
  - Add base transform coverage if a combined map+mutate base is introduced.
- `tests/unit_test/transforms/test_transform_accuracy.py`
  - Add/adjust structure and parity tests for mapper/mutator behaviour.
- `tests/transformers/models/test_moe_prefill_blocked.py`
  - Replace manual `__qeff_init__` setup with transform application and retain prefill-blocked parity/export coverage.

## Acceptance Criteria

- MoE mapper transforms do not call code that mutates expert weights.
- MoE weight mutation is idempotent and isolated to the mutator transform.
- `OptimizedMoETransform.apply(...)` remains safe to call before every export.
- Decode and prefill MoE forwards use shared flavour functions unless a model has a documented semantic exception.
- Existing HF PyTorch → QEff PyTorch parity is preserved for all migrated MoE models.
- ONNX export hash params capture all MoE graph-shaping options.
- Non-MoE models are unaffected.

## Test Plan

Run narrow tests first, then widen:

1. `pytest -n auto tests/base/test_pytorch_transforms.py`
2. `pytest -n auto tests/unit_test/transforms/test_transform_accuracy.py -k "MoE or moe"`
3. `pytest -n auto tests/transformers/models/test_moe_prefill_blocked.py`
4. If `tests/test_model_quickcheck.py` exists, run the MoE-relevant quickcheck selection; otherwise report the focused tests above as the fallback acceptance gate.
5. Run `pre-commit run --all-files` before PR handoff.

Do not claim export/compile parity from export success alone; keep explicit HF PyTorch → QEff PyTorch comparisons for every migrated MoE path.

## Risks and Mitigations

- **Risk:** separating mapping from mutation changes transform ordering.
  - **Mitigation:** add structure tests for ordering and keep `OptimizedMoETransform` as a single facade during migration.
- **Risk:** removing aliases breaks hidden tests or older runtime code.
  - **Mitigation:** remove aliases in stages and assert no production code references them before deletion.
- **Risk:** quantized expert transforms must run before MoE weight canonicalization.
  - **Mitigation:** document and test ordering with FP8/MXFP4/AWQ/GPTQ paths.
- **Risk:** VLM language decoder paths may bypass causal-LM-only wiring.
  - **Mitigation:** keep the transform scanning `model.modules()` and include VLM MoE tests.
- **Risk:** chunked prefill export depends on static packed-chunk counts.
  - **Mitigation:** keep hash-param and ONNX function-count tests for chunked exports.

## Proposed PR Breakdown

1. Add MoE adapter specs and the weight mutator with tests, while preserving existing model hooks.
2. Convert one lower-risk model family, preferably Mixtral or GraniteMoE, to prove the pattern.
3. Convert Qwen3/GLM4/Qwen3-VL/Qwen3.5 prefill-blocked families and update tests to stop calling `__qeff_init__` manually.
4. Convert GPT-OSS/Gemma4/Llama4/DeepSeek/Grok paths as applicable, respecting quantized expert ordering.
5. Remove now-dead MoE `__qeff_init__` methods, aliases, duplicated forwards, and duplicate prefill-blocking helpers.
