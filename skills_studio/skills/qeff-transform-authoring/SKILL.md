---
name: qeff-transform-authoring
description: Add, modify, or review QEfficient transform code. Use when the user wants a new PyTorch transform, module mapping, method mapper, module mutator, quantization transform, prefill/sampler/SPD/blocking transform, ONNX-adjacent transform wiring, or tests for transform registration/parity in `QEfficient/base/pytorch_transforms.py`, `QEfficient/transformers/models/pytorch_transforms.py`, `QEfficient/transformers/quantizers/quant_transforms.py`, `QEfficient/base/modeling_qeff.py`, or `QEfficient/transformers/models/modeling_auto.py`.
---

# QEff Transform Authoring

## Overview
Use this skill to add transforms with the same architecture and safety expectations as the existing QEfficient transform pipeline. Prefer the smallest transform type that matches the behavior: mapper for class or method substitution, mutator for weight/state conversion, bespoke `apply` only when extra runtime inputs or non-local edits are required.

## First Checks
1. Read `references/transform-patterns.md` before editing transform code.
2. Inspect the target model/module and decide whether the transform must preserve weights, mutate weights, bind methods, or depend on compile/runtime options.
3. Confirm registration path before implementing: constructor-time `_pytorch_transforms`, explicit calls in `modeling_auto.py`, compile-time `QEFFBaseModel.transform`, or ONNX export transforms.
4. If the task downloads or loads a model, ask for `HF_HUB_CACHE`; if it exports or compiles artifacts, ask for `QEFF_HOME`.

## Choose The Transform Type
- Use `ModuleMappingTransform` for in-place class replacement where parameters and buffers stay valid. It matches exact `type(module)`, assigns `module.__class__`, and invokes `__qeff_init__` if present.
- Use `ExternalModuleMapperTransform` when the upstream class cannot be imported reliably or only methods need replacing. It binds replacement callables with `MethodType` and may still run `__qeff_init__`.
- Use `ModuleMutatorTransform` when the original module object must be replaced or weights/buffers must be unpacked, dequantized, reshaped, split, packed, or otherwise changed.
- Use a bespoke `PytorchTransform.apply(model, ...) -> (model, transformed)` only when the transform needs extra arguments, wraps the whole model, edits config, changes selected layers by policy, or runs only during compile/export modes.

## Implementation Workflow
1. Add QEff replacement behavior first.
   - For mappers, add or reuse QEff classes/methods near the relevant model wrapper.
   - For mutators, implement deterministic `mutate(original_module, parent_module)` and return a fully initialized replacement module.
2. Wire the transform in the narrowest registry.
   - Quantizer loading behavior maps through `QEfficient/transformers/quantizers/auto.py`; `from_pretrained` temporarily replaces Transformers quantizer/config registries with QEff entries.
   - Generic model class swaps: `QEfficient/transformers/models/pytorch_transforms.py`.
   - Quantized weight conversions: `QEfficient/transformers/quantizers/quant_transforms.py`.
   - Model wrapper pipelines: `_pytorch_transforms` in `QEfficient/transformers/models/modeling_auto.py`.
   - Compile-parameter transforms: `QEFFBaseModel.transform` in `QEfficient/base/modeling_qeff.py` only when the transform depends on compile options.
3. Preserve transform ordering.
   - Quantization mutators must run before `KVCacheTransform` if QEff wrappers expect dequantized/custom modules.
   - `CustomOpsTransform` and `KVCacheTransform` are constructor-time transforms for most model wrappers.
   - Prefill, sampler, SPD, pooling, VLM offload, and blocking transforms are applied explicitly where their mode flags are known.
4. Keep mapper mutations intentional.
   - Mappers should usually preserve parameters/buffers by only replacing class or methods.
   - If `__qeff_init__` creates, reshapes, or copies state, document and test that mutation; otherwise use a mutator.
5. Return the correct `(model, transformed)` tuple and make repeated application safe when practical.

## Testing Workflow
- Add a structure test for every new transform: importability, mapping or `_match_class`, registration in `_pytorch_transforms`, and ordering if ordering matters.
- Add a minimal functional test that applies the transform to a tiny local module or config-derived model and asserts the expected class/method/module replacement.
- For mapper transforms, prove HF/QEff PyTorch behavior is unchanged with logits, greedy token, finite output, or a focused forward parity check.
- For mutators, verify replacement object identity changes where expected and validate weight/buffer shapes, dtype, bias handling, and numerical equivalence or dequantization correctness.
- For mode-dependent bespoke transforms, test both disabled and enabled paths and assert unsupported models raise clear errors.

## References
- `references/transform-patterns.md` — transform taxonomy, registration sites, examples, and test locations.
