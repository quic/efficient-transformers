# QEfficient Transform Patterns

## Core PyTorch Transform Contract
- `QEfficient/base/pytorch_transforms.py` defines `PytorchTransform.apply(model) -> (model, transformed)` and forbids instantiation.
- `QEFFBaseModel.__init__` applies each class in `self._pytorch_transforms` immediately after dtype normalization and warns if none apply.
- `QEFFBaseModel.transform` applies compile-parameter-dependent transforms such as KV-head replication and attention blocking.

## Transform Types
- `ModuleMappingTransform`: exact-type class mapping via `_module_mapping = {HFClass: QEffClass}`; replaces `module.__class__`, preserves module object/parameters/buffers, and calls `__qeff_init__` when the replacement defines it. Examples: `CustomOpsTransform`, `KVCacheTransform`, `PrefillOnlyTransform`, `T5ModelTransform`, `TextClassificationTransform`, VLM offload transforms.
- `ExternalModuleMapperTransform`: method mapping for external/unvendored modules via `_match_class_replace_method` or `_match_string_replace_method`; binds methods with `MethodType` and calls `__qeff_init__` if present. Examples: `KVCacheExternalModuleMapperTransform`, `PrefillOnlyExternalModuleMapperTransform`, `RevertPrefillOnlyExternalModuleMapperTransform`.
- `ModuleMutatorTransform`: recursive child replacement for modules matching `_match_class`; `mutate` returns a new module and may change weights/buffers. Examples: AWQ/GPTQ to `QuantLinearORT`, FP8 dequantization to `nn.Linear`, MXFP4 expert dequantization.
- Bespoke transform classes: custom `apply` signatures or whole-model edits. Examples: `ReplicateKVHeadTransform`, `SpDTransform`, `SamplerTransform`, `PoolingTransform`, `BlockingAttentionTransform`, `SplitGateUpWeightsTransform`.

## Current Registration Sites
- Generic constructor-time transforms live in `_pytorch_transforms` lists in `QEfficient/transformers/models/modeling_auto.py`.
- `QEFFAutoModel` uses custom ops plus AWQ/GPTQ mutators for general models.
- sequence classification uses custom ops plus `TextClassificationTransform`.
- causal/VLM wrappers include quantization mutators before `CustomOpsTransform`, `KVCacheTransform`, and external mappers as needed.
- Prefill toggles are applied explicitly with `PrefillOnly*` and revert transforms when the prefill mode changes.
- `SamplerTransform` and `SpDTransform` are applied where `qaic_config` mode flags are available.
- `PoolingTransform` wraps embedding models during `QEFFAutoModel.__init__` when a pooling option is provided.
- `BlockingAttentionTransform` is applied by `QEFFBaseModel.transform` after compile-time blocking config is built.

## Mapping Paths By Transform Family
- Quantizer selection is mapped before/while loading models: `QEfficient/transformers/quantizers/auto.py` replaces Transformers `AUTO_QUANTIZER_MAPPING` and `AUTO_QUANTIZATION_CONFIG_MAPPING` with QEff quantizer/config classes. Normal QEff loads use `@with_replaced_quantizers` on `QEFFTransformersBase.from_pretrained`; manual flows can call `replace_transformers_quantizers()`/`undo_transformers_quantizers()`.
- Quantized module shape is then handled by QEff quantizers and mutators: quantizers may insert placeholder/dequant modules during weight loading, and `quant_transforms.py` mutators convert loaded AWQ/GPTQ/FP8/MXFP4 modules into runtime-friendly QEff or Torch modules.
- Non-quant PyTorch mapping is declared in `QEfficient/transformers/models/pytorch_transforms.py`: class mappers use `_module_mapping`, external method mappers use class/string method maps, and bespoke transforms implement their own `apply`.
- Model applicability is model-wise, not automatic globally: add the transform to the relevant `_pytorch_transforms` list in `modeling_auto.py` so only that wrapper family runs it and ordering stays explicit.
- Mode-dependent transforms are not just list entries: prefill, sampler, SPD, pooling, VLM offload, and blocking transforms are applied where their mode flags or compile-time config are available.

## Mapper Guidelines
- Prefer exact class mappings when the upstream class is imported in-tree and full class behavior should change.
- Prefer external method mapping when classes come from dynamic/external model code or the model cannot be safely imported as a static dependency.
- Keep mapper replacements state-preserving unless `__qeff_init__` is explicitly needed. If `__qeff_init__` mutates buffers/parameters/config, add tests that prove the mutation is correct and safe to re-run or guarded.
- Remember `ModuleMappingTransform` uses `type(module)`, not `isinstance`; subclasses are not matched unless explicitly mapped.

## Mutator Guidelines
- Use mutators for true weight/layout transformations: unpacking quantized weights, dequantizing FP8/MXFP4, replacing fused expert modules, packing custom operators, or changing module object identity.
- Copy bias, device, dtype, config fields, and train/eval-relevant flags intentionally.
- Recurse behavior is supplied by `ModuleMutatorTransform.apply`; implement only `_match_class` and `mutate` unless traversal policy itself must change.
- If a mutator uses parent context, handle `parent_module is None` for root matches.

## Test Hotspots
- `tests/base/test_pytorch_transforms.py`: minimal mapper/mutator contract tests.
- `tests/transformers/test_pytorch_transforms.py`: custom ops, KV cache, SPD, AWQ/GPTQ functional coverage.
- `tests/unit_test/transforms/test_quantization_transforms.py`: quant transform importability, structure, ordering, and auto-model integration.
- `tests/unit_test/transforms/test_transform_accuracy.py`: mapper replacement, parity, pooling, sampler, SPD, VLM offload, blocking, and architecture coverage.
- `tests/unit_test/transforms/test_speculative_decoding.py`: SPD mode behavior and error paths.
- `tests/unit_test/models/test_new_arch_accuracy.py`: architecture-specific transform coverage for newer model families.
