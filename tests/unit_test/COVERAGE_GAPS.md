# QEfficient Unit Test Coverage Gaps & Roadmap to ~100% Coverage

**Generated**: 2026-03-11  
**Last Updated**: 2026-03-11 (third gap-fill review)  
**Baseline**: `tests/unit_test/` as of commit `388a1f6`  
**Estimated coverage before any gap-fill**: ~40–45%  
**Estimated coverage after all gap-fills**: ~90–92%  
**Total tests**: **1,286** across 24 test files

---

## Table of Contents

1. [Complete Gap-Fill History](#1-complete-gap-fill-history)
2. [Updated Coverage by Category](#2-updated-coverage-by-category)
3. [Remaining Gaps (to reach 95%+)](#3-remaining-gaps-to-reach-95)
4. [Priority Matrix (Remaining Work)](#4-priority-matrix-remaining-work)
5. [Full Test Inventory](#5-full-test-inventory)

---

## 1. Complete Gap-Fill History

### Round 1 Gap-Fill

| New/Updated File | Tests Added | Key Gaps Closed |
|-----------------|-------------|-----------------|
| `models/test_sliding_window_cache.py` (NEW) | 28 | `QEffSlidingWindowCache` (0→100%), `update3D` (0→100%), GPT-OSS chunked methods, CCL path |
| `utils/test_auto_model_api.py` (NEW) | 41 | `QEFFAutoModel` encoder, `QEFFAutoModelForCTC`, specialization API, prefill toggle |
| `utils/test_error_handling.py` (NEW) | 23 | 12/15 error paths now covered |
| `utils/test_padding_and_shapes.py` (NEW) | 25 | `get_padding_shape_from_config`, `sampler_utils`, `hash_utils` (all 0→covered) |
| `transforms/test_speculative_decoding.py` (UPDATED) | +11 | `tlm_forward` execution, Qwen2 SpD, `post_processing` registry |
| `transforms/test_transform_accuracy.py` (UPDATED) | +5 | Max/avg/custom pooling modes |
| `transforms/test_onnx_transforms.py` (UPDATED) | +6 | FP16Clip functional clipping, `RenameFunctionOutputsTransform` |

**Coverage after Round 1**: ~42% → ~72–78%

---

### Round 2 Gap-Fill

| New/Updated File | Tests Added | Key Gaps Closed |
|-----------------|-------------|-----------------|
| `models/test_new_arch_accuracy.py` (NEW) | **70** | 11 previously-uncovered CausalLM architectures + GPT-OSS + Grok1 transform structure |

**Architectures now covered by `test_new_arch_accuracy.py`:**

| Architecture | Tests | What's Covered |
|-------------|-------|----------------|
| **Gemma3** (text) | 5 | KV transform replacement, custom ops, greedy token preserved, finite outputs |
| **Qwen3** | 5 | KV transform replacement, custom ops, greedy token preserved, finite outputs |
| **Qwen3-MoE** | 4 | KV transform replacement, sparse MoE block replacement, finite outputs |
| **GPTBigCode** | 4 | KV transform replacement, finite outputs, module mapping |
| **Starcoder2** | 4 | KV transform replacement, greedy token preserved, finite outputs |
| **Granite** | 5 | KV transform replacement, custom ops, greedy token preserved, finite outputs |
| **GraniteMoE** | 3 | KV transform replacement, finite outputs |
| **OLMo2** | 5 | KV transform replacement, custom ops, greedy token preserved, finite outputs |
| **MPT** | 4 | KV transform replacement, finite outputs, module mapping |
| **CodeGen** | 4 | KV transform replacement, finite outputs, module mapping |
| **GPTJ** | 4 | KV transform replacement, finite outputs, module mapping |
| **GPT-OSS** (structure) | 5 | In KV cache mapping, PrefillOnly mapping, maps to QEff variants |
| **Grok1** (structure) | 6 | In external mapper transform, forward method callable |
| **Mapping completeness** | 12 | All 12 new archs verified in `KVCacheTransform.module_mapping` |

**Coverage after Round 2**: ~72–78% → **~85–88%**

---

### Round 3 Gap-Fill

| Updated File | Old Tests | New Tests | Delta | Gaps Closed |
|-------------|-----------|-----------|-------|-------------|
| `models/test_hybrid_cache_correctness.py` | 35 | **45** | +10 | GAP C: `full_cache_update_chunked`, `sliding_window_update_chunked`, `from_legacy_cache` classmethods |
| `transforms/test_transform_accuracy.py` | 99 | **119** | +20 | GAP D: `VlmKVOffloadTransform`, `VlmNoKVOffloadTransform`, `KVCacheExternalModuleMapperTransform` |
| `transforms/test_onnx_transforms.py` | 26 | **40** | +14 | GAP E: `SplitTensorsTransform` functional, `CustomOpTransform` structure (RMSNorm node) |
| `utils/test_auto_model_api.py` | 41 | **60** | +19 | GAP F: CCL mode (`comp_ctx_lengths`), prefill state change structure |
| `transforms/test_peft_transforms.py` | 18 | **31** | +13 | GAP G: LoRA accuracy vs base, `AdapterWeightsToInputsTransform` structure |
| `utils/test_padding_and_shapes.py` | 25 | **46** | +21 | GAP H: `process_ccl_specializations`, `automatic_ccl_generation`, helper functions |
| `transforms/test_speculative_decoding.py` | 33 | **41** | +8 | GAP I: SpD ONNX structure, `build_and_attach_mlp` |

**New tests added in Round 3**: +105  
**Coverage after Round 3**: ~85–88% → **~90–92%**

**New classes added in Round 3:**

| New Test Class | File | Tests | What's Covered |
|---------------|------|-------|----------------|
| `TestQEffHybridCacheForGPTOSSChunkedMethods` | `test_hybrid_cache_correctness.py` | 7 | `full_cache_update_chunked`, `sliding_window_update_chunked` with position/shape verification |
| `TestFromLegacyCacheClassmethods` | `test_hybrid_cache_correctness.py` | 5 | `from_legacy_cache` on all 3 hybrid cache classes |
| `TestVlmKVOffloadTransform` | `test_transform_accuracy.py` | 5 | Importable, has mapping, maps `MllamaTextCrossAttention` → `QEffMllamaTextCrossAttentionTwoQPC` |
| `TestVlmNoKVOffloadTransform` | `test_transform_accuracy.py` | 6 | Importable, has mapping, maps to `QEffMllamaTextCrossAttentionSingleQPC`, differs from offload |
| `TestKVCacheExternalModuleMapperTransform` | `test_transform_accuracy.py` | 11 | InternVL/Molmo/Grok1 string-based mappings, forward callables, `get_dummy_inputs`, RMSLayerNorm |
| `TestSplitTensorsTransformFunctional` | `test_onnx_transforms.py` | 5 | `apply` populates mapping, assigns correct file name, stores tensor proto |
| `TestCustomOpTransformStructure` | `test_onnx_transforms.py` | 9 | `_custom_ops` dict, RMSNorm/CtxScatter/CtxGather keys, `to_function_proto`, ONNX apply |
| `TestQEFFAutoModelForCausalLMCCL` | `test_auto_model_api.py` | 8 | `comp_ctx_lengths` in prefill/decode specialization results |
| `TestQEFFAutoModelForCausalLMPrefillStateChange` | `test_auto_model_api.py` | 11 | `prefill()` signature, `PrefillOnlyTransform`/`RevertPrefillOnlyTransform` structure |
| `TestLoRAAccuracyVsBase` | `test_peft_transforms.py` | 5 | LoRA with zero B = same as base; non-zero B = different; trainable params subset |
| `TestAdapterWeightsToInputsTransformStructure` | `test_peft_transforms.py` | 8 | `apply` classmethod, in `_onnx_transforms`, `export`/`compile` methods exist |
| `TestCheckCCLSpecializations` | `test_padding_and_shapes.py` | 8 | `process_ccl_specializations` returns lists, last element ≤ ctx_len, explicit/partial lists |
| `TestAutomaticCCLGeneration` | `test_padding_and_shapes.py` | 5 | `automatic_ccl_generation` returns 3-tuple, `mapped_cl` is multiple of 1024 |
| `TestCCLHelperFunctions` | `test_padding_and_shapes.py` | 8 | `next_multiple_of_1024`, `build_doubling_list`, `is_power_of_two`, `floor_to_1000` |
| `TestSpDONNXStructure` | `test_speculative_decoding.py` | 8 | `build_and_attach_mlp` importable/callable/signature, `build_and_attach_turbo`, ONNX export |

---

## 2. Updated Coverage by Category

| Category | Round 0 | Round 1 | Round 2 | Round 3 | Notes |
|----------|---------|---------|---------|---------|-------|
| **CausalLM models** | 38% | 38% | **85%** | **85%** | No new arch tests in Round 3; Llama4 still missing |
| **VLM models** | 0% | 0% | **0%** | **0%** | No VLM unit tests added in any round |
| **Cache classes** | 55% | 88% | **88%** | **~95%** | HybridChunked methods + from_legacy_cache classmethods |
| **PyTorch transforms** | 47% | 68% | **68%** | **~88%** | VlmKVOffload + VlmNoKVOffload + ExternalModuleMapper |
| **ONNX transforms** | 40% | 72% | **72%** | **~90%** | SplitTensors functional + CustomOpTransform structure |
| **QEFFAutoModel API** | 40% | 78% | **78%** | **~92%** | CCL mode + prefill state change |
| **Utility functions** | 33% | 75% | **75%** | **~90%** | CCL specializations + automatic_ccl_generation + helpers |
| **Error handling** | 20% | 82% | **82%** | **82%** | No new error tests in Round 3 |
| **SpD functional** | 30% | 78% | **78%** | **~90%** | SpD ONNX structure + build_and_attach_mlp |
| **PEFT functional** | 15% | 15% | **15%** | **~65%** | LoRA accuracy + AdapterWeightsToInputsTransform structure |
| **Overall** | **~42%** | **~75%** | **~85–88%** | **~90–92%** | |

---

## 3. Remaining Gaps (to reach 95%+)

Only **2 gaps remain**. Addressing both would push coverage to ~95%.

---

### 🔴 GAP A — VLM / Multimodal Models (HIGHEST REMAINING IMPACT)

**Status**: ❌ NOT addressed in any round  
**Impact**: 0% unit test coverage for all 9 VLM architectures  
**Coverage impact if fixed**: +4–5% overall

| Model | Key Components Untested | Risk |
|-------|------------------------|------|
| **Mllama** (Llama3.2-Vision) | `VlmKVOffloadTransform` apply (actual module replacement), cross-attention KV | 🔴 HIGH |
| **Qwen2.5-VL** | Vision transformer + text decoder | 🔴 HIGH |
| **Gemma3** (multimodal) | `QEffGemma3ForConditionalGeneration` | 🔴 HIGH |
| **Llama4** (multimodal) | Vision + text | 🔴 HIGH |
| **Llava / LlavaNext** | Decoder wrapper pattern | 🟡 MEDIUM |
| **Mistral3** | Pixtral vision encoder | 🟡 MEDIUM |
| **InternVL** | `KVCacheExternalModuleMapperTransform` apply (actual) | 🟡 MEDIUM |
| **Molmo** | External module mapper apply (actual) | 🟡 MEDIUM |

**Required**: Create `models/test_vlm_accuracy.py`:
```python
class TestQEffAutoModelForImageTextToTextDispatch:
    def test_dual_qpc_dispatch_when_kv_offload_true(self): ...
    def test_single_qpc_dispatch_when_kv_offload_false(self): ...
    def test_dual_qpc_has_vision_and_lang_models(self): ...
    def test_single_qpc_has_single_model(self): ...

class TestMultimodalUtilityMixin:
    def test_auto_correct_inputs_raises_on_missing_pixel_values(self): ...
    def test_auto_correct_inputs_raises_on_wrong_image_shape(self): ...
    def test_auto_correct_inputs_passes_with_valid_inputs(self): ...

class TestVlmKVOffloadTransformApply:
    def test_mllama_cross_attention_replaced_with_two_qpc_variant(self): ...
    def test_vlm_no_kv_offload_transform_replaces_with_single_qpc_variant(self): ...
    def test_transform_module_mapping_contains_mllama_cross_attention(self): ...
```

**Estimated tests needed**: ~25

---

### 🔴 GAP B — Llama4 (Text) Architecture (HIGH IMPACT)

**Status**: ❌ NOT addressed  
**Impact**: Llama4 is a new high-priority architecture (MoE + chunked attention) with zero tests  
**Coverage impact if fixed**: +1% overall

**Required**: Add to `models/test_new_arch_accuracy.py`:
```python
def make_tiny_llama4():
    from transformers import Llama4Config, Llama4ForCausalLM
    cfg = Llama4Config(num_hidden_layers=2, num_attention_heads=4, ...)
    return Llama4ForCausalLM(cfg).eval(), cfg

class TestLlama4TextAccuracy:
    def test_llama4_kv_transform_replaces_attention(self): ...
    def test_llama4_kv_transform_for_causal_lm_replaced(self): ...
    def test_llama4_combined_transforms_produce_finite_outputs(self): ...
    def test_llama4_in_kv_cache_transform_mapping(self): ...
```

**Estimated tests needed**: ~5

---

## 4. Priority Matrix (Remaining Work)

| Priority | Gap | New Test File(s) | Est. Tests | Effort | Coverage Impact |
|----------|-----|-----------------|------------|--------|-----------------|
| 🔴 **P1** | VLM dispatch + transforms (0% unit coverage) | `models/test_vlm_accuracy.py` | ~25 | Medium | +4–5% overall |
| 🔴 **P1** | Llama4 text architecture | `models/test_new_arch_accuracy.py` | ~5 | Low | +1% overall |

**Total remaining tests to reach ~95%: ~30 tests**

---

## 5. Full Test Inventory

### Current Test Count (1,286 total)

| File | Tests | Status |
|------|-------|--------|
| `utils/test_cloud.py` | 147 | ✅ |
| `utils/test_diffusers.py` | 109 | ✅ |
| `utils/test_generation.py` | 107 | ✅ |
| `transforms/test_transform_accuracy.py` | 119 | ✅ UPDATED (Round 3) |
| `utils/test_modeling_registry.py` | 98 | ✅ |
| `models/test_new_arch_accuracy.py` | 70 | ✅ NEW (Round 2) |
| `utils/test_auto_model_api.py` | 60 | ✅ UPDATED (Round 3) |
| `models/test_causal_lm_accuracy.py` | 62 | ✅ |
| `utils/test_padding_and_shapes.py` | 46 | ✅ UPDATED (Round 3) |
| `transforms/test_onnx_transforms.py` | 40 | ✅ UPDATED (Round 3) |
| `e2e/test_vlm_e2e.py` | 39 | ✅ |
| `utils/test_input_handler.py` | 39 | ✅ |
| `transforms/test_quantization_transforms.py` | 35 | ✅ |
| `transforms/test_peft_transforms.py` | 31 | ✅ UPDATED (Round 3) |
| `models/test_hybrid_cache_correctness.py` | 45 | ✅ UPDATED (Round 3) |
| `transforms/test_speculative_decoding.py` | 41 | ✅ UPDATED (Round 3) |
| `models/test_prefill_decode_kv_handoff.py` | 33 | ✅ |
| `models/test_sliding_window_cache.py` | 28 | ✅ NEW (Round 1) |
| `models/test_cache_correctness.py` | 26 | ✅ |
| `models/test_gemma2_accuracy.py` | 24 | ✅ |
| `utils/test_error_handling.py` | 23 | ✅ NEW (Round 1) |
| `e2e/test_speech_e2e.py` | 22 | ✅ |
| `e2e/test_embedding_e2e.py` | 22 | ✅ |
| `e2e/test_seq_classification_e2e.py` | 20 | ✅ |
| **TOTAL** | **1,286** | |

### Coverage Statistics

| Category | Source Classes/Methods | Covered | Coverage % |
|----------|----------------------|---------|------------|
| CausalLM models | ~22 architectures | 19 fully + 2 structure-only | **85%** |
| VLM models | ~9 architectures | 0 (unit) | **0%** |
| Cache classes | 8 classes, ~30 methods | 8 classes, ~28 methods | **~95%** |
| PyTorch transforms | 15 transforms | 13 (structure), 11 (functional) | **~88%** |
| ONNX transforms | 5 transforms | 5 (4 functional) | **~90%** |
| QEFFAutoModel API | ~30 methods | ~27 | **~92%** |
| Utility functions | ~15 modules | ~13 | **~90%** |
| Error handling | ~15 error paths | ~12 | **82%** |
| SpD functional | ~8 components | ~7 | **~90%** |
| PEFT functional | ~6 components | ~4 | **~65%** |
| **Overall** | | | **~90–92%** |

### Target State (to reach 95%+)

| Category | Current | Target | Tests Needed |
|----------|---------|--------|-------------|
| CausalLM models | 85% | 95% | ~5 (Llama4) |
| VLM models | 0% | 65% | ~25 |
| Cache classes | ~95% | 98% | — |
| PyTorch transforms | ~88% | 92% | — |
| ONNX transforms | ~90% | 95% | — |
| QEFFAutoModel API | ~92% | 95% | — |
| Utility functions | ~90% | 95% | — |
| Error handling | 82% | 90% | ~5 |
| SpD functional | ~90% | 95% | — |
| PEFT functional | ~65% | 80% | ~5 |
| **Overall** | **~90–92%** | **~95%** | **~30** |

---

## Appendix: Key Source Files Referenced

| Source File | Key Classes / Functions |
|------------|------------------------|
| `QEfficient/transformers/cache_utils.py` | `QEffDynamicLayer`, `QEffDynamicCache`, `QEffSlidingWindowCache`, `QEffHybridCache`, `QEffHybridChunkedCache`, `QEffHybridCacheForGPTOSS`, `QEffEncoderDecoderCache`, `InvalidIndexProvider` |
| `QEfficient/transformers/models/modeling_auto.py` | `QEFFAutoModelForCausalLM`, `QEFFAutoModelForImageTextToText`, `QEFFAutoModel`, `QEFFAutoModelForCTC`, `QEFFAutoModelForSequenceClassification`, `QEFFAutoModelForSpeechSeq2Seq` |
| `QEfficient/transformers/models/pytorch_transforms.py` | All PyTorch transforms including `VlmKVOffloadTransform`, `VlmNoKVOffloadTransform`, `KVCacheExternalModuleMapperTransform` |
| `QEfficient/base/onnx_transforms.py` | `FP16ClipTransform`, `SplitTensorsTransform`, `CustomOpTransform`, `RenameFunctionOutputsTransform` |
| `QEfficient/transformers/modeling_utils.py` | `get_padding_shape_from_config`, `_create_causal_mask`, model registries |
| `QEfficient/utils/sampler_utils.py` | `get_sampling_inputs_and_outputs` |
| `QEfficient/utils/hash_utils.py` | `compute_hash` |
| `QEfficient/transformers/post_processing.py` | `model_type_registry`, `build_and_attach_mlp`, `build_and_attach_turbo` |
| `QEfficient/peft/onnx_transforms.py` | `AdapterWeightsToInputsTransform` |
| `QEfficient/transformers/embeddings/` | `PooledModel`, `validate_user_pooling_function` |
| `QEfficient/utils/check_ccl_specializations.py` | `process_ccl_specializations`, `automatic_ccl_generation`, `next_multiple_of_1024`, `build_doubling_list`, `is_power_of_two`, `floor_to_1000` |
| `QEfficient/transformers/models/mllama/modeling_mllama.py` | `QEffMllamaTextCrossAttentionTwoQPC`, `QEffMllamaTextCrossAttentionSingleQPC` |
