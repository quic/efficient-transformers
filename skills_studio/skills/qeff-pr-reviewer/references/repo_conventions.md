# QEfficient repo conventions (reviewer's quick reference)

The contract a PR is being judged against. When you're unsure whether something is convention, grep here first.

## Top-level layout

| Dir | Purpose |
|---|---|
| `QEfficient/base/` | `QEFFBaseModel`, ONNX transform base classes, pytorch transform base classes, abstract export/compile workflow, `common.py` (where `QEFFCommonLoader` lives) |
| `QEfficient/transformers/` | Per-model implementations under `models/<model>/`, the Auto class registry (`models/modeling_auto.py`), the transform map (`models/pytorch_transforms.py`), `cache_utils.py`, `modeling_utils.py` (DYNAMIC_SEQ_LEN_SUPPORTED_MODEL_ARCH lives here), `sampler/` |
| `QEfficient/exporter/` | ONNX export utilities |
| `QEfficient/generation/` | Inference loops: `text_generation_inference`, `vlm_generation`, `embedding_handler`, `cloud_infer` |
| `QEfficient/compile/` | QNN compile orchestration |
| `QEfficient/peft/` | PEFT/LoRA adapter support; `auto.py` and `lora/auto.py` define their own `model_hash` |
| `QEfficient/finetune/` | Fine-tuning entry points |
| `QEfficient/customop/` | Custom torch/ONNX ops; `ctx_scatter_gather.py` and `ctx_scatter_gather_cb.py` provide the 3D / Generalized / BlockedKV / CB variant matrix |
| `QEfficient/cloud/` | Cloud AI 100 session API |
| `QEfficient/diffusers/` | Diffusion pipelines (`pipelines/<x>/pipeline_<x>.py`) |
| `QEfficient/utils/` | Constants, logging, `hash_utils.py`, `_utils.py` (where `create_model_params` and `KWARGS_INCLUSION_LIST` live), generate-inputs helpers — keep it thin |
| `tests/` | 4-stage parity tests + unit tests + `configs/*_model_configs.json` |
| `examples/` | Per-task runnable examples + `onboarding_guide/causallm/` (canonical onboarding template) |
| `docs/source/` | Sphinx + Markdown; `validate.md` is the model support matrix |
| `scripts/` | Debug/analysis utilities, not packaged |

When new code lands in a directory whose existing files don't match its purpose (e.g. ONNX surgery in `utils/`, generic cache logic referencing a specific model), raise a placement concern; cite the dir's stated purpose from this table.

## Layering / change-placement contract (read this on every PR)

This is the single most-violated rule in agent-authored PRs. The repo is layered, and a change must land at the layer that *actually owns* it. Putting model-specific behavior into a shared/base layer is the default failure mode — it compiles, passes the author's one test, and quietly degrades the architecture for every other model.

**Decision tree — "who needs this change?"**

1. **One specific model** (its attention, forward, RoPE, cache shaping, dummy inputs, output names, a quirk of that arch) → it belongs in that model's `QEfficient/transformers/models/<model>/modeling_<model>.py`. Never in a base class, never in the Auto registry, never behind a `model_type ==` branch in generic code.
2. **A handful of models, but not all** (a helper two or three families share) → `QEfficient/transformers/modeling_utils.py` (or a clearly-scoped shared helper module), imported by those models. Not a base class.
3. **Generation / inference-loop behavior** (decode loop, sampling glue, streaming, KV bookkeeping at runtime, VLM input assembly) → `QEfficient/generation/` (`text_generation_inference.py`, `vlm_generation.py`, `embedding_handler.py`, `cloud_infer.py`). Not `modeling_auto.py`'s `generate()` body.
4. **ONNX export surgery** → `QEfficient/exporter/`. **Compile orchestration** → `QEfficient/compile/`. **Thin constants / hashing / logging** → `QEfficient/utils/`.
5. **Genuinely needed by ALL models** — part of the abstract export → compile → runtime lifecycle every Auto class inherits → only then a base class (`QEfficient/base/modeling_qeff.py`, `base/common.py`, `base/pytorch_transforms.py`, `base/onnx_transforms.py`) or the shared Auto registry (`QEfficient/transformers/models/modeling_auto.py`, `cache_utils.py`).

**The base/shared layer is the highest bar, not the default landing spot.** These files are inherited by every model; a change here ripples everywhere and is the hardest to revert:

- `QEfficient/base/modeling_qeff.py` — `QEFFBaseModel`, abstract export/compile workflow.
- `QEfficient/base/common.py` — `QEFFCommonLoader` dispatch.
- `QEfficient/base/pytorch_transforms.py`, `QEfficient/base/onnx_transforms.py` — transform base classes.
- `QEfficient/transformers/models/modeling_auto.py` — the Auto class registry.
- `QEfficient/transformers/cache_utils.py` — generic cache abstractions.

**Rules for a change touching the base/shared layer:**

- **Adding *any* method, kwarg, or branch to a base class requires an explicit "all models need this" justification.** Review it hard. If only one model (or a few) exercises it, it's misplaced — relocate to `modeling_<x>.py` or `modeling_utils.py`. (Severity: Issue → Blocker if the misplaced code also couples the base layer to a specific model.)
- **Model-specific conditionals in generic code are a Blocker**: `if model_type == "<x>"`, `isinstance(self.model, <SpecificModel>)`, `<SpecificModel>._start`, or any name of a concrete model/config inside a base class, a generic Auto method, or `cache_utils.py`. The generic layer must not know which model it's running. (This is the same coupling failure as the cross-class-state check below, generalized to control flow.)
- **Generation logic in `modeling_auto.py`** instead of `QEfficient/generation/` → Issue; ask to move it.
- **A generic helper dumped into a base class** instead of `modeling_utils.py` → Issue.
- **Legitimate base-class change** (a new lifecycle hook every model uses, a fix to the shared export/compile/hash machinery, a new transform base capability) is fine — but the PR must show it's generic, and the change should not embed any single model's name or quirk.

When a PR's diff is concentrated in the base/shared files, that is itself a yellow flag: most well-scoped model work touches `models/<model>/`, `tests/`, `examples/`, `docs/` and adds only a few mapping/import lines to `pytorch_transforms.py` / `modeling_auto.py`. A feature PR that rewrites `modeling_qeff.py` or grows `modeling_auto.py` by hundreds of lines almost always has logic that belongs one layer down. Verify every base/shared hunk against the decision tree above and say where each misplaced piece should live.

## License header (mandatory on every `.py`)

```python
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
```

The header is BSD-3-Clause. Verify with `grep -q "SPDX-License-Identifier: BSD-3-Clause" <file>`. Don't byte-equal-check against a canonical block — small dash-count drift exists in main and is not a defect.

`#!/usr/bin/env python3` shebangs on `QEfficient/<anything that's imported>.py` are wrong — those modules are imported, not executed. Reserve shebangs for `examples/` and `scripts/`.

## Auto class architecture (`QEfficient/transformers/models/modeling_auto.py`)

Inheritance is split:

- `QEFFBaseModel` (abstract) — directly subclassed by `QEFFAutoModelForCausalLM` and a few PEFT variants.
- `QEFFTransformersBase(QEFFBaseModel)` — intermediate base. Subclassed by `QEFFAutoModel` (embeddings), `QEFFAutoModelForSequenceClassification`, `QEFFAutoModelForSpeechSeq2Seq`, `QEFFAutoModelForCTC`.
- `QEFFAutoModelForImageTextToText` — **factory class via `__new__`**, not a `QEFFBaseModel` subclass. Returns either `_QEffAutoModelForImageTextToTextDualQPC` or `_QEFFAutoModelForImageTextToTextSingleQPC`.
- `QEFFCommonLoader` — lives in `QEfficient/base/common.py`, dispatches to the right Auto class.
- PEFT variants — `QEffAutoPeftModelForCausalLM` (`peft/auto.py`) and `QEffAutoLoraModelForCausalLM` (`peft/lora/auto.py`) — these are the *only* classes with an actual `model_hash` property.

### Required class vars on every Auto class

- `_pytorch_transforms: List[PytorchTransform]` — applied in `QEFFBaseModel.__init__` via the transform loop.
- `_onnx_transforms: List[BaseOnnxTransform]` — consumed by `_export`.
- `_hf_auto_class` — what `from_pretrained` instantiates.

### Inherited from `QEFFBaseModel`

`transform`, `_export`, `_compile`, `get_onnx_path`, the `model_name` property. Subclasses provide `from_pretrained`, `export`, `compile`, `generate`. Overriding the inherited helpers is rare and worth flagging.

### Hash plumbing

Cache keys are SHA256 truncated to 16 chars (NOT MD5). Two hashes get derived during a normal run:

- Export hash: `create_export_hash(...)` in `QEfficient/utils/hash_utils.py:53`. Inputs include `self.hash_params` plus the exporter kwargs.
- Compile hash: `hash_dict_params(compile_hash_params)` from `QEFFBaseModel._compile` (`QEfficient/base/modeling_qeff.py:823-832`). `compile_hash_params` is a fixed dict: `command`, `specializations`, `custom_io`, `mdp_ts_num_devices`, `mdp_ts_json`, `num_speculative_tokens`, `prefill_only`.

Rules for a graph-affecting kwarg in a PR:

1. Plumbed through `__init__` / `from_pretrained` → must be in `KWARGS_INCLUSION_LIST` in `QEfficient/utils/constants.py`. The current list is small: `state_dict`, `revision`, `key_mapping`, `commit_hash`, `adapter_kwargs`, `adapter_name`, `gguf_file`, `pretrained_model_name_or_path`, `attn_implementation`, `_attn_implementation`, `qaic_config`. Anything else is silently dropped from the hash by `create_model_params` (`utils/_utils.py:516`).
2. Or the Auto class explicitly assigns `self.hash_params[<key>] = <value>` (the dominant pattern — see `modeling_auto.py` for many examples).
3. Compile-time switch → must enter `compile_hash_params`.

If none of those apply, the cache poisons: users hit a stale QPC after toggling and silently get wrong results. The export hash *does* incorporate the transform class names (`applied_transform_names` feeds `create_export_hash`), so a pre-fix and post-fix checkout land in different hashed dirs — but a kwarg that never enters any of the three plumbing points above is invisible regardless. This is a recurring bite: when reviewing, trace every new graph-affecting kwarg to one of the three sinks.

For PEFT: `model_hash` reads `model.active_peft_config.to_dict()`. Re-exporting the same base model with different adapters MUST produce different hashes; verify if the PR touches `QEfficient/peft/`.

### `continuous_batching` vs `DYNAMIC_SEQ_LEN_SUPPORTED_MODEL_ARCH`

These are separate gates:

- `continuous_batching=True` and `full_batch_size` — set on the QEFFAutoModelForCausalLM instance, gates the CB code path and KV cache shape.
- `DYNAMIC_SEQ_LEN_SUPPORTED_MODEL_ARCH` (`QEfficient/transformers/modeling_utils.py:196`) — gates the *full-batch dynamic seq-len* padding path used at `modeling_auto.py` ~line 3314 and `utils/generate_inputs.py:180`. Currently `{"gemma3", "llama4", "gemma3_text", "llama4_text"}`.

Adding a new model to `DYNAMIC_SEQ_LEN_SUPPORTED_MODEL_ARCH` requires more than just a list edit — the dummy-input path must handle the dynamic seq-len shape.

### `get_dummy_inputs` / `get_onnx_dynamic_axes` / `get_output_names` triplet

The Auto class delegates to the patched HF model: `self.model.get_dummy_inputs(...)`, `self.model.get_onnx_dynamic_axes(...)`, `self.model.get_output_names()`. Three rules:

- The keys returned by `get_dummy_inputs` must match the keys used in `get_onnx_dynamic_axes`.
- The names returned by `get_output_names` must match what the patched `forward()` actually returns.
- A mismatch surfaces as `torch.onnx.export` failure or — worse — as silent name mismatches at compile time.

### Model-specific hooks

Model-specific behavior reaches the generic layer through hooks, not conditionals. This is the positive model behind the whole layering contract: **the generic Auto class stays model-agnostic by probing the patched HF model for an optional method and delegating to it**, never by branching on the model's identity. The triplet above is one instance; others in `modeling_auto.py` include `get_specializations`, `get_npi_file`, `generate_npi_file`, `get_dummy_pkv_cache`, `get_pkv_dynamic_axes`. The pattern is always:

```python
if hasattr(self.model, "get_specializations"):
    specializations, compiler_options = self.model.get_specializations(...)
```

A model needing custom export/compile behavior **adds the method to its `models/<model>/modeling_<model>.py`**; the generic layer calls it behind `hasattr`. PR #271's review thread is the canonical statement of this discipline (maintainer rejected hardcoded constants in `modeling_auto.py` in favor of model-side `get_dummy_inputs` / `get_specializations` hooks). When a PR adds model-specific behavior, the right shape is a new hook, not a new `if model_type ==` branch. The single inline leak in-tree is `modeling_auto.py:1987` (`model_type == "molmo"`) — it should have been a hook.

**Arch-set registries (`SPECIALIZED_DISAGG_SERVING_MODEL_ARCH`, `DYNAMIC_SEQ_LEN_SUPPORTED_MODEL_ARCH`, `EXTERNAL_MODEL_CLASS_MAPPING`, all in `modeling_utils.py`) are a tolerated middle ground, not the ideal.** They are better than an inline literal (the model list is data, not control flow), and existing ones are accepted. But they are still "enumerate the models that get this behavior," which doesn't scale and re-introduces a model-awareness the generic layer would be better without. So:

- Do **not** flag `model_type in <SET>` membership as the inline-literal Blocker — it isn't one.
- When a PR *adds a model to such a set*, ask whether the same gate could be a model-side hook or a property derived from `config` (e.g. "does this model define `get_specializations` with a disagg path?" rather than "is this model in the disagg set?"). If a cleaner mechanism exists, recommend it and raise the registry growth as an **Issue** — don't endorse the set as the correct long-term design.
- Verify the model-side machinery the set membership implies actually exists. A set-only edit (string added, no dummy-input / prefill-transform / `retain_full_kv` support on the model) compiles and silently mis-serves — **Blocker**.

### `comp_ctx_lengths_prefill` vs `comp_ctx_lengths_decode`

Two separate kwargs (`modeling_auto.py:2997`, `3610-3611`, `1482-1483`). Processed by `process_ccl_specializations` (`modeling_auto.py:1575`). Easy to confuse: passing the same list to both is almost always wrong and changes the QPC's spec table.

## Pytorch transforms (`QEfficient/transformers/models/pytorch_transforms.py`)

Concrete transforms (verified list):

- `CustomOpsTransform` — RMSNorm and similar to AIC custom kernels.
- `KVCacheTransform` — attention/layer/model classes for KV cache injection.
- `PrefillOnlyTransform` / `PrefillOnlyChunkedTransform` — prefill-mode swaps.
- `RevertPrefillKeepAttentionTransform` / `RevertPrefillOnlyTransform` — inverses (see asymmetry note below).
- `ReplicateKVHeadTransform` — KV-head replication for grouped-query.
- `SpDTransform` — disaggregated serving (vision/lang split).
- `SamplerTransform` — on-device sampling logits processor.
- `VlmKVOffloadTransform` / `VlmNoKVOffloadTransform` — VLM KV memory variants.
- `KVCacheExternalModuleMapperTransform`, `PrefillOnlyExternalModuleMapperTransform`, `RevertPrefillOnlyExternalModuleMapperTransform` — string-keyed mappers (used by DeepseekV3, Molmo, Grok1).
- `T5ModelTransform`, `TextClassificationTransform`, `PoolingTransform`, `BlockingAttentionTransform`.

### PrefillOnly/Revert asymmetry (recurring landmine)

`RevertPrefillOnlyTransform` auto-derives its map by inverting `PrefillOnlyChunkedTransform._module_mapping` (`{v: k for k, v in ...}`), so it covers every arch automatically. But `RevertPrefillKeepAttentionTransform` is **hand-written** — adding a new architecture to `PrefillOnlyChunkedTransform` without also adding its inverse entry here silently leaks the chunked-prefill forward into the decode QPC (disagg-serving compiles the same `nn.Module` three times in place via `module.__class__ = repl_module`, so an unreverted transform persists into the decode compile). Symptom: works in unified mode, wrong/degraded output in 3-QPC disagg mode. Always check this pair when a PR touches chunked prefill.

### Three registration mechanisms

Not every transform uses `_module_mapping`:

1. `_module_mapping: Dict[type, type]` — `ModuleMappingTransform` subclasses (most transforms).
2. `_match_string_replace_method` — `ExternalModuleMapperTransform` subclasses match HF class **name strings**, not class objects.
3. Explicit invocation from Auto-class methods — e.g. `prefill()` in `modeling_auto.py:2885` calls `PrefillOnlyExternalModuleMapperTransform.apply` directly.

A new `QEff*` class can be registered through any of these. When checking that a class is wired up, grep all three.

The cardinal sin: a new `QEff<X>` class added but registered through none of the three. The HF class will be used instead and the override silently no-ops.

## ONNX transforms (`QEfficient/base/onnx_transforms.py`)

Verified production transforms:

- `BaseOnnxTransform` — abstract base.
- `FP16ClipTransform` — clip FP32 to FP16 range.
- `SplitTensorsTransform` — split large external weights (>2GB protobuf bug: embedding models export a `PooledModel.onnx` over the ~2GB protobuf limit and fail to load on-device; this transform marks large initializers EXTERNAL so they spill to `*.onnx.data` sidecars). This line has a drop-then-restore history — it has been removed as collateral during unrelated `_onnx_transforms` refactors more than once. When a PR touches `_onnx_transforms`, confirm `SplitTensorsTransform` isn't silently dropped.
- `CustomOpTransform` — register custom op prototypes.
- `RemovePrefix` — strip path prefixes from graph names.
- `RenameFunctionOutputsTransform` — rename outputs of decoder-related functions.
- `AdapterWeightsToInputsTransform` — adapter weight surgery.
- `OnnxTransformPipeline` — sequencer.

`apply` returns `Tuple[ModelProto, bool]` (model, applied). Subclasses override `apply(model, **kwargs)`. Idempotence and "applied iff something changed" are part of the contract.

A transform that renames graph inputs/outputs (like `RemovePrefix`) breaks the contract that `compile()` consumes the same names `export()` produced. Renames need: (a) opt-in flag, (b) round-trip test that downstream consumers still resolve names, (c) idempotence check.

## QEff class rules

`QEff<X>` classes that subclass HF Attention/DecoderLayer/Model/ForCausalLM should not define `__init__` — the transform does `module.__class__ = QEff<X>` on a live HF instance and a new `__init__` would re-initialize weights.

Legitimate exceptions (don't flag these):

- `QEff*RotaryEmbedding` — pre-populates cos/sin caches via `_set_cos_sin_cache`. Many examples: `modeling_llama.py:45`, `modeling_qwen3_moe.py`, etc.
- Wrapper classes that subclass `nn.Module` directly — `QEffMllamaVisionEncoder` (`modeling_mllama.py:736`), `QEffQwen_2_5_vl_EncoderWrapper`, `QEffQwen_2_5_vl_DecoderWrapper`. Need their own `__init__`.
- Custom cache subclasses — `QEffQwen3_5MoeDynamicCache(Cache)`. Need `__init__`.
- The `__qeff_init__` hook (`base/pytorch_transforms.py:75`) — `ModuleMappingTransform.apply` calls this automatically when present. Use it for post-swap initialization. Active in deepseek_v3, mixtral_moe, molmo, grok_1.

Flag a `__init__` only on a class that subclasses an HF Attention/DecoderLayer/Model/ForCausalLM directly.

## Per-model file structure

Most transformer text/VLM models follow:

```
QEfficient/transformers/models/<model>/
├── __init__.py            # Re-exports of QEff* classes used by pytorch_transforms.py
└── modeling_<model>.py    # Implementation
```

Exceptions to know about:

- `gemma3/` has a `configs/` subdir with yaml configs (and matching `[tool.setuptools.package-data]` entry in `pyproject.toml`).
- PEFT-aware variants live in `QEfficient/peft/` and `QEfficient/peft/lora/`.
- Diffusion models live in `QEfficient/diffusers/pipelines/<x>/pipeline_<x>.py`.
- Speech models — `whisper/`, `wav2vec2/`.

When in doubt, list the actual `models/` directory rather than assume layout.

`modeling_<model>.py` skeleton:

1. License header (BSD-3-Clause, 7 lines).
2. Imports, isort-ordered: stdlib → third-party (transformers, torch) → first-party (`QEfficient.*`).
3. `QEff<HFClassName><Suffix>` classes overriding only `forward` (and minimal helpers; respect `__init__` rules above).
4. Optional: `QEffPrefillChunked<HFClassName><Suffix>` for MoE-aware chunked prefill.

`__init__.py` re-exports the public surface used by `pytorch_transforms.py`. A new model dir whose `__init__.py` is header-only is incomplete — the transform map can't import what's not exported.

## Onboarding-completeness contract

When a PR onboards a new model, "it exports and a token matches" is not the bar. A new QEff model is expected to ship the default support that every model carries, unless the model architecture genuinely cannot support a given capability — in which case the PR must say so and why. Missing support with no explanation is a review finding, not an acceptable gap.

1. **Continuous batching.** The model supports `continuous_batching=True` / `full_batch_size` (CB KV-cache shaping, `batch_index` threaded through the decoder/attention forward into the cache update). Baseline expectation — absence without justification is a Blocker.
2. **Tri-dtype export.** The model exports in **FP32, FP16, and BF16**. This is the direct consequence of the "no hardcoded dtype" rule — every graph-feeding tensor derives its dtype from `config.torch_dtype`, not a literal `torch.float32`. Baseline expectation — a model that only works in one precision (without a hardware/architecture reason stated) is a Blocker.
3. **Subfunction enablement.** The model exposes the repeated-layer hook used for ONNX subfunction extraction (the `get_*` method on the model that returns the set of repeated decoder-layer classes — see `modeling_llama.py`, `modeling_gemma3.py`, `modeling_starcoder2.py`). Subfunction extraction shrinks the exported graph; a new model without it = Issue.
4. **VLM multi-resolution / vision-size.** For vision-language models, the export handles multiple input resolutions / vision sizes (`img_size`, dynamic vision shapes in `get_specializations` / dynamic axes). Exception: models like **Gemma4** that internally normalize any resolution to a constant tensor don't need it — note the exception rather than flagging. Missing multi-resolution on a VLM that needs it = Issue.
5. **MoE disaggregated serving.** For MoE models, the model supports disaggregated (prefill/decode-split) serving: the prefill transform + `retain_full_kv` path, and membership in `SPECIALIZED_DISAGG_SERVING_MODEL_ARCH` **with** the corresponding model-side machinery (not just the string added to the set). Missing for a new MoE = Issue → Blocker if the set was edited without the machinery.
6. **RAM / disk blast radius.** QEff is constrained on host RAM during export. Any modeling or architecture change must not inflate export memory or on-disk artifact size beyond what the model genuinely needs — watch for materialized full attention biases, redundant fp32 weight copies, un-split large external weights (`SplitTensorsTransform`), or extra ONNX initializers. A change that needlessly raises peak export RAM or artifact size = Issue, Blocker if it risks export OOM.
7. **PR description.** The PR body describes what's changing relative to the existing architecture, why, and which of the above defaults are (and aren't) supported. A new-model PR with an empty or content-free body is an Issue (the reviewer must not infer scope from the diff).

A clean onboarding PR either ships all applicable items above, or names the ones it intentionally omits with a one-line reason. The reviewer's job is to check each item and call out silent omissions.

## Tests

CONTRIBUTING.md (line 57, with original typo): *"verify all 4 pipeline stages (PyTorch HF → KV → ORT → AI 100) and make sure tokens are matching with refernce PyTorch HF"* [sic].

- `tests/transformers/models/causal_lm_models/test_causal_lm_models.py` — 4-stage parity for text LMs.
- `tests/transformers/models/image_text_to_text/test_image_text_to_text_models.py` — same for VLMs. **Note:** the KV-vs-HF and ORT-vs-HF assertions in this file are currently commented out, so it validates only HF-vs-AI100 (2 of 4 stages) for VLMs. A passing VLM test is not full 4-stage proof.
- `tests/unit_test/` — fast unit tests (transform replacement, finite-output smoke).
- `tests/transformers/test_pytorch_transforms.py` — generic transform-mapping sanity. `tests/transformers/subfunction/` — subfunction-extraction tests.

**Dummy models, not full-size, are the CI bar.** Full-size models take far too long to run in CI, so the suite is split by marker: `@pytest.mark.full_layers` (real model, run rarely), `@pytest.mark.few_layers`, and `@pytest.mark.dummy_layers` (tiny config — `num_hidden_layers: 1`, small hidden/heads/vocab). A new model must add a **dummy** config entry and run under `dummy_layers`. Enforce dummy coverage; do not let a PR rely on a full-size run for CI. A `num_hidden_layers: 1` config is therefore *expected*, not a test-defeating smell — it only hides a bug when the change under review is itself cross-layer.

**Skip-lists are a last resort.** `ModelConfig.SKIPPED_MODELS` / `FULL_MODEL_TESTS_TO_SKIP` (`QEfficient/utils/test_utils.py`) deselect a model from parity entirely. Adding a model here is rare and legitimate only for a known upstream HF-transformers defect that blocks the model — and must be justified in the PR body. A PR that adds a config entry *and* skip-lists the same model runs zero parity (Blocker — gaming the check).

Config files: `tests/configs/causal_model_configs.json`, `tests/configs/image_text_model_configs.json`. Each entry needs `{model_name, model_type, additional_params}`. `additional_params` typically contains `max_position_embeddings`, `num_hidden_layers`, `num_attention_heads`, `hidden_size`, `intermediate_size`, `vocab_size`, `num_key_value_heads`. A skeleton entry without these will pass file-load but fail the test.

Beyond parity, a new model's QEff-specific surface (transform, cache, subfunction, CB) should have **unit-test** coverage in `tests/unit_test/` or the relevant `tests/transformers/<area>/`. And the added tests must cover the cases the change introduces (CB, each export dtype, subfunction, disagg/MoE, multi-resolution/VLM) **without** inflating CI time — redundant params, a full-size model in the dummy path, or a combinatorial parametrize explosion is its own finding.

Watch for parameters chosen so the buggy path never executes — e.g. a grouped-router test parametrized with `n_group=1` degenerates the group mask to a global top-k no-op and hides a router divergence from HF. Single-token sequences hide sequence-state bugs. The bar is a config that exercises the regime the change actually touches, plus numerical parity against unmodified HF (not QEff-vs-QEff).

## Examples

`examples/<task>/[models/<model>/]<name>.py`. Onboarding template at `examples/onboarding_guide/causallm/`.

Required structure:

1. License header.
2. Imports.
3. `argparse` for `--model-name`, `--prompt`, `--prefill-seq-len`, `--ctx-len`, `--generation-len`, `--num-cores`, `--device-group` (subset as applicable).
4. `model = QEFFAutoModelForCausalLM.from_pretrained(args.model_name, ...)` (or right Auto class).
5. `model.export(...)`, `model.compile(...)`, `model.generate(...)`.
6. `if __name__ == "__main__": main()`.

Examples must run. Examples must use the public surface in `QEfficient/__init__.py`, not internals (`from QEfficient._private...` is wrong).

## Docs

- `docs/source/validate.md` — model support matrix; new model gets a row in the right table with the HF model card link. A row that's just `TBD` / `—` / `?` is not compliance.
- `docs/source/features_enablement.md` — feature flags / runtime knobs.
- `docs/source/index.rst` — Sphinx TOC. Usually no edit for a new model file.

## Coding-style signals (`pyproject.toml`)

```toml
[tool.ruff]
line-length = 120
target-version = "py310"
lint.extend-select = ["I"]
```

- Line length 120.
- isort active.
- Pre-commit + DCO sign-off enforced.
- Type hints expected on public functions.
- Docstrings: terse or absent. A bad docstring is worse than no docstring.
- ruff `target-version = py310` while `requires-python = ">=3.8,<3.13"` — py3.10-only syntax (match statements, runtime `|` unions in type expressions) is a real defect even though ruff won't catch it.
- Pinned `transformers==5.5.4`. New imports must verify the symbol exists in this version.

## What a "good" model-onboarding PR looks like

- ~10–25 files, ~2-3k net lines.
- 1 `modeling_<x>.py` (~1000-1500 lines is normal for full model + attention + MoE + RoPE).
- 1 `__init__.py` with the exports.
- ~20-60 lines added to `pytorch_transforms.py` (imports + mapping entries).
- ~10-50 lines added to `modeling_auto.py` if a custom Auto class hook is needed.
- 1-3 example files under `examples/<task>/models/<model>/`.
- 1 row in `tests/configs/<task>_model_configs.json`.
- 1 row (or section) in `docs/source/validate.md`.
- Unit tests if the model has unusual transform requirements.
- Multiple commits, DCO sign-off, atomic changes.

A PR claiming to "add model X" with 35 files and 10k lines is suspect — either bundling unrelated work or auto-generated.
