# Model Family Map

Use this file to classify an incoming Hugging Face model before writing code.

## Read This First
- Inspect the upstream config or model card for:
  - `model_type`
  - `architectures`
  - `num_attention_heads`
  - `num_key_value_heads`
  - RoPE or rotary settings
  - `sliding_window`
  - chunked attention or disaggregated serving hints
  - MoE or routed expert parameters
  - hybrid or alternating cache behavior
  - multimodal sub-configs such as `text_config`, `vision_config`, or `audio_config`
  - quantization notes such as AWQ or GPTQ

## Family Selection

### Plain causal LM
- Start from:
  - `QEfficient/transformers/models/llama/modeling_llama.py`
  - nearby RoPE/GQA families already in the repo, such as Mistral, Qwen2, Granite, Olmo2, Starcoder2
- Use when:
  - decoder-only causal LM
  - no routed experts
  - no hybrid cache
  - no multimodal tower
  - no MLA/compressed-cache path

### MoE without MLA
- Start from:
  - `QEfficient/transformers/models/gpt_oss/modeling_gpt_oss.py`
  - `QEfficient/transformers/models/mixtral_moe/modeling_mixtral.py`
  - `QEfficient/transformers/models/qwen3_moe/modeling_qwen3_moe.py`
- Use when:
  - routed experts materially change decode or prefill execution
  - top-k routing or expert packing must be preserved in export/runtime
- Notes:
  - `gpt_oss` is the strongest local reference for decode and prefill separation, chunking, and sliding-window-aware specialized serving.
  - `mixtral` and `qwen3_moe` are lighter references for sparse MoE without MLA-specific cache changes.

### Hybrid cache or dynamic sequence families
- Start from:
  - `QEfficient/transformers/models/gemma3/modeling_gemma3.py`
  - `QEfficient/transformers/models/llama4/modeling_llama4.py`
- Use when:
  - attention/cache policy changes across layers
  - text-side and full-model export paths diverge
  - dynamic sequence support is model-defined rather than uniform
- Repo hint:
  - `QEfficient/transformers/modeling_utils.py` already marks dynamic-sequence model families.

### VLM and multimodal
- Start from:
  - `QEfficient/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
  - `QEfficient/transformers/models/internvl/modeling_internvl.py`
  - `QEfficient/transformers/models/gemma3/modeling_gemma3.py`
  - `QEfficient/transformers/models/mllama/modeling_mllama.py`
- Use when:
  - text and vision towers are fused
  - image tokens are inserted into text-side generation
  - export may need text-only fallback for runtime parity

### MLA, absorption, compressed cache
- Start from:
  - `QEfficient/transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py`
  - `QEfficient/transformers/cache_utils.py`
  - `QEfficient/utils/generate_inputs.py`
  - `QEfficient/utils/run_utils.py`
- Use when:
  - compressed KV and rope-side cache are first-class
  - attention uses absorption or fused query/key-space tricks
  - export/runtime require nonstandard retained-state shapes
- Notes:
  - Reuse DeepSeek-style MLA ideas only after matching them against the current local GLM/cache abstractions.
  - Keep decode and prefill behavior explicit. Do not assume a prefill-only trick generalizes to decode.

### Whisper, CTC, embeddings, and non-causal families
- Start from:
  - `QEfficient/transformers/models/whisper/modeling_whisper.py`
  - the auto-model wrappers in `QEfficient/transformers/models/modeling_auto.py`
- Use when:
  - encoder-decoder speech model
  - CTC model
  - embedding or classification model
- Notes:
  - These often need export coverage more than causal runtime parity.

## Wiring Checklist
- Add the QEff wrapper under the right `QEfficient/transformers/models/<family>/` directory.
- Update `QEfficient/transformers/models/pytorch_transforms.py` so the upstream modules are replaced by QEff modules.
- Update `QEfficient/transformers/modeling_utils.py`:
  - supported architectures
  - module mapping
  - any family-specific dynamic-sequence or specialized-serving sets
- Update `QEfficient/transformers/models/modeling_auto.py` when:
  - export example inputs differ
  - retained-state names or shapes differ
  - compile/generate glue needs family-specific handling
- Update cache helpers only if the model introduces new retained-state semantics.

## Tiny Model Policy
- Prefer an existing tiny Hub model for the exact architecture.
- If the exact architecture lacks a tiny checkpoint, choose the smallest compatible random checkpoint that still exercises the same path.
- If no usable checkpoint exists, construct a tiny config in tests or from the upstream config with reduced hidden size, layer count, head count, and vocab size while preserving the feature under test.
- Do not validate a specialized path with a tiny model that bypasses that path.
