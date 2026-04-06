# Speculative Decoding Examples

Accelerate text generation using speculative decoding techniques on Qualcomm Cloud AI 100.

Speculative decoding improves inference speed by generating multiple candidate tokens in parallel and validating them with the target model, reducing sequential forward passes required for text generation.

## Authentication

For private/gated models, export your HuggingFace token:
```bash
export HF_TOKEN=<your_huggingface_token>
```

## Quick Start

```bash
# Draft-based: Use small draft model + large target model
python draft_based.py \
    --draft-model-name "meta-llama/Llama-3.2-1B" \
    --target-model-name "meta-llama/Llama-3.2-1B" \
    --num-speculative-tokens 4

# Prompt Lookup: N-gram matching without draft model
python prompt_lookup.py \
    --target-model-name "meta-llama/Llama-3.2-1B" \
    --num-speculative-tokens 3 \
    --max-ngram-size 3

# Multi-Projection: Built-in speculation for Turbo models (requires speculator_config.json)
# Note: TinyLlama does not support multi-projection - use actual Turbo models
python multi_projection.py \
    --pretrained-model-name-or-path "meta-llama/Llama-3.1-8B-Turbo"
```

## Available Scripts

### draft_based.py - Two-Model Speculative Decoding

**How It Works:**
1. **Draft Phase**: Small, fast model generates `N` candidate tokens sequentially
2. **Validation Phase**: Large target model scores all candidates in a single forward pass
3. **Acceptance**: Greedily accept tokens until first mismatch, then sample from target distribution
4. **Iteration**: Repeat with accepted tokens + one additional target token

This approach achieves speedup when draft model is 3-8x faster than target model.

**Basic Usage:**
```bash
python draft_based.py \
    --draft-model-name "meta-llama/Llama-3.2-1B" \
    --target-model-name "meta-llama/Llama-3.2-8B" \
    --num-speculative-tokens 4 \
    --prefill-seq-len 32 \
    --ctx-len 128
```

**Multi-Device Deployment:**
```bash
python draft_based.py \
    --draft-model-name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --target-model-name "meta-llama/Llama-3.1-70B" \
    --target-device-group 0,1,2,3 \
    --draft-device-group 4,5 \
    --num-speculative-tokens 6
```

**Key Features:**
- Uses `qaic_config={"speculative_model_type": "target"}` for target model compilation
- Draft model uses fewer cores (5) vs target model (11) by default
- Supports both regular batching and continuous batching modes
- Implements "bonus token" handling for multi-batch scenarios

**Recommended Model Pairs:**
- `TinyLlama-1.1B` → `Llama-3.1-8B` (8x size ratio)
- `Llama-3.2-1B` → `Llama-3.2-8B` (8x size ratio)
- `Llama-3.1-8B` → `Llama-3.1-70B` (9x size ratio)

### prompt_lookup.py - N-gram Pattern Matching

**How It Works:**
1. **Pattern Search**: Sliding window searches input context for n-gram matches
2. **Candidate Generation**: When match found, extract following tokens as candidates
3. **Fallback**: If no match, pad with dummy tokens (no speculation benefit)
4. **Validation**: Target model scores candidates like draft-based approach

Most effective for repetitive text patterns, code with common structures, or templated content.

**Basic Usage:**
```bash
python prompt_lookup.py \
    --target-model-name "meta-llama/Llama-3.2-8B" \
    --num-speculative-tokens 3 \
    --max-ngram-size 3 \
    --prefill-seq-len 256 \
    --ctx-len 1024
```

**Optimized for Repetitive Content:**
```bash
python prompt_lookup.py \
    --target-model-name "meta-llama/Llama-3.1-8B" \
    --prompts "Write code with repeated patterns: for i in range(10): print(i)" \
    --num-speculative-tokens 5 \
    --max-ngram-size 4 \
    --ctx-len 2048
```

**Key Features:**
- Implements `find_candidate_pred_tokens()` for n-gram matching
- Maintains `all_ids` array to track full context for pattern matching
- Default prompts designed for repetitive patterns (e.g., "hello, good morning to you")
- Uses `fill_tok=-1` for padding when no matches found
- No separate draft model required - uses n-gram pattern matching instead

**Key Parameters:**
- `--max-ngram-size`: Larger values (3-5) better for structured text
- `--num-speculative-tokens`: Reduce if acceptance rate is low
- Longer context lengths improve pattern matching opportunities

### multi_projection.py - Turbo Model Speculation

**How It Works:**
1. **Multi-Head Projection**: Model has multiple projection heads generating token candidates
2. **Single Forward Pass**: All candidates generated simultaneously in one inference
3. **Built-in Validation**: Model internally scores and ranks candidates
4. **Optimized Architecture**: Specifically designed for speculative decoding

Requires models with `speculative_config` and multi-projection architecture.

**Basic Usage:**
```bash
python multi_projection.py \
    --pretrained-model-name-or-path "meta-llama/Llama-3.1-8B-Turbo" \
    --prefill-seq-len 32 \
    --ctx-len 128
```

**Continuous Batching:**
```bash
python multi_projection.py \
    --pretrained-model-name-or-path "meta-llama/Llama-3.1-8B-Turbo" \
    --full-batch-size 4 \
    --device-group 0,1,2,3 \
    --ignore-eos-token
```

**Key Features:**
- Uses `qaic_config={"speculative_model_type": "turbo"}` for compilation
- Automatically extracts `num_speculative_tokens` from model's `speculative_config`
- Generates 4D logits tensor: `[batch, num_logits, num_logits, vocab_size]`
- No separate draft model required - speculation built into architecture


## Common Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--prefill-seq-len` | Prefill chunk size | 32 | 128-256 |
| `--ctx-len` | Max context length | 128 | 512-2048 |
| `--num-speculative-tokens` | Candidates per iteration | 3-4 | 3-6 |
| `--device-group` | Device allocation | `[0]` | Multi-device for large models |
| `--full-batch-size` | Continuous batching | None | 2-8 for throughput |

## Performance Metrics Explained

All scripts output detailed metrics:

```
Avg TLM+DLM TTFT = 0.15          # Time to first token (seconds)
Decode Throughput = 125.67       # Tokens/second during generation
E2E Throughput = 98.23           # Overall tokens/second including prefill
Avg number of accepted tokens = 2.8  # Speculation effectiveness
```



## Documentation

- [Speculative Decoding Guide](https://quic.github.io/efficient-transformers/source/features_enablement.html#speculative-decoding)
- [QEff Auto Classes](https://quic.github.io/efficient-transformers/source/qeff_autoclasses.html)
- [Performance Optimization](https://quic.github.io/efficient-transformers/source/features_enablement.html)
