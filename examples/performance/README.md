# Performance Optimization Examples

Examples demonstrating performance optimization techniques for Qualcomm Cloud AI 100.

## Authentication

For private/gated models, export your HuggingFace token:
```bash
export HF_TOKEN=<your_huggingface_token>
```

## Available Examples

### Speculative Decoding

Accelerate text generation using speculative decoding techniques.

#### draft_based.py
Draft-based speculative decoding with separate draft and target models.

**Basic Usage:**
```bash
python speculative_decoding/draft_based.py \
    --target-model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --draft-model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --num-speculative-tokens 4
```

**Advanced Usage:**
```bash
python speculative_decoding/draft_based.py \
    --target-model-name meta-llama/Llama-3.1-8B \
    --draft-model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --num-speculative-tokens 4 \
    --prefill-seq-len 32 \
    --ctx-len 128 \
    --target-device-group 0,1 \
    --draft-device-group 2
```
errors in this example


#### prompt_lookup.py
Prompt Lookup Decoding (PLD) - N-gram based speculation without a draft model.

**Basic Usage:**
```bash
python speculative_decoding/prompt_lookup.py \
    --target-model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --num-speculative-tokens 3 \
    --max-ngram-size 3
```

#### multi_projection.py
Multi-projection speculative decoding (Turbo models).

**Basic Usage:**
```bash
python speculative_decoding/multi_projection.py \
    --pretrained-model-name-or-path TinyLlama/TinyLlama-1.1B-Chat-v1.0
```
error 

### On-Device Sampling

Control sampling parameters directly on the AI 100 hardware.

#### on_device_sampling.py
Configure sampling parameters (temperature, top-k, top-p, etc.) on-device.

**Basic Usage:**
```bash
python on_device_sampling.py \
    --model-name meta-llama/Llama-3.1-8B \
    --num-cores 16 \
    --prompt-len 128 \
    --ctx-len 256
```

**Advanced Usage with Sampling Parameters:**
```bash
python on_device_sampling.py \
    --model-name meta-llama/Llama-3.1-8B \
    --prompt-len 128 \
    --ctx-len 256 \
    --full-batch-size 2 \
    --device-group 0,1,2,3 \
    --num-cores 16 \
    --mxint8-kv-cache \
    --mxfp6-matmul \
    --override-qaic-config "aic_include_sampler:true aic_return_pdfs:false max_top_k_ids:512" \
    --repetition-penalty 1.9 \
    --temperature 0.67 \
    --top-k 54720 \
    --top-p 0.89
```

## Performance Tips

1. **Speculative Decoding**: Best for long-form generation where draft model is much faster than target
2. **Prompt Lookup**: No draft model needed, works well for repetitive patterns
3. **Multi-Projection**: Optimal for models with built-in speculation support
4. **On-Device Sampling**: Reduces host-device communication overhead
5. **C++ Execution**: Maximum performance for production deployments

## Documentation

- [QEff Auto Classes](https://quic.github.io/efficient-transformers/source/qeff_autoclasses.html)
- [Performance Features](https://quic.github.io/efficient-transformers/source/features_enablement.html)
- [Quick Start Guide](https://quic.github.io/efficient-transformers/source/quick_start.html)
