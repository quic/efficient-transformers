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

### Compute-Context-Length

Calculating Context-Length dynamically during inference for getting the best related performance within each window of context-length

#### compute_context_length/basic_inference.py
Configure CCL parameters: 1) ccl-enabled: to activate CCL feature, 2) comp-ctx-lengths-prefill: list of context length to be used during prefilling, and 3) comp-ctx-lengths-decode: list of context lengths to be used during decoding.

**Usage for Text-only models:**
```bash
python compute_context_length/basic_inference.py \
    --model-name meta-llama/Llama-3.1-8B \
    --num-cores 16 \
    --prefill-seq-len 32 \
    --ctx-len 1024 \
    --ccl-enabled \
    --comp-ctx-lengths-prefill 500,1000 \
    --comp-ctx-lengths-decode 512,1024
```

**Usage for VLM models such as mllama and llava:**
```bash
python compute_context_length/vlm_inference.py \
    --model-name meta-llama/Llama-3.2-11B-Vision-Instruct \
    --hf-token "" \
    --num-cores 16 \
    --prefill-seq-len 32 \
    --ctx-len 8192 \
    --img-size 560 \
    --ccl-enabled \
    --comp-ctx-lengths-prefill 4096 \
    --comp-ctx-lengths-decode 6144,8192
```

**Usage with other MoE and Multimodal models:**
For various models available in compute_context_length directory such as gemma3, gpt_oss, granite_vision, internvl, llama4_cb, llama4_multi_image, llama4, mistral3, molmo, qwen2_5_vl, qwen2_5_vl_cb, and qwen3moe, use the related inference script and only change the model-name and ccl configuration in the related script. The following is an example of each model:
```bash
python compute_context_length/gemma3.py
python compute_context_length/gpt_oss.py
python compute_context_length/granite_vision.py
python compute_context_length/internvl.py
python compute_context_length/llama4_cb.py
python compute_context_length/llama4_multi_image.py
python compute_context_length/llama4.py
python compute_context_length/mistral3.py
python compute_context_length/molmo.py
python compute_context_length/qwen2_5_vl.py
python compute_context_length/qwen2_5_vl_cb.py
python compute_context_length/qwen3moe.py
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
