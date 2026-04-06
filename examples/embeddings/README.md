# Embedding Examples

Examples for running text embedding models on Qualcomm Cloud AI 100.

## Authentication

For private/gated models, export your HuggingFace token:
```bash
export HF_TOKEN=<your_huggingface_token>
```

## Supported Models

**QEff Auto Class:** `QEFFAutoModel`

For the complete list of supported embedding models, see the [Validated Models - Embedding Section](../../docs/source/validate.md#embedding-models).

Popular model families include:
- BERT-based (BGE, E5)
- MPNet
- Mistral-based
- NomicBERT
- Qwen2
- RoBERTa (Granite)
- XLM-RoBERTa (multilingual)

## Available Examples

### text_embeddings.py
Generate text embeddings using transformer models.

**Usage:**
```bash
# With default parameters
python text_embeddings.py

# With custom parameters
python text_embeddings.py \
    --model-name sentence-transformers/all-MiniLM-L6-v2 \
    --sentences "This is an example sentence" \
    --pooling max \
    --num-cores 16 \
    --seq-len "32,64"
```

**Parameters:**
- `--model-name`: HuggingFace embedding model ID (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `--sentences`: Input text to generate embeddings for (default: `"This is an example sentence"`)
- `--pooling`: Pooling strategy - `max`, `mean`, or `none` (default: `max`)
- `--num-cores`: Number of cores (default: `16`)
- `--seq-len`: Sequence length(s) - single int or comma-separated list (default: `"32,64"`)

This example:
- Uses `sentence-transformers/all-MiniLM-L6-v2` by default
- Demonstrates custom pooling strategies (max pooling)
- Compiles for multiple sequence lengths [32, 64]
- Outputs text embeddings
- Works with various embedding model families (BERT, MPNet, Mistral-based, etc.)

## Pooling Strategies

The example supports different pooling strategies:
- **max**: Max pooling over token embeddings
- **mean**: Mean pooling over token embeddings  
- **custom**: Pass your own pooling function

## Documentation

- [QEff Auto Classes](https://quic.github.io/efficient-transformers/source/qeff_autoclasses.html)
- [Validated Embedding Models](https://quic.github.io/efficient-transformers/source/validate.html#embedding-models)
- [Quick Start Guide](https://quic.github.io/efficient-transformers/source/quick_start.html)
