# Qwen3-VL Embedding Inference

This directory contains an AI100 example for running Qwen3-VL embedding models with QEfficient and printing query-document similarity scores.

Supported models:
- `Qwen/Qwen3-VL-Embedding-8B`

## What this example does

- Loads Qwen3-VL embedding model from Hugging Face (or local snapshot path).
- Uses QEff dual-QPC execution (vision encoder + language model).
- Runs the same queries against multiple text/image documents.
- Prints the query-document similarity matrix.

## Required package

- `qwen-vl-utils>=0.0.14`

```bash
pip install "qwen-vl-utils>=0.0.14"
```

## Scripts

- `qwen3_vl_embedding.py` - runnable example that explicitly shows:
  - `QEFFAutoModelForImageTextToText.from_pretrained(...)`
  - `model.compile(...)` arguments for QPC generation
  - AI100 embedding call flow
- `embedding_model.py` - Qwen3-VL-specific helper logic (prompting/tokenization/runtime glue).

## Run

```bash
python examples/embeddings/qwen3vl/qwen3_vl_embedding.py \
  --model-name Qwen/Qwen3-VL-Embedding-8B
```

With compile parameters:

```bash
python examples/embeddings/qwen3vl/qwen3_vl_embedding.py \
  --model-name Qwen/Qwen3-VL-Embedding-8B \
  --ctx-len 2048 \
  --num-cores 16 \
  --num-devices 1 \
  --compile-prefill-seq-len 4096 \
  --mxfp6-matmul
```
