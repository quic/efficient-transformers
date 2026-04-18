# Qwen3-VL Reranker Inference

This directory contains an AI100 example for running Qwen3-VL reranker models with QEfficient and printing per-document relevance scores.

Supported models:
- `Qwen/Qwen3-VL-Reranker-2B`
- `Qwen/Qwen3-VL-Reranker-8B`

## What this example does

- Loads Qwen3-VL reranker from Hugging Face (or local snapshot path).
- Uses QEff dual-QPC execution (vision encoder + language model).
- Runs the same query against multiple text/image documents.
- Prints one score per document in input order.

## Required package

- `qwen-vl-utils>=0.0.14`

```bash
pip install "qwen-vl-utils>=0.0.14"
```

## Script

- `qwen3_vl_reranker.py`

## Run

```bash
python examples/image_text_to_text/models/qwen3vl/reranker/qwen3_vl_reranker.py \
  --model-name Qwen/Qwen3-VL-Reranker-2B
```

Or run with 8B:

```bash
python examples/image_text_to_text/models/qwen3vl/reranker/qwen3_vl_reranker.py \
  --model-name Qwen/Qwen3-VL-Reranker-8B
```

With compile parameters:

```bash
python examples/image_text_to_text/models/qwen3vl/reranker/qwen3_vl_reranker.py \
  --model-name Qwen/Qwen3-VL-Reranker-2B \
  --ctx-len 2048 \
  --num-cores 16 \
  --num-devices 1 \
  --compile-prefill-seq-len 4096 \
  --mxfp6-matmul
```
