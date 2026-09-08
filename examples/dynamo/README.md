# Dynamo Examples

Examples in this directory use the `torch.export` / dynamo ONNX export path.
Use this section when validating dynamo-specific export behavior or features
that depend on the dynamo path, such as ONNX subfunctions and weight-free
export.

## Available Examples

| Example | Description |
|---|---|
| [causal_lm](causal_lm/) | Export, compile, and run text-only CausalLM models with dynamo. |
| [image_text_to_text](image_text_to_text/) | Export, compile, and run dual-QPC VLM models with dynamo. |

## Prerequisites

Install QEfficient from the repository root:

```bash
pip install -e .
```

Install the dependencies for the example you want to run. For CausalLM:

```bash
pip install -r examples/dynamo/causal_lm/requirements.txt
```

For image-text-to-text VLMs:

```bash
pip install -r examples/dynamo/image_text_to_text/requirements.txt
```

For private or gated Hugging Face models, export your token before running the
examples:

```bash
export HF_TOKEN=<your_token>
```

## CausalLM Quick Start

```bash
python examples/dynamo/causal_lm/basic_dynamo_inference.py \
    --model-name Qwen/Qwen2-1.5B-Instruct \
    --prompt "My name is" \
    --prefill-seq-len 32 \
    --ctx-len 128 \
    --num-cores 16
```

Enable weight-free export with the same script:

```bash
python examples/dynamo/causal_lm/basic_dynamo_inference.py \
    --model-name Qwen/Qwen2-1.5B-Instruct \
    --prompt "My name is" \
    --prefill-seq-len 128 \
    --ctx-len 128 \
    --num-cores 16 \
    --weight-free
```

## Image-Text-To-Text Quick Start

```bash
python examples/dynamo/image_text_to_text/qwen3_vl_moe_dynamo_inference.py
```

The VLM example defaults to dual-QPC Qwen3-VL-MoE with weight-free export,
ONNX subfunctions, and a reduced-layer bring-up config. See
`examples/dynamo/image_text_to_text/README.md` for the full argument list.
