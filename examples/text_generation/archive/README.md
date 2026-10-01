# Archived example scripts

These scripts were moved from their original locations during the
text-generation consolidation. Their contents are preserved from mainline;
they are historical references and may require updating for future APIs.

Use [`../basic_inference.py`](../basic_inference.py)
for maintained recipes. See the [text-generation guide](../README.md)
for complete commands and `--help-advanced` for all options.

| Original path under `examples/` | Archived script | Canonical controls |
| --- | --- | --- |
| `text_generation/batch_blocking_example.py` | [batch_blocking_example.py](text_generation/batch_blocking_example.py) | `--enable-blocking`, `--num-batch-blocks` |
| `text_generation/blocked_attention_inference.py` | [blocked_attention_inference.py](text_generation/blocked_attention_inference.py) | `--enable-blocking`, `--blocking-mode` |
| `text_generation/cli_examples.sh` | [cli_examples.sh](text_generation/cli_examples.sh) | Canonical recipes and the packaged cloud CLI |
| `text_generation/continuous_batching.py` | [continuous_batching.py](text_generation/continuous_batching.py) | `--continuous-batching`, `--full-batch-size` |
| `text_generation/gguf_models.py` | [gguf_models.py](text_generation/gguf_models.py) | `--gguf-file` |
| `text_generation/gpt_oss_blocked_example.py` | [gpt_oss_blocked_example.py](text_generation/gpt_oss_blocked_example.py) | `--model-name`, blocking and stage controls |
| `text_generation/gpt_oss_blocked_prefill_headpar_example.py` | [gpt_oss_blocked_prefill_headpar_example.py](text_generation/gpt_oss_blocked_prefill_headpar_example.py) | `--disaggregated`, prefill/decode blocking controls |
| `text_generation/moe_inference.py` | [moe_inference.py](text_generation/moe_inference.py) | `--model-name`, `--use-onnx-subfunctions` |
| `disagg_serving/gpt_oss_disagg_mode_cb_chunking_with_kv_share.py` | [gpt_oss_disagg_mode_cb_chunking_with_kv_share.py](disagg_serving/gpt_oss_disagg_mode_cb_chunking_with_kv_share.py) | `--disaggregated`, `--full-batch-size`, MDP controls |
| `disagg_serving/qwen3moe_disagg_mode_cb_chunking_with_kv_share.py` | [qwen3moe_disagg_mode_cb_chunking_with_kv_share.py](disagg_serving/qwen3moe_disagg_mode_cb_chunking_with_kv_share.py) | `--disaggregated`, `--full-batch-size`, MDP controls |

The original `basic_inference.py` is superseded in place. Historical GLM4 and
Kimi-K2 bespoke recipes are preserved under [`advanced/`](advanced/).
