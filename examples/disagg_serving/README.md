# We should be using disaggragate serving for GPTOSS model for best performance
 - GPT-OSS model has 128/4 for 120b and 32/4 ratio of total_experts/experts_per_tok
 - We use read all experts only once always strategy in prefill-only model
 - And we treat weights activtions meaning read only chosen experts for decode-only model

# Prefill-only model
## Blocking default behviour when `prefill_only=True` in compile API
 - NUM_Q_BLOCKS=<int> set number of Q blocks in attention 
 - NUM_FFN_BLOCKS=<int> set number of blocks in FFN
 - ENABLE_OPT_SWA="0" or "1" to enable/disable optimized SWA. when enabled we will be using only valid KVs for given block in Attention reducing MACs
 - prefix_caching is not supported with this mode

## Chunking pass `enable_chunking=True` and `prefill_only=True` in compile API
 - Optimized SWA i.e. reading only valid KV as per diagonal attention mask is enabled for this version by default
 - This model can be used for prefix_caching by passing `kv_cache_batch_size=<int>` in compile API

## Multi-Device Pipeline (MDP) for prefill-only model
MDP allows the prefill model to be partitioned across multiple devices in a pipeline-parallel fashion.
Pass the options below to `qeff_model.compile()` together with `prefill_only=True`.

| Option | Type | Description |
|---|---|---|
| `mdp_num_partitions` | `int` | Number of pipeline-parallel partitions. Values > 1 cause the compiler to generate a partition config. `mdp_ts_num_devices` must be evenly divisible by `mdp_num_partitions`; if it is not, floor-division is used to assign devices per partition and the remainder devices are unused/unassigned. |
| `mdp_strategy` | `str` | Partitioning strategy: `"onnx"` derives cuts directly from the ONNX graph; `"intersection"` intersects ONNX graph nodes with a prior compiler dump to refine the cuts. |
| `mdp_compiler_dump_path` | `str` | Path to the JSON produced by a prior compiler dump run. Required only when `mdp_strategy="intersection"`. |

See [`qwen3_vl_mdp_compile.py`](qwen3_vl_mdp_compile.py) for a standalone compile-only script that validates MDP compilation for `Qwen/Qwen3-VL-30B-A3B-Instruct` using these options.

### Example — ONNX strategy (recommended starting point)
```python
qeff_model.compile(
    prefill_only=True,
    mdp_ts_num_devices=4,
    mdp_num_partitions=2,
    mdp_strategy="onnx",
)
```

### Example — intersection strategy (uses a prior compiler dump to refine partitions)
```python
qeff_model.compile(
    prefill_only=True,
    mdp_ts_num_devices=4,
    mdp_num_partitions=2,
    mdp_strategy="intersection",
    mdp_compiler_dump_path="/path/to/compiler_dump.json",
)
```

# Decode-only model
## Retain Sliding window length of KV for sliding window layers, default behavour when `prefill_seq_len=1` in compile API
 - This reduces the amount of DDR used by the model
 - CB is enabled for this version pass `continous_batching=True` in `from_pretrained` call and strictly pass `full_batch_size=<int>` and optinally `kv_cache_batch_size=<int>` if needed
## Full KV for sliding window layers pass `retain_full_kv=True` along with `prefill_seq_len=1` in compile API
 - This uses higher DDR as we are retaining ctx_len KV even for sliding window layers but will be reading only sliding window len kv in attention
 - CB is enabled for this version pass `continous_batching=True` in `from_pretrained` call and strictly pass `full_batch_size=<int>` and optinally `kv_cache_batch_size=<int>` if needed
 - This is enabled for the usecase of multi-turn chat, where we will be running prefill-> decode and then use cache of prefill as well as decode combined to again run prefill, so we want to retain full KV for sliding window layers


NOTE:
* decode-only model currently fails compilation with `use_onnx_subfunctions=True` so avoid using it
* 120B model needs NPI, there are two versions of NPI one with and without subfunction both are uploaded here, pass it as `node_precision_info=<path to file>`
* It is advised to use `use_onnx_subfunctions=True` with prefill-only model, otherwise the compilation times are too high, with this the model is supposed to export and fail during compile as it needs assert sdk, so user is supposed to run this compilation manually by pasting the command printed in the error

