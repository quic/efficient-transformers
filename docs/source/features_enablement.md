# Fetaures Enablement Guide
Below guide highlights the steps to enable supported features in QEfficient.

(id-continuous-batching)=
## Continuous Batching

Users can compile a model utilizing the continuous batching feature by specifying full_batch_size <full_batch_size_value> in the infer and compiler APIs. If full_batch_size is not provided, the model will be compiled in the regular way.

When enabling continuous batching, batch size should not be specified.

Users can leverage multi-Qranium and other supported features along with continuous batching.

```bash
python -m QEfficient.cloud.infer --model_name TinyLlama/TinyLlama_v1.1 --prompt_len 32 --ctx_len 128 --num_cores 16 --device_group [0] --prompt "My name is|The flat earth theory is the belief that|The sun rises from" --mxfp6 --mos 1 --aic_enable_depth_first --full_batch_size 3
```

---

(id-multi-qranium-inference)=
## Multi-Qranium Inference

You can also enable MQ, just based on the number of devices. Based on the `--device-group` as input it will create TS config on the fly. If `--device-group [0,1]` it will create TS config for 2 devices and use it for compilation, if `--device-group [0]` then TS compilation is skipped and single soc execution is enabled.

```bash
python -m QEfficient.cloud.infer --model_name Salesforce/codegen-2B-mono --batch_size 1 --prompt_len 32 --ctx_len 128 --mxfp6 --num_cores 16 --device-group [0,1] --prompt "def fibonacci(n):" --mos 2 --aic_enable_depth_first
```

Above step will save the `qpc` files under `efficient-transformers/qeff_models/{model_card_name}`, you can use the execute API to run for different prompts. This will automatically pick the pre-compiled `qpc` files.

```bash
python -m QEfficient.cloud.execute --model_name Salesforce/codegen-2B-mono --qpc-path qeff_models/Salesforce/codegen-2B-mono/qpc_16cores_1BS_32PL_128CL_2devices_mxfp6/qpcs --prompt "def binary_search(array: np.array, k: int):" --device-group [0,1]
```

To disable MQ, just pass single soc like below, below step will compile the model again and reuse the `ONNX` file as only compilation argument are different from above commands.

```bash
python -m QEfficient.cloud.infer --model_name gpt2 --batch_size 1 --prompt_len 32 --ctx_len 128 --mxfp6 --num_cores 16 --device-group [0] --prompt "My name is" --mos 1 --aic_enable_depth_first
```

---

(id-qnn-compilation-via-python-api)=
## QNN Compilation via Python API

Users can also use python API to export, compile and execute onnx models using QNN SDK.

```Python
# We can now export the modified models to ONNX framework
# This will generate single ONNX Model for both Prefill and Decode Variations which are optimized for
# Cloud AI 100 Platform.
from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM

# Model-Card name (This is HF Model Card name) : https://huggingface.co/gpt2-xl
model_name = "gpt2"  # Similar, we can change model name and generate corresponding models, if we have added the support in the lib.

qeff_model = AutoModelForCausalLM.from_pretrained(model_name)

qnn_config_file_path = "QEfficient/compile/qnn_config.json"

generated_qpc_path = qeff_model.compile(
    num_cores=14,
    mxfp6=True,
    enable_qnn=True,
    qnn_config = qnn_config_file_path # QNN compilation configuration is passed.
)
```
---

(id-draft-based-speculative-decoding)=
## Draft-Based Speculative Decoding
Draft-based speculative decoding is a technique where a small Draft Language Model (DLM) makes `num_speculative_tokens` autoregressive speculations ahead of the Target Language Model (TLM). The objective is to predict what the TLM would have predicted if it would have been used instead of the DLM. This approach is beneficial when the autoregressive decode phase of the TLM is memory bound and thus, we can leverage the extra computing resources of our hardware by batching the speculations of the DLM as an input to TLM to validate the speculations.

To export and compile both DLM/TLM, add corresponding `qaic_config` and `num_speculative_tokens` for TLM and export DLM as you would any other QEfficient LLM model:

```Python
from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM

tlm_name = "meta-llama/Llama-2-70b-chat-hf"
dlm_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
k = 3 # DLM will make `k` speculations
qaic_config = dict(speculative_model_type="target")
tlm = AutoModelForCausalLM.from_pretrained(tlm_name, qaic_config=qaic_config)
dlm = AutoModelForCausalLM.from_pretrained(dlm_name)
tlm.compile(num_speculative_tokens=k)
dlm.compile()
```

The `qaic_config` dictionary is fed during the instantiation of the model because slight changes to the ONNX graph are required. Once complete, the user can specify `num_speculative_tokens` to define the actual number of speculations that the TLM will take as input during the decode phase. As for the DLM, no new changes are required at the ONNX or compile level.

---

(id-mdp-multi-device-partitioning)=
## Multi-Device Partitioning (MDP)

MDP splits a model graph across multiple devices at compile time. A partition
config is a JSON file listing the graph nodes assigned to each device.
**Compilation time scales with the number of nodes in that config** — keeping
the node list compact is critical for acceptable compile times.

### MDP compile flows

| Flow | Recommended? | Notes |
|------|:---:|-------|
| Final compile — QEff MDP with `mdp_strategy="onnx"` | ✅ Yes | Standard path; uses ONNX subfunction names as node keys. |
| Compiler dump-only pass (`mdp_dump_partition_config`) | ✅ Yes (Step 1 only) | Produces a JSON used as input to intersection; no QPC emitted. Do **not** pass `mdp_num_partitions` here. |
| Final intersection compile (`mdp_strategy="intersection"`) | ✅ Yes (no-subfunction models) | Trims QEff node list to compiler-known nodes before final compile. Preferred for no-subfunction flows. |
| No-subfunction compile without intersection | ⚠️ Not recommended | Compiler MDP dump can have one entry per low-level op, making the node list enormous and compilation **extremely slow**. |
| Passing `mdp_num_partitions` during the dump-only pass | ❌ Do not use | `mdp_num_partitions` triggers disaggregated MDP generation; omit it during the dump pass to get a plain compiler dump. |

### MDP compile options reference

| Option | Purpose | When to set | Warnings / Notes |
|--------|---------|-------------|-----------------|
| `num_devices` | Number of devices to compile for | Always set to the number of target devices. | Controls overall number of devices |
| `mdp_num_partitions` | Number of pipeline-parallel partitions (disaggregated MDP). | Set only during the final compile (Step 2), not the dump pass. | When `> 1`, triggers full node-list generation from the ONNX graph. |
| `mdp_strategy` | Selects MDP node-list generation strategy: `"onnx"` or `"intersection"`. | Set to `"intersection"` when using a compiler dump for a no-subfunction model. | `"intersection"` requires `mdp_compiler_dump_path` to also be set. |
| `mdp_compiler_dump_path` | Path to the compiler MDP dump produced in Step 1. | Set during the intersection compile (Step 2). | Required when `mdp_strategy="intersection"`. The compiler dump is **not** the desired final partitioning; it is used only to identify valid node names. |
| `mdp_dump_partition_config` | Compiler option telling `qaic-compile` to write its own MDP layout to this path instead of producing a QPC. | Set only during the dump-only pass (Step 1). | Passed as a `**compiler_options` kwarg. Omit `mdp_num_partitions` when using this. |
| `use_onnx_subfunctions` | Export the ONNX model with subgraph subfunctions, reducing node count. | Set `True` when ONNX subfunctions are acceptable for the model. | Without subfunctions the compiler dump grows very large; the intersection flow is then strongly recommended. |

### Two-pass flow for no-subfunction models

When a model is exported without ONNX subfunctions, the compiler MDP dump
contains one entry per low-level operation rather than per subgraph. Passing
that full node list directly to the compiler makes compilation **extremely slow**.

Use this two-pass flow instead:

1. **Dump pass** — generate the compiler MDP dump (no QPC produced).
2. **Intersection compile** — QEff intersects its own MDP config with the
   compiler dump, trimming it to only nodes the compiler recognises, then
   performs the final compile.

#### Step 1 — Generate the compiler MDP dump

Pass `mdp_dump_partition_config` and **omit** `mdp_num_partitions`. The
compiler writes its partition layout to the file; no QPC is produced. The
resulting file is used only as input to intersection — it does not represent
the desired final partitioning.

#### Step 2 — Intersection compile

Set `mdp_num_partitions`, point `mdp_compiler_dump_path` at the dump from
Step 1, and set `mdp_strategy="intersection"`. QEff generates its own MDP
config, intersects it with the compiler dump, and performs the final compile.

#### Complete working example

A self-contained script demonstrating both the `mdp_strategy="onnx"` direct
compile and the two-pass `mdp_strategy="intersection"` flow (including the
dump-pass required to generate `mdp_compiler_dump_path`) is available here:

[`examples/disagg_serving/qwen3_vl_mdp_compile.py`](../../examples/disagg_serving/qwen3_vl_mdp_compile.py)
