# Fetaures Enablement Guide
Below guide highlights the steps to enable supported features in QEfficient.

(id-continuous-batching)=
## Continuous Batching

Users can compile a model utilizing the continuous batching feature by specifying full_batch_size <full_batch_size_value> in the infer and compiler APIs. If full_batch_size is not provided, the model will be compiled in the regular way.

When enabling continuous batching, batch size should not be specified.

Users can leverage multi-Qranium and other supported features along with continuous batching.

```bash
python -m QEfficient.cloud.infer --model_name TinyLlama/TinyLlama_v1.1 --prompt_len 32 --ctx_len 128 --num_cores 16 --device_group [0] --prompt "My name is|The flat earth
theory is the belief that|The sun rises from" --mxfp6 --mos 1 --aic_enable_depth_first --full_batch_size 3
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

generated_qpc_path = qeff_model.compile(
    num_cores=14,
    mxfp6=True,
    enable_qnn=True,
    qnn_config = qnn_config_file_path # QNN compilation configuration is passed.
)

qeff_model.generate(prompts=["My name is"])
```
---

(id-draft-based-speculative-decoding)=
## Draft-Based Speculative Decoding
Draft-based speculative decoding is a technique where a small Draft Language Model (DLM) makes `num_speculative_tokens` autoregressive speculations ahead of the Target Language Model (TLM). The objective is to predict what the TLM would have predicted if it would have been used instead of the DLM. This approach is beneficial when the autoregressive decode phase of the TLM is memory bound and thus, we can leverage the extra computing resources of our hardware by batching the speculations of the DLM as an input to TLM to validate the speculations.

To export and compile both DLM/TLM, add corresponding `qaic_config` and `num_speculative_tokens` for TLM and export DLM as you would any other QEfficient LLM model:

```Python
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