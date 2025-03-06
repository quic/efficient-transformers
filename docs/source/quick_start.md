# Quick Start

QEfficient Library was designed with one goal:

**To make onboarding of models inference straightforward for any Transformer architecture, while leveraging the complete power of Cloud AI platform**

To achieve this, we have 2 levels of APIs, with different levels of abstraction.
1. Command line interface abstracts away complex details, offering a simpler interface. They're ideal for quick development and prototyping. If you're new to a technology or want to minimize coding effort.

2. Python high level APIs offer more granular control, ideal for when customization is necessary.

## Supported Features

| Feature | Impact |
| --- | --- |
| Context Length Specializations (upcoming) | Increases the maximum context length that models can handle, allowing for better performance on tasks requiring long sequences of text. |
| Swift KV (upcoming) | Reduces computational overhead during inference by optimizing key-value pair processing, leading to improved throughput. |
| Block Attention (in progress) | Reduces inference latency and computational cost by dividing context into blocks and reusing key-value states, particularly useful in RAG. |
| [Vision Language Model](QEFFAutoModelForImageTextToText) | Provides support for the AutoModelForImageTextToText class from the transformers library, enabling advanced vision-language tasks. Refer [sample script](https://github.com/quic/efficient-transformers/blob/main/examples/image_text_to_text_inference.py) for more **details**. |
| [Speech Sequence to Sequence Model](QEFFAutoModelForSpeechSeq2Seq) | Provides support for the QEFFAutoModelForSpeechSeq2Seq Facilitates speech-to-text sequence models. Refer [sample script](https://github.com/quic/efficient-transformers/blob/main/examples/speech_to_text/run_whisper_speech_to_text.py) for more **details**. |
| Support for FP8 Execution | Enables execution with FP8 precision, significantly improving performance and reducing memory usage for computational tasks. |
| Prefill caching  | Enhances inference speed by caching key-value pairs for shared prefixes, reducing redundant computations and improving efficiency. |
|Prompt-Lookup Decoding | Speeds up text generation by using overlapping parts of the input prompt and the generated text, making the process faster without losing quality. Refer [sample script](https://github.com/quic/efficient-transformers/blob/main/examples/pld_spd_inference.py) for more **details**.|
| [PEFT LoRA support](QEffAutoPeftModelForCausalLM) | Enables parameter-efficient fine-tuning using low-rank adaptation techniques, reducing the computational and memory requirements for fine-tuning large models. Refer [sample script](https://github.com/quic/efficient-transformers/blob/main/examples/peft_models.py) for more **details**. |
| [QNN support](#qnn-compilation) | Enables compilation using QNN SDK, making Qeff adaptable for various backends in the future. |
| [Embedding model support](QEFFAutoModel) | Facilitates the generation of vector embeddings for retrieval tasks. |
| [Speculative Decoding](#draft-based-speculative-decoding) | Accelerates text generation by using a draft model to generate preliminary predictions, which are then verified by the target model, reducing latency and improving efficiency. Refer [sample script](https://github.com/quic/efficient-transformers/blob/main/examples/draft_spd_inference.py) for more **details**. |
| [Finite lorax](QEffAutoLoraModelForCausalLM) | Users can activate multiple LoRA adapters and compile them with the base model. At runtime, they can specify which prompt should use which adapter, enabling mixed adapter usage within the same batch. Refer [sample script](https://github.com/quic/efficient-transformers/blob/main/examples/lora_models.py) for more **details**. |
| Python and CPP Inferencing API support | Provides flexibility while running inference with Qeff and enabling integration with various applications and improving accessibility for developers. Refer [sample script](https://github.com/quic/efficient-transformers/blob/main/examples/cpp_execution/text_inference_using_cpp.py) for more **details**.|
| [Continuous batching](#continuous-batching) | Optimizes throughput and latency by dynamically batching requests, ensuring efficient use of computational resources. |
| AWQ and GPTQ support | Supports advanced quantization techniques, improving model efficiency and performance on AI 100. |
| Support serving successive requests in same session | An API that yields tokens as they are generated, facilitating seamless integration with various applications and enhancing accessibility for developers. |
| Perplexity calculation | A script for computing the perplexity of a model, allowing for the evaluation of model performance and comparison across different models and datasets. Refer [sample script](https://github.com/quic/efficient-transformers/blob/main/scripts/perplexity_computation/calculate_perplexity.py) for more **details**. |
| KV Heads Replication Script| A sample script for replicating key-value (KV) heads for the Llama-3-8B-Instruct model, running inference with the original model, replicating KV heads, validating changes, and exporting the modified model to ONNX format. Refer [sample script](https://github.com/quic/efficient-transformers/blob/main/scripts/replicate_kv_head/replicate_kv_heads.py) for more **details**.|

## Transformed models and QPC storage

By default, the library exported models and Qaic Program Container (QPC) files, which are compiled and inference-ready model binaries generated by the compiler, are stored in `~/.cache/qeff_cache`. You can customize this storage path using the following environment variables:

1. **QEFF_HOME**: If this variable is set, its path will be used for storing models and QPC files.
2. **XDG_CACHE_HOME**: If `QEFF_HOME` is not set but `XDG_CACHE_HOME` is provided, this path will be used instead. Note that setting `XDG_CACHE_HOME` will reroute the entire `~/.cache` directory to the specified folder, including HF models.
3. **Default**: If neither `QEFF_HOME` nor `XDG_CACHE_HOME` are set, the default path `~/.cache/qeff_cache` will be used.


## Command Line Interface

```{NOTE}
Use ``bash terminal``, else if using ``ZSH terminal`` then ``device_group``should be in single quotes e.g.  ``'--device_group [0]'``
```

### QEfficient.cloud.infer

This is the single e2e CLI API, which takes `model_card` name as input along with other compilation arguments. Check [Infer API doc](infer_api) for more details.

* HuggingFace model files Download → Optimize for Cloud AI 100 → Export to `ONNX` → Compile on Cloud AI 100 → [Execute](#qefficientcloudexecute)
* It skips the export/compile stage based if `ONNX` or `qpc` files are found. If you use infer second time with different compilation arguments, it will automatically skip `ONNX` model creation and directly jump to compile stage.


```bash
# Check out the options using the help
python -m QEfficient.cloud.infer --help
python -m QEfficient.cloud.infer --model_name gpt2 --batch_size 1 --prompt_len 32 --ctx_len 128 --mxfp6 --num_cores 16 --device_group [0] --prompt "My name is" --mos 1 --aic_enable_depth_first
```
If executing for batch size>1,
You can pass input prompts in single string but separate with pipe (|) symbol". Example below

```bash
python -m QEfficient.cloud.infer --model_name gpt2 --batch_size 3 --prompt_len 32 --ctx_len 128 --num_cores 16 --device_group [0] --prompt "My name is|The flat earth
theory is the belief that|The sun rises from" --mxfp6 --mos 1 --aic_enable_depth_first
```

You can also pass path of txt file with input prompts when you want to run inference on lot of prompts, Example below, sample txt file(prompts.txt) is present in examples folder.

```bash
python -m QEfficient.cloud.infer --model_name gpt2 --batch_size 3 --prompt_len 32 --ctx_len 128 --num_cores 16 --device_group [0] --prompts_txt_file_path examples/prompts.txt --mxfp6 --mos 1 --aic_enable_depth_first
```

### QEfficient.cloud.execute
You can first run `infer` API and then use `execute` to run the pre-compiled model on Cloud AI 100 cards.
Once we have compiled the QPC, we can now use the precompiled QPC in execute API to run for different prompts. Make sure to pass same `--device_group` as used during infer. Refer [Execute API doc](execute_api) for more details.

```bash
python -m QEfficient.cloud.execute --model_name gpt2 --qpc_path qeff_models/gpt2/qpc_16cores_1BS_32PL_128CL_1devices_mxfp6/qpcs --prompt "Once upon a time in" --device_group [0]
```

### QEfficient.cloud.finetune
You can run the finetune with set of predefined existing datasets on QAIC using the eager pipeline

```bash
python -m QEfficient.cloud.finetune --device qaic:0 --use-peft --output_dir ./meta-sam --num_epochs 2 --context_length 256 
```
For more details on finetune, checkout the subsection.

### Multi-Qranium Inference
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

### Continuous Batching

Users can compile a model utilizing the continuous batching feature by specifying full_batch_size <full_batch_size_value> in the infer and compiler APIs. If full_batch_size is not provided, the model will be compiled in the regular way.

When enabling continuous batching, batch size should not be specified.

Users can leverage multi-Qranium and other supported features along with continuous batching.

```bash
python -m QEfficient.cloud.infer --model_name TinyLlama/TinyLlama_v1.1 --prompt_len 32 --ctx_len 128 --num_cores 16 --device_group [0] --prompt "My name is|The flat earth
theory is the belief that|The sun rises from" --mxfp6 --mos 1 --aic_enable_depth_first --full_batch_size 3
```

### QNN Compilation

Users can compile a model with QNN SDK by following the steps below:

* Set QNN SDK Path: export $QNN_SDK_ROOT=/path/to/qnn_sdk_folder
* Enabled QNN by passing enable_qnn flag, add --enable_qnn in the cli command.
* An optional config file can be passed to override the default parameters.

**CLI Inference Command**

Without QNN Config
```bash
python -m QEfficient.cloud.infer --model_name gpt2 --batch_size 1 --prompt_len 32 --ctx_len 128 --mxfp6 --num_cores 16 --device_group [0] --prompt "My name is" --mos 1 --aic_enable_depth_first --enable_qnn
```

With QNN Config
```bash
python -m QEfficient.cloud.infer --model_name gpt2 --batch_size 1 --prompt_len 32 --ctx_len 128 --mxfp6 --num_cores 16 --device_group [0] --prompt "My name is" --mos 1 --aic_enable_depth_first --enable_qnn QEfficient/compile/qnn_config.json
````

**CLI Compile Command**

Users can also use `compile` API to compile pre exported onnx models using QNN SDK.

Without QNN Config
```bash
python -m QEfficient.cloud.compile --onnx_path <path to gpt2 onnx file> --qpc-path <path to save qpc files> --batch_size 1 --prompt_len 32 --ctx_len 128 --mxfp6 --num_cores 16 --device_group [0] --prompt "My name is" --mos 1 --aic_enable_depth_first --enable_qnn
```

With QNN Config
```bash
python -m QEfficient.cloud.compile --onnx_path <path to gpt2 onnx file> --qpc-path <path to save qpc files> --batch_size 1 --prompt_len 32 --ctx_len 128 --mxfp6 --num_cores 16 --device_group [0] --prompt "My name is" --mos 1 --aic_enable_depth_first --enable_qnn QEfficient/compile/qnn_config.json
````

**CLI Execute Command**

Once we have compiled the QPC using `infer` or `compile` API, we can now use the precompiled QPC in `execute` API to run for different prompts.

Make sure to pass same `--device_group` as used during infer. Refer [Execute API doc](execute_api) for more details.

```bash
python -m QEfficient.cloud.execute --model_name gpt2 --qpc_path qeff_models/gpt2/qpc_qnn_16cores_1BS_32PL_128CL_1devices_mxfp6/qpcs --prompt "Once upon a time in" --device_group [0]
```

**QNN Compilation via Python API**

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

**Users can also take advantage of features like multi-Qranium inference and continuous batching with QNN SDK Compilation.**

## Python API

### 1.  Model download and Optimize for Cloud AI 100
If your models falls into the model architectures that are [already supported](validated_models), Below steps should work fine.
Please raise an [issue](https://github.com/quic/efficient-transformers/issues), in case of trouble.



```Python
# Initiate the Original Transformer model
# import os

from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM

# Please uncomment and use appropriate Cache Directory for transformers, in case you don't want to use default ~/.cache dir.
# os.environ["TRANSFORMERS_CACHE"] = "/local/mnt/workspace/hf_cache"

# ROOT_DIR = os.path.dirname(os.path.abspath(""))
# CACHE_DIR = os.path.join(ROOT_DIR, "tmp") #, you can use a different location for just one model by passing this param as cache_dir in below API.

# Model-Card name (This is HF Model Card name) : https://huggingface.co/gpt2-xl
model_name = "gpt2"  # Similar, we can change model name and generate corresponding models, if we have added the support in the lib.

qeff_model = AutoModelForCausalLM.from_pretrained(model_name)
print(f"{model_name} optimized for AI 100 \n", qeff_model)
```

### 2. Export and Compile with one API

Use the qualcomm_efficient_converter API to export the KV transformed Model to ONNX and Verify on Torch.

```Python
# We can now export the modified models to ONNX framework
# This will generate single ONNX Model for both Prefill and Decode Variations which are optimized for
# Cloud AI 100 Platform.

# While generating the ONNX model, this will clip the overflow constants to fp16
# Verify the model on ONNXRuntime vs Pytorch

# Then generate inputs and customio yaml file required for compilation.
# Compile the model for provided compilation arguments
# Please use platform SDk to Check num_cores for your card.

generated_qpc_path = qeff_model.compile(
    num_cores=14,
    mxfp6_matmul=True,
)
```

### 3. Execute

Benchmark the model on Cloud AI 100, run the infer API to print tokens and tok/sec

```Python
# post compilation, we can print the latency stats for the kv models, We provide API to print token and Latency stats on AI 100
# We need the compiled prefill and decode qpc to compute the token generated, This is based on Greedy Sampling Approach
tokenizer = AutoTokenizer.from_pretrained(model_name)
qeff_model.generate(prompts=["My name is"],tokenizer=tokenizer)
```
End to End demo examples for various models are available in **notebooks** directory. Please check them out.

### Draft-Based Speculative Decoding
Draft-based speculative decoding is a technique where a small Draft Language Model (DLM) makes `num_speculative_tokens` autoregressive speculations ahead of the Target Language Model (TLM). The objective is to predict what the TLM would have predicted if it would have been used instead of the DLM. This approach is beneficial when the autoregressive decode phase of the TLM is memory bound and thus, we can leverage the extra computing resources of our hardware by batching the speculations of the DLM as an input to TLM to validate the speculations.

To export and compile both DLM/TLM, add corresponding `is_tlm` and `num_speculative_tokens` for TLM and export DLM as you would any other QEfficient LLM model:

```Python
tlm_name = "meta-llama/Llama-2-70b-chat-hf"
dlm_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
k = 3 # DLM will make `k` speculations
tlm = AutoModelForCausalLM.from_pretrained(tlm_name, is_tlm=True)
dlm = AutoModelForCausalLM.from_pretrained(dlm_name)
tlm.compile(num_speculative_tokens=k)
dlm.compile()
```

The `is_tlm` flag is fed during the instantiation of the model because slight changes to the ONNX graph are required. Once complete, the user can specify `num_speculative_tokens` to define the actual number of speculations that the TLM will take as input during the decode phase. As for the DLM, no new changes are required at the ONNX or compile level.
