# Quick Start Guide

QEfficient Library was designed with one goal: 

**To make onboarding of models inference straightforward for any Transformer architecture, while leveraging the complete power of Cloud AI platform**

To achieve this, we have 2 levels of APIs, with different levels of abstraction.
1. High-level APIs abstract away complex details, offering a simpler interface. They're ideal for quick development and prototyping. If you're new to a technology or want to minimize coding effort, high-level APIs are more user-friendly.

2. Low-level APIs offer more granular control, ideal for when customization is necessary. These are particularly useful for users who are trying their own models, not hosted on HF but are implemented based on Transformers.

In summary:

* Choose high-level APIs for quick development, simplicity, and ease of use.
* Opt for low-level APIs when you need fine-tuned control, optimization, or advanced customization.

 
# Using High Level API

## 1. Use QEfficient.cloud.infer 

This is the single e2e python api in the library, which takes model_card name as input along with other compile args if necessary and does everything in one go. 

* Torch Download → Optimize for Cloud AI 100 → Export to ONNX → Verify (CPU) → Compile on Cloud AI 100 → [Execute](#2-use-of-qefficientcloudexecute)
* It skips the ONNX export/compile stage if ONNX file or qpc is found on given path.


```bash
# Check out the options using the help menu
python -m QEfficient.cloud.infer --help
python -m QEfficient.cloud.infer --model_name gpt2 --batch_size 1 --prompt_len 32 --ctx_len 128 --mxfp6 --num_cores 16 --device_group [0] --prompt "My name is" --mos 1 --aic_enable_depth_first  

# If executing for batch size>1,

# Either pass input prompts in single string but seperate with pipe (|) symbol". Example below

python -m QEfficient.cloud.infer --model_name gpt2 --batch_size 3 --prompt_len 32 --ctx_len 128 --num_cores 16 --device_group [0] --prompt "My name is|The flat earth 
theory is the belief that|The sun rises from" --mxfp6 --mos 1 --aic_enable_depth_first

# Or pass path of txt file with input prompts, Example below, sample txt file(prompts.txt) is present in examples folder .

python -m QEfficient.cloud.infer --model_name gpt2 --batch_size 3 --prompt_len 32 --ctx_len 128 --num_cores 16 --device_group [0] --prompts_txt_file_path examples/prompts.txt --mxfp6 --mos 1 --aic_enable_depth_first  
 ```
## 2. Use of QEfficient.cloud.execute
Once we have compiled the QPC, we can now use the precompiled QPC in execute API to run for different prompts, like below:

```bash
python -m QEfficient.cloud.execute --model_name gpt2 --qpc_path qeff_models/gpt2/qpc_16cores_1BS_32PL_128CL_1devices_mxfp6/qpcs --prompt "Once upon a time in" --device_group [0]  
```

We can also enable MQ, just based on the number of devices. Based on the "--device_group" as input it will create TS config on the fly. If "--device_group [0,1]" it will create TS config for 2 devices and use it for compilation, if "--device_group [0]" then TS compilation is skipped and single soc execution is enabled.

```bash
python -m QEfficient.cloud.infer --model_name Salesforce/codegen-2B-mono --batch_size 1 --prompt_len 32 --ctx_len 128 --mxfp6 --num_cores 16 --device-group [0,1] --prompt "def fibonacci(n):" --mos 1 --aic_enable_depth_first  
 
# Once qpc is saved, you can use the execute API to run for different prompts
python -m QEfficient.cloud.execute --model_name Salesforce/codegen-2B-mono --qpc-path qeff_models/Salesforce/codegen-2B-mono/qpc_16cores_1BS_32PL_128CL_2devices_mxfp6/qpcs --prompt "def binary_search(array: np.array, k: int):" --device-group [0,1] 
 
# To disable MQ, just pass single soc like below:
python -m QEfficient.cloud.infer --model_name gpt2 --batch_size 1 --prompt_len 32 --ctx_len 128 --mxfp6 --num_cores 16 --device-group [0] --prompt "My name is" --mos 1 --aic_enable_depth_first
```

# Using Low Level API

### 1.  Model download and transform

Initialize QEfficient and transform the models, Check the list of supported architectures in the repo.

```Python
# Initiate the Orignal Transformer model
import os

from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM

# Please uncomment and use appropriate Cache Directory for transformers, in case you don't want to use default ~/.cache dir.
# os.environ["TRANSFORMERS_CACHE"] = "/local/mnt/workspace/hf_cache"

# ROOT_DIR = os.path.dirname(os.path.abspath(""))
# CACHE_DIR = os.path.join(ROOT_DIR, "tmp") #, you can use a different location for just one model by passing this param as cache_dir in below API.

# Model-Card name to be onboarded (This is HF Model Card name) : https://huggingface.co/gpt2-xl
model_name = "gpt2"  # Similar, we can change model name and generate corresponding models, if we have added the support in the lib.

qeff_model = AutoModelForCausalLM.from_pretrained(model_name)
print(f"{model_name} optmized for AI 100 \n", qeff_model)
```

### 2. ONNX export of transformed model

use the qualcomm_efficient_converter API to export the KV transformed Model to ONNX and Verify on Torch.

```Python
import QEfficient
from QEfficient.utils import load_hf_tokenizer
# We can now export the modified models to Onnx framework
# This will generate single Onnx Model for both Prefill and Decode Variations which are optimized for
# Cloud AI 100 Platform.

# This will generate Onnx model, clip the overflow constants to fp16
# Verify the model on Onnxruntime vs Pytorch
# Then generate inputs and customio yaml file required for compilation.

# We can generate the KV Style models with the flag "kv"
# Bertstyle models do not have any optimization w.r.t KV cache changes and are unoptimized version.
# It is recommended to use kv=True for better performance.
tokenizer = load_hf_tokenizer(model_name, use_cache=True)
base_path, onnx_path = QEfficient.export(
    model_name=model_name,
    model_kv=qeff_model,
    tokenizer=tokenizer,
    kv=True,
    form_factor="cloud",
)
```

### 3. Compile on Cloud AI 100

Once, the model is exported, Compile the model on Cloud AI 100 and generate QPC.

```Python
# Please use platform SDk to Check num_cores for your card.

generated_qpc_path = QEfficient.compile(
    onnx_path=onnx_path,
    num_cores=14,  # You can use `/opt/qti-aic/tools/qaic-util | grep "Nsp Total"` from Apps SDK for this. 
    qpc_path=os.path.dirname(base_path),
    mxfp6=False,
    device_group=[0],
)
```

### 4. Run Benchmark 

Benchmark the model on Cloud AI 100, run the infer API to print tokens and tok/sec

```Python
from QEfficient.generation.text_generation_inference import get_compilation_dims

# post compilation, we can print the latency stats for the kv models, We provide API to print token and Latency stats on AI 100
# We need the compiled prefill and decode qpc to compute the token generated, This is based on Greedy Sampling Approach
batch_size, ctx_len = get_compilation_dims(generated_qpc_path)
QEfficient.cloud_ai_100_exec_kv(batch_size=batch_size, tokenizer=tokenizer, qpc_path=generated_qpc_path, device_id=[0], prompt=["My name is"], ctx_len=ctx_len)
```
