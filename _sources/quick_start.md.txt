## Quick Start Guide

QEfficient Library was designed with one goal: **to make onboarding of models inference straightforward for any Transformer architecture, while leveraging the complete power of Cloud AI platform**

To achieve this, we have 2 levels of APIs, with different levels of abstraction.
1. High-level APIs abstract away complex details, offering a simpler interface. They're ideal for quick development and prototyping. If you're new to a technology or want to minimize coding effort, high-level APIs are more user-friendly.

2. Low-level APIs offer more granular control, ideal for when customization is necessary. These are particularly useful for users who are trying their own models, not hosted on HF but are implemented based on Transformers.

In summary:

* Choose high-level APIs for quick development, simplicity, and ease of use.
* Opt for low-level APIs when you need fine-tuned control, optimization, or advanced customization.

 
# Using High Level API

### 1. Use QEfficient.cloud.infer 

This is the single e2e python api in the library, which takes model_card name as input along with other compile args if necessary and does everything in one go. 

* Torch Download → Optimize for Cloud AI 100 → Export to ONNX → Verify (CPU) → Compile on Cloud AI 100 → [Execute](#2-use-of-qefficientcloudexecute)
* Its skips the ONNX export/compile stage if ONNX file or qpc found on path


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
### 2. Use of QEfficient.cloud.execute
Once we have compiled the QPC, we can now use the precompiled QPC in execute API to run for different prompts, like below:

```bash
python -m QEfficient.cloud.execute --model_name gpt2 --qpc_path qeff_models/gpt2/qpc_16cores_1BS_32PL_128CL_1devices_mxfp6/qpcs --prompt "Once upon a time in" --device_group [0]  
```

We can also enable MQ, just based on the number of devices. Based on the "--device-group" as input it will create TS config on the fly. If "--device-group [0,1]" it will create TS config for 2 devices and use it for compilation, if "--device-group 0" then TS compilation is skipped and single soc execution is enabled.

```bash
python -m QEfficient.cloud.infer --model_name Salesforce/codegen-2B-mono --batch_size 1 --prompt_len 32 --ctx_len 128 --mxfp6 --num_cores 16 --device-group [0,1] --prompt "def fibonacci(n):" --mos 2 --aic_enable_depth_first  
 
# Once qpc is saved, you can use the execute API to run for different prompts
python -m QEfficient.cloud.execute --model_name Salesforce/codegen-2B-mono --qpc-path qeff_models/Salesforce/codegen-2B-mono/qpc_16cores_1BS_32PL_128CL_2devices_mxfp6/qpcs --prompt "def binary_search(array: np.array, k: int):" --device-group [0,1] 
 
# To disable MQ, just pass single soc like below:
python -m QEfficient.cloud.infer --model_name gpt2 --batch_size 1 --prompt_len 32 --ctx_len 128 --mxfp6 --num_cores 16 --device-group [0] --prompt "My name is" --mos 1 --aic_enable_depth_first
```

| High Level APIs | Single SoC | Tensor Slicing         |
|-----------------|------------|-------------------|
| QEfficient.cloud.infer           | python -m QEfficient.cloud.infer --model_name {model_name}  --batch_size 1 --prompt_len 128 --ctx_len 1024 --num_cores 16 --device-group [0] --prompt "My name is" --mxfp6 --hf_token  {xyz}  --mos 1 --aic_enable_depth_first |  python -m QEfficient.cloud.infer --model_name {model}  --batch_size 1 --prompt_len 128 --ctx_len 1024--num_cores 16 --device-group [0,1,2,3] --prompt "My name is" --mxfp6 --hf_token  {xyz}  --mos 4 --aic_enable_depth_first |
| QEfficient.cloud.execute  |   python -m QEfficient.cloud.execute --model_name {model}  --device_group [0] --qpc_path  {path}  --prompt "My name is"  --hf_token  {xyz}   |  python -m QEfficient.cloud.execute --model_name {model}  --device_group [0,1,2,3] --qpc_path  {path}  --prompt "My name is"  --hf_token  {xyz}   |

**Note: Replace {model} ,  {path}  and  {xyz}  with preffered model card name, qpc path and hf token respectively.**

# Using Low Level API

### 1.  Model download and transform

Initialize QEfficient and transform the models, Check the list of supported architectures in the repo.

```bash
# Initiate the Orignal Transformer model
import os
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
import QEfficient
from transformers import AutoTokenizer
from QEfficient.utils import hf_download
from QEfficient.utils.constants import Constants
# Please uncomment and use appropriate Cache Directory for transformers, in case you don't want to use default ~/.cache dir.
# os.environ["TRANSFORMERS_CACHE"] = "/local/mnt/workspace/hf_cache"

ROOT_DIR = os.path.dirname(os.path.abspath(""))

# Model-Card name to be onboarded (This is HF Model Card name) : https://huggingface.co/gpt2-xl

model_name = "gpt2" 

# Similar, we can change model name and generate corresponding models, if we have added the support in the lib.

model_hf_path = hf_download(repo_id=model_name, cache_dir=Constants.CACHE_DIR, ignore_pattrens=["*.txt", "*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf"])
model_hf = GPT2LMHeadModel.from_pretrained(model_hf_path, use_cache=True)
model_hf.eval()
print(f"{model_name} from hugging-face \n", model_hf)

# Easy and minimal api to update the model
model_transformed = QEfficient.transform(model_hf, type="Transformers", form_factor="cloud")

model_transformed.eval()
print("Model after Optimized transformations \n", model_transformed)
```

### 2. ONNX export of transformed model

use the qualcomm_efficient_converter API to export the KV transformed Model to ONNX and Verify on Torch.

```bash
from QEfficient.exporter.export_hf_to_cloud_ai_100 import qualcomm_efficient_converter

# We can now export the modified models to  ONNX framework
# This will generate single ONNX Model for both Prefill and Decode Variations which are optimized for
# Cloud AI 100 Platform.

# This will generate  ONNX model, clip the overflow constants to fp16
# Verify the model on  ONNXRuntime vs Pytorch
# Then generate inputs and custom_io.yaml file required for compilation.

# We can generate the KV Style models with the flag "kv"
# Bertstyle models do not have any optimization w.r.t KV cache changes and are unoptimized version.
# It is recommended to use kv=True for better performance.

# For custom models defined on the Hub in their own modeling files. We need `trust_remote_code` option
# Should be set to `True` in `AutoTokenizer` for repositories you trust.
tokenizer = AutoTokenizer.from_pretrained(model_hf_path, use_cache=True, padding_side="left")
base_path, onnx_path = qualcomm_efficient_converter(
    model_kv=model_transformed,
    model_name=model_name,
    kv=True,
    form_factor="cloud",
    return_path=True,
    tokenizer=tokenizer,
)
```

### 3. Compile on Cloud AI 100

Once, the model is exported, Compile the model on Cloud AI 100 and generate QPC.

```bash
# Please use platform SDk to Check num_cores for your card.
from QEfficient.cloud.compile import main as compile

generated_qpc_path = compile(
    onnx_path=onnx_path,
    num_cores=14,
    qpc_path=base_path,
    device_group=[0],
    mxfp6=True,
)
```

### 4. Run Benchmark 

Benchmark the model on Cloud AI 100, run the infer API to print tokens and tok/sec

```bash
from QEfficient.generation.text_generation_inference import cloud_ai_100_exec_kv, get_compilation_batch_size

# post compilation, we can print the latency stats for the kv models, We provide API to print token and Latency stats on AI 100
# We need the compiled prefill and decode qpc to compute the token generated, This is based on Greedy Sampling Approach
batch_size = get_compilation_batch_size(generated_qpc_path)
cloud_ai_100_exec_kv(batch_size=batch_size, tokenizer=tokenizer, qpc_path=generated_qpc_path, device_id=[0], prompt="My name is")
```
