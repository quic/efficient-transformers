# HF-Based QEfficient Finetune Module

The **QEfficient Fine-Tune Module** is a component of the QEfficient project focused on high-quality, production-grade fine-tuning pipelines. It leverages the Hugging Face ecosystem (Transformers, TRL) and supports QAIC (Qualcomm® AI) environments for accelerated training and inference.

***

## Highlights

*   **SFT-first design** using `trl.SFTTrainer` with PEFT (LoRA/QLoRA) and mixed precision.
*   **Typed Config Manager**: centralized YAML with validation, overrides, and profile inheritance.
*   **Component Registry**: plug-and-play registries for models, tokenizers, datasets, trainers, optimizers, and callbacks.
*   **Dataset support**: JSON/JSONL, CSV, and HF Hub datasets; supports instruction–response based chat schemas.
*   **Parallelism**: This stack currently supports `Data Parallelism (DDP)` for single and multi node devices and `Pipeline Parallelism (PP)`. 
*   **Reproducibility**: experiment tracking hooks, seed control, and deterministic data loaders (where supported).

***

## Getting Started

### Installation

Install the same prerequisites as **QEfficient**, plus **QAIC PyTorch Eager mode** as needed.

*   QEfficient Library: <https://github.com/quic/efficient-transformers/>

If QEfficient is already installed, install `torch_qaic`, `transformers` and (optionally) `accelerate` for QAIC:

```bash
# torch_qaic (example wheel path — adjust to your environment)
pip install /opt/qti-aic/integrations/torch_qaic/py310/torch_qaic-0.1.0-cp310-cp310-linux_x86_64.whl

# Install transformers with QAIC backend support
# TODO : Create transformer.whl
git clone https://github.com/quic-swatia/transformers.git
cd transformers 
git checkout version-4.55.0 && pip install -e .

# accelerate 
pip install /opt/qti-aic/integrations/accelerate/py310/accelerate-1.10.0-py3-none-any.whl
```

Before training, export environment variables commonly used in HF and QAIC environments:

```bash
# Allow remote code in datasets that require it (use only if you trust the source)
export HF_DATASETS_TRUST_REMOTE_CODE=True

# QAIC debugging and device logs
export QAIC_DEVICE_LOG_LEVEL=0   # Device-level logs
export QAIC_DEBUG=1              # Show CPU fallback ops, etc.

# Set temp directory
export TMPDIR = $HOME/tmp
```

### Step-by-Step Guide to run a fine-tuning job

For Docker-based environments, use the provided `torch-qaic-env` environment.

```bash
source /opt/torch-qaic-env/bin/activate
git clone https://github.com/quic/efficient-transformers.git
git checkout ft_experimental
cd efficient-transformers
pip install -e .
pip install   --index-url https://download.pytorch.org/whl/cpu   --extra-index-url     https://devpi.qualcomm.com/qcom/dev/+simple   --trusted-host devpi.qualcomm.com   "torch==2.9.1+cpu"   "torchvision==0.24.1+cpu"   "torchaudio==2.9.1+cpu"
pip install trl==0.22.0
git clone https://github.com/quic-swatia/transformers.git
cd transformers 
git checkout version-4.55.0 && pip install -e .
cd .. && QAIC_VISIBLE_DEVICES=0 python QEfficient/cloud/finetune_experimental.py QEfficient/finetune/experimental/configs/sft_single_device_gsm8k_config.yaml

```

> **Note**  
> If you’re using the `torch-qaic-env` Docker environment, `torch_qaic` and `accelerate` may already be installed.

***
## Finetuning

### Sample Launch Commands

**Single device using yaml file**
```bash
QAIC_VISIBLE_DEVICES=0 python QEfficient/cloud/finetune_experimental.py QEfficient/finetune/experimental/configs/sft_single_device_gsm8k_config.yaml

#As Module
QAIC_VISIBLE_DEVICES=0 python -m QEfficient.cloud.finetune_experimental QEfficient/finetune/experimental/configs/sft_single_device_gsm8k_config.yaml
```

**Single device using CLI flags**
```bash
QAIC_VISIBLE_DEVICES=0 python -m QEfficient.cloud.finetune_experimental --device qaic --lora_r 16 --target_modules q_proj, v_proj --gradient_checkpointing True --dataset_name "yahma/alpaca-cleaned" --completion_template {output} --prompt_func QEfficient.finetune.experimental.preprocessing.alpaca_func:create_alpaca_prompt

```
**Distributed (Using TorchRun)**
```bash
QAIC_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m QEfficient.cloud.finetune_experimental QEfficient/finetune/experimental/configs/sft_ddp_config.yaml
```

**Distributed (Using Accelerate)**
```bash
QAIC_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 -m QEfficient.cloud.finetune_experimental QEfficient/finetune/experimental/configs/sft_ddp_config.yaml
```

***
## Component Registry
The training script uses a component registry to manage different components like models, optimizers, and datasets. This allows for easy swapping of components without modifying core logic.

To register a new component, use the `@registery` decorator.
See `Experimental/core/component_registry.py` for more details on how to register components and their usage in the training pipeline. 

***
## Configuration

The configuration system uses YAML files with typed validation. It supports:
*   **Overrides**: Command-line arguments override config values.
*   **Profiles**: Inherit from base profiles and override specific settings.
*   **Validation**: Ensures all required fields are present and types match.

See `experimental/core/config_manger.py` for more details on configuration management.
Detailed configuration documentation is available in 
[Training Configuration](#training-configuration).

## Prepare Data

This module supports both custom dataset loaders and Hugging Face datasets. You can also define prompt templates or formatting functions in your configuration. Examples of prompt function in [Prompt Function Examples](#example-prompt-functions).

### Registering Datasets

Register your dataset using  `Component Factory`:

```python
# QEfficient/finetune/experimental/core/datasets.py
import json
from torch.utils.data import Dataset
from QEfficient.finetune.experimental.core.component_registry import registry  

@registry.dataset( "my_custom_dataset")
class MyCustomDataset(BaseDataset):
    def __init__(self,
        dataset_name: str,
        split: str,
        **kwargs):
        self.json_file_path = kwargs.get("json_path", None)
        self.dataset_name = dataset_name
        self.split = split

        if self.json_file_path:
            # Load dataset from JSON file
            self.dataset = load_dataset("json", data_files=self.json_file_path, split="train")
        else:
            self.dataset = load_dataset(self.dataset_name, split=self.split)       
        self.template = kwargs.get(prompt_template,None) or 
        "### Instruction:\n{prompt}\n### Response:\n{response}"

    def __len__(self):
        return self.dataset.num_rows
    
    def preprocess(self, example):
        return self.template.format(**example)  # Safe string formatting with placeholders.

    def __getitem__(self, idx):
        example = self.dataset.select(indices=[int(idx)])[0]
        # Apply preprocessing (templating) on the fly
        processed_example = self.preprocess(example)
        return processed_example
```

#### Using json_file with Prompt Function/ Prompt Template
```yaml
dataset:
  dataset_name: my_custom_dataset
  dataset_type: my_custom_dataset
  split_train: train
  json_file_path: data/my_train.jsonl
  prompt_template: |
    ### Instruction:
    {prompt}
    ### Response:
    {response}
```

#### Using a Hugging Face Dataset with a Prompt Function/ Prompt Template

In your config, reference an HF dataset and a template function name:

```yaml
dataset:
  dataset_name: "yahma/alpaca-cleaned"
  split_train: "train"
  prompt_func: "QEfficient.finetune.experimental.preprocessing.alpaca_func:create_alpaca_prompt"
```

Define the function (e.g., in `preprocess/alpaca_func.py`):

```python
#preprocess/alpaca_func.py
def format_alpaca(example):
    # Expect keys: 'instruction' and 'output'
    return f"### Instruction:\n{example['instruction']}\n### Response:\n{example['output']}"
```
```
Tips:
Ensure your dataset's rows have keys that match the placeholders used in "prompt_template" or "prompt func".
Configure it in YAML (avoid Python f-strings inside YAML; use "{prompt}/{response}" placeholders)
```
***
## Parallelism

The training script supports multiple parallelism strategies:

## Data Parallelism
Distribute batches across devices.Configure this via `ddp` in the config.
 ```bash
   ddp_config:
    ddp_backend: "qccl"
    ddp_find_unused_parameters: False
    ddp_bucket_cap_mb: 25
    ddp_broadcast_buffers: null
    ddp_timeout: 1800
 ```
With the same sft_ddp_config.yaml, we can perform single node multi-device DDP and multimode DDP by changing the torchrun command
 
**For DDP in a single server**:
```bash
QAIC_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 -m QEfficient.cloud.finetune_experimental QEfficient/finetune/experimental/configs/sft_ddp_config.yaml
``` 
where nproc-per-node is number of workers(QAIC devices) running locally.

**For DDP across multiple servers(MULTINODE DDP for RACK LEVEL Finetuning)**:

This enables scaling training across multiple nodes.

Use servers with compatible/same network interface(eg:ethernet).

And supported only for linux servers now. Use servers connected to same switch for benefits in time while scaling.

*  On host server (i.e. the server which we are going to treat as the master and we’ll use the ip addr of this server as the master addr):

    ```bash
    QAIC_VISIBLE_DEVICES=0,1 GLOO_SOCKET_IFNAME=* torchrun --nnodes=2 --nproc-per-node=2 --node-rank=0 --master_addr=* --master_port=8888 -m QEfficient.cloud.finetune_experimental QEfficient/finetune/experimental/configs/sft_ddp_config.yaml
    ```

*  On client server:

    ```bash
    QAIC_VISIBLE_DEVICES=0,1 GLOO_SOCKET_IFNAME=* torchrun --nnodes=2 --nproc-per-node=2 --node-rank=1 --master_addr=* --master_port=8888 -m QEfficient.cloud.finetune_experimental QEfficient/finetune/experimental/configs/sft_ddp_config.yaml
    ```

*  Use servers with compatible/same network interface(eg:ethernet).
*  PYTHONUNBUFFERED: make python prints unbuffered, especially useful to identify progress (or lack thereof) for distributed tasks.This is optional and not compulsory
*  GLOO_SOCKET_IFNAME: specify which network interface gloo (and indirectly qccl) uses for inter-host communication (eg: eno1, eth0 etc)
*  --nnodes: total number of hosts participating in the task
*  --nproc-per-node: number of processes launched on this host, usually coincides with number of accelerators on this host
*  --master_addr: ip of the host designated with node_rank=0 ($ ip addr) 
*  --master_port: port on which host will be listening for other nodes to connect. (eg: 8888, 8000 etc).Use node-rank 0 on the host server and node-rank 1 on client server(for dual server setup).
*  When running distributed training across multiple servers, the --node-rank parameter must be assigned a unique value for each server, starting from 0 and incrementing by 1 for each additional server. For a setup with N servers it range from 0 to N-1.

***

## Pipeline Parallelism (PP)

Pipeline Parallelism splits a model's layers across multiple devices so that a model too large to fit on a single device can still be trained. 

### How it works

PP is controlled by a single parameter: **`pp_degree`**.

| `pp_degree` value | Behaviour |
|---|---|
| `1` (default) | PP disabled — standard single-device training |
| `> 1` | Model is split into `pp_degree` stages, one per device |

When `pp_degree > 1` the framework:
1. Reads the model's layer count and architecture from its HuggingFace config.
2. Distributes transformer layers as evenly as possible across stages (surplus layers go to the first stages).
3. Pins the embedding (`model.embed_tokens`) to the first stage and the final norm (`model.norm`) to the last stage.
4. When `pp_degree == num_available_devices`, uses HuggingFace's `device_map="auto"` for automatic placement. Otherwise a custom per-layer dict is built.

### Configuration parameter

Add `pp_degree` under the `training` section of your YAML config or pass it as a CLI flag.

```yaml
# training section of your config YAML
training:
  device: "qaic"       # or "cuda"
  pp_degree: 2         # split model into 2 pipeline stages
```

> **Note:** `pp_degree` must be ≤ the number of locally available devices. The total devices consumed per node is `pp_degree` (for PP-only) or `LOCAL_WORLD_SIZE × pp_degree` (for PP + DDP).

### Launch commands

**PP only — single process, 2 stages (via YAML)**
```bash
python -m QEfficient.cloud.finetune_experimental configs/sample_pp_config.yaml
```
where `sample_pp_config.yaml` contains `pp_degree: 2` under `training:`.

**PP only — single process, 2 stages (via CLI flags)**
```bash
python -m QEfficient.cloud.finetune_experimental \
    --model_name meta-llama/Llama-3.2-1B \
    --device qaic \
    --pp_degree 2
```



### Notes

- PP is currently verified primarily for **Llama-family** models. Other architectures with different layer naming conventions may need adjustments in `device_map_utils.py`.

***
