# HF-Based QEfficient Finetune Module

The **QEfficient Fine-Tune Module** is a component of the QEfficient project focused on high-quality, production-grade fine-tuning pipelines. It leverages the Hugging Face ecosystem (Transformers, TRL) and supports QAIC (Qualcomm® AI) environments for accelerated training and inference.

***

## Highlights

*   **SFT-first design** using `trl.SFTTrainer` with PEFT (LoRA/QLoRA) and mixed precision.
*   **Typed Config Manager**: centralized YAML with validation, overrides, and profile inheritance.
*   **Component Registry**: plug-and-play registries for models, tokenizers, datasets, trainers, optimizers, and callbacks.
*   **Dataset support**: JSON/JSONL, CSV, and HF Hub datasets; supports instruction–response based chat schemas.
*   **Parallelism**: This stack currently supports `Data Parallelism (DDP)` for single and multi node devices and `Pipeline Parallelism (PP)`. 
*   **Multi-Node Finetuning**: Supports Multi node finetuning, which can be scaled across multiple servers.
*   **Reproducibility**: experiment tracking hooks, seed control, and deterministic data loaders (where supported).

***

## Getting Started

Transformer's Trainer `https://huggingface.co/docs/transformers/main/en/main_classes/trainer#trainer` class goes hand-in-hand with the TrainingArguments class `https://huggingface.co/docs/transformers/v5.2.0/en/main_classes/trainer#transformers.TrainingArguments`, which offers a wide range of options to customize how a model is trained.
Since this stack is based on HF's Trainer class. Please refer to above docs to configure config.yaml file for finetuning. 

### Installation (ENV set up)

Install the same prerequisites as **QEfficient**, additionally **QAIC PyTorch Eager mode** as needed.

*   QEfficient Library: <https://github.com/quic/efficient-transformers/>

If QEfficient is already installed, install `torch_qaic`, `transformers` and (optionally) `accelerate` for QAIC:

```bash
# torch_qaic (example wheel path for python 3.10 — adjust to your environment)
pip install /opt/qti-aic/integrations/torch_qaic/py310/torch_qaic-0.1.0-cp310-cp310-linux_x86_64.whl

# Install transformers with QAIC backend support
# Note: Upstream changes to transformer library
git clone https://github.com/quic-swatia/transformers.git
cd transformers 
git checkout version-4.55.0 && pip install -e .

# accelerate (example wheel path for python 3.10)
pip install /opt/qti-aic/integrations/accelerate/py310/accelerate-1.10.0-py3-none-any.whl
```

Before training, set environment variables commonly used in HF and QAIC environments:

```bash
# Allow remote code in datasets that require it (use only if you trust the source)
export HF_DATASETS_TRUST_REMOTE_CODE=True

# QAIC debugging and device logs
export QAIC_DEVICE_LOG_LEVEL=0   # Device-level logs
export QAIC_DEBUG=1              # Show CPU fallback ops, etc.

# Set temp directory
export TMPDIR=$HOME/tmp
```

### Step-by-Step Guide to run a fine-tuning job

> **Note**  
> If you’re using the pre-built `torch-qaic-env` from the Docker image for QAIC SDK, `torch_qaic` and `accelerate` whl are already installed inside it.

#### For QAIC Finetuning
For Docker-based environments, use the pre-built `torch-qaic-env` environment.

```bash
python -m venv finetune_env
source finetune_env/bin/activate
git clone https://github.com/quic/efficient-transformers.git
cd efficient-transformers
git checkout ft_experimental      #Can remove this once merged to mainline
pip install -e .
pip install   --index-url https://download.pytorch.org/whl/cpu \
--extra-index-url     https://devpi.qualcomm.com/qcom/dev/+simple \
--trusted-host devpi.qualcomm.com   "torch==2.9.1+cpu"  \
"torchvision==0.24.1+cpu"   "torchaudio==2.9.1+cpu"
pip install trl==0.22.0
cd .. && git clone https://github.com/quic-swatia/transformers.git
cd transformers 
git checkout version-4.55.0 && pip install -e .
pip install datasets==4.5.0
cd .. && cd efficient-transformers
QAIC_VISIBLE_DEVICES=0 python QEfficient/cloud/finetune_experimental.py \
QEfficient/finetune/experimental/configs/sft_single_device_gsm8k_config.yaml

```

#### For CUDA Finetuning

```bash
python -m venv finetune_env
source finetune_env/bin/activate
git clone https://github.com/quic/efficient-transformers.git
cd efficient-transformers
git checkout ft_experimental   #Can remove this once merged to mainline
pip install -e .
pip install torch==2.9.1 torchvision==0.24.1 \
torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
pip install trl==0.22.0
cd .. && git clone https://github.com/quic-swatia/transformers.git
cd transformers 
git checkout version-4.55.0 && pip install -e .
pip install datasets==4.5.0
cd .. && cd efficient-transformers
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 -m QEfficient.cloud.finetune_experimental \
--device cuda --num_epochs 1 --model_name meta-llama/Llama-3.2-3B \
--dataset_name  yahma/alpaca-cleaned --train_batch_size 1 \
--gradient_accumulation_steps 768 \
--prompt_func QEfficient.finetune.experimental.preprocessing.alpaca_func:create_alpaca_prompt \
--completion_template {output}
```

***
## Finetuning Guide

### Sample Launch Commands

**Single device (via YAML file)**

```bash
QAIC_VISIBLE_DEVICES=0 python QEfficient/cloud/finetune_experimental.py \
QEfficient/finetune/experimental/configs/sft_single_device_gsm8k_config.yaml

#As Module
QAIC_VISIBLE_DEVICES=0 python -m QEfficient.cloud.finetune_experimental \
QEfficient/finetune/experimental/configs/sft_single_device_gsm8k_config.yaml
```

**Single device (via CLI flags)**

```bash
QAIC_VISIBLE_DEVICES=0 python -m QEfficient.cloud.finetune_experimental \
--device qaic --lora_r 16 --target_modules q_proj, v_proj \
--gradient_checkpointing True --dataset_name "yahma/alpaca-cleaned" \
--completion_template {output} \
--prompt_func QEfficient.finetune.experimental.preprocessing.alpaca_func:create_alpaca_prompt

```

**Distributed Data Parallelism (Via TorchRun)**
#### If the tokenizer was used before forking processes (for DDP), which can cause deadlocks.
```bash
export TOKENIZERS_PARALLELISM=false
```

```bash
QAIC_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m QEfficient.cloud.finetune_experimental QEfficient/finetune/experimental/configs/sft_ddp_config.yaml
```

**Distributed Data Parallelism(Via Accelerate)**
```bash
QAIC_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 -m QEfficient.cloud.finetune_experimental QEfficient/finetune/experimental/configs/sft_ddp_config.yaml
```

***
## Component Registry
The training script uses a component registry to manage different components like models, optimizers, and datasets. This allows for easy swapping of components without modifying core logic.

To register a new component, use the `@registry` decorator.
See `QEfficient/finetune/experimental/core/component_registry.py` for more details on how to register components and their usage in the training pipeline. 

***
## Configuration

The configuration system uses YAML files with typed validation. It supports:
*   **Overrides**: Command-line arguments override config values.
*   **Profiles**: Inherit from base profiles and override specific settings.
*   **Validation**: Ensures all required fields are present and types match.

See `QEfficient/finetune/experimental/core/config_manager.py` for more details on configuration management.
Detailed configuration documentation is available in 
[Training Configuration](#training-configuration).

***
## Prepare Data

This module supports both custom dataset loaders and Hugging Face datasets. You can also define prompt templates or formatting functions in your configuration. Examples of prompt function in [Prompt Function Examples](#example-prompt-functions).
See `QEfficient/finetune/experimental/examples` for more details on how to register our own custom dataset

#### Using a Hugging Face Dataset with a Prompt Function/ Prompt Template

In your config, reference an HF dataset and a template function name:

```yaml
dataset:
  dataset_name: "yahma/alpaca-cleaned"
  split_train: "train"
  prompt_func: "QEfficient.finetune.experimental.preprocessing.alpaca_func:create_alpaca_prompt"
  completion_template: "{output}" # Template for completion field in dataset
```

Define the function (e.g., in `QEfficient/finetune/experimental/preprocessing/alpaca_func.py`):

```python
#preprocessing/alpaca_func.py
def create_alpaca_prompt(row):
    return prompt_no_input(row) if row["input"] == "" else prompt_input(row)
```

In your config, reference an HF dataset and a prompt template:

```yaml
dataset:
  dataset_name: "openai/gsm8k"
  config_name: "main"  # available config_name for gsm8k dataset: ["main", "socratic"]
  train_split: "train"
  prompt_template: "Solve the following math problem step by step:\n\n{'question'}\n\nAnswer:\n"
  completion_template: "{answer}"
```

Notes: 
*  The pipeline expects input data in JSON format. If your custom dataset is in JSONL or any other format, please convert it to JSON as a one‑time preprocessing step. After conversion, simply provide the JSON file path in your config.yaml.
*  Ensure your dataset's rows have keys that match the placeholders used in "prompt_template" or "prompt func". Configure it in YAML (avoid Python f-strings inside YAML; use "{prompt}/{response}" placeholders)

***
## Parallelism

The training script supports multiple parallelism strategies:

### Data Parallelism (DDP)
Distribute batches across devices.Configure this via `ddp` in the config.
 ```yaml
   ddp_config:
    ddp_backend: "qccl"
    ddp_find_unused_parameters: False
    ddp_bucket_cap_mb: 25
    ddp_broadcast_buffers: null
    ddp_timeout: 1800
 ```
With the same sft_ddp_config.yaml, we can perform single node multi-device DDP and multinode DDP by changing the torchrun command
 
**For DDP in a single server**:
```bash
QAIC_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 -m QEfficient.cloud.finetune_experimental QEfficient/finetune/experimental/configs/sft_ddp_config.yaml
``` 
where nproc-per-node is number of workers(QAIC devices) running locally.

**DDP across multiple servers(MULTINODE DDP for RACK LEVEL Finetuning)**:

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

### Pipeline Parallelism (PP)

Pipeline Parallelism splits a model's layers across multiple devices so that a model too large to fit on a single device can still be trained. 

#### How it works

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

#### Configuration parameter

Add `pp_degree` under the `training` section of your YAML config or pass it as a CLI flag.

```yaml
# training section of your config YAML
training:
  device: "qaic"       # or "cuda"
  pp_degree: 2         # split model into 2 pipeline stages
```
> **Note:** `pp_degree` must be ≤ the number of locally available devices. The total devices consumed per node is `pp_degree` (for PP-only) or `LOCAL_WORLD_SIZE × pp_degree` (for PP + DDP). Where LOCAL_WORLD_SIZE = number of processes per node. For example, add 'pp_degree: 2' as explained above in the existing yaml file: sft_single_device_gsm8k_config.yaml and use below commands. 

#### Launch commands

**PP only — single process, 2 stages (via YAML)**
```bash
python -m QEfficient.cloud.finetune_experimental QEfficient/finetune/experimental/configs/sft_single_device_gsm8k_config.yaml
```
where user can configure `pp_degree: 2` under `training:` section for the input config file `sft_single_device_gsm8k_config.yaml` to enable pipeline parallelism of degree 2.

**PP only — single process, 2 stages (via CLI flags)**
```bash
QAIC_VISIBLE_DEVICES=0,1 python -m QEfficient.cloud.finetune_experimental \
--device qaic --lora_r 16 --target_modules q_proj, v_proj \
--gradient_checkpointing True --dataset_name "yahma/alpaca-cleaned" \
--completion_template {output} \
--prompt_func QEfficient.finetune.experimental.preprocessing.alpaca_func:create_alpaca_prompt \
--pp_degree 2
```


#### Notes

- PP is currently verified primarily for **Llama-family** models. Other architectures with different layer naming conventions may need adjustments in `device_map_utils.py`.

***

## To run the Finetune project tests

Install following plugins:
```bash
pip install pytest pytest-mock
```

```bash
QAIC_VISIBLE_DEVICES=0 python -m pytest QEfficient/finetune/experimental/tests/
```

To run two of the pipeline parallelism tests, 2 devices are required:

```bash
QAIC_VISIBLE_DEVICES=0,1 python -m pytest QEfficient/finetune/experimental/tests/test_pipeline_parallelism.py
```

If we pass only one device, two of these tests get skipped and remaining tests run successfully. 
