# HF-Based QEfficient Finetune Module

The **QEfficient Fine-Tune Module** is a component of the QEfficient project focused on high-quality, production-grade fine-tuning pipelines. It leverages the Hugging Face ecosystem (Transformers, TRL) and supports QAIC (Qualcomm® AI) environments for accelerated training and inference.

***

## Highlights

*   **SFT-first design** using `trl.SFTTrainer` with PEFT (LoRA/QLoRA) and mixed precision.
*   **Typed Config Manager**: centralized YAML with validation, overrides, and profile inheritance.
*   **Component Registry**: plug-and-play registries for models, tokenizers, datasets, trainers, optimizers, and callbacks.
*   **Dataset support**: JSON/JSONL, CSV, and HF Hub datasets; supports instruction–response and multi-turn chat schemas.
*   **Parallelism**: `accelerate`, **DeepSpeed**, and **FSDP** for multi-GPU and sharded training.
*   **Reproducibility**: experiment tracking hooks, seed control, and deterministic data loaders (where supported).

***

## Getting Started

### Installation

Install the same prerequisites as **QEfficient**, plus **QAIC PyTorch Eager mode** as needed.

*   QEfficient Library: <https://github.com/quic/efficient-transformers/tree/ft_experimental#>

If QEfficient is already installed, install `torch_qaic`, `transformers` and (optionally) `accelerate` for QAIC:

```bash
# torch_qaic (example wheel path — adjust to your environment)
pip install /opt/qti-aic/integrations/torch_qaic/py310/torch_qaic-0.1.0-cp310-cp310-linux_x86_64.whl

# transformers 
git clone https://github.com/quic-meetkuma/transformers/tree/qaic_support_transformer_20_12_2025
cd transformers && pip install -e .

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
```


> **Note**  
> If you’re using the `torch_qaic_env` Docker environment, `torch_qaic`,`transformers` and `accelerate` may already be installed.

***
## Finetuning

### Launch Commands

**Single device using yaml file**
```bash
python finetune_experimental.py configs/sample_config.yaml

#As Module
python -m finetune_experimental configs/sample_config.yaml
```

**Single device using CLI flags**
```bash
python finetune_experimental.py --device qaic --lora_r 16 --target_modules q_proj, v_proj --gradient_checkpointing True
```
**Distributed (TorchRun)**
```bash
torchrun --nproc_per_node=4 finetune_experimental.py configs/distributed_config.yaml
```

**Distributed (Accelerate)**
```bash
accelerate launch --num_processes 4 finetune_experimental.py configs/distributed_config.yaml
```

## Inference
```bash
python infer.py configs/inference.yaml 
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

See `Experimental/core/config_manger.py` for more details on configuration management.
Detailed configuration documentation is available in 
[Training Configuration](#training-configuration).

## Prepare Data

This module supports both custom dataset loaders and Hugging Face datasets. You can also define prompt templates or formatting functions in your configuration. Examples of prompt function in [Prompt Function Examples](#example-prompt-functions).

### Registering Datasets

Register your dataset using  `registry/datasets.py`:

```python
# registry/datasets.py
import json
from torch.utils.data import Dataset
from .base import register  # your registry base

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
  dataset_name: "tatsu-lab/alpaca"
  split_train: "train"
  prompt_func: "preprocess.alpaca_func:format_alpaca"
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

- **Data Parallelism**: Distribute batches across devices.Configure this via `ddp` in the config.
 ```bash
   ddp_config:
    ddp_backend: "qccl"
    ddp_find_unused_parameters: False
    ddp_bucket_cap_mb: 25
    ddp_broadcast_buffers: null
    ddp_timeout: 1800
 ```
- **FSDP**: Fully Sharded Data Parallelism (FSDP) is supported for model sharding.
```bash
  fsdp: "full_shard"
  fsdp_config: "./configs/accelerate/fsdp_config.yaml"
  fsdp_config: "./configs/accelerate/fsdp_tp_parallelism_config.yaml"
```
- **Pipeline Parallelism**: Split model layers across devices.
- **Tensor Parallelism**: Split tensors across devices.

***