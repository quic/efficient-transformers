# HF-Based QEfficient Finetune Module

The **QEfficient Fine-Tune Module** is part of the QEfficient project for production-grade fine-tuning pipelines. It uses the Hugging Face ecosystem, including Transformers and TRL, and supports QAIC (Qualcomm AI) environments for accelerated training and inference.

## Highlights

- **SFT-first design** using `trl.SFTTrainer` with PEFT (LoRA/QLoRA) and mixed precision.
- **Typed config manager** with centralized YAML with validation, overrides, and profile inheritance.
- **Component registry** plug-and-play registries for models, tokenizers, datasets, trainers, optimizers, and callbacks.
- **Dataset support** for JSON, JSONL, CSV, and HF Hub datasets; including instruction-response chat schemas.
- **Parallelism support** for Data Parallelism (DDP) and Pipeline Parallelism (PP).
- **Multi-node finetuning** that can scale across multiple servers.
- **Reproducibility** through experiment tracking hooks, seed control, and deterministic data loaders (where supported).

## Workflow Overview

| Step | Stage | What happens | Common entry points |
| --- | --- | --- | --- |
| 1 | Install | Set up the QEfficient stack and QAIC-specific dependencies. | `pip install`, Docker |
| 2 | Configure | Create the YAML config and choose the training mode. | PEFT, model, dataset, optimizer settings |
| 3 | Train | Launch the finetuning job. | `QEfficient finetune` CLI, `torchrun`, `accelerate` |
| 4 | Inference | Run the trained checkpoint for generation validation. | `QEfficient/finetune/experimental/examples/example_inference.py` |
| 5 | Evaluate | Measure quality and regression signals. | Validation loss, perplexity, `lm-eval-harness` |

Recommended flow:

1. Install the QEfficient stack and any QAIC-specific dependencies.
2. Configure a YAML file for PEFT, including model, dataset, optimizer, and runtime settings.
3. Train with the module entry point, `torchrun`, or `accelerate` depending on the target parallelism mode.
4. Run inference with the trained checkpoint to validate generation quality and prompt handling.
5. Evaluate the run with validation loss, perplexity, or `lm-eval-harness` style benchmarks.
***
## Getting Started

PEFT finetuning freezes the base model and trains small low-rank adapters. PEFT supports memory-efficient model trainingWith the help of [LoRA](https://arxiv.org/abs/2106.09685) , trainable parameters are reduced to less than 1% of the original model, which lowers memory and storage costs.

The Hugging Face `Trainer` class and `TrainingArguments` class provide the main configuration surface for this stack.
Transformer's [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#trainer) class goes hand-in-hand with the [TrainingArguments](https://huggingface.co/docs/transformers/v5.2.0/en/main_classes/trainer#transformers.TrainingArguments) class, which offers a wide range of options to customize how a model is trained.
Since this stack is based on HF's Trainer class. Please refer to above docs to configure the config.yaml file for finetuning. 
For TRL-based supervised finetuning -`SFTTrainer`, see the [TRL SFTTrainer docs](https://huggingface.co/docs/trl/v0.22.2/en/sft_trainer#trl.SFTTrainer).
`SFTConfig` includes only the parameters specific to SFT training; for the full list of training arguments, refer to the [`transformers.TrainingArguments`](https://huggingface.co/docs/transformers/v5.2.0/en/main_classes/trainer#transformers.TrainingArguments) documentation.

### Installation

Install the same prerequisites as **QEfficient**, plus **QAIC PyTorch Eager mode** as needed.

- QEfficient Library: <https://github.com/quic/efficient-transformers/>

If QEfficient is already installed, install `torch_qaic`, `transformers`, and optionally `accelerate` for QAIC:

```bash
# torch_qaic (example wheel path for Python 3.10; adjust to your environment)
pip install /opt/qti-aic/integrations/torch_qaic/py310/torch_qaic-0.1.0-cp310-cp310-linux_x86_64.whl

# Install transformers with QAIC backend support
# Note: Upstream changes to transformer library
git clone https://github.com/quic-akuruvil/transformers.git
cd transformers
git checkout version-4.57.3 && pip install -e .

# accelerate (example wheel path for Python 3.10)
pip install /opt/qti-aic/integrations/accelerate/py310/accelerate-1.10.0-py3-none-any.whl
```

### Environment Variables

Set the following environment variables before training:

```bash
# Allow remote code in datasets that require it
export HF_DATASETS_TRUST_REMOTE_CODE=True

# QAIC debugging and device logs
export QAIC_DEVICE_LOG_LEVEL=0
export QAIC_DEBUG=1

# Set temp directory
export TMPDIR=$HOME/tmp
```

## Step-by-Step Guide

> **Note**
> If you are using the pre-built `torch-qaic-env` from the Docker image for QAIC SDK, `torch_qaic` and `accelerate` wheels are already installed.

### QAIC Finetuning

For Docker-based environments, use the pre-built `torch-qaic-env` environment.

```bash
python -m venv finetune_env
source finetune_env/bin/activate
git clone https://github.com/quic/efficient-transformers.git
cd efficient-transformers
git checkout ft_experimental_v1  # Can remove this once merged to mainline
pip install -e .
pip install --index-url https://download.pytorch.org/whl/cpu \
--extra-index-url https://devpi.qualcomm.com/qcom/dev/+simple \
--trusted-host devpi.qualcomm.com \
"torch==2.9.1+cpu" \
"torchvision==0.24.1+cpu" \
"torchaudio==2.9.1+cpu"
pip install trl==0.22.0
cd .. && git clone https://github.com/quic-akuruvil/transformers.git
cd transformers
git checkout version-4.57.3 && pip install -e .
pip install datasets==4.5.0
cd .. && cd efficient-transformers
QAIC_VISIBLE_DEVICES=0 python QEfficient/cloud/finetune_experimental.py \
QEfficient/finetune/experimental/configs/sft_single_device_gsm8k_config.yaml
```

### CUDA Finetuning

```bash
python -m venv finetune_env
source finetune_env/bin/activate
git clone https://github.com/quic/efficient-transformers.git
cd efficient-transformers
git checkout ft_experimental_v1  # Can remove this once merged to mainline
pip install -e .
pip install torch==2.9.1 torchvision==0.24.1 \
torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
pip install trl==0.22.0
cd .. && git clone https://github.com/quic-akuruvil/transformers.git
cd transformers
git checkout version-4.57.3 && pip install -e .
pip install datasets==4.5.0
cd .. && cd efficient-transformers
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 -m QEfficient.cloud.finetune_experimental \
  --device cuda --num_epochs 1 --model_name meta-llama/Llama-3.2-3B \
  --dataset_name yahma/alpaca-cleaned --train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --prompt_func QEfficient.finetune.experimental.preprocessing.alpaca_func:create_alpaca_prompt \
  --completion_template {output}
```


## Finetuning Guide

### Sample Launch Commands

#### Single Device via YAML

```bash
QAIC_VISIBLE_DEVICES=0 python QEfficient/cloud/finetune_experimental.py \
  QEfficient/finetune/experimental/configs/sft_single_device_gsm8k_config.yaml

# As module
QAIC_VISIBLE_DEVICES=0 python -m QEfficient.cloud.finetune_experimental \
  QEfficient/finetune/experimental/configs/sft_single_device_gsm8k_config.yaml
```

#### Single Device via CLI Flags

```bash
QAIC_VISIBLE_DEVICES=0 python -m QEfficient.cloud.finetune_experimental \
  --device qaic --peft_config <peft_config.json> \
  --gradient_checkpointing True --dataset_name "yahma/alpaca-cleaned" \
  --completion_template {output} \
  --prompt_func QEfficient.finetune.experimental.preprocessing.alpaca_func:create_alpaca_prompt
```

#### Single Device via combination of YAML and CLI Args
When a field is present in both the YAML and CLI args, the CLI value
  takes precedence. CLI > YAML > class defaults
```bash
QAIC_VISIBLE_DEVICES=0 python QEfficient/cloud/finetune_experimental.py \
  QEfficient/finetune/experimental/configs/sft_single_device_gsm8k_config.yaml \
  --log_file_name='results' --dataset_num_samples=100
```

#### Distributed Data Parallelism via TorchRun

If the tokenizer was used before forking processes for DDP, disable tokenizer parallelism to avoid deadlocks:

```bash
export TOKENIZERS_PARALLELISM=false
```

```bash
QAIC_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m QEfficient.cloud.finetune_experimental \
  QEfficient/finetune/experimental/configs/sft_ddp_config.yaml
```

#### Distributed Data Parallelism via Accelerate

```bash
QAIC_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 -m QEfficient.cloud.finetune_experimental \
  QEfficient/finetune/experimental/configs/sft_ddp_config.yaml
```

### Post-PEFT Inference

After finetuning, run the adapter checkpoint with the experimental HF generate entrypoint.
The wrapper script `QEfficient/finetune/experimental/examples/example_inference.py` is a thin
launcher around `QEfficient.finetune.experimental.inference.main()`.

```bash
python QEfficient/finetune/experimental/examples/example_inference.py \
  --base_model_path <base_model_or_hf_id> \
  --adapter_path <path_to_checkpoint-xxxx> \
  --prompt "A store has 12 apples and buys 8 more. How many apples does it have now?" \
  --prompt_template <prompt_template>
```
prompt-template formats the input into the same shape the model saw during finetuning.
If the adapter checkpoint directory also contains tokenizer artifacts, the script will use them;
otherwise it falls back to the tokenizer from the config or the base model tokenizer.

<!-- If the model is finetuned on GSM8K dataset, GSM8K-style prompts should ask a short arithmetic word problem and expect a step-by-step or final numeric answer. -->

## Component Registry

The training script uses a component registry to manage models, optimizers, datasets, and other pluggable parts. This allows components to be swapped without changing the core training logic.

To register a new component, use the `@registry` decorator.
See `QEfficient/finetune/experimental/core/component_registry.py` for details on registration and usage in the training pipeline.

## Configuration

The configuration system uses YAML files with typed validation. It supports:

- **Overrides**: command-line arguments override config values.
- **Profiles**: inherit from base profiles and override specific settings.
- **Validation**: ensure all required fields are present and types match.

See `QEfficient/finetune/experimental/core/config_manager.py` for more details on configuration management.

Detailed configuration documentation is available in the training configuration section of the docs.

## Prepare Data

This module supports both custom dataset loaders and Hugging Face datasets. You can also define prompt templates or formatting functions in your configuration.
Examples of prompt function in [Prompt Function Examples](#example-prompt-functions).
See `QEfficient/finetune/experimental/examples` for more details on how to register our own custom dataset.
### Hugging Face Dataset with a Prompt Function

In your config, reference an HF dataset and a template function name:

```yaml
dataset:
  dataset_name: "yahma/alpaca-cleaned"
  split_train: "train"
  prompt_func: "QEfficient.finetune.experimental.preprocessing.alpaca_func:create_alpaca_prompt"
  completion_template: "{output}" # Template for completion field in dataset
```

Define the function, for example in `QEfficient/finetune/experimental/preprocessing/alpaca_func.py`:

```python
# preprocessing/alpaca_func.py
def create_alpaca_prompt(row):
    return prompt_no_input(row) if row["input"] == "" else prompt_input(row)
```

### Hugging Face Dataset with a Prompt Template

In your config, reference an HF dataset and a prompt template:

```yaml
dataset:
  dataset_name: "openai/gsm8k"
  config_name: "main"  # Available config_name values: ["main", "socratic"]
  train_split: "train"
  prompt_template: "Solve the following math problem step by step:\n\n{'question'}\n\nAnswer:\n"
  completion_template: "{answer}"
```

### Notes

- The pipeline expects input data in JSON format. If your custom dataset is in JSONL or any other format, convert it to JSON as a one-time preprocessing step and then provide the JSON file path in your config.
- Ensure dataset rows have keys that match the placeholders used in `prompt_template` or `prompt_func`. Configure this in YAML and avoid Python f-strings inside YAML; use `{prompt}` and `{response}` placeholders.

## Parallelism

The training script supports multiple parallelism strategies.

### Data Parallelism (DDP)

Distribute batches across devices. Configure this via `ddp` in the config.

```yaml
ddp_config:
  ddp_backend: "qccl"
  ddp_find_unused_parameters: False
  ddp_bucket_cap_mb: 25
  ddp_broadcast_buffers: null
  ddp_timeout: 1800
```

With the same `sft_ddp_config.yaml`, you can perform single-node multi-device DDP and multi-node DDP by changing the `torchrun` command.

#### Single Server

```bash
QAIC_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 -m QEfficient.cloud.finetune_experimental \
  QEfficient/finetune/experimental/configs/sft_ddp_config.yaml
```

`nproc-per-node` is the number of workers (QAIC devices) running locally.

#### Multi-Server

This enables scaling training across multiple nodes.

Use servers with the same or compatible network interface, for example Ethernet.Use servers connected to same switch for benefits in time while scaling. The current setup is supported only on Linux servers.

Host server:
i.e. the server which we are going to treat as the master and we’ll use the ip addr of this server as the master address.
```bash
QAIC_VISIBLE_DEVICES=0,1 GLOO_SOCKET_IFNAME=* torchrun --nnodes=2 --nproc-per-node=2 --node-rank=0 \
  --master_addr=* --master_port=8888 -m QEfficient.cloud.finetune_experimental \
  QEfficient/finetune/experimental/configs/sft_ddp_config.yaml
```

Client server:

```bash
QAIC_VISIBLE_DEVICES=0,1 GLOO_SOCKET_IFNAME=* torchrun --nnodes=2 --nproc-per-node=2 --node-rank=1 \
  --master_addr=* --master_port=8888 -m QEfficient.cloud.finetune_experimental \
  QEfficient/finetune/experimental/configs/sft_ddp_config.yaml
```

- `PYTHONUNBUFFERED`: makes Python prints unbuffered, which is useful for identifying progress in distributed tasks.This is optional and not compulsory to set.
- `GLOO_SOCKET_IFNAME`: specifies which network interface Gloo uses for inter-host communication, for example `eno1` or `eth0`.
- `--nnodes`: total number of hosts participating in the task.
- `--nproc-per-node`: number of processes launched on this host, usually matching the number of accelerators on the host.
- `--master_addr`: IP of the host designated as `node_rank=0`.
- `--master_port`: port used by the host to listen for other nodes. Common values are `8888` or `8000`.Use node-rank 0 on the host server and node-rank 1 on client server(for dual server setup).
- `--node-rank` must be unique per server, starting from `0` and incrementing by `1` for each additional server.For a setup with N servers it range from 0 to N-1.


### Pipeline Parallelism (PP)

Pipeline Parallelism splits a model's layers across multiple devices so that a model too large to fit on a single device can still be trained.

#### How It Works

PP is controlled by a single parameter: `pp_degree`.

| `pp_degree` value | Behaviour |
| --- | --- |
| `1` (default) | PP disabled, standard single-device training |
| `> 1` | Model is split into `pp_degree` stages, one per device |

When `pp_degree > 1`, the framework:

1. Reads the model's layer count and architecture from its Hugging Face config.
2. Distributes transformer layers as evenly as possible across stages, with surplus layers assigned to the first stages.
3. Pins the embedding (`model.embed_tokens`) to the first stage and the final norm (`model.norm`) to the last stage.
4. Uses Hugging Face `device_map="auto"` when `pp_degree == num_available_devices` for automatic placement; otherwise, it builds a custom per-layer dictionary.

#### Configuration

Add `pp_degree` under the `training` section of your YAML config or pass it as a CLI flag.

```yaml
# training section of your config YAML
training:
  device: "qaic"  # or "cuda"
  pp_degree: 2    # split model into 2 pipeline stages
```

> **Note:** `pp_degree` must be less than or equal to the number of locally available devices. The total devices consumed per node is `pp_degree` for PP-only or `LOCAL_WORLD_SIZE × pp_degree` for PP + DDP, where `LOCAL_WORLD_SIZE` is the number of processes per node. Configure 'pp_degree: 2' as explained above in the existing yaml file: sft_single_device_gsm8k_config.yaml to enable PP with 2 stages and use below commands.

#### Launch Commands

##### PP Only via YAML

```bash
python -m QEfficient.cloud.finetune_experimental \
  QEfficient/finetune/experimental/configs/sft_single_device_gsm8k_config.yaml
```

Set `pp_degree: 2` under the `training:` section in `sft_single_device_gsm8k_config.yaml` to enable pipeline parallelism with degree 2.

##### PP Only via CLI Flags

```bash
QAIC_VISIBLE_DEVICES=0,1 python -m QEfficient.cloud.finetune_experimental \
  --device qaic --use_peft True \
  --lora_r 16 --target_modules q_proj v_proj \
  --gradient_checkpointing True --dataset_name "yahma/alpaca-cleaned" \
  --completion_template {output} \
  --prompt_func QEfficient.finetune.experimental.preprocessing.alpaca_func:create_alpaca_prompt \
  --pp_degree 2
```

#### Notes

- PP is currently verified primarily for **Llama-family** models. Other architectures with different layer naming conventions may need adjustments in `device_map_utils.py`.

## Running Tests

Install the required plugins:

```bash
pip install pytest pytest-mock
```

Run the full test suite:

```bash
QAIC_VISIBLE_DEVICES=0 python -m pytest QEfficient/finetune/experimental/tests/
```

Run the pipeline parallelism tests with two devices:

```bash
QAIC_VISIBLE_DEVICES=0,1 python -m pytest QEfficient/finetune/experimental/tests/test_pipeline_parallelism.py
```

If only one device is provided, two of these tests are skipped and the remaining tests run successfully.
