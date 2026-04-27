# Training Configuration
(training-configuration)=
## Overview

This configuration file defines the setup for fine-tuning a Hugging Face causal language model using **LoRA (Low-Rank Adaptation)** and **PEFT (Parameter-Efficient Fine-Tuning)** techniques. It also includes dataset, training, optimizer, and scheduler settings.

***
## 1. Model Configuration

Model-related parameters for loading and fine-tuning.

| Parameter | Default | Description |
| --- | --- | --- |
| `model_type` | `hf` | Type of model. Use `hf` to load from Hugging Face. For a custom model, inherit from `BaseModel`, register it under a key, and use that key here. |
| `auto_class_name` | `AutoModelForCausalLM` | Auto class used to load the model. |
| `model_name` | `HuggingFaceTB/SmolLM-135M` | Pretrained model to fine-tune. |
| `load_in_4bit` | `false` | If `true`, loads model in 4-bit quantization for memory efficiency. |
| `use_cache` | `false` | Uses the past key/values cache for faster decoding during generation. |
| `attn_implementation` | `"sdpa"` | Attention implementation. Common values: `sdpa`, `eager`. |
| `device_map` | `None` | Specifies how to distribute the model across devices. |
| `use_peft` | `true` | Enables PEFT for parameter-efficient fine-tuning. |
| `peft_config` | - | Defines LoRA parameters when `use_peft` is true. |

PEFT sub-parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `lora_r` | `8` | Rank for LoRA adapters. |
| `lora_alpha` | `16` | Scaling factor for LoRA updates. |
| `lora_dropout` | `0.1` | Dropout applied to LoRA layers. |
| `target_modules` | `["q_proj", "v_proj"]` | Modules to apply LoRA to. |
| `bias` | `None` | Bias handling (`none`, `all`, `lora_only`). |
| `task_type` | `CAUSAL_LM` | Task type, for example `CAUSAL_LM` or `SEQ_2_SEQ_LM`. |
| `peft_type` | `LORA` | Fine-tuning method, for example `LORA` or `IA3`. |

***


## 2. Dataset Configuration

This section defines parameters for dataset handling during fine-tuning with Hugging Face models. It covers dataset type, splits, prompt formatting, and DataLoader settings.

| Parameter | Default | Description |
| --- | --- | --- |
| `tokenizer_name` | `HuggingFaceTB/SmolLM-135M` | Matches model name. |
| `dataset_type` | `seq_completion` | Used for sequence continuation tasks. |
| `dataset_name` | `knkarthick/samsum` | Dataset name for training. |
| `json_file_path` | `None` | Path to a custom JSON file. Takes precedence over `dataset_name`. |
| `train_split/test_split` | `train/test` | Train and test split names when loading from Hugging Face. |
| `split_ratio` | `0.8` | Used for splitting the train/test dataset when only train split is provided. |
| `prompt_func` | - | Python function to format prompts from a dataset row. |
| `prompt_template` | - | Template for formatting prompts from dataset rows. |
| `completion_func` | - | Python function to format completions from a dataset row. |
| `completion_template` | - | String pattern for the target completion. |
| `dataset_subset` | `default` | Selects a specific dataset configuration when multiple variants exist. |
| `max_seq_length` | `512` | Maximum tokenized sequence length. |
| `input_columns` | `["text"]` | Column names containing input text. |
| `target_column` | `None` | Column containing target labels. |
| `train_batch_size` | `1` | Per-device batch size during training. |
| `eval_batch_size` | `1` | Per-device batch size during evaluation. |
| `collate_fn` | `dynamic_padding` | Collation function used to build batches. |
| `dataset_disc_style` | `None` | Style remix category used during preprocessing. |
| `group_by_length` | `true` | Groups samples of similar lengths for efficient batching. |
| `length_column_name` | `input_ids` | Column name used to determine sequence length. |
| `num_workers` | `4` | Number of subprocesses used for data loading. |
| `dataloader_pin_memory` | `true` | Pins memory for faster GPU transfer. |
| `dataloader_drop_last` | `false` | Drops the last incomplete batch. |
| `dataset_num_samples` | `-1` | Number of samples to use. If `-1`, all samples are used. |
| `dataloader_prefetch_factor` | `1` | Number of batches loaded in advance by the DataLoader. |
| `dataloader_persistent_workers` | `true` | Keeps workers alive between epochs. |
| `dataloader_num_workers` | `1` | Number of DataLoader workers. |

Notes:

- If both `prompt_template` and `prompt_func` are provided, `prompt_template` takes precedence.
- If both `completion_template` and `completion_func` are provided, `completion_template` takes precedence.


***
### Example Dataset Configs 

#### **1. Alpaca (yahma/alpaca-cleaned)**

```yaml
dataset:
  tokenizer_name: "meta-llama/Llama-3.2-1B"
  dataset_type: "sft_dataset"
  dataset_name: "yahma/alpaca-cleaned"
  train_split: "train"
  test_split: "test"
  max_seq_length: 512
  prompt_func: "QEfficient.finetune.experimental.preprocessing.alpaca_func:create_alpaca_prompt"
  completion_template: "{output}"

```
(example-prompt-functions)=
### Prompt Function Example

```python
# Alpaca
#preprocess/alpaca_func.py
def prompt_no_input(row):
    return ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n").format_map(row)


def prompt_input(row):
    return ("Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n").format_map(row)


def create_alpaca_prompt(row):
    return prompt_no_input(row) if row["input"] == "" else prompt_input(row)
```
***

#### **2. Samsum (knkarthick/samsum)**

```yaml
dataset:
  tokenizer_name: "meta-llama/Llama-3.2-1B"
  dataset_type: "sft_dataset"
  dataset_name: "knkarthick/samsum"
  train_split: "train"
  test_split: "test"
  prompt_template: "Summarize the following conversation:\n\n{'dialogue'}\n\nSummary:\n"
  completion_template: "{summary}"

```

***
#### **3. gsm8k (openai/gsm8k)**

```yaml
dataset:
  tokenizer_name: "meta-llama/Llama-3.2-1B"
  dataset_type: "sft_dataset"
  dataset_name: "openai/gsm8k"
  config_name: "main"  # available config_name for gsm8k dataset: ["main", "socratic"]
  train_split: "train"
  test_split: "test"
  prompt_template: "Solve the following math problem step by step:\n\n{'question'}\n\nAnswer:\n"
  completion_template: "{answer}"

```
***

#### **4. Style-Remix (hallisky/DiSC)**

```yaml
dataset:
  dataset_type: "sft_dataset"
  dataset_name: "hallisky/DiSC" 
  prompt_template: "### Original:{original} \n ### Rewrite:\n" 
  completion_template: "{generation}"     
  dataset_disc_style: "sarcasm_more" 

```
***

## 3. Training Configuration

This section defines core parameters for fine-tuning and evaluation.

| Parameter | Default | Description |
| --- | --- | --- |
| `type` | `sft` | Training type. `sft` uses TRL's `SFTTrainer`. `base` uses `transformers.Trainer`. |
| `output_dir` | `./training_results` | Directory where model checkpoints and logs are saved. |
| `log_file_name`| None | The log_file output name. If it is None, the training log file will be saved as training_logs_<{datetime.now():%Y%m%d_%H%M%S}>.txt format.
| `overwrite_output_dir` | `false` | Whether to overwrite the output directory if it already exists. |
| `do_eval` | `true` | Enables evaluation during training. |
| `eval_strategy` | `epoch` | When to run evaluation. |
| `gradient_accumulation_steps` | `1` | Accumulates gradients over multiple steps. |
| `dtype` | `fp16` | Mixed precision setting. |
| `seed` | `42` | Random seed for reproducibility. |
| `device` | `qaic` | Device to use for training. |
| `per_device_train_batch_size` | `1` | Batch size per device during training. |
| `per_device_eval_batch_size` | `1` | Batch size per device during evaluation. |
| `num_train_epochs` | `1` | Total number of training epochs. |
| `max_steps` | `-1` | If > 0, sets total number of training steps. |
| `log_level` | `info` | Logging verbosity. |
| `log_on_each_node` | `true` | Whether to log on each node in distributed setups. |
| `logging_strategy` | `steps` | Logging strategy. |
| `logging_steps` | `10` | Steps between logging events. |
| `save_strategy` | `epoch` | Checkpoint save strategy. |
| `save_steps` | `100` | Steps between checkpoints when `save_strategy="steps"`. |
| `save_total_limit` | `5` | Maximum number of checkpoints to keep. |
| `metric_for_best_model` | `eval_loss` | Metric used to determine the best model. |
| `include_num_input_tokens_seen` | `true` | Logs the number of input tokens processed. |
| `average_tokens_across_devices` | `true` | Averages token counts across devices in distributed training. |
| `fsdp_config` | `None` | FSDP configuration dictionary. |
| `deepspeed_config` | `None` | DeepSpeed configuration dictionary. |
| `accelerator_config` | `None` | Accelerate configuration dictionary. |
| `use_cpu` | `false` | Whether to explicitly run training on CPU. |
| `restore_callback_states_from_checkpoint` | - | Whether to restore callback states from checkpoint. |
| `gradient_checkpointing` | - | Saves memory by recomputing activations during backward pass. |
| `gradient_checkpointing_kwargs.preserve_rng_state` | `true` | Preserves RNG state during checkpointing. |
| `gradient_checkpointing_kwargs.use_reentrant` | `false` | Uses reentrant gradient checkpointing. |
| `ddp_config.ddp_backend` | `qccl` | Backend for distributed communication. |
| `ddp_config.ddp_find_unused_parameters` | `false` | Detects unused parameters during backward pass. |
| `ddp_config.ddp_bucket_cap_mb` | `25` | Size of gradient buckets for communication. |
| `ddp_config.ddp_broadcast_buffers` | `true` | Broadcasts model buffers across ranks. |
| `ddp_config.ddp_timeout` | `1800` | Timeout in seconds for DDP operations. |
| `torch_compile` | `false` | Wraps the model with `torch.compile()`. |
| `report_to` | `tensorboard` | Logging frameworks to use. |
| `resume_from_checkpoint` | - | Path to a checkpoint to resume training from. |
| `disable_tqdm` | `false` | Disables the progress bar. |

Optional distributed configs: FSDP, DeepSpeed, or DDP for multi-QAIC or large-scale training.

📁 **Output Directory Structure**

    output_dir/
    │
    ├── checkpoints/              # Saved model checkpoints (checkpoint-*)
    │
    ├── runs/                     # TensorBoard logs
    │   └── events.out.tfevents.* 
    │
    ├── logs/                     # Logs from other backends
    ├── <log_file_name>.txt       # Training log file    



***

## 4. Optimizer & Scheduler

| Parameter | Default | Description |
| --- | --- | --- |
| `optimizer` | `adamw` | Optimizer for weight-decoupled regularization. |
| `optimizer.lr` | - | Initial learning rate, for example `5e-5`. |
| `optimizer.weight_decay` | `0.01` | Regularization strength. |
| `scheduler` | `cosine` | Learning rate decay strategy. |
| `scheduler.warmup_steps` | - | Number of warmup steps or a ratio of total steps. |

Hugging Face reference for learning-rate schedules:
https://huggingface.co/docs/transformers/v5.0.0rc1/en/main_classes/optimizer_schedules#transformers.SchedulerType
 
***

## 5. Callbacks

Callbacks allow custom actions during training, such as logging, early stopping, or hardware profiling. Once these callbacks are registered, the trainer class calls them based on training state. If a callback defines `on_epoch_end`, it runs at the end of each epoch.

| Callback | Description |
| --- | --- |
| `early_stopping` | Stops training if there is no improvement in a monitored metric for a defined patience period. |
| `early_stopping_patience` | Number of consecutive evaluation steps or epochs without significant improvement before stopping. |
| `early_stopping_threshold` | Minimum change in the monitored metric required to qualify as improvement. |
| `train_logger` | Logs per epoch time, training metric (perplexity),training loss, evaluation metrics and loss etc.|
| `enhanced_progressbar` | More informative progress bar with additional metrics. |
| `default_flow` | Handles the default behavior for logging, saving, and evaluation. |
| `Printer` | Displays progress and logs. Used when tqdm is disabled. |
| `JSONLoggerCallback` | Logs training metrics to a JSON file. |
| `tensorboard` | Enables logging of metrics and losses to TensorBoard. |
| `QAICProfilerCallback` | Profiles QAIC devices over a specified training step range. |
| `QAICOpByOpVerifierCallback` | Verifies QAIC operations step-by-step for correctness and debugging. |

**References to some commonly used Hugging Face callbacks**:
https://huggingface.co/docs/transformers/en/main_classes/callback
***
