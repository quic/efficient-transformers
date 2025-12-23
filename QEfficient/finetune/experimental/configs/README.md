---
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
---
***

# Training Configuration with LoRA Finetuning

## Overview

This configuration file defines the setup for fine-tuning a Hugging Face causal language model using **LoRA (Low-Rank Adaptation)** and **PEFT (Parameter-Efficient Fine-Tuning)** techniques. It also includes dataset, training, optimizer, and scheduler settings.

***
### 1. Model Configuration

Model-related parameters for loading and fine-tuning.

*   **model\_type**: `hf` → Type of model (`hf` for Hugging Face, `custom` for custom models).
*   **auto\_class\_name**: `AutoModelForCausalLM` → AutoClass used to load the model.
*   **model\_name**: `HuggingFaceTB/SmolLM-135M` → Pretrained model to fine-tune.
*   **load\_in\_4bit**: `false` → If `true`, loads model in 4-bit quantization for memory efficiency.
*   **use\_peft**: `true` → Enables PEFT for parameter-efficient fine-tuning.
*   **peft\_config**: Defines LoRA parameters when `use_peft` is `true`:
    *   `lora_r`: Rank for LoRA adapters.
    *   `lora_alpha`: Scaling factor for LoRA updates.
    *   `lora_dropout`: Dropout applied to LoRA layers.
    *   `target_modules`: Modules to apply LoRA (e.g., `q_proj`, `v_proj`).
    *   `bias`: Bias handling (`none`, `all`, `lora_only`).
    *   `task_type`: `CAUSAL_LM` → Task type (e.g., `CAUSAL_LM`, `SEQ_2_SEQ_LM`).
    *   `peft_type`: `LORA` → Fine-tuning method (e.g., `LORA`, `IA3`).

***


### 2. Dataset Configuration

This section defines parameters for dataset handling during fine-tuning with Hugging Face models. It covers dataset type, splits, prompt formatting, and DataLoader settings.

*   **tokenizer\_name**: Matches model name.
*   **dataset\_type**: `seq_completion` → Used for sequence continuation tasks, where the model     predicts the next tokens given an input text (e.g., summarization, text generation).
*   **dataset\_name**: Dataset name for training.
*   **train\_split/test\_split**: Defines splits.
*   **split\_ratio**: For spliting the train/test dataset, only if train split is provided.
*   **prompt\_func**: Python function to format prompts.
*   **completion\_template**: `{output}` → string pattern that tells the fine-tuning pipeline which part of the dataset should be treated as the target output (completion) for the model to learn.

### Example Dataset Configs

### **1. Alpaca (yahma/alpaca-cleaned)**

```yaml
dataset:
  tokenizer_name: "meta-llama/Llama-3.2-1B"
  dataset_type: "seq_completion"
  dataset_name: "yahma/alpaca-cleaned"
  train_split: "train"
  test_split: "test"
  max_seq_length: 512
  prompt_func: "alpaca_func:create_alpaca_prompt"
  completion_template: "{output}"

```

***

### **2. Samsum (knkarthick/samsum)**

```yaml
dataset:
  tokenizer_name: "meta-llama/Llama-3.2-1B"
  dataset_type: "seq_completion"
  dataset_name: "knkarthick/samsum"
  train_split: "train"
  test_split: "test"
  prompt_func: "samsum_func:create_samsum_prompt"
  completion_template: "{summary}"

```

***
### **3. gsm8k (openai/gsm8k)**

```yaml
dataset:
  tokenizer_name: "meta-llama/Llama-3.2-1B"
  dataset_type: "seq_completion"
  dataset_name: "openai/gsm8k"
  train_split: "train"
  test_split: "test"
  prompt_func: "gsm8k_func:create_gsm8k_prompt"
  completion_template: "{answer}"

```

 ***

***
### **4. grammar (grammar_dataset)**

```yaml
dataset:
  tokenizer_name: "meta-llama/Llama-3.2-1B"
  dataset_type: "seq_completion"
  dataset_name: "grammar"
  train_split: "train"
  split_ratio: 0.8
  prompt_func: "gsm8k_func:create_grammar_prompt"
  completion_template: "{target}"
```

 *** 
### Prompt Function Examples

```python
# Alpaca
def create_alpaca_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n Response:\n"

# Samsum
def create_samsum_prompt(example):
    return f"Summarize the following conversation:\n\n{example['dialogue']}\n\nSummary:\n"

#gsm8K
def create_gsm8k_prompt(example):
    return f"Solve the following math problem step by step:\n\n{example['question']}\n\nAnswer:\n"

#grammar
def create_grammar_prompt(example):
    return f"Correct the grammar in the following sentence:\n\n{example['input']}\n\nCorrected:\n"
  
```


***

### 3. Training Configuration

This section defines core parameters for fine-tuning and evaluation.

*   **type**: `sft` → Specifies training type; `sft` means Supervised Fine-Tuning.
*   **output\_dir**: Directory where model checkpoints and logs are saved.
*   **do\_eval**: Enables evaluation during training.
*   **eval\_strategy**: `epoch` → When to run evaluation (e.g., per epoch or steps).
*   **gradient\_accumulation\_steps**: Accumulate gradients over multiple steps to simulate larger batch size.
*   **dtype**: `fp16` → Mixed precision for faster training and reduced memory usage.
*   **gradient\_checkpointing**: Saves memory by recomputing activations during backward pass (slower but memory-efficient).
*   **torch\_compile**: Wraps your model with torch.compile() (PyTorch 2.0+) to fuse ops, reduce Python overhead, and generate optimized kernels—often yielding speed-ups without code changes.
*   **Optional distributed configs**: FSDP, DeepSpeed, or DDP for multi-QAIC or large-scale training.
*    **resume_from_checkpoint**: Path to a checkpoint to resume training from.
*    **disable_tqdm**: False by default; set to True to disable progress bar (if running in Notebook).

***

### 4. Optimizer & Scheduler

*   **optimizer**: `adamw` – Optimizer for weight-decoupled regularization; options: `adamw`, `adam`, `sgd`.
    *   **lr**: Initial learning rate (e.g., `5e-5` for fine-tuning).
    *   **weight\_decay**: Regularization strength (commonly `0.01`).

*   **scheduler**: `cosine` – Learning rate decay strategy; options: `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup`, `inverse_sqrt`.
    *   **warmup\_steps**: Number of steps or ratio (e.g., `100` steps or `0.05` for 5% of total steps).
    *   Stabilizes early training and improves convergence.

***

### 5. Callbacks

Callbacks allow custom actions during training, such as logging, early stopping, or hardware profiling.

*   **early\_stopping**: Stops training if no improvement in a monitored metric for a defined patience period.
*   **tensorboard**: Enables logging of metrics and losses to TensorBoard for visualization.
*   **QAICProfilerCallback**: Profiles QAIC devices over a specified training step range to monitor performance and resource usage.
*   **QAICOpByOpVerifierCallback**: Verifies QAIC operations step-by-step during a specified training range for correctness and debugging.

***

