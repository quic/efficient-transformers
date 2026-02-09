# Training Configuration
(training-configuration)=
## Overview

This configuration file defines the setup for fine-tuning a Hugging Face causal language model using **LoRA (Low-Rank Adaptation)** and **PEFT (Parameter-Efficient Fine-Tuning)** techniques. It also includes dataset, training, optimizer, and scheduler settings.

***
## 1. Model Configuration

Model-related parameters for loading and fine-tuning.

*   **model\_type**: `default = hf` → Type of model (Use `hf` to load the model from huggingface. If the user has some custom model then user should inherit from BaseModel class and register the class under a particular key and use the key here).
*   **auto\_class\_name**: `default = AutoModelForCausalLM` → AutoClass used to load the model (Only if `model_type : hf`).
*   **model\_name**: `default = HuggingFaceTB/SmolLM-135M` → Pretrained model to fine-tune (Only if `model_type : hf`).
*   **load\_in\_4bit**: `default = false` → If `true`, loads model in 4-bit quantization for memory efficiency.
*   **use_cache**: `default = false`: Whether to use the **past key/values cache** in the model for faster decoding during generation.  
    *Enabling this can significantly speed up autoregressive decoding by reusing previous attention computations.*

*   **attn_implementation**: `default = "sdpa"`: The attention implementation to use. Common options:
    *   `"sdpa"` → Scaled Dot-Product Attention (optimized for speed and memory).
    *   `"eager"` → Standard eager-mode attention (simpler, but slower).

*   **device_map**: `default= None`: Specifies how to distribute the model across devices.
    *   `"auto"` → Automatically spreads layers across available GPUs/CPUs for memory efficiency.
    *   `None` → No distribution; model stays on the default device.

*   **use\_peft**:`default = true` → Enables PEFT for parameter-efficient fine-tuning.
*   **peft\_config**: Defines LoRA parameters when `use_peft` is true`:
    *   **lora_r**: `default = 8` Rank for LoRA adapters.
    *   **lora_alpha**: `default = 16` Scaling factor for LoRA updates.
    *   **lora_dropout**: `default = 0.1` Dropout applied to LoRA layers.
    *   **target_modules**: `dafault = ["q_proj", "v_proj"]` Modules to apply LoRA (e.g., `q_proj`, `v_proj`,`o_proj`,`k_proj`,`up_proj`,`down_proj`,`gate_proj`).
    *   **bias**: `default = None` Bias handling (`none`, `all`, `lora_only`).
    *   **task_type**: `default = CAUSAL_LM` → Task type (e.g., `CAUSAL_LM`, `SEQ_2_SEQ_LM`).
    *   **peft_type**: `default = LORA` → Fine-tuning method (e.g., `LORA`, `IA3`).

***


## 2. Dataset Configuration

This section defines parameters for dataset handling during fine-tuning with Hugging Face models. It covers dataset type, splits, prompt formatting, and DataLoader settings.

*   **tokenizer\_name**: `default = "HuggingFaceTB/SmolLM-135M"` → Matches model name.
*   **dataset\_type**: `default = "seq_completion"` → Used for sequence continuation tasks, where the language model learns to generate the correct output (completion) step by step, given an input (prompt).
*   **dataset\_name**: `default = "knkarthick/samsum"` → Dataset name for training.
*   **json_file_path**: `default = None`→ Path to a custom JSON file containing the dataset.
If provided, this takes precedence over dataset_name.
*   **train\_split/test\_split**: `default = train/test` → Names of train and test splits to be used in case of dataset being loaded from Huggingface using dataset_name argument.
*   **split\_ratio**: `default = 0.8` → For spliting the train/test dataset, only if train split is provided.
*   **prompt\_func**: Path to python function to format prompts. Use when you need complex preprocessing or conditional logic to build the final prompt string from a dataset row (e.g alpaca dataset).
*   **prompt\_template**: Template for formatting prompts from dataset rows.Prompt_template should contain the column names which are available in the dataset.

     **Note** :prompt_func and prompt_template cannot be used together. Please specify only one of these options at a time.
*  **completion\_func**: Path to python function to format completions. Use when you need complex preprocessing or conditional logic to build the final completion string from a dataset row.
*   **completion\_template**: string pattern that tells the fine-tuning pipeline which part of the dataset should be treated as the target output (completion) for the model to learn.

     **Note** : completion_func and completion_template cannot be used together. Please specify only one of these options at a time.
*   **dataset_subset**: `default = "default"` → dataset_subset is used to pick a specific configuration of a dataset when the dataset provides multiple variants. The default is "default" but you can specify something like "en", "movies", "cleaned", etc., depending on the dataset.
*   **max_seq_length**: `default = 512` → Maximum sequence length for tokenization. Longer inputs are truncated; shorter inputs may be padded depending on the collation.
*   **input_columns**: `default = ["text"]` → Column names that contain input text to be tokenized.
*   **target_column**: `default=None` → Column containing target labels (classification/regression). Set to `None` for generation-only workloads.
*   **train_batch_size**: `default = 1` → Per-device batch size during training.
*   **eval_batch_size**: `default = 1` → Per-device batch size during evaluation.
*   **collate_fn**: `default = "dynamic_padding"` → Collation function used to build batches (e.g., dynamic padding to match the longest sequence in the batch).
*   **group_by_length**: `default = true` → Whether to group samples of similar lengths together for efficient batching.
*   **length_column_name**: `default = "input_ids"` → Column name used to determine sequence length for grouping (commonly the token IDs field).
*   **num_workers**: `default = 4` → Number of subprocesses to use for data loading.
*   **dataloader_pin_memory**: `default = true` → Whether to pin memory for faster GPU transfer.
*   **dataloader_drop_last**: `default = false` → Whether to drop the last incomplete batch.

*   **dataloader_prefetch_factor**: `default = 1` → Number of batches loaded in advance by the DataLoader to overlap I/O with computations.

*   **dataloader_persistent_workers**: `default = true` → Whether to keep workers alive between epochs.
*   **dataloader_num_workers**: `default = 1` → Number of workers used by the **DataLoader** to load batches in parallel.


***
### Example Dataset Configs 

#### **1. Alpaca (yahma/alpaca-cleaned)**

```yaml
dataset:
  tokenizer_name: "meta-llama/Llama-3.2-1B"
  dataset_type: "seq_completion"
  dataset_name: "yahma/alpaca-cleaned"
  train_split: "train"
  test_split: "test"
  max_seq_length: 512
  prompt_func: "preprocess/alpaca_func:create_alpaca_prompt"
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
  dataset_type: "seq_completion"
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
  dataset_type: "seq_completion"
  dataset_name: "openai/gsm8k"
  train_split: "train"
  test_split: "test"
  prompt_template: "Solve the following math problem step by step:\n\n{'question'}\n\nAnswer:\n"
  completion_template: "{answer}"

```

***
#### **4. grammar (grammar_dataset)**

```yaml
dataset:
  tokenizer_name: "meta-llama/Llama-3.2-1B"
  dataset_type: "seq_completion"
  dataset_name: "grammar"
  train_split: "train"
  split_ratio: 0.8
  prompt_template: f"Correct the grammar in the following sentence:\n\n{'input'}\n\nCorrected:\n"
  completion_template: "{target}"
```

***

## 3. Training Configuration

This section defines core parameters for fine-tuning and evaluation.

*   **type**: `default = sft` → Specifies training type; `sft` will use trl's SFTTrainer infrastructure to perform PEFT based SFT training. `base' will use transformers' Trainer infrastructure. If user has written and registered some custom trainer then the same can be called by mentioning the registration key name here.
*   **output\_dir**: `default = "./training_results"` → Directory where model checkpoints and logs are saved.
*   **overwrite\_output\_dir**: `default = false` → Whether to overwrite the output directory if it already exists.
*   **do\_eval**: `default = true` → Enables evaluation during training.
*   **eval\_strategy**: `default = epoch` → When to run evaluation (e.g., per epoch or steps. In case of `steps` eval_strategy, include `eval_steps` to specify number of steps at which evaluation to be performed).
*   **gradient\_accumulation\_steps**: `default = 1` → Accumulate gradients over multiple steps to simulate larger batch size.
*   **dtype**: `default = fp16` → Mixed precision for faster training and reduced memory usage. FP16 dtype is recommended while training on QAIC backend.
*   **seed**: `default = 42` → Random seed for reproducibility.
*   **device**: `default = "qaic"` → The device to use for training (e.g., `"cuda"`, `"cpu"`, `"qaic"`).
*   **per\_device\_train\_batch\_size**: `default = 1` → Batch size per device during training.
*   **per\_device\_eval\_batch\_size**: `default = 1` → Batch size per device during evaluation.
*   **num\_train\_epochs**: `default = 1` → Total number of training epochs.
*   **max\_steps**: `default = -1` → If > 0, sets total number of training steps (overrides `num_train_epochs`).
*   **log\_level**: `default = "info"` → Logging verbosity (`"debug"`, `"info"`, `"warning"`, `"error"`).
*   **log\_on\_each\_node**: `default = true` → Whether to log on each node in distributed setups.
*   **logging\_strategy**: `default = "steps"` → Logging strategy (`"no"`, `"steps"`, `"epoch"`).
*   **logging\_steps**: `default = 10` → Steps between logging events.
*   **save\_strategy**: `default = "epoch"` → Checkpoint save strategy (`"no"`, `"steps"`, `"epoch"`).
*   **save\_steps**: `default = 100` → Steps between checkpoints (if `save_strategy="steps"`).
*   **save\_total\_limit**: `default = 5` → Maximum number of checkpoints to keep (older ones are deleted).
*   **metric\_for\_best\_model**: `default = "eval_loss"` → Metric used to determine the best model.
*   **include\_num\_input\_tokens\_seen**: `default = true` → Log the number of input tokens processed.
*   **average\_tokens\_across\_devices**: `default = true` → Average token counts across devices in distributed training.
*   **fsdp\_config**: `default = false` → FSDP configuration dictionary.

*   **deepspeed\_config**: `default = false` → DeepSpeed configuration dictionary.

*   **accelerator\_config**: `default = false` → Accelerate configuration dictionary.

*   **ddp\_config**: DDP configuration dictionary.

*   **use\_cpu**: `default = false` → Whether to explicitly run training on CPU.
*   **restore\_callback\_states\_from\_checkpoint**: → Whether to restore callback states from checkpoint.

*   **gradient\_checkpointing**: Saves memory by recomputing activations during backward pass (slower but memory-efficient).
*  **gradient_checkpointing_kwargs** :

   *  **preserve_rng_state**: `default = true` → Controls whether to preserve the RNG (Random Number Generator) state during checkpointing. Preserving RNG state ensures reproducibility of stochastic operations (e.g., dropout) when recomputing activations during backward passes.
   *  **use_reentrant**: `default = false`  → Determines whether to use reentrant gradient checkpointing. Reentrant checkpointing uses PyTorch's built-in mechanism for recomputation, which can reduce memory usage but may have limitations with certain custom autograd functions.
*  **ddp\_config**: Arguments for Distributed Data Parallel (DDP) training.
     *   **ddp\_backend**: `default = "qccl"` → Backend for distributed communication. Common options: `"nccl"` for GPU, `"gloo"` for CPU, `"qccl"` for QAIC.
     *   **ddp\_find\_unused\_parameters**: `default = false` → Whether to detect unused parameters during backward pass.
     *   **ddp\_bucket\_cap\_mb**: `default = 25` → Size (in MB) of gradient buckets for communication. Larger buckets reduce communication overhead but increase memory usage.
     *   **ddp\_broadcast\_buffers**: `default = true` → Whether to broadcast model buffers (e.g., BatchNorm stats) across all ranks. Use `null` or `false` to skip for speed if safe.
     *   **ddp\_timeout**: `default = 1800` → Timeout (in seconds) for DDP operations. Increase for large models or slow networks.
 
*   **torch\_compile**: `default = true` → Wraps your model with torch.compile() (PyTorch 2.0+) to fuse ops, reduce Python overhead, and generate optimized kernels—often yielding speed-ups without code changes.
*   **Optional distributed configs**: FSDP, DeepSpeed, or DDP for multi-QAIC or large-scale training.
*    **resume_from_checkpoint**: Path to a checkpoint to resume training from.
*    **disable_tqdm**: `default = false` → set to `true` to disable progress bar (if running in Notebook).


***

## 4. Optimizer & Scheduler

*   **optimizer**: `adamw`  → Optimizer for weight-decoupled regularization; options: `adamw`, `adam`, `sgd`.
    *   **lr**: Initial learning rate (e.g., `5e-5` for fine-tuning).
    *   **weight\_decay**: Regularization strength (commonly `0.01`).

*   **scheduler**: `cosine`  → Learning rate decay strategy; options: `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup`, `inverse_sqrt`.
    *   **warmup\_steps**: Number of steps or ratio (e.g., `100` steps or `0.05` for 5% of total steps). Warmup is a technique where the learning rate starts small and gradually increases to the target value during the initial phase of training to stabilize optimization. Stabilizes early training and improves convergence.

**Huggingface document for the reference and visualization of LRs**:
https://huggingface.co/docs/transformers/v5.0.0rc1/en/main_classes/optimizer_schedules#transformers.SchedulerType
 
***

## 5. Callbacks

Callbacks allow custom actions during training, such as logging, early stopping, or hardware profiling. Once these callbacks are registered, the trainer class will call these callbacks based on the state of the training. If a callback has "on_epoch_end" method defined then this method will be executed at the end of each epoch.

*   **early\_stopping**:  
    Stops training if there is no improvement in a monitored metric for a defined patience period.
    *   **early\_stopping\_patience**: `3` → The number of consecutive evaluation steps or epochs without significant improvement after which training will stop early.
    *   **early\_stopping\_threshold**: `0.01` → The minimum change in the monitored metric required to qualify as an improvement.
*   **enhanced_progressbar**: A more informative progress bar that shows additional metrics like loss, accuracy, etc. It also provides better visualization of training progress. 
*   **default_flow**: Handles the default behavior for logging, saving and evaluation. 
*   **Printer**: Display progress and print the logs (`Printer` is used if you deactivate tqdm through the TrainingArguments, otherwise it’s `enhanced_progressbar`).   
*   **JSONLoggerCallback**: Logs training metrics to a JSON file. This is useful for tracking training progress and results. 
*   **tensorboard**: Enables logging of metrics and losses to TensorBoard for visualization.
*   **QAICProfilerCallback**: Profiles QAIC devices over a specified training step range to monitor performance and resource usage.
*   **QAICOpByOpVerifierCallback**: Verifies QAIC operations step-by-step during a specified training range for correctness and debugging.

**References to some commonly used Hugging Face callbacks**:
https://huggingface.co/docs/transformers/en/main_classes/callback
***