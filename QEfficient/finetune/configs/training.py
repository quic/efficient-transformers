# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str = "meta-llama/Llama-3.2-1B"
    tokenizer_name: str = "meta-llama/Llama-3.2-1B"
    run_validation: bool = True
    batch_size_training: int = 1
    context_length: int = 512
    gradient_accumulation_steps: int = 4
    num_epochs: int = 1
    max_train_step: int = 0
    max_eval_step: int = 0
    device: str = "qaic:0"
    num_workers_dataloader: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.0
    gamma: float = 0.85  # multiplicatively decay the learning rate by gamma after each epoch
    seed: int = 42
    use_fp16: bool = True
    use_autocast: bool = False
    val_batch_size: int = 1
    dataset = "samsum_dataset"
    peft_method: str = "lora"
    use_peft: bool = True  # use parameter efficient fine tuning
    from_peft_checkpoint: str = ""  # if not empty and use_peft=True, will load the peft checkpoint and resume the fine-tuning on that checkpoint
    output_dir: str = "meta-llama-samsum"
    num_freeze_layers: int = 1
    one_qaic: bool = False
    save_model: bool = True
    save_metrics: bool = True  # saves training metrics to a json file for later plotting
    intermediate_step_save: int = 1000
    batching_strategy: str = "packing"

    # TODO: vbaddi: Uncomment post adding qaic to Pytorch Profiler
    # flop_counter: bool = False # Enable flop counter to measure model throughput, can not be used with pytorch profiler at the same time.
    # flop_counter_start: int = 3 # The step to start profiling, default is 3, which means after 3 steps of warmup stage, the profiler will start to count flops.
    use_profiler: bool = False  # Enable pytorch profiler, can not be used with flop counter at the same time.
    # profiler_dir: str = "PATH/to/save/profiler/results" # will be used if using profiler
