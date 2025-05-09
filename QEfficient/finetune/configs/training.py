# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
from dataclasses import dataclass


# Configuration Classes
@dataclass
class TrainConfig:
    """Training configuration for model fine-tuning.

    Attributes:
        model_name (str): Name of the pre-trained model to fine-tune (default: "meta-llama/Llama-3.2-1B").
        tokenizer_name (str): Name of the tokenizer (defaults to model_name if None).
        run_validation (bool): Whether to run validation during training (default: True).
        batch_size_training (int): Batch size for training (default: 1).
        context_length (Optional[int]): Maximum sequence length for inputs (default: None).
        gradient_accumulation_steps (int): Steps for gradient accumulation (default: 4).
        gradient checkpointing (bool): Enable gradient checkpointing to save the memory by compromising the speed. (default: False).
        num_epochs (int): Number of training epochs (default: 1).
        max_train_step (int): Maximum training steps (default: 0, unlimited if 0).
        max_eval_step (int): Maximum evaluation steps (default: 0, unlimited if 0).
        device (str): Device to train on (default: "qaic").
        num_workers_dataloader (int): Number of workers for data loading (default: 1).
        lr (float): Learning rate (default: 3e-4).
        weight_decay (float): Weight decay for optimizer (default: 0.0).
        gamma (float): Learning rate decay factor (default: 0.85).
        seed (int): Random seed for reproducibility (default: 42).
        use_fp16 (bool): Use mixed precision training (default: True).
        use_autocast (bool): Use autocast for mixed precision (default: True).
        val_batch_size (int): Batch size for validation (default: 1).
        dataset (str): Dataset name for training (default: "samsum_dataset").
        task_type (str): Type of task for which the finetuning is to be done. Options: "generation" and "seq_classification". (default: "generation")
        peft_method (str): Parameter-efficient fine-tuning method (default: "lora").
        use_peft (bool): Whether to use PEFT (default: True).
        from_peft_checkpoint (str): Path to PEFT checkpoint (default: "").
        output_dir (str): Directory to save outputs (default: "meta-llama-samsum").
        num_freeze_layers (int): Number of layers to freeze (default: 1).
        one_qaic (bool): Use single QAIC device (default: False).
        save_model (bool): Save the trained model (default: True).
        save_metrics (bool): Save training metrics (default: True).
        intermediate_step_save (int): Steps between intermediate saves (default: 1000).
        batching_strategy (str): Batching strategy (default: "packing").
        enable_sorting_for_ddp (bool): Sort data for DDP (default: True).
        convergence_counter (int): Steps to check convergence (default: 5).
        convergence_loss (float): Loss threshold for convergence (default: 1e-4).
        use_profiler (bool): Enable profiling (default: False).
        enable_ddp (bool): Enable distributed data parallel (default: False).
        dist_backend (str): Backend for distributed training (default: "cpu:gloo,qaic:qccl,cuda:gloo").
        grad_scaler (bool): Use gradient scaler (default: True).
        dump_root_dir (str): Directory for mismatch dumps (default: "meta-llama-samsum-mismatches/step_").
        opByOpVerifier (bool): Enable operation-by-operation verification (default: False).
    """

    model_name: str = "meta-llama/Llama-3.2-1B"
    tokenizer_name: str = None  # if not passed as an argument, it uses the value of model_name
    run_validation: bool = True
    batch_size_training: int = 1
    context_length: int = None
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = False
    num_epochs: int = 1
    max_train_step: int = 0
    max_eval_step: int = 0
    device: str = "qaic"
    num_workers_dataloader: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.0
    gamma: float = 0.85  # multiplicatively decay the learning rate by gamma after each epoch
    seed: int = 42
    use_fp16: bool = True
    use_autocast: bool = True
    val_batch_size: int = 1
    dataset = "samsum_dataset"
    task_type = "generation"  # "generation" / "seq_classification"
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
    enable_sorting_for_ddp: bool = True
    convergence_counter: int = 5  # its value should be >= 1, stop fine tuning when loss <= convergence_loss (defined below) for #convergence_counter steps
    convergence_loss: float = (
        1e-4  # if loss value is <= convergence_loss for #convergence_counter consecutive steps, fine tuning stops
    )

    # TODO: vbaddi: Uncomment post adding qaic to Pytorch Profiler
    # flop_counter: bool = False # Enable flop counter to measure model throughput, can not be used with pytorch profiler at the same time.
    # flop_counter_start: int = 3 # The step to start profiling, default is 3, which means after 3 steps of warmup stage, the profiler will start to count flops.
    use_profiler: bool = False  # Enable pytorch profiler, can not be used with flop counter at the same time.
    # profiler_dir: str = "PATH/to/save/profiler/results" # will be used if using profiler

    # dist-related
    enable_ddp: bool = False
    dist_backend: str = "cpu:gloo,qaic:qccl,cuda:gloo"

    grad_scaler: bool = True
    dump_root_dir: str = "meta-llama-samsum-mismatches/step_"
    opByOpVerifier: bool = False
