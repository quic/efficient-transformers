# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse

from QEfficient.finetune.dataset.dataset_config import DATASET_PREPROC
from QEfficient.finetune.utils.helper import BATCHING_STRATEGY, DEVICE, PEFT_METHOD, TASK_TYPE


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_finetune_parser():
    parser = argparse.ArgumentParser(
        description="Finetune command, the model is downloaded from Huggingface, finetuned on Cloud AI 100 and checkpoints are saved."
    )
    parser.add_argument(
        "--model_name",
        "--model-name",
        required=False,
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Name of the pre-trained model to fine-tune",
    )
    parser.add_argument(
        "--tokenizer_name",
        "--tokenizer-name",
        required=False,
        type=str,
        default=None,
        help="Name of the tokenizer,if not passed as an argument, it uses the value of model_name",
    )
    parser.add_argument(
        "--run_validation",
        "--run-validation",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="To run validation during training",
    )
    parser.add_argument(
        "--train_batch_size", "--train-batch-size", required=False, type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--val_batch_size", "--val-batch-size", required=False, type=int, default=1, help="Batch size for validation"
    )
    parser.add_argument(
        "--context_length",
        "--context-length",
        required=False,
        type=int,
        default=None,
        help="Maximum sequence length for inputs",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        "--gradient-accumulation-steps",
        required=False,
        type=int,
        default=4,
        help="Steps for gradient accumulation",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        "--gradient-checkpointing",
        action="store_true",
        help="Use gradient checkpointing",
    )
    parser.add_argument(
        "--use_autocast",
        "--use-autocast",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Use autocast for mixed precision",
    )
    parser.add_argument(
        "--grad_scaler", "--grad-scaler", type=str2bool, nargs="?", const=True, default=True, help="Use gradient scaler"
    )
    parser.add_argument(
        "--num_epochs", "--num-epochs", required=False, type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_train_step",
        "--max-train-step",
        required=False,
        type=int,
        default=0,
        help="Maximum training steps, unlimited if 0",
    )
    parser.add_argument(
        "--max_eval_step",
        "--max-eval-step",
        required=False,
        type=int,
        default=0,
        help="Maximum evaluation steps, unlimited if 0",
    )
    parser.add_argument("--device", required=False, type=str, default="qaic", choices=DEVICE, help="Device to train on")
    parser.add_argument(
        "--num_workers_dataloader",
        "--num-workers-dataloader",
        required=False,
        type=int,
        default=1,
        help="Number of workers for dataloader",
    )
    parser.add_argument("--lr", required=False, type=float, default=3e-4, help="Learning rate ")
    parser.add_argument(
        "--weight_decay", "--weight-decay", required=False, type=float, default=0.0, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--gamma",
        required=False,
        type=float,
        default=0.85,
        help="Learning rate decay factor, multiplicatively decays the learning rate by gamma after each epoch",
    )
    parser.add_argument("--seed", required=False, type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--dataset",
        required=False,
        default="alpaca_dataset",
        type=str,
        choices=DATASET_PREPROC.keys(),
        help="Dataset name to be used for finetuning (default: %(default)s)",
    )
    parser.add_argument(
        "--task_type",
        "--task-type",
        required=False,
        type=str,
        default="generation",
        choices=TASK_TYPE,
        help="Task used for finetuning. Use 'generation' for decoder based models and 'seq_classification' for encoder based models.",
    )
    parser.add_argument(
        "--use_peft",
        "--use-peft",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether to use PEFT(parameter efficient fine tuning)",
    )
    parser.add_argument(
        "--peft_method",
        "--peft-method",
        required=False,
        type=str,
        default="lora",
        choices=PEFT_METHOD,
        help="Parameter efficient finetuning technique to be used. Currently only 'lora' is supported.",
    )
    parser.add_argument(
        "--from_peft_checkpoint",
        "--from-peft-checkpoint",
        required=False,
        type=str,
        default="",
        help="Path to load PEFT checkpoint and resume the fine-tuning on that checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        required=False,
        type=str,
        default="training_results",
        help="Directory to save outputs of training",
    )
    parser.add_argument(
        "--save_model",
        "--save-model",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Save the final trained model checkpoints",
    )
    parser.add_argument(
        "--save_metrics",
        "--save-metrics",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Save training metrics to a json file for later plotting",
    )
    parser.add_argument(
        "--intermediate_step_save",
        "--intermediate-step-save",
        required=False,
        type=int,
        default=1000,
        help="Steps between intermediate saves of checkpoint",
    )
    parser.add_argument(
        "--batching_strategy",
        "--batching-strategy",
        required=False,
        type=str,
        default="padding",
        choices=BATCHING_STRATEGY,
        help="Strategy for making batches of data points. Packing groups data points into batches by minimizing unnecessary empty spaces. Padding adds extra values (often zeros) to batch sequences so they align in size. Currently only padding is supported which is by default.",
    )
    parser.add_argument(
        "--enable_sorting_for_ddp",
        "--enable_sorting-for-ddp",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Sort the data points according to sequence length for DDP",
    )
    parser.add_argument(
        "--convergence_counter",
        "--convergence-counter",
        required=False,
        type=int,
        default=5,
        help="Steps to check convergence, its value should be >= 1, stop fine tuning when loss <= convergence_loss (defined below) for #convergence_counter steps",
    )
    parser.add_argument(
        "--convergence_loss",
        "--convergence-loss",
        required=False,
        type=float,
        default=1e-4,
        help="Loss threshold for convergence, if loss value is <= convergence_loss for #convergence_counter consecutive steps, fine tuning stops",
    )
    parser.add_argument(
        "--use_profiler",
        "--use-profiler",
        action="store_true",
        help="Enable profiling for the operations during pytorch eager mode execution.",
    )
    parser.add_argument(
        "--enable_ddp",
        "--enable-ddp",
        action="store_true",
        help="Enable distributed data parallel training. This will load the replicas of model on given number of devices and train the model. This should be used using torchrun interface. Please check docs for exact usage.",
    )
    parser.add_argument(
        "--opByOpVerifier",
        action="store_true",
        help=argparse.SUPPRESS,
        # This is for debugging purpose only.
        # Enables operation-by-operation verification w.r.t reference device(cpu).
        # It is a context manager interface that captures and verifies each operator against reference device.
        # In case results of test & reference do not match under given tolerances, a standalone unittest is generated at dump_root_dir.
    )

    return parser
