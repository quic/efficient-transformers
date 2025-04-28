import argparse


def get_finetune_parser():
    parser = argparse.ArgumentParser(
        description="Finetune command, the model will be downloaded from Huggingface and finetuned on Cloud AI 100 and weights are saved."
    )
    parser.add_argument(
        "--model-name",
        "--model_name",
        required=False,
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Name of the pre-trained model to fine-tune",
    )
    parser.add_argument(
        "--tokenizer-name",
        "--tokenizer_name",
        required=False,
        type=str,
        default=None,
        help="Name of the tokenizer,if not passed as an argument, it uses the value of model_name",
    )
    parser.add_argument(
        "--run-validation",
        "--run_validation",
        required=False,
        type=bool,
        default=True,
        help="To run validation during training",
    )
    parser.add_argument(
        "--train-batch-size", "--train_batch_size", required=False, type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--val-batch-size", "--val_batch_size", required=False, type=int, default=1, help="Batch size for validation"
    )
    parser.add_argument(
        "--context-length",
        "--context_length",
        required=False,
        type=int,
        default=None,
        help="Maximum sequence length for inputs",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        "--gradient_accumulation_steps",
        required=False,
        type=int,
        default=4,
        help="Steps for gradient accumulation",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        "--gradient_checkpointing",
        required=False,
        type=bool,
        default=False,
        help="Use gradient checkpointing",
    )
    parser.add_argument(
        "--num-epochs", "--num_epochs", required=False, type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--max-train-step",
        "--max_train_step",
        required=False,
        type=int,
        default=0,
        help="Maximum training steps, unlimited if 0",
    )
    parser.add_argument(
        "--max-eval-step",
        "--max_eval_step",
        required=False,
        type=int,
        default=0,
        help="Maximum evaluation steps, unlimited if 0",
    )
    parser.add_argument("--device", required=False, type=str, default="qaic", help="Device to train on")
    parser.add_argument(
        "--num-workers-dataloader",
        "--num_workers_dataloader",
        required=False,
        type=int,
        default=1,
        help="Number of workers for dataloader",
    )
    parser.add_argument("--lr", required=False, type=float, default=3e-4, help="Learning rate ")
    parser.add_argument(
        "--weight-decay", "--weight_decay", required=False, type=float, default=0.0, help="Weight decay for optimizer"
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
        "--use-autocast",
        "--use_autocast",
        required=False,
        type=bool,
        default=True,
        help="Use autocast for mixed precision",
    )

    parser.add_argument(
        "--dataset", required=False, type=str, default="samsum_dataset", help="Dataset name for finetuning"
    )
    parser.add_argument(
        "--task-type",
        "--task_type",
        required=False,
        type=str,
        default="generation",
        help="generation/seq_classification",
    )
    parser.add_argument(
        "--peft-method",
        "--peft_method",
        required=False,
        type=str,
        default="lora",
        help="Parameter-efficient fine-tuning method",
    )
    parser.add_argument(
        "--use-peft",
        "--use_peft",
        required=False,
        type=bool,
        default=True,
        help="Whether to use PEFT(parameter efficient fine tuning)",
    )
    parser.add_argument(
        "--from-peft-checkpoint",
        "--from_peft_checkpoint",
        required=False,
        type=str,
        default="",
        help="Path to load PEFT checkpoint and resume the fine-tuning on that checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        required=False,
        type=str,
        default="meta-llama-samsum",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--num-freeze-layers",
        "--num_freeze_layers",
        required=False,
        type=int,
        default=1,
        help="Number of layers to freeze",
    )
    parser.add_argument(
        "--save-model", "--save_model", required=False, type=bool, default=True, help="Save the trained model"
    )
    parser.add_argument(
        "--save-metrics",
        "--save_metrics",
        required=False,
        type=bool,
        default=True,
        help="Save training metrics to a json file for later plotting",
    )
    parser.add_argument(
        "--intermediate-step-save",
        "--intermediate_step_save",
        required=False,
        type=int,
        default=1000,
        help="Steps between intermediate saves",
    )
    parser.add_argument(
        "--batching-strategy",
        "--batching_strategy",
        required=False,
        type=str,
        default="packing",
        help="Batching strategy",
    )
    parser.add_argument(
        "--enable-sorting-for-ddp",
        "--enable_sorting_for_ddp",
        required=False,
        type=bool,
        default=True,
        help="Sort data for DDP",
    )
    parser.add_argument(
        "--convergence-counter",
        "--convergence_counter",
        required=False,
        type=int,
        default=5,
        help="Steps to check convergence, its value should be >= 1, stop fine tuning when loss <= convergence_loss (defined below) for #convergence_counter steps",
    )
    parser.add_argument(
        "--convergence-loss",
        "--convergence_loss",
        required=False,
        type=float,
        default=1e-4,
        help="Loss threshold for convergence, if loss value is <= convergence_loss for #convergence_counter consecutive steps, fine tuning stops",
    )
    parser.add_argument(
        "--use-profiler", "--use_profiler", required=False, type=bool, default=False, help="Enable profiling"
    )
    parser.add_argument(
        "--enable-ddp", "--enable_ddp", required=False, type=int, default=1000, help="Enable distributed data parallel"
    )
    parser.add_argument(
        "--dist-backend",
        "--dist_backend",
        required=False,
        type=str,
        default="cpu:gloo,qaic:qccl,cuda:gloo",
        help="Backend for distributed training",
    )
    parser.add_argument(
        "--grad-scaler", "--grad_scaler", required=False, type=bool, default=True, help="Use gradient scaler"
    )
    parser.add_argument(
        "--dump-root-dir",
        "--dump_root_dir",
        required=False,
        type=str,
        default="meta-llama-samsum-mismatches/step_",
        help="Directory for mismatch dumps",
    )
    parser.add_argument(
        "--opByOpVerifier", required=False, type=bool, default=True, help="Enable operation-by-operation verification"
    )

    return parser
