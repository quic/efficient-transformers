# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Configuration manager for handling all training configurations.
Provides centralized configuration loading, validation, and management.
"""

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from transformers.hf_argparser import HfArgumentParser

from QEfficient.finetune.experimental.core.component_registry import registry


@dataclass
class OptimizerConfig:
    """Configuration for optimizers."""

    optimizer_name: str = field(
        default="adamw",
        metadata={"help": "The name of the optimizer to use."},
    )
    lr: float = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for the optimizer."},
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "The weight decay to apply (if any)."},
    )


@dataclass
class SchedulerConfig:
    """Configuration for learning rate schedulers."""

    scheduler_name: str = field(
        default="cosine",
        metadata={"help": "The name of the scheduler to use (e.g., 'linear', 'cosine')."},
    )
    warmup_steps: int = field(
        default=100,
        metadata={
            "help": "Number of steps for the warmup phase. If provided "
            "value is within [0-1) range then it will be interpreted as "
            "ratio of total training steps for the warmup phase."
        },
    )


@dataclass
class DatasetConfig:
    """Configuration for datasets."""

    tokenizer_name: str = field(
        default="HuggingFaceTB/SmolLM-135M",
        metadata={"help": "The name or path of the tokenizer to use."},
    )
    dataset_type: str = field(
        default="seq_completion",
        metadata={"help": "The type of dataset (e.g., 'seq_completion')."},
    )
    dataset_name: str = field(
        default="knkarthick/samsum",
        metadata={"help": "The name or path of the dataset."},
    )
    dataset_subset: str = field(
        default="default",
        metadata={"help": "The subset of the dataset to use, if applicable."},
    )
    train_split: str = field(
        default="train",
        metadata={"help": "The name of the training split."},
    )
    test_split: str = field(
        default="test",
        metadata={"help": "The name of the test/validation split."},
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "The maximum sequence length for tokenization."},
    )
    split_ratio: float = field(
        default=0.8,
        metadata={"help": "Ratio for train/test split, used when only train_split is provided."},
    )
    input_columns: list[str] = field(
        default_factory=lambda: ["text"],
        metadata={"help": "List of column names containing input text."},
    )
    target_column: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the column containing target labels (if applicable)."},
    )
    train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device during training."},
    )
    eval_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device during evaluation."},
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of workers for dataset processing."},
    )
    collate_fn: str = field(
        default="dynamic_padding",
        metadata={"help": "The collation function to use (e.g., 'dynamic_padding')."},
    )
    group_by_length: bool = field(
        default=True,
        metadata={"help": "Whether to group samples by length to minimize padding."},
    )
    length_column_name: str = field(
        default="input_ids",
        metadata={"help": "The column name containing the length of the input sequences."},
    )
    dataloader_pin_memory: bool = field(
        default=True,
        metadata={"help": "Whether to pin GPU memory for dataloaders."},
    )
    dataloader_persistent_workers: bool = field(
        default=True,
        metadata={"help": "Whether to keep dataloader workers alive across epochs."},
    )
    dataloader_prefetch_factor: int = field(
        default=1,
        metadata={"help": "Number of samples loaded in advance by each worker."},
    )
    dataloader_drop_last: bool = field(
        default=False,
        metadata={"help": "Whether to drop the last incomplete batch."},
    )
    dataloader_num_workers: int = field(
        default=1,
        metadata={"help": "Number of workers for the DataLoader."},
    )


@dataclass
class PeftConfig:
    """Configuration for PEFT (Parameter-Efficient Fine-Tuning) methods."""

    lora_r: int = field(
        default=8,
        metadata={"help": "Lora attention dimension."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "Lora alpha."},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout probability for Lora layers."},
    )
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"],
        metadata={"help": "The modules to apply Lora to."},
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for Lora ('none', 'all', 'lora_only')."},
    )
    task_type: str = field(
        default="CAUSAL_LM",
        metadata={"help": "The task type for PEFT (e.g., 'CAUSAL_LM', 'SEQ_2_SEQ_LM')."},
    )
    peft_type: str = field(
        default="LORA",
        metadata={"help": "The PEFT method to use (e.g., 'LORA', 'IA3')."},
    )


@dataclass
class ModelConfig:
    """Configuration for models."""

    model_name: str = field(
        default="HuggingFaceTB/SmolLM-135M",
        metadata={"help": "The name or path of the pretrained model."},
    )
    model_type: str = field(
        default="hf",
        metadata={"help": "The type of model ('hf' for Hugging Face, 'custom' for custom models)."},
    )
    auto_class_name: str = field(
        default="AutoModelForCausalLM",
        metadata={"help": "The AutoClass name to load the model (e.g., 'AutoModelForCausalLM')."},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to load the model in 4-bit quantization."},
    )
    use_peft: bool = field(
        default=True,
        metadata={"help": "Whether to use PEFT (Parameter-Efficient Fine-Tuning)."},
    )
    peft_config: Optional[PeftConfig] = field(
        default_factory=PeftConfig,
        metadata={"help": "Configuration for PEFT."},
    )
    use_cache: bool = field(
        default=False,
        metadata={"help": "Whether to use the past key/values in the model for faster decoding."},
    )
    attn_implementation: str = field(
        default="sdpa",
        metadata={"help": "The attention implementation to use (e.g., 'sdpa', 'eager')."},
    )
    device_map: Optional[str] = field(
        default=None,
        metadata={"help": "The device map to use for model distribution (e.g., 'auto')."},
    )


@dataclass
class CallbackConfig:
    """Configuration for callbacks."""

    callbacks: Dict[str, Dict[str, Any]] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of callback configurations, keyed by callback name."},
    )


@dataclass
class GradientCheckpointingKwargs:
    """Arguments for gradient checkpointing."""

    preserve_rng_state: bool = field(
        default=True,
        metadata={"help": "Whether to preserve the RNG state when checkpointing."},
    )
    use_reenrant: bool = field(
        default=False,
        metadata={"help": "Whether to use reentrant gradient checkpointing."},
    )


@dataclass
class DdpConfig:
    """Arguments for Distributed Data Parallel (DDP) training."""

    ddp_backend: str = field(
        default="qccl",
        metadata={"help": "The DDP backend to use (e.g., 'nccl', 'gloo', 'qccl')."},
    )
    ddp_find_unused_parameters: bool = field(
        default=True,
        metadata={"help": "Whether to find unused parameters in DDP."},
    )
    ddp_bucket_cap_mb: Optional[int] = field(
        default=25,
        metadata={"help": "The bucket size in MB for DDP communication."},
    )
    ddp_broadcast_buffers: bool = field(
        default=True,
        metadata={"help": "Whether to broadcast buffers in DDP."},
    )
    ddp_timeout: int = field(
        default=1800,
        metadata={"help": "Timeout for DDP operations in seconds."},
    )


@dataclass
class TrainingConfig:
    """Configuration for training."""

    type: str = field(
        default="sft",
        metadata={"help": "The type of training (e.g., 'sft' for Supervised Fine-Tuning)."},
    )
    output_dir: str = field(
        default="./training_results",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={"help": "Whether to overwrite the output directory."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility."},
    )

    do_eval: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation during training."},
    )
    eval_strategy: str = field(
        default="epoch",
        metadata={"help": "The evaluation strategy to use ('no', 'steps', 'epoch')."},
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Number of update steps between two evaluations."},
    )

    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device during training."},
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device during evaluation."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Total number of training epochs to perform."},
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform."},
    )

    log_level: str = field(
        default="info",
        metadata={"help": "Set the verbosity level of the logs ('debug', 'info', 'warning', 'error')."},
    )
    log_on_each_node: bool = field(
        default=True,
        metadata={"help": "Whether to log on each node in a distributed setup."},
    )
    logging_strategy: str = field(
        default="steps",
        metadata={"help": "The logging strategy to use ('no', 'steps', 'epoch')."},
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Number of update steps between two loggings."},
    )

    save_strategy: str = field(
        default="epoch",
        metadata={"help": "The checkpoint save strategy to use ('no', 'steps', 'epoch')."},
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Number of update steps between two checkpoints (if save_strategy is 'steps')."},
    )
    save_total_limit: int = field(
        default=5,
        metadata={"help": "Limit the total amount of checkpoints. Deletes older checkpoints to stay within limit."},
    )
    metric_for_best_model: str = field(
        default="eval_loss",
        metadata={"help": "The metric to use to compare two models ('eval_loss', etc.)."},
    )

    dtype: str = field(
        default="fp16",
        metadata={"help": "The data type to use for training (e.g., 'fp16', 'bf16')."},
    )

    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether to use gradient checkpointing."},
    )
    gradient_checkpointing_kwargs: Optional[GradientCheckpointingKwargs] = field(
        default_factory=GradientCheckpointingKwargs,
        metadata={"help": "Arguments for gradient checkpointing."},
    )

    torch_compile: bool = field(
        default=True,
        metadata={"help": "Whether to compile the model with `torch.compile`."},
    )
    include_tokens_per_second: bool = field(
        default=True,
        metadata={"help": "Whether to include tokens per second in logs."},
    )
    include_num_input_tokens_seen: bool = field(
        default=True,
        metadata={"help": "Whether to include the number of input tokens seen in logs."},
    )
    average_tokens_across_devices: bool = field(
        default=True,
        metadata={"help": "Whether to average tokens across devices in distributed training."},
    )

    disable_tqdm: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to disable the tqdm progress bar."},
    )
    fsdp_config: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "FSDP configuration dictionary."},
    )
    deepspeed_config: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "DeepSpeed configuration dictionary."},
    )
    accelerator_config: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "Accelerate configuration dictionary."},
    )
    ddp_config: Optional[DdpConfig] = field(
        default_factory=DdpConfig,
        metadata={"help": "DDP configuration dictionary."},
    )
    use_cpu: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to explicitly run training on CPU."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a checkpoint to resume training from."},
    )
    restore_callback_states_from_checkpoint: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to restore callback states from checkpoint."},
    )


@dataclass
class MasterConfig:
    """Main training configuration."""

    model: ModelConfig = field(default_factory=ModelConfig, metadata={"help": "Configuration for the model."})

    dataset: DatasetConfig = field(default_factory=DatasetConfig, metadata={"help": "Configuration for the dataset."})

    optimizers: OptimizerConfig = field(
        default_factory=OptimizerConfig, metadata={"help": "Configuration for optimizers."}
    )

    scheduler: SchedulerConfig = field(
        default_factory=SchedulerConfig, metadata={"help": "Configuration for the learning rate scheduler."}
    )

    callbacks: CallbackConfig = field(default_factory=CallbackConfig, metadata={"help": "Configuration for callbacks."})

    training: TrainingConfig = field(
        default_factory=TrainingConfig, metadata={"help": "Configuration for training parameters."}
    )

    extra_params: Dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Additional top-level parameters not explicitly defined."}
    )


def parse_arguments(config_path: Optional[str] = None) -> MasterConfig:
    """Create argument parser for the new finetuning interface."""
    parser = HfArgumentParser(MasterConfig)

    if config_path:
        config_path = os.path.abspath(config_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not (config_path.endswith(".yaml") or config_path.endswith(".yml")):
            raise ValueError(f"Expected a .yaml/.yml file, got: {config_path}")

        try:
            (master_config,) = parser.parse_yaml_file(yaml_file=config_path)
            return master_config
        except Exception as e:
            raise ValueError(f"Failed to parse YAML config '{config_path}': {e}")

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        master_config = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))[0]
    else:
        master_config = parser.parse_args_into_dataclasses()

    return master_config


class ConfigManager:
    """Manages configuration loading, validation, and updates."""

    def __init__(self, config: MasterConfig):
        """
        Initialize ConfigManager with either:
        - Path to config file (str or Path)
        - Configuration dictionary
        - None (creates empty config)
        """
        self.config = config

    def load_config(self, config_path: Union[str, Path]) -> None:
        """Load configuration from file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if config_path.suffix.lower() in [".yaml", ".yml"]:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
        elif config_path.suffix.lower() == ".json":
            with open(config_path, "r") as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

        self.update_config(config_dict)

    def update_config(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with dictionary values."""
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                if isinstance(value, dict) and hasattr(getattr(self.config, key), "__dataclass_fields__"):
                    # Special handling for callbacks
                    if key in ["callbacks", "optimizers", "loss_functions"]:
                        nested_config = getattr(self.config, key)
                        for component_name, component_dict in value.items():
                            if isinstance(component_dict, dict):
                                getattr(nested_config, key)[component_name] = component_dict
                            else:
                                getattr(nested_config, "extra_params")[component_name] = nested_config.extra_params[
                                    component_name
                                ] = component_dict
                    else:
                        # Update nested dataclass
                        nested_config = getattr(self.config, key)
                        for nested_key, nested_value in value.items():
                            if hasattr(nested_config, nested_key):
                                setattr(getattr(self.config, key), nested_key, nested_value)
                            elif hasattr(nested_config, "extra_params"):
                                getattr(getattr(self.config, key), "extra_params")[nested_key] = nested_value
                else:
                    setattr(self.config, key, value)
            else:
                # Store unknown parameters in extra_params
                self.config.extra_params[key] = value

    def save_config(self, output_path: Union[str, Path]) -> None:
        """Save current configuration to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.config

        if output_path.suffix.lower() in [".yaml", ".yml"]:
            with open(output_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif output_path.suffix.lower() == ".json":
            with open(output_path, "w") as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported output file format: {output_path.suffix}")

    def validate_config(self) -> None:
        """Validate configuration parameters."""
        errors = []

        # Validate model configuration
        if not self.config.model.model_name:
            errors.append("Model name is required")

        # Validate dataset configuration
        if not self.config.dataset.dataset_name:
            errors.append("Dataset name is required")

        # Validate training parameters
        if self.config.dataset.train_batch_size <= 0:
            errors.append("Train batch size must be positive")

        if self.config.dataset.eval_batch_size <= 0:
            errors.append("Validation batch size must be positive")

        if self.config.training.num_train_epochs <= 0:
            errors.append("Number of epochs must be positive")

        if self.config.training.gradient_accumulation_steps <= 0:
            errors.append("Gradient accumulation steps must be positive")

        # Validate device configuration
        valid_devices = ["cpu", "cuda", "qaic"]
        if self.config.training.device not in valid_devices:
            errors.append(f"Device must be one of {valid_devices}")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))

    def get_callback_config(self) -> Dict[str, Any]:
        """Get callback configuration as dictionary."""
        return self.config.callbacks

    def get_optimizer_config(self) -> Dict[str, Any]:
        """Get optimizer configuration as dictionary."""
        return self.config.optimizers

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration as dictionary."""
        return self.config.training

    def get_scheduler_config(self) -> Dict[str, Any]:
        """Get scheduler configuration as dictionary."""
        return self.config.scheduler

    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration as dictionary."""
        return self.config.dataset

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration as dictionary."""
        return self.config.model

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self.config)

    def __getattr__(self, name: str) -> Any:
        """Allow direct access to config attributes."""
        if hasattr(self.config, name):
            return getattr(self.config, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


def create_trainer_config(name: str, **dependencies) -> tuple:
    """
    Create trainer configuration based on registered trainer modules.

    Args:
        name: Name of the trainer type
        **dependencies: Any dependencies needed to configure the trainer

    Returns:
        tuple: (trainer_class, args_class, additional_kwargs)
    """
    config = registry.get_trainer_module(name)

    # Process required kwargs based on available dependencies
    additional_kwargs = {}
    for kwarg, default in config["required_kwargs"].items():
        if kwarg in dependencies:
            additional_kwargs[kwarg] = dependencies[kwarg]
        elif default != "REQUIRED":
            additional_kwargs[kwarg] = default

    # Check for missing required arguments
    for kwarg, default in config["required_kwargs"].items():
        if kwarg not in additional_kwargs and default == "REQUIRED":
            raise ValueError(f"Required argument '{kwarg}' not provided for trainer '{name}'")

    return config["trainer_cls"], config["args_cls"], additional_kwargs
