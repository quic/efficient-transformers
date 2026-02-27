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
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from transformers.hf_argparser import HfArgumentParser

from QEfficient.finetune.experimental.core.logger import Logger

logger = Logger(__name__)


@dataclass
class OptimizerConfig:
    """Configuration for optimizers."""

    optimizer_name: str = field(
        default="AdamW",
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
    warmup_ratio: int = field(
        default=0.1,
        metadata={"help": "ratio of total training steps for the warmup phase. value is within [0-1) range."},
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
        default="yahma/alpaca-cleaned",
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
    input_columns: List[str] = field(
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
    prompt_template: str = field(
        default=None,
        metadata={"help": "Template for formatting prompts (e.g., 'User: {input} Assistant: ')."},
    )
    prompt_func: str = field(
        default=None,
        metadata={"help": "Function for formatting prompts (e.g., 'User: {input} Assistant: ')."},
    )
    completion_template: str = field(
        default=None,
        metadata={"help": "Template for formatting output completions (e.g., '{output}')."},
    )
    completion_func: str = field(
        default=None,
        metadata={"help": "Function for formatting output completions (e.g., '{output}')."},
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
    config_name: str = field(
        default="default",
        metadata={"help": "Name of the hf configuration file."},
    )
    json_file_path: str = field(default=None, metadata={"help": "Path to a JSON file containing data."})


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
    target_modules: List[str] = field(
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
        default=None,
        metadata={"help": "The DDP backend to use (e.g., 'nccl', 'gloo', 'qccl')."},
    )
    ddp_find_unused_parameters: bool = field(
        default=False,
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
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether to use gradient checkpointing."},
    )
    gradient_checkpointing_kwargs: Optional[GradientCheckpointingKwargs] = field(
        default_factory=GradientCheckpointingKwargs,
        metadata={"help": "Arguments for gradient checkpointing."},
    )
    device: str = field(
        default="qaic",
        metadata={"help": "The device to use for training ('cuda', 'cpu', etc.)."},
    )
    torch_dtype: str = field(
        default="fp16",
        metadata={"help": "The torch data type to use for model weights (e.g., 'fp32', 'fp16', 'bf16')."},
    )
    torch_compile: bool = field(
        default=False,
        metadata={"help": "Whether to compile the model with `torch.compile`."},
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
        default=False,
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
    report_to: Optional[List[str]] = field(
        default=None,
        metadata={"help": "The list of integrations to report the results and logs to."},
    )
    completion_only_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to compute loss only on completion tokens."},
    )
    pp_degree: int = field(
        default=1,
        metadata={"help": "Pipeline parallelism degree (number of pipeline stages). Set > 1 to enable PP."},
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


class ConfigManager:
    """Manages configuration loading, validation, and updates."""

    def __init__(self, config: Optional[MasterConfig] = None, config_path: Optional[str] = None):
        """
        Initialize ConfigManager with either:
        - Path to config file (str or Path)
        - Configuration dictionary
        """
        if config:
            self.config = config
        else:
            self.config = MasterConfig()

        if config_path and not config:
            logger.log_rank_zero("Loading configuration from config_path...")
            config_path = os.path.abspath(config_path)
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            if not (config_path.endswith(".yaml") or config_path.endswith(".yml")):
                raise ValueError(f"Expected a .yaml/.yml file, got: {config_path}")
            try:
                self.load_config(config_path)
            except Exception as e:
                raise ValueError(f"Failed to parse YAML config '{config_path}': {e}")

        elif config and not config_path:
            logger.log_rank_zero("Loading configuration from config object...")

        elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            logger.log_rank_zero("Loading configuration from config_path from CLI...")
            config_path = os.path.abspath(sys.argv[1])
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            try:
                self.load_config(config_path)
            except Exception as e:
                raise ValueError(f"Failed to parse YAML config '{config_path}': {e}")

        elif len(sys.argv) > 2:
            logger.log_rank_zero("Loading configuration flags from CLI...")
            parser = HfArgumentParser(
                (
                    TrainingConfig,
                    ModelConfig,
                    DatasetConfig,
                    OptimizerConfig,
                    SchedulerConfig,
                    CallbackConfig,
                    PeftConfig,
                    DdpConfig,
                    GradientCheckpointingKwargs,
                )
            )
            train_args, model_args, data_args, opt_args, schd_args, call_args, peft_args, ddp_args, gck_args, extra = (
                parser.parse_args_into_dataclasses(return_remaining_strings=True)
            )
            train_args.ddp_config = ddp_args
            train_args.gradient_checkpointing_kwargs = gck_args
            model_args.peft_config = peft_args
            self.config = MasterConfig(
                model=model_args,
                dataset=data_args,
                training=train_args,
                callbacks=call_args,
                optimizers=opt_args,
                scheduler=schd_args,
                extra_params=extra,
            )

        else:
            logger.log_rank_zero("Using default configuration...")
        self.config = asdict(self.config)
        self.config = MasterConfig(**self.config)
        # Validate loaded config
        try:
            self.validate_config()
        except Exception as e:
            logger.log_rank_zero(f"Config validation failed with error: {e}")

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

    def _ensure_extra_params(self, obj) -> Dict[str, Any]:
        """Ensure obj.extra_params exists and is a dict; return it."""
        ep = getattr(obj, "extra_params", None)
        if ep is None:
            setattr(obj, "extra_params", {})
            ep = obj.extra_params
        if not isinstance(ep, dict):
            raise TypeError("extra_params must be a dict.")
        return ep

    def _stash_top_level_extra(self, section: str, nested_key: str, value: Any) -> None:
        """Store unknown nested values under MasterConfig.extra_params['section.nested_key']."""
        ep = self._ensure_extra_params(self.config)
        ep[f"{section}.{nested_key}"] = value

    def update_config(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with dictionary values."""

        SPECIAL_KEYS = {"callbacks"}

        for key, value in config_dict.items():
            if hasattr(self.config, key):
                target = getattr(self.config, key)

                # Special handling for callbacks (dict inside CallbackConfig)
                if key in SPECIAL_KEYS and isinstance(value, dict):
                    if is_dataclass(target) and hasattr(target, "callbacks") and isinstance(target.callbacks, dict):
                        for component_name, component_cfg in value.items():
                            target.callbacks[component_name] = component_cfg
                    elif isinstance(target, dict):
                        target.update(value)
                    else:
                        self._stash_top_level_extra(key, "__all__", value)
                    continue

                if isinstance(value, dict) and is_dataclass(target):
                    known = {f.name for f in fields(target)}
                    for nested_key, nested_value in value.items():
                        if nested_key in known:
                            setattr(target, nested_key, nested_value)
                        else:
                            self._stash_top_level_extra(key, nested_key, nested_value)
                    continue

                if isinstance(value, dict) and isinstance(target, dict):
                    target.update(value)
                    continue
                setattr(self.config, key, value)

            else:
                ep = self._ensure_extra_params(self.config)
                ep[key] = value

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

    def _push(self, errs: List[str], cond: bool, msg: str) -> None:
        """Append msg to errs if cond is True."""
        if cond:
            errs.append(msg)

    def validate_config(self) -> None:
        """
        Validate configuration parameters for MasterConfig.
        """
        cfg = self.config
        errors: List[str] = []

        model = getattr(cfg, "model", {})
        dataset = getattr(cfg, "dataset", {})
        training = getattr(cfg, "training", {})

        # ---------- Model ----------
        self._push(errors, not model.get("model_name"), "model.model_name is required.")
        # Device
        valid_devices = ["cpu", "cuda", "qaic"]
        training_device = model.get("device", "qaic")
        if training_device not in valid_devices:
            self._push(errors, training_device not in valid_devices, f"training.device must be one of {valid_devices}.")
        # PEFT validation
        if model.get("use_peft"):
            pc = model.get("peft_config", {})
            self._push(errors, not isinstance(pc, dict), "model.peft_config must be a dict when use_peft=True.")
            if isinstance(pc, dict):
                self._push(
                    errors,
                    not isinstance(pc.get("lora_r", 0), int) or pc.get("lora_r", 0) <= 0,
                    "model.peft_config.lora_r must be a positive integer.",
                )
                self._push(
                    errors,
                    not isinstance(pc.get("lora_alpha", 0), int) or pc.get("lora_alpha", 0) <= 0,
                    "model.peft_config.lora_alpha must be a positive integer.",
                )
                self._push(
                    errors,
                    not (0.0 <= float(pc.get("lora_dropout", 0.0)) < 1.0),
                    "model.peft_config.lora_dropout must be in [0,1).",
                )

        # ---------- Dataset ----------
        self._push(errors, not dataset.get("dataset_name"), "dataset.dataset_name is required.")
        self._push(errors, not dataset.get("tokenizer_name"), "dataset.tokenizer_name is required.")

        # ---------- Training ----------
        # torch_dtype validation
        torch_dtype = training.get("torch_dtype")
        valid_dtypes = {"fp16", "bf16", "fp32"}
        self._push(
            errors,
            not torch_dtype,
            "training.torch_dtype is required.",
        )
        self._push(
            errors,
            torch_dtype and torch_dtype not in valid_dtypes,
            f"training.torch_dtype must be one of {valid_dtypes}.",
        )

        # Batch sizes
        self._push(
            errors,
            training.get("per_device_train_batch_size", 1) <= 0,
            "training.per_device_train_batch_size must be positive.",
        )
        self._push(
            errors,
            training.get("per_device_eval_batch_size", 1) <= 0,
            "training.per_device_eval_batch_size must be positive.",
        )

        # Epochs / steps
        n_epochs = training.get("num_train_epochs", 1)
        self._push(
            errors,
            n_epochs <= 0,
            "Either training.num_train_epochs > 0  must be set.",
        )

        # Gradient accumulation
        self._push(
            errors,
            training.get("gradient_accumulation_steps", 1) <= 0,
            "training.gradient_accumulation_steps must be positive.",
        )

        # Logging / saving configs
        self._push(errors, training.get("logging_steps", 0) < 0, "training.logging_steps must be >= 0.")
        self._push(errors, training.get("save_total_limit", 0) < 0, "training.save_total_limit must be >= 0.")

        # Pipeline Parallelism (PP) config
        pp_degree = training.get("pp_degree", 1)
        self._push(
            errors,
            not isinstance(pp_degree, int) or pp_degree < 1,
            "training.pp_degree must be a positive integer (default 1 = no PP; > 1 enables PP).",
        )

        # DDP config
        ddp = training.get("ddp_config", {})
        if isinstance(ddp, dict):
            backend = ddp.get("ddp_backend")
            # Accept qccl for Qualcomm, nccl for CUDA, gloo for CPU
            self._push(
                errors,
                backend not in {"qccl", "nccl", "gloo", None},
                "training.ddp_config.ddp_backend must be one of {'qccl','nccl','gloo'} or omitted.",
            )

        # ---------- Final ----------
        if errors:
            # Join messages with bullet points for readability
            raise ValueError("Configuration validation failed:\n- " + "\n- ".join(errors))

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
        """
        Get model configuration as dictionary.

        Automatically handles torch_dtype conversion from training config if not set in model config.
        """
        model_config = self.config.model

        # Get torch_dtype from training config and convert
        # To do: check if it can be moved from training config to model config instead
        if model_config.get("torch_dtype") is None:
            training_config = self.get_training_config()
            training_dtype = training_config.get("torch_dtype")
            if training_dtype:
                # Convert from training format (fp16/bf16) to model format (float16/bfloat16)
                dtype_mapping = {"fp16": "float16", "bf16": "bfloat16"}
                model_config["torch_dtype"] = dtype_mapping.get(training_dtype, "auto")

        return model_config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self.config)

    def __getattr__(self, name: str) -> Any:
        """Allow direct access to config attributes."""
        if hasattr(self.config, name):
            return getattr(self.config, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
