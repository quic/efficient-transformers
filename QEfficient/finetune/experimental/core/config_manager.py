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
import logging
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Union

import yaml
from transformers.hf_argparser import HfArgumentParser

from QEfficient.finetune.experimental.core.logger import Logger
from QEfficient.finetune.experimental.core.utils import constants
from QEfficient.finetune.experimental.core.utils.dist_utils import is_main_process
from QEfficient.utils.device_utils import is_nsp_free

logger = Logger(__name__)


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
        default="sft_dataset",
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
    dataset_num_samples: int = field(
        default=-1,
        metadata={"help": "Number of samples to use from the dataset. -1 means all samples."},
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
        default=None,
        metadata={"help": "The collation function to use (e.g., 'dynamic_padding')."},
    )
    dataset_disc_style: str = field(
        default=None,
        metadata={"help": "Style of dataset"},
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
    remove_samples_with_empty_columns: bool = field(
        default=True,
        metadata={"help": "Whether to remove samples with empty columns."},
    )
    config_name: str = field(
        default="default",
        metadata={"help": "Name of the hf configuration file."},
    )
    json_file_path: str = field(default=None, metadata={"help": "Path to a JSON file containing data."})
    data_seed: int = field(default=42, metadata={"help": "Seed for data shuffling and sampling."})


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
    torch_dtype: str = field(
        default="fp16",
        metadata={"help": "The torch data type to use for model weights (e.g., 'fp32', 'fp16', 'bf16')."},
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
    use_reentrant: bool = field(
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
    log_file_name: str = field(
        default=None,
        metadata={"help": "The log_file output name."},
    )
    # overwrite_output_dir: bool = field(
    #     default=False,
    #     metadata={"help": "Whether to overwrite the output directory."},
    # )
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
    fp16: bool = field(
        default=False,
        metadata={"help": "Enable fp16 mixed precision/autocast (GradScaler path)."},
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Enable bf16 mixed precision/autocast."},
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
        default="tensorboard",
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
    tp_degree: int = field(
        default=1,
        metadata={"help": "Tensor parallelism degree (number of pipeline stages). Set > 1 to enable TP."},
    )
    ddp_degree: int = field(
        default=1,
        metadata={"help": "Data parallelism degree (number of pipeline stages). Set > 1 to enable DDP."},
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

    def __init__(
        self,
        config: Optional[MasterConfig] = None,
        config_path: Optional[str] = None,
        cli_args: Optional[List[str]] = None,
    ):
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
            self._load_config_file(config_path)
            if cli_args:
                self._apply_cli_overrides(cli_args)

        elif config and not config_path:
            logger.log_rank_zero("Loading configuration from config object...")

        elif cli_args is not None:
            self._load_from_cli_args(cli_args)
        else:
            logger.log_rank_zero("Using default configuration...")
        self.config = asdict(self.config)
        self.config = MasterConfig(**self.config)
        # Validate loaded config
        try:
            self.validate_config()
        except Exception as e:
            logger.log_rank_zero(f"Config validation failed with error: {e}")

    def _build_cli_parser(self) -> HfArgumentParser:
        return HfArgumentParser(
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

    @staticmethod
    def _looks_like_config_path(arg: str) -> bool:
        return bool(arg) and arg.endswith((".yaml", ".yml", ".json"))

    @staticmethod
    def _provided_cli_keys(cli_args: List[str]) -> Set[str]:
        keys: Set[str] = set()
        for token in cli_args:
            if not token.startswith("--"):
                continue
            key = token[2:]
            if not key:
                continue
            keys.add(key.split("=", 1)[0].replace("-", "_"))
        return keys

    def _load_config_file(self, config_path: Union[str, Path]) -> None:
        config_path = os.path.abspath(str(config_path))
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not self._looks_like_config_path(config_path):
            raise ValueError(f"Expected a .yaml/.yml/.json file, got: {config_path}")
        try:
            self.load_config(config_path)
        except Exception as e:
            raise ValueError(f"Failed to parse YAML config '{config_path}': {e}")

    def _load_from_cli_args(self, cli_args: List[str]) -> None:
        if not cli_args:
            logger.log_rank_zero("Using default configuration...")
            return

        if self._looks_like_config_path(cli_args[0]):
            logger.log_rank_zero("Loading configuration from config_path from CLI...")
            self._load_config_file(cli_args[0])
            if len(cli_args) > 1:
                self._apply_cli_overrides(cli_args[1:])
            return

        logger.log_rank_zero("Loading configuration flags from CLI...")
        self._apply_cli_overrides(cli_args)

    def _apply_cli_overrides(self, cli_args: List[str]) -> None:
        parser = self._build_cli_parser()
        train_args, model_args, data_args, opt_args, schd_args, call_args, peft_args, ddp_args, gck_args, extra = (
            parser.parse_args_into_dataclasses(args=cli_args, return_remaining_strings=True, look_for_args_file=False)
        )

        provided_keys = self._provided_cli_keys(cli_args)
        updates: Dict[str, Any] = {}

        def add_section(section_name: str, dataclass_obj: Any, excluded: Optional[Set[str]] = None) -> None:
            excluded = excluded or set()
            section_updates: Dict[str, Any] = {}
            for f in fields(dataclass_obj):
                if f.name in excluded:
                    continue
                if f.name in provided_keys:
                    section_updates[f.name] = getattr(dataclass_obj, f.name)
            if section_updates:
                updates[section_name] = section_updates

        add_section("training", train_args, excluded={"ddp_config", "gradient_checkpointing_kwargs"})
        add_section("model", model_args, excluded={"peft_config"})
        add_section("dataset", data_args)
        add_section("optimizers", opt_args)
        add_section("scheduler", schd_args)
        add_section("callbacks", call_args)

        peft_updates = {f.name: getattr(peft_args, f.name) for f in fields(peft_args) if f.name in provided_keys}
        if peft_updates:
            updates.setdefault("model", {})
            updates["model"]["peft_config"] = peft_updates

        ddp_updates = {f.name: getattr(ddp_args, f.name) for f in fields(ddp_args) if f.name in provided_keys}
        if ddp_updates:
            updates.setdefault("training", {})
            updates["training"]["ddp_config"] = ddp_updates

        gck_updates = {f.name: getattr(gck_args, f.name) for f in fields(gck_args) if f.name in provided_keys}
        if gck_updates:
            updates.setdefault("training", {})
            updates["training"]["gradient_checkpointing_kwargs"] = gck_updates

        if updates:
            self.update_config(updates)

        if extra:
            self._ensure_extra_params(self.config)["cli_remaining_args"] = extra

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

    def _merge_dataclass_inplace(self, dc_obj: Any, updates: Dict[str, Any], parent_path: str = "") -> None:
        """
        Recursively merge 'updates' (dict) into the dataclass instance 'dc_obj',
        preserving defaults by updating nested dataclasses/dicts in place.
        """
        if not is_dataclass(dc_obj):
            raise TypeError("dc_obj must be a dataclass instance")
        field_names = {f.name for f in fields(dc_obj)}
        for key, value in updates.items():
            path = f"{parent_path}.{key}" if parent_path else key

            if key not in field_names:
                self._stash_top_level_extra(parent_path or "__root__", key, value)
                continue

            current = getattr(dc_obj, key)

            # Case A: current is dataclass, incoming is dict -> deep merge
            if is_dataclass(current) and isinstance(value, Mapping):
                self._merge_dataclass_inplace(current, value, path)

            # Case B: both dicts -> shallow update
            elif isinstance(current, dict) and isinstance(value, Mapping):
                current.update(value)

            # Case C: both lists -> by default replace; switch to extend if desired
            elif isinstance(current, list) and isinstance(value, list):
                setattr(dc_obj, key, value)

            # Case D: simple assignment
            else:
                setattr(dc_obj, key, value)

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
                self._merge_dataclass_inplace(target, value, parent_path=key)
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
        if training_device == "qaic":
            try:
                import torch_qaic  # noqa: F401

                logger.log_rank_zero("torch_qaic package found. Using QAIC devices...")
                if is_main_process():
                    is_nsp_free()

            except ImportError as e:
                logger.log_rank_zero(
                    f"Unable to import 'torch_qaic' package due to exception: {e}. Moving ahead without the torch_qaic extension.",
                    logging.WARNING,
                )
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

        # ---------- Model + Training ----------
        # model.torch_dtype validation
        torch_dtype = model.get("torch_dtype")
        valid_dtypes = {"fp16", "bf16", "fp32"}
        self._push(
            errors,
            not torch_dtype,
            "model.torch_dtype is required.",
        )
        self._push(
            errors,
            torch_dtype and torch_dtype not in valid_dtypes,
            f"model.torch_dtype must be one of {valid_dtypes}.",
        )
        self._push(
            errors,
            bool(training.get("fp16", False)) and bool(training.get("bf16", False)),
            "At most one of training.fp16 and training.bf16 can be True.",
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

        # Pipeline / Tensor / Data parallelism config
        pp_degree = training.get("pp_degree", 1)
        tp_degree = training.get("tp_degree", 1)
        ddp_degree = training.get("ddp_degree", 1)

        self._push(
            errors,
            not isinstance(pp_degree, int) or pp_degree < 1,
            "training.pp_degree must be a positive integer (default 1 = no PP; > 1 enables PP).",
        )
        self._push(
            errors,
            not isinstance(tp_degree, int) or tp_degree < 1,
            "training.tp_degree must be a positive integer (default 1 = no TP; > 1 enables TP).",
        )
        self._push(
            errors,
            not isinstance(ddp_degree, int) or ddp_degree < 1,
            "training.ddp_degree must be a positive integer (default 1 = no DDP; > 1 enables DDP).",
        )

        # Supported modes:
        #  - PP only
        #  - DDP only (single-server / multi-server)
        #  - TP only (single-server)
        #  - TP + DDP (single-server)
        if isinstance(pp_degree, int) and isinstance(tp_degree, int) and isinstance(ddp_degree, int):
            self._push(
                errors,
                pp_degree > 1 and tp_degree > 1,
                "Unsupported parallelism combination: TP cannot be combined with PP. "
                "Supported modes are PP only, DDP only, TP only, or TP+DDP (single-server).",
            )
            self._push(
                errors,
                pp_degree > 1 and ddp_degree > 1,
                "Unsupported parallelism combination: DDP cannot be combined with PP. "
                "Supported modes are PP only, DDP only, TP only, or TP+DDP (single-server).",
            )

        # Launcher world-size metadata (if present in environment).
        local_world_size = -1
        if "LOCAL_WORLD_SIZE" in os.environ:
            try:
                local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            except ValueError:
                local_world_size = -1
            self._push(
                errors,
                local_world_size < 1,
                f"Invalid LOCAL_WORLD_SIZE={os.environ.get('LOCAL_WORLD_SIZE')!r}; expected a positive integer.",
            )

        world_size = -1
        if "WORLD_SIZE" in os.environ:
            try:
                world_size = int(os.environ["WORLD_SIZE"])
            except ValueError:
                world_size = -1
            self._push(
                errors,
                world_size < 1,
                f"Invalid WORLD_SIZE={os.environ.get('WORLD_SIZE')!r}; expected a positive integer.",
            )

        # LOCAL_WORLD_SIZE consistency checks (single-node process count).
        # Only enforce degree product when TP/PP is active; for pure DDP we rely
        # on launcher-provided world sizes and do not force ddp_degree matching.
        if (
            local_world_size > 0
            and isinstance(pp_degree, int)
            and isinstance(tp_degree, int)
            and isinstance(ddp_degree, int)
            and (pp_degree > 1 or tp_degree > 1)
        ):
            expected_world_size = pp_degree * tp_degree * ddp_degree
            self._push(
                errors,
                expected_world_size != local_world_size,
                "Parallelism degree mismatch: pp_degree * tp_degree * ddp_degree "
                f"must equal LOCAL_WORLD_SIZE ({pp_degree} * {tp_degree} * {ddp_degree} = {expected_world_size}, "
                f"LOCAL_WORLD_SIZE={local_world_size}).",
            )

        # WORLD_SIZE product checks are required only when TP/PP is active.
        if (
            world_size > 0
            and isinstance(pp_degree, int)
            and isinstance(tp_degree, int)
            and isinstance(ddp_degree, int)
            and (pp_degree > 1 or tp_degree > 1)
        ):
            expected_world_size = pp_degree * tp_degree * ddp_degree
            self._push(
                errors,
                expected_world_size != world_size,
                "Parallelism degree mismatch for TP/PP modes: pp_degree * tp_degree * ddp_degree "
                f"must equal WORLD_SIZE ({pp_degree} * {tp_degree} * {ddp_degree} = {expected_world_size}, "
                f"WORLD_SIZE={world_size}).",
            )

        if local_world_size > 0 and world_size > 0:
            multi_server = world_size > local_world_size
            self._push(
                errors,
                multi_server and tp_degree > 1,
                "Unsupported parallelism combination: TP and TP+DDP are supported only on a single server. "
                "Detected multi-server launch from WORLD_SIZE > LOCAL_WORLD_SIZE.",
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

        Converts model.torch_dtype from config format (fp16/bf16/fp32)
        to HF format (float16/bfloat16/float32) for from_pretrained kwargs.
        """
        model_config = dict(self.config.model)

        model_dtype = model_config.get("torch_dtype")
        if model_dtype:
            dtype_mapping = constants.DTYPE_MAPPING
            model_config["torch_dtype"] = dtype_mapping.get(model_dtype, "auto")

        return model_config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self.config)

    def __getattr__(self, name: str) -> Any:
        """Allow direct access to config attributes."""
        if hasattr(self.config, name):
            return getattr(self.config, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
