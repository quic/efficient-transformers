# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Main entry point for fine-tuning LLMs using the experimental finetune framework.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from QEfficient.finetune.experimental.core.callbacks import replace_progress_callback
from QEfficient.finetune.experimental.core.component_registry import ComponentFactory
from QEfficient.finetune.experimental.core.config_manager import (
    ConfigManager,
)
from QEfficient.finetune.experimental.core.dataset import SFTDataset  # noqa: F401
from QEfficient.finetune.experimental.core.logger import Logger
from QEfficient.finetune.experimental.core.model import HFModel  # noqa: F401
from QEfficient.finetune.experimental.core.optimizer import prepare_optimizer
from QEfficient.finetune.experimental.core.trainer import sft_trainer  # noqa: F401
from QEfficient.finetune.experimental.core.utils.peft_utils import convert_peft_config_to_lora_config
from QEfficient.finetune.experimental.core.utils.training_config_utils import prepare_training_config

logger = Logger(__name__)

# Try importing QAIC-specific module, proceed without it if it's unavailable
try:
    import torch_qaic  # noqa: F401
except ImportError as e:
    logger.log_rank_zero(
        f"Unable to import 'torch_qaic' package due to exception: {e}. Moving ahead without the torch_qaic extension.",
        level="warning",
    )


class FineTuningPipeline:
    """
    Main pipeline class for fine-tuning LLMs.
    """

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the fine-tuning pipeline with configuration.

        Args:
            config_manager: ConfigManager instance with loaded and validated configuration
        """
        self.config_manager = config_manager
        self.config = self.config_manager.config
        self.output_dir = Path(self.config.training["output_dir"])
        self._setup_environment()

    def _setup_environment(self) -> None:
        """Set up environment variables for output directories."""
        os.environ["OUTPUT_DIR"] = str(self.output_dir)
        os.environ["TRACKIO_DIR"] = str(self.output_dir / "trackio_logs")
        os.environ["TENSORBOARD_LOGGING_DIR"] = str(self.output_dir)

    def _create_datasets(self) -> Tuple[Any, Any]:
        """
        Create training and evaluation datasets.

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        dataset_config = self.config_manager.get_dataset_config()

        dataset_type = dataset_config.get("dataset_type")
        dataset_name = dataset_config.get("dataset_name")
        train_split = dataset_config.get("train_split", "train")
        test_split = dataset_config.get("test_split", "test")
        seed = self.config.training["seed"]

        # Create a copy of dataset_config excluding keys that are passed explicitly
        # to avoid duplicate keyword arguments when unpacking
        excluded_keys = ("dataset_type", "dataset_name", "split", "seed", "train_split", "test_split")
        dataset_config_copy = {k: v for k, v in dataset_config.items() if k not in excluded_keys}

        # Helper function to create a dataset for a specific split
        def create_dataset_for_split(split_name: str) -> Any:
            return ComponentFactory.create_dataset(
                dataset_type=dataset_type,
                dataset_name=dataset_name,
                split=split_name,
                seed=seed,
                **dataset_config_copy,
            )

        # Create training and evaluation datasets using config values
        train_dataset = create_dataset_for_split(train_split)
        eval_dataset = create_dataset_for_split(test_split)

        return train_dataset, eval_dataset

    def _create_model(self) -> Any:
        """
        Create and load the model instance.

        Returns:
            Model instance with loaded model and tokenizer
        """
        # Get model config as dict
        model_config = self.config_manager.get_model_config()

        # Extract required fields
        model_type = model_config.pop("model_type")
        model_name = model_config.pop("model_name")

        # Filter out PEFT-related fields, these shouldn't be passed to model creation
        excluded_keys = {"use_peft", "peft_config"}
        model_config_kwargs = {k: v for k, v in model_config.items() if k not in excluded_keys}

        model_instance = ComponentFactory.create_model(model_type, model_name, **model_config_kwargs)
        return model_instance

    def _create_optimizer(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Create optimizer configuration.

        Returns:
            Tuple of (optimizer_class, optimizer_kwargs)
        """
        optimizer_config = self.config_manager.get_optimizer_config()
        return prepare_optimizer(optimizer_config)

    def _create_callbacks(self) -> List[Any]:
        """
        Create callback instances from configuration.

        Returns:
            List of callback instances
        """
        callback_config = self.config_manager.get_callback_config()
        callbacks = []

        # callback_config.callbacks is a dictionary of callback configurations
        for callback_name, callback_kwargs in callback_config["callbacks"].items():
            try:
                callback_instance = ComponentFactory.create_callback(callback_name, **callback_kwargs)
                callbacks.append(callback_instance)
            except ValueError as e:
                logger.log_rank_zero(f"Warning: Failed to create callback '{callback_name}': {e}", level="warning")

        return callbacks

    def _create_trainer(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
        eval_dataset: Any,
        optimizer_cls_and_kwargs: Tuple[Any, Dict[str, Any]],
        callbacks: List[Any],
        training_config: Dict[str, Any],
    ) -> Any:
        """
        Create and configure the trainer instance.

        Args:
            model: The model to train
            tokenizer: Tokenizer for processing
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            optimizer_cls_and_kwargs: Optimizer class and kwargs tuple
            callbacks: List of callbacks
            training_config: Training configuration dictionary

        Returns:
            Trainer instance
        """
        trainer_type = training_config.pop("type")

        # Get PEFT config if enabled
        model_config_dict = self.config_manager.get_model_config()
        peft_config = None
        if model_config_dict.get("use_peft", False):
            peft_config_dataclass = model_config_dict.get("peft_config")
            if peft_config_dataclass is not None:
                peft_config = convert_peft_config_to_lora_config(peft_config_dataclass)

        # Build dependencies for trainer configuration
        dependencies = {}
        if peft_config is not None:
            dependencies["peft_config"] = peft_config
        trainer_cls, args_cls, additional_kwargs = ComponentFactory.create_trainer_config(trainer_type, **dependencies)

        # Clean up training config: remove fields that shouldn't be passed to TrainingArguments
        training_config.pop("device", None)
        # Note: torch_dtype was already converted to fp16/bf16 flag in prepare_training_config
        training_config.pop("deepspeed_config", None)
        training_config.pop("torch_dtype", None)

        # Create trainer arguments instance
        args = args_cls(**training_config)
        # Initialize trainer
        trainer = trainer_cls(
            model=model,
            processing_class=tokenizer,
            args=args,
            compute_loss_func=None,
            train_dataset=train_dataset.dataset,
            eval_dataset=eval_dataset.dataset,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            callbacks=callbacks,
            **additional_kwargs,
        )

        replace_progress_callback(trainer, callbacks, logger)

        return trainer

    def run(self) -> None:
        """
        Execute the complete fine-tuning pipeline.
        """
        # Validate configuration
        self.config_manager.validate_config()

        # Prepare training configuration
        training_config = prepare_training_config(config_manager=self.config_manager)

        # Create datasets
        logger.log_rank_zero("Creating datasets...")
        train_dataset, eval_dataset = self._create_datasets()

        # Create model and tokenizer
        logger.log_rank_zero("Loading model and tokenizer...")
        model_instance = self._create_model()
        model = model_instance.model
        tokenizer = model_instance.tokenizer

        # Create optimizer
        logger.log_rank_zero("Preparing optimizer...")
        optimizer_cls_and_kwargs = self._create_optimizer()

        # Create callbacks
        logger.log_rank_zero("Creating callbacks...")
        callbacks = self._create_callbacks()

        # Create trainer
        logger.log_rank_zero("Initializing trainer...")
        trainer = self._create_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            callbacks=callbacks,
            training_config=training_config,
        )

        # Start training
        logger.log_rank_zero("Starting training...")
        trainer.train()


def main():
    """
    Main entry point for fine-tuning.

    Parses command-line arguments or config file and runs the fine-tuning pipeline.
    """
    # ConfigManager now handles argument parsing internally via its __init__
    # It will automatically detect and parse:
    # - Command-line args (if len(sys.argv) > 1)
    # - Config file path (if sys.argv[1] ends with .yaml)
    # - Or use defaults if no args provided
    config_manager = ConfigManager()

    # Create and run pipeline - pass ConfigManager directly to avoid redundant wrapping
    pipeline = FineTuningPipeline(config_manager)
    pipeline.run()


if __name__ == "__main__":
    main()
