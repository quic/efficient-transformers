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
from typing import Any, Dict, List, Optional, Tuple

from QEfficient.finetune.experimental.core.callbacks import create_callbacks, replace_progress_callback
from QEfficient.finetune.experimental.core.component_registry import ComponentFactory, registry
from QEfficient.finetune.experimental.core.utils.peft_utils import convert_peft_config_to_lora_config
from QEfficient.finetune.experimental.core.utils.training_config_utils import prepare_training_config
from QEfficient.finetune.experimental.core.config_manager import (
    ConfigManager,
    MasterConfig,
    PeftConfig,
    create_trainer_config,
    parse_arguments,
)
from QEfficient.finetune.experimental.core.optimizer import prepare_optimizer
from QEfficient.finetune.experimental.core.logger import Logger

logger = Logger(__name__)


class FineTuningPipeline:
    """
    Main pipeline class for fine-tuning LLMs.
    """

    def __init__(self, config: MasterConfig):
        """
        Initialize the fine-tuning pipeline with configuration.

        Args:
            config: Master configuration object containing all training parameters
        """
        self.config = config
        self.config_manager = ConfigManager(config)
        self.output_dir = Path(config.training.output_dir)
        self._setup_environment()

    def _setup_environment(self) -> None:
        """Set up environment variables for output directories."""
        os.environ["OUTPUT_DIR"] = str(self.output_dir)
        os.environ["TRACKIO_DIR"] = str(self.output_dir / "trackio_logs")
        os.environ["TENSORBOARD_LOGGING_DIR"] = str(self.output_dir)

    def _prepare_training_config(self) -> Dict[str, Any]:
        """
        Prepare and validate training configuration.

        Returns:
            Dictionary of training arguments ready for trainer initialization
        """
        return prepare_training_config(
            config_manager=self.config_manager,
            include_num_input_tokens_seen=False,
            use_cpu=False,
        )

    def _create_datasets(self) -> Tuple[Any, Any]:
        """
        Create training and evaluation datasets.

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        dataset_config = self.config_manager.get_dataset_config()
        dataset_type = dataset_config.get("dataset_type")
        seed = self.config.training.seed

        # Helper function to create a dataset for a specific split
        def create_dataset_for_split(split: str) -> Any:
            # Extract split-specific configuration
            split_key = f"{split}_split" if split == "test" else "train_split"
            split_name = dataset_config.get(split_key, split)

            return ComponentFactory.create_dataset(
                dataset_type=dataset_type,
                dataset_name=dataset_config["dataset_name"],
                split=split_name,
                seed=seed,
                **dataset_config,
            )

        train_dataset = create_dataset_for_split("train")
        eval_dataset = create_dataset_for_split("test")

        return train_dataset, eval_dataset

    def _create_model(self) -> Any:
        """
        Create and load the model instance.

        Returns:
            Model instance with loaded model and tokenizer
        """
        # Get model config as dict and create mutable copy to avoid mutating original
        model_config = dict(self.config_manager.get_model_config())
        model_type = model_config.pop("model_type")
        model_name = model_config.pop("model_name")

        # Handle dtype conversion for model
        training_config_dict = self.config_manager.get_training_config()
        model_dtype = training_config_dict.get("dtype")
        model_config["dtype"] = {"fp16": "float16", "bf16": "bfloat16"}.get(model_dtype, "auto")

        model_instance = ComponentFactory.create_model(model_type, model_name, **model_config)
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
        for callback_name, callback_kwargs in callback_config.callbacks.items():
            try:
                callback_instance = create_callbacks(callback_name, **callback_kwargs)
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

        # Get trainer configuration
        dependencies = {}
        if peft_config is not None:
            dependencies["peft_config"] = peft_config

        trainer_cls, args_cls, additional_kwargs = create_trainer_config(trainer_type, **dependencies)

        # Create trainer arguments instance
        args = args_cls(**training_config)

        # Initialize trainer
        trainer = trainer_cls(
            model=model,
            processing_class=tokenizer,
            args=args,
            compute_loss_func=None,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
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
        training_config = self._prepare_training_config()

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
    # Parse arguments/config
    master_config = parse_arguments()

    # Create and run pipeline
    pipeline = FineTuningPipeline(master_config)
    pipeline.run()


if __name__ == "__main__":
    main()
