# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
from typing import Optional

from peft import get_peft_model
from transformers import Trainer, TrainingArguments

from QEfficient.finetune.experimental.core.component_registry import registry
from QEfficient.finetune.experimental.core.config_manager import PeftConfig


@registry.trainer_module(name="base", args_cls=TrainingArguments, required_kwargs={"peft_config": PeftConfig})
class BaseTrainer(Trainer):
    """
    Extended Trainer class that supports PEFT (Parameter-Efficient Fine-Tuning).

    This trainer extends the standard HuggingFace Trainer to optionally apply
    PEFT configurations to the model before training.
    """

    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        peft_config: Optional[PeftConfig] = None,
        **kwargs,
    ):
        """
        Initialize the BaseTrainer with optional PEFT support.

        Args:
            model: The model to train
            args: Training arguments
            data_collator: Data collator for batching
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            processing_class: Tokenizer or processor
            model_init: Function to initialize model
            compute_metrics: Function to compute metrics
            callbacks: List of callbacks
            optimizers: Tuple of (optimizer, scheduler)
            preprocess_logits_for_metrics: Function to preprocess logits
            peft_config: Optional PEFT configuration. If provided, the model will be
                        wrapped with PEFT before training.
            **kwargs: Additional keyword arguments
        """
        # Apply PEFT to model if peft_config is provided
        if peft_config is not None and model is not None:
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        # Initialize the parent Trainer class
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **kwargs,
        )
