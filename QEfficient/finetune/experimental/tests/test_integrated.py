# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
End-to-end integration tests for the new experimental finetuning pipeline.
Tests the complete workflow using all components from the core/ directory.
"""

import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional

import pytest
import torch
from datasets import load_dataset

from QEfficient.finetune.experimental.core.component_registry import registry
from QEfficient.finetune.experimental.core.config_manager import (
    ConfigManager,
    DatasetConfig,
    MasterConfig,
    ModelConfig,
    OptimizerConfig,
    PeftConfig,
    SchedulerConfig,
    TrainingConfig,
)
from QEfficient.finetune.experimental.core.dataset import SFTDataset
from QEfficient.finetune.experimental.core.model import HFModel
from QEfficient.finetune.experimental.core.utils.constants import (
    HF_DATASET_ALPACA,
    HF_DATASET_GSM8K,
    HF_DATASET_GSM8K_CONFIG,
    HF_DATASET_IMDB,
    TEST_DATASET_SUBSET_SIZE,
    TEST_LEARNING_RATE,
    TEST_LOGGING_STEPS,
    TEST_LORA_ALPHA,
    TEST_LORA_BIAS,
    TEST_LORA_DROPOUT,
    TEST_LORA_R,
    TEST_LORA_TARGET_MODULES_BERT,
    TEST_LORA_TARGET_MODULES_LLAMA,
    TEST_MAX_SEQ_LENGTH_CAUSAL,
    TEST_MAX_SEQ_LENGTH_SEQ_CLS,
    TEST_MAX_STEPS,
    TEST_MODEL_LLAMA,
    TEST_NUM_HIDDEN_LAYERS,
    TEST_NUM_TRAIN_EPOCHS,
    TEST_PER_DEVICE_BATCH_SIZE,
    TEST_SEED,
    TEST_WARMUP_STEPS,
    TEST_WEIGHT_DECAY,
    AutoClassName,
    DatasetType,
    TaskType,
)
from QEfficient.utils.logging_utils import logger

# ============================================================================
# Test Configuration Dataclasses
# ============================================================================


@dataclass
class TestModelConfig:
    """Dataclass for test model configuration."""

    model_name: str
    task_type: TaskType
    use_peft: bool
    target_modules: list[str]


@dataclass
class TestDatasetConfig:
    """Dataclass for test dataset configuration."""

    dataset_name: str
    hf_dataset_name: str
    hf_dataset_config: Optional[str]
    prompt_template: str
    completion_template: str
    max_seq_length: int


@dataclass
class TestTrainingConfig:
    """Dataclass for test training configuration."""

    max_eval_step: int
    max_train_step: int
    config_name: str


# ============================================================================
# Test Configuration Constants
# ============================================================================

# Model configurations
LLAMA_MODEL_CONFIG = TestModelConfig(
    model_name=TEST_MODEL_LLAMA,
    task_type=TaskType.CAUSAL_LM,
    use_peft=True,
    target_modules=TEST_LORA_TARGET_MODULES_LLAMA,
)

BERT_MODEL_CONFIG = TestModelConfig(
    model_name="google-bert/bert-base-uncased",
    task_type=TaskType.SEQ_CLS,
    use_peft=False,
    target_modules=TEST_LORA_TARGET_MODULES_BERT,
)

# Dataset configurations
GSM8K_DATASET_CONFIG = TestDatasetConfig(
    dataset_name="gsm8k",
    hf_dataset_name=HF_DATASET_GSM8K,
    hf_dataset_config=HF_DATASET_GSM8K_CONFIG,
    prompt_template="Question: {question}\nAnswer: ",
    completion_template="{answer}",
    max_seq_length=TEST_MAX_SEQ_LENGTH_CAUSAL,
)

ALPACA_DATASET_CONFIG = TestDatasetConfig(
    dataset_name="alpaca",
    hf_dataset_name=HF_DATASET_ALPACA,
    hf_dataset_config=None,
    prompt_template="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    completion_template="{output}",
    max_seq_length=TEST_MAX_SEQ_LENGTH_CAUSAL,
)

IMDB_DATASET_CONFIG = TestDatasetConfig(
    dataset_name="imdb",
    hf_dataset_name=HF_DATASET_IMDB,
    hf_dataset_config=None,
    prompt_template="Review: {text}\nSentiment: ",
    completion_template="{label}",
    max_seq_length=TEST_MAX_SEQ_LENGTH_SEQ_CLS,
)


# ============================================================================
# Helper Functions
# ============================================================================


def create_master_config(
    model_config: TestModelConfig,
    dataset_config: TestDatasetConfig,
    output_dir: str,
) -> MasterConfig:
    """
    Create a MasterConfig instance from test configurations.

    Args:
        model_config: Test model configuration
        dataset_config: Test dataset configuration
        output_dir: Output directory for training results

    Returns:
        MasterConfig instance
    """
    # Determine auto_class_name and dataset_type based on task type
    if model_config.task_type == TaskType.CAUSAL_LM:
        auto_class_name = AutoClassName.CAUSAL_LM.value
        dataset_type = DatasetType.SEQ_COMPLETION.value
    elif model_config.task_type == TaskType.SEQ_CLS:
        auto_class_name = AutoClassName.SEQ_CLS.value
        dataset_type = DatasetType.SEQ_CLASSIFICATION.value
    else:
        raise ValueError(f"Unsupported task type: {model_config.task_type}")

    return MasterConfig(
        model=ModelConfig(
            model_name=model_config.model_name,
            model_type="hf",
            auto_class_name=auto_class_name,
            use_peft=model_config.use_peft,
            use_cache=False,
            attn_implementation="eager",
            device_map=None,
            peft_config=PeftConfig(
                lora_r=TEST_LORA_R,
                lora_alpha=TEST_LORA_ALPHA,
                lora_dropout=TEST_LORA_DROPOUT,
                target_modules=model_config.target_modules,
                bias=TEST_LORA_BIAS,
                task_type=model_config.task_type.value,
                peft_type="LORA",
            )
            if model_config.use_peft
            else None,
        ),
        dataset=DatasetConfig(
            tokenizer_name=model_config.model_name,
            dataset_type=dataset_type,
            dataset_name=dataset_config.dataset_name,
            max_seq_length=dataset_config.max_seq_length,
            train_batch_size=TEST_PER_DEVICE_BATCH_SIZE,
            eval_batch_size=TEST_PER_DEVICE_BATCH_SIZE,
            num_workers=1,
        ),
        optimizers=OptimizerConfig(
            optimizer_name="AdamW",
            lr=TEST_LEARNING_RATE,
            weight_decay=TEST_WEIGHT_DECAY,
        ),
        scheduler=SchedulerConfig(
            scheduler_name="cosine",
            warmup_steps=TEST_WARMUP_STEPS,
        ),
        training=TrainingConfig(
            type="sft",  # Using the "type" field from TrainingConfig
            output_dir=output_dir,
            num_train_epochs=TEST_NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=TEST_PER_DEVICE_BATCH_SIZE,
            per_device_eval_batch_size=TEST_PER_DEVICE_BATCH_SIZE,
            logging_steps=TEST_LOGGING_STEPS,
            save_strategy="no",
            eval_strategy="no",
            seed=TEST_SEED,
            max_steps=TEST_MAX_STEPS,
        ),
    )


def load_and_prepare_dataset(
    dataset_config: TestDatasetConfig,
    output_dir: str,
) -> tuple[SFTDataset, Callable]:
    """
    Load and prepare a dataset for training.

    Args:
        dataset_config: Test dataset configuration
        output_dir: Directory to save temporary dataset files

    Returns:
        Tuple of (SFTDataset instance, formatting function)
    """
    if dataset_config.hf_dataset_config:
        hf_dataset = load_dataset(
            dataset_config.hf_dataset_name,
            dataset_config.hf_dataset_config,
            split="train",
        )
    else:
        hf_dataset = load_dataset(dataset_config.hf_dataset_name, split="train")

    # Use small subset for testing
    hf_subset = hf_dataset.select(range(TEST_DATASET_SUBSET_SIZE))

    # Save to temporary JSON file
    json_path = os.path.join(output_dir, f"{dataset_config.dataset_name}.json")
    hf_subset.to_json(json_path)

    # Create SFTDataset instance
    dataset = SFTDataset(
        dataset_name=dataset_config.dataset_name,
        split="train",
        json_file_path=json_path,
        prompt_template=dataset_config.prompt_template,
        completion_template=dataset_config.completion_template,
    )

    # Create formatting function
    def formatting_func(example):
        prompt = dataset._preprocess_sample(example)
        return prompt["prompt"] + prompt["completion"]

    return dataset, formatting_func


def create_model_and_tokenizer(
    model_config: ModelConfig,
) -> tuple[torch.nn.Module, any]:
    """
    Create model and tokenizer instances.

    Args:
        model_config: Model configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    # Adjust model config for faster testing
    model_config_kwargs = {"num_hidden_layers": TEST_NUM_HIDDEN_LAYERS}

    # Create HFModel instance
    hf_model = HFModel(
        model_name=model_config.model_name,
        auto_class_name=model_config.auto_class_name,
        use_cache=model_config.use_cache,
        attn_implementation=model_config.attn_implementation,
        device_map=model_config.device_map,
        **model_config_kwargs,
    )

    # Load model and tokenizer
    model = hf_model.load_model()
    tokenizer = hf_model.load_tokenizer()

    return model, tokenizer


def create_peft_config(peft_config: Optional[PeftConfig]) -> Optional[LoraConfig]:
    """
    Create PEFT configuration from config dataclass.

    Args:
        peft_config: PEFT configuration dataclass

    Returns:
        LoraConfig instance or None
    """
    if peft_config is None:
        return None

    return LoraConfig(
        r=peft_config.lora_r,
        lora_alpha=peft_config.lora_alpha,
        lora_dropout=peft_config.lora_dropout,
        target_modules=peft_config.target_modules,
        bias=peft_config.bias,
    )


def create_sft_trainer(
    model: torch.nn.Module,
    tokenizer: any,
    dataset: SFTDataset,
    formatting_func: Callable,
    master_config: MasterConfig,
    peft_config: Optional[LoraConfig],
):
    """
    Create SFT trainer using ComponentRegistry.

    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        dataset: Dataset instance
        formatting_func: Formatting function for dataset
        master_config: Master configuration
        peft_config: PEFT configuration

    Returns:
        Trainer instance
    """
    # Get trainer type from training config
    trainer_type = master_config.training.type

    # Get trainer module from registry
    trainer_module = registry.get_trainer_module(trainer_type)
    if trainer_module is None:
        raise ValueError(f"Trainer type '{trainer_type}' not found in registry")

    # Get the trainer class and args class
    trainer_cls = trainer_module["trainer_cls"]
    args_cls = trainer_module["args_cls"]

    # Create training arguments using the args class
    training_config = master_config.training
    sft_config = args_cls(
        output_dir=training_config.output_dir,
        max_length=master_config.dataset.max_seq_length,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        num_train_epochs=training_config.num_train_epochs,
        max_steps=training_config.max_steps,
        logging_steps=training_config.logging_steps,
        save_strategy=training_config.save_strategy,
        eval_strategy=training_config.eval_strategy,
        seed=training_config.seed,
        bf16=False,
        fp16=True,
        report_to="none",
    )

    # Create trainer instance
    trainer = trainer_cls(
        model=model,
        args=sft_config,
        train_dataset=dataset.dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        formatting_func=formatting_func,
    )

    return trainer


def run_training(trainer, config_name: str):
    """
    Run training and return results.

    Args:
        trainer: Trainer instance
        config_name: Configuration name for logging

    Returns:
        Training result
    """
    logger.warning(f"Starting training for {config_name}...")
    train_result = trainer.train()
    logger.warning(f"Training completed for {config_name}!")
    return train_result


def verify_training_results(train_result):
    """
    Verify training results.

    Args:
        train_result: Training result object
    """
    assert train_result is not None
    assert hasattr(train_result, "training_loss")
    logger.warning(f"Training loss: {train_result.training_loss:.4f}")


def run_inference_causal_lm(model, tokenizer):
    """
    Run inference for causal language models.

    Args:
        model: Model instance
        tokenizer: Tokenizer instance
    """
    test_prompt = "Test prompt for generation"
    inputs = tokenizer(test_prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.warning(f"Generated text: {generated_text}")


# ============================================================================
# Test Classes
# ============================================================================


class TestCausalLMIntegration:
    """Integration tests for Causal Language Modeling tasks."""

    def setup_method(self):
        """Setup method executed before each test."""
        self.test_output_dir = tempfile.mkdtemp(prefix="test_ft_causal_lm_")
        logger.info(f"Created test directory: {self.test_output_dir}")

    def teardown_method(self):
        """Teardown method executed after each test."""
        if os.path.exists(self.test_output_dir):
            try:
                shutil.rmtree(self.test_output_dir)
                logger.info(f"Cleaned up test directory: {self.test_output_dir}")
            except Exception as e:
                logger.warning(f"Warning: Failed to clean up {self.test_output_dir}: {e}")

    @pytest.mark.parametrize(
        "dataset_config,config_name",
        [
            pytest.param(
                GSM8K_DATASET_CONFIG,
                "llama_3.2_1B_gsm8k",
                id="llama_gsm8k",
            ),
            pytest.param(
                ALPACA_DATASET_CONFIG,
                "llama_3.2_1B_alpaca",
                id="llama_alpaca",
            ),
        ],
    )
    def test_llama_causal_lm(self, dataset_config: TestDatasetConfig, config_name: str):
        """
        Test Llama model with different datasets for causal language modeling.

        Args:
            dataset_config: Dataset configuration
            config_name: Configuration name for logging
        """
        # Create master configuration
        master_config = create_master_config(
            model_config=LLAMA_MODEL_CONFIG,
            dataset_config=dataset_config,
            output_dir=self.test_output_dir,
        )

        # Validate configuration
        config_manager = ConfigManager(master_config)
        config_manager.validate_config()

        # Load model and tokenizer
        model_config = config_manager.get_model_config()
        model, tokenizer = create_model_and_tokenizer(model_config)
        logger.warning(f"Model loaded: {model_config.model_name}")

        # Load and prepare dataset
        dataset, formatting_func = load_and_prepare_dataset(
            dataset_config=dataset_config,
            output_dir=self.test_output_dir,
        )
        logger.warning(f"Dataset loaded: {len(dataset)} samples")

        # Create PEFT config if needed
        peft_config = None
        if model_config.use_peft and model_config.peft_config:
            peft_config = create_peft_config(model_config.peft_config)

        # Create trainer using ComponentRegistry
        trainer = create_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            formatting_func=formatting_func,
            master_config=master_config,
            peft_config=peft_config,
        )
        logger.warning("Trainer instantiated")

        # Run training
        train_result = run_training(trainer, config_name)

        # Verify training results
        verify_training_results(train_result)

        # Test inference
        run_inference_causal_lm(model, tokenizer)
