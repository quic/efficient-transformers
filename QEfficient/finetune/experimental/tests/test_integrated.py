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

import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest
import torch

from QEfficient.cloud.finetune_experimental import FineTuningPipeline
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
from QEfficient.finetune.experimental.core.logger import Logger
from QEfficient.finetune.experimental.tests import reference_data as ref_data
from QEfficient.finetune.experimental.tests.constants import (
    HF_DATASET_ALPACA,
    HF_DATASET_ALPACA_CONFIG,
    HF_DATASET_GSM8K,
    HF_DATASET_GSM8K_CONFIG,
    HF_DATASET_IMDB,
    LOSS_ATOL,
    METRIC_ATOL,
    TEST_DATASET_SUBSET_SIZE,
    TEST_EVAL_STEPS,
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
    TEST_NUM_TRAIN_EPOCHS,
    TEST_PER_DEVICE_BATCH_SIZE,
    TEST_SEED,
    TEST_WEIGHT_DECAY,
    TRAIN_EVAL_EPOCH_LOSS_DIFF_THRESHOLD,
    AutoClassName,
    DatasetType,
    TaskType,
)
from QEfficient.finetune.utils.helper import get_rank, get_world_size

logger = Logger(__name__)
# ============================================================================
# Test Configuration Dataclasses
# ============================================================================


def clean_up(path):
    if os.path.isdir(path) and os.path.exists(path):
        shutil.rmtree(path)
    if os.path.isfile(path):
        os.remove(path)


def assert_list_close(ref_list, actual_list, atol, name, scenario_key, current_world_size, current_rank):
    """
    Asserts that two lists of floats are numerically close element-wise.
    If not close, reports the step numbers and the differences at those steps.
    """
    # --- Initial Checks ---
    assert actual_list is not None and isinstance(actual_list, list), (
        f"Actual {name} data is missing or not a list for scenario '{scenario_key}'."
    )
    assert len(ref_list) == len(actual_list), (
        f"{name} length mismatch for scenario '{scenario_key}' (WS: {current_world_size}, Rank: {current_rank}). "
        f"Expected {len(ref_list)} elements, but got {len(actual_list)}."
    )

    # --- Convert to NumPy arrays for efficient comparison ---
    ref_arr = np.array(ref_list)
    actual_arr = np.array(actual_list)

    # --- Check if all elements are close using np.allclose ---
    # This is the primary assertion that will fail if any deviation is too large
    if not np.allclose(ref_arr, actual_arr, atol=atol):
        # If not all close, identify the specific deviations
        deviated_indices = np.where(~np.isclose(ref_arr, actual_arr, atol=atol))[0]
        deviation_details = []
        for idx in deviated_indices:
            ref_val = ref_arr[idx]
            actual_val = actual_arr[idx]
            diff = actual_val - ref_val
            deviation_details.append(f"Step {idx}: Ref={ref_val:.6f}, Actual={actual_val:.6f}, Diff={diff:.6f}")

        # Calculate max_diff
        max_diff = np.max(np.abs(ref_arr - actual_arr))

        # --- Report detailed deviation in the AssertionError ---
        error_message = (
            f"{name} deviated too much for scenario '{scenario_key}' "
            f"(WS: {current_world_size}, Rank: {current_rank}).\n"
            f"Max Difference: {max_diff:.6f}, Allowed Tolerance: {atol:.6f}.\n"
            f"Deviations found at {len(deviated_indices)} steps:\n" + "\n".join(deviation_details) + "\n"
            f"Reference (first 10): {ref_list[:10]}...\n"
            f"Actual    (first 10): {actual_list[:10]}..."
        )
        assert False, error_message  # Force the assertion to fail with the custom message
    else:
        # If all close, report success and max_diff for printing
        max_diff = np.max(np.abs(ref_arr - actual_arr))
        print(f"  ✅ {name} PASSED. Max Diff: {max_diff:.6f}")


def get_reference_metrics(
    scenario_key,
):
    reference_data = ref_data.REFERENCE_DATA.get(scenario_key)
    if reference_data is None:
        pytest.fail(f"Reference data for scenario '{scenario_key}' not found in REFERENCE_DATA.")
    current_world_size = get_world_size()
    current_rank = get_rank()
    if current_world_size > 1:
        rank_reference_data = reference_data.get("rank_data", {}).get(str(current_rank))
        if rank_reference_data is None:
            pytest.fail(f"Reference data for rank {current_rank} not found in distributed scenario '{scenario_key}'.")
        ref_train_losses = rank_reference_data["train_step_losses"]
        ref_eval_losses = rank_reference_data["eval_step_losses"]
        ref_train_metrics = rank_reference_data["train_step_metrics"]
        ref_eval_metrics = rank_reference_data["eval_step_metrics"]
    else:  # Single device or world_size=1
        ref_train_losses = reference_data["train_step_losses"]
        ref_eval_losses = reference_data["eval_step_losses"]
        ref_train_metrics = reference_data["train_step_metrics"]
        ref_eval_metrics = reference_data["eval_step_metrics"]

    all_ref_metrices = {
        "ref_train_losses": ref_train_losses,
        "ref_eval_losses": ref_eval_losses,
        "ref_train_metrics": ref_train_metrics,
        "ref_eval_metrics": ref_eval_metrics,
    }

    all_config_spy = {
        "current_world_size": current_world_size,
        "current_rank": current_rank,
    }
    return all_ref_metrices, all_config_spy


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
    completion_template: str
    max_seq_length: int
    prompt_template: Optional[str] = None
    prompt_func: Optional[str] = None


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
    dataset_name="openai/gsm8k",
    hf_dataset_name=HF_DATASET_GSM8K,
    hf_dataset_config=HF_DATASET_GSM8K_CONFIG,
    prompt_template="Solve the following math problem step by step.\n\n### Question:\n{question}\n\n### Answer:\n",
    completion_template="{answer}",
    max_seq_length=TEST_MAX_SEQ_LENGTH_CAUSAL,
)

ALPACA_DATASET_CONFIG = TestDatasetConfig(
    dataset_name="yahma/alpaca-cleaned",
    hf_dataset_name=HF_DATASET_ALPACA,
    hf_dataset_config=HF_DATASET_ALPACA_CONFIG,
    prompt_func="QEfficient.finetune.experimental.preprocessing.alpaca_func:create_alpaca_prompt",
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
        dataset_type = DatasetType.SFT_DATASET.value
    elif model_config.task_type == TaskType.SEQ_CLS:
        auto_class_name = AutoClassName.SEQ_CLS.value
        dataset_type = DatasetType.SFT_DATASET.value
    else:
        raise ValueError(f"Unsupported task type: {model_config.task_type}")
    return MasterConfig(
        model=ModelConfig(
            model_name=model_config.model_name,
            model_type="hf",
            auto_class_name=auto_class_name,
            use_peft=model_config.use_peft,
            use_cache=False,
            attn_implementation="sdpa",
            device_map=None,
            torch_dtype="fp16",
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
            prompt_template=dataset_config.prompt_template if dataset_config.prompt_template else None,
            prompt_func=dataset_config.prompt_func if dataset_config.prompt_func else None,
            completion_template=dataset_config.completion_template,
            test_split="test",
            config_name=dataset_config.hf_dataset_config,
            dataset_num_samples=TEST_DATASET_SUBSET_SIZE,
            data_seed=TEST_SEED,
        ),
        optimizers=OptimizerConfig(
            optimizer_name="adamw",
            lr=TEST_LEARNING_RATE,
            weight_decay=TEST_WEIGHT_DECAY,
        ),
        scheduler=SchedulerConfig(
            scheduler_name="cosine",
        ),
        training=TrainingConfig(
            type="sft",  # Using the "type" field from TrainingConfig
            output_dir=output_dir,
            num_train_epochs=TEST_NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=TEST_PER_DEVICE_BATCH_SIZE,
            per_device_eval_batch_size=TEST_PER_DEVICE_BATCH_SIZE,
            logging_strategy="steps",
            logging_steps=TEST_LOGGING_STEPS,
            gradient_accumulation_steps=1,
            eval_strategy="steps",
            max_steps=TEST_MAX_STEPS,
            eval_steps=TEST_EVAL_STEPS,
            seed=TEST_SEED,
            fp16=False,
            bf16=False,
        ),
    )


def run_training(trainer, config_name):
    """
    Run training and return results.

    Args:
        trainer: Trainer instance
        config_name: Configuration name for logging

    Returns:
        Training result, Evaluation result
    """
    logger.info(f"Starting training for {config_name}...")
    trainer.train()
    logger.info(f"Training completed for {config_name}!")
    train_step_loss = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    eval_step_loss = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]
    train_step_metric = [math.exp(x) for x in train_step_loss]
    eval_step_metric = [math.exp(x) for x in eval_step_loss]
    final_train_loss = train_step_loss[-1] if train_step_loss else float("inf")
    final_eval_loss = eval_step_loss[-1] if eval_step_loss else float("inf")
    return final_eval_loss, final_train_loss, train_step_loss, eval_step_loss, train_step_metric, eval_step_metric


def verify_training_results(train_result, eval_result):
    """
    Verify training results.

    Args:
        train_result: Training result object
        eval_result: Evaluation result dictionary
    """
    assert train_result is not None
    assert eval_result is not None
    logger.info(f"Training loss: {train_result:.4f}")
    logger.info(f"Evaluation loss: {eval_result:.4f}")
    assert abs(train_result - eval_result) < TRAIN_EVAL_EPOCH_LOSS_DIFF_THRESHOLD


def run_inference_causal_lm(model, tokenizer):
    """
    Run inference for causal language models.

    Args:
        model: Model instance
        tokenizer: Tokenizer instance
    """
    test_prompt = "Test prompt for generation."
    texts = tokenizer(test_prompt, return_tensors="pt")
    texts = texts.to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **texts,
            temperature=0.4,
            max_new_tokens=10,
            do_sample=False,
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Generated text: {generated_text}")


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
        "dataset_config,config_name,scenario_key",
        [
            pytest.param(
                GSM8K_DATASET_CONFIG,
                "llama_3.2_1B_gsm8k",
                "llama_3.2_1B_config_gsm8k_single_device",
                id="llama_gsm8k",
            ),
            pytest.param(
                ALPACA_DATASET_CONFIG,
                "llama_3.2_1B_alpaca",
                "llama_3.2_1B_config_alpaca_single_device",
                id="llama_alpaca",
            ),
        ],
    )
    def test_llama_causal_lm(self, dataset_config: TestDatasetConfig, scenario_key: str, config_name: str):
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
        config_manager = ConfigManager(master_config)
        pipeline = FineTuningPipeline(config_manager)
        model, tokenizer = pipeline.get_model_and_tokenizer()
        trainer = pipeline.get_trainer()
        # Verify model and tokenizer are loaded correctly
        assert model is not None, "Model should be loaded"
        assert tokenizer is not None, "Tokenizer should be loaded"
        assert hasattr(model, "generate"), "Model should have generate method"
        assert hasattr(tokenizer, "decode"), "Tokenizer should have decode method"
        logger.info(f"Model and tokenizer loaded successfully for {config_name}")
        # Verify model parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params:,}")
        # Run training
        final_eval_loss, final_train_loss, train_step_loss, eval_step_loss, train_step_metric, eval_step_metric = (
            run_training(trainer, config_name)
        )
        all_ref_metrices, all_config_spy = get_reference_metrics(scenario_key)
        verify_training_results(final_train_loss, final_eval_loss)
        run_inference_causal_lm(model, tokenizer)

        # Test inference
        # Assertions for step-level values using the helper function
        assert_list_close(
            all_ref_metrices["ref_train_losses"],
            train_step_loss,
            LOSS_ATOL,
            "Train Step Losses",
            scenario_key,
            all_config_spy["current_world_size"],
            all_config_spy["current_rank"],
        )
        assert_list_close(
            all_ref_metrices["ref_eval_losses"],
            eval_step_loss,
            LOSS_ATOL,
            "Eval Step Losses",
            scenario_key,
            all_config_spy["current_world_size"],
            all_config_spy["current_rank"],
        )
        assert_list_close(
            all_ref_metrices["ref_train_metrics"],
            train_step_metric,
            METRIC_ATOL,
            "Train Step Metrics",
            scenario_key,
            all_config_spy["current_world_size"],
            all_config_spy["current_rank"],
        )
        assert_list_close(
            all_ref_metrices["ref_eval_metrics"],
            eval_step_metric,
            METRIC_ATOL,
            "Eval Step Metrics",
            scenario_key,
            all_config_spy["current_world_size"],
            all_config_spy["current_rank"],
        )
        clean_up("qaic-dumps")
