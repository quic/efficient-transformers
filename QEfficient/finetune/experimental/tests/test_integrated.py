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

import importlib.util
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
# Update Reference Data Flag
# ============================================================================

UPDATE_REFERENCE = os.getenv("UPDATE_REFERENCE") == "1"

# Path to reference_data.py — resolved relative to this file
REFERENCE_DATA_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "reference_data.py",
)

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
    if not np.allclose(ref_arr, actual_arr, atol=atol):
        deviated_indices = np.where(~np.isclose(ref_arr, actual_arr, atol=atol))[0]
        deviation_details = []
        for idx in deviated_indices:
            ref_val = ref_arr[idx]
            actual_val = actual_arr[idx]
            diff = actual_val - ref_val
            deviation_details.append(f"Step {idx}: Ref={ref_val:.6f}, Actual={actual_val:.6f}, Diff={diff:.6f}")

        max_diff = np.max(np.abs(ref_arr - actual_arr))
        error_message = (
            f"{name} deviated too much for scenario '{scenario_key}' "
            f"(WS: {current_world_size}, Rank: {current_rank}).\n"
            f"Max Difference: {max_diff:.6f}, Allowed Tolerance: {atol:.6f}.\n"
            f"Deviations found at {len(deviated_indices)} steps:\n" + "\n".join(deviation_details) + "\n"
            f"Reference (first 10): {ref_list[:10]}...\n"
            f"Actual    (first 10): {actual_list[:10]}..."
        )
        assert False, error_message
    else:
        max_diff = np.max(np.abs(ref_arr - actual_arr))
        print(f"  ✅ {name} PASSED. Max Diff: {max_diff:.6f}")


def get_reference_metrics(scenario_key):
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
    else:
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


# ============================================================================
# Reference Data Update Utilities
# ============================================================================


def _load_current_reference_data() -> dict:
    """
    Dynamically load the current REFERENCE_DATA dict from reference_data.py.
    This ensures we always read the latest on-disk state before updating.
    """
    spec = importlib.util.spec_from_file_location("reference_data_live", REFERENCE_DATA_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.REFERENCE_DATA)


def _format_float_list(values: list, indent: int = 12) -> str:
    """
    Format a list of floats as a Python list literal with one value per line,
    matching the style of the existing reference_data.py.
    """
    pad = " " * indent
    inner = f",\n{pad}".join(repr(float(v)) for v in values)
    return f"[\n{pad}{inner},\n{' ' * (indent - 4)}]"


def _build_scenario_block(scenario_key: str, scenario_data: dict) -> str:
    """
    Render a single scenario entry as a formatted Python dict literal string,
    matching the style of the existing reference_data.py.
    """
    description = scenario_data.get("description", f"Baseline for {scenario_key}")

    train_losses_str = _format_float_list(scenario_data.get("train_step_losses", []))
    eval_losses_str = _format_float_list(scenario_data.get("eval_step_losses", []))
    train_metrics_str = _format_float_list(scenario_data.get("train_step_metrics", []))
    eval_metrics_str = _format_float_list(scenario_data.get("eval_step_metrics", []))

    # Handle optional distributed rank_data
    rank_data_block = ""
    if "rank_data" in scenario_data:
        rank_lines = ['        "rank_data": {\n']
        for rank_key, rank_vals in scenario_data["rank_data"].items():
            rank_train_losses = _format_float_list(rank_vals.get("train_step_losses", []), indent=20)
            rank_eval_losses = _format_float_list(rank_vals.get("eval_step_losses", []), indent=20)
            rank_train_metrics = _format_float_list(rank_vals.get("train_step_metrics", []), indent=20)
            rank_eval_metrics = _format_float_list(rank_vals.get("eval_step_metrics", []), indent=20)
            rank_lines.append(f'            "{rank_key}": {{\n')
            rank_lines.append(f'                "train_step_losses": {rank_train_losses},\n')
            rank_lines.append(f'                "eval_step_losses": {rank_eval_losses},\n')
            rank_lines.append(f'                "train_step_metrics": {rank_train_metrics},\n')
            rank_lines.append(f'                "eval_step_metrics": {rank_eval_metrics},\n')
            rank_lines.append("            },\n")
        rank_lines.append("        },\n")
        rank_data_block = "".join(rank_lines)

    block = (
        f"    # {description}\n"
        f'    "{scenario_key}": {{\n'
        f'        "description": "{description}",\n'
        f'        "train_step_losses": {train_losses_str},\n'
        f'        "eval_step_losses": {eval_losses_str},\n'
        f'        "train_step_metrics": {train_metrics_str},\n'
        f'        "eval_step_metrics": {eval_metrics_str},\n'
        f"{rank_data_block}"
        f"    }},\n"
    )
    return block


def _write_reference_data_file(updated_data: dict):
    """
    Write the full updated REFERENCE_DATA back to reference_data.py.
    Uses an atomic temp-file write to avoid corrupting the file on failure.
    Preserves the original copyright header and module docstring.
    """
    header = (
        "# " + "-" * 77 + "\n"
        "#\n"
        "# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.\n"
        "# SPDX-License-Identifier: BSD-3-Clause\n"
        "#\n"
        "# " + "-" * 77 + "\n"
        '"""Reference data for the finetune tests from SDK version - 1.22.0.32"""\n'
        "# A dictionary to hold all reference data for all test sets.\n"
    )

    scenario_blocks = "".join(_build_scenario_block(key, val) for key, val in updated_data.items())

    content = f"{header}REFERENCE_DATA = {{\n{scenario_blocks}}}\n"

    # Atomic write: write to a temp file first, then replace
    tmp_path = REFERENCE_DATA_FILE + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            f.write(content)
        os.replace(tmp_path, REFERENCE_DATA_FILE)
        logger.info(f"✅ reference_data.py updated at: {REFERENCE_DATA_FILE}")
    except Exception as e:
        # Clean up temp file if something went wrong
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Failed to write reference_data.py: {e}") from e


def update_reference_data(
    scenario_key: str,
    train_step_losses: list,
    eval_step_losses: list,
    train_step_metrics: list,
    eval_step_metrics: list,
):
    """
    Update a single scenario entry in reference_data.py with new baseline values.

    For single-device runs (world_size == 1): updates top-level lists directly.
    For distributed runs (world_size > 1):    updates under rank_data[str(rank)].

    Args:
        scenario_key:        Key in REFERENCE_DATA (e.g. 'llama_3.2_1B_config_alpaca_single_device')
        train_step_losses:   New train step loss values  (plain float list from trainer.state.log_history)
        eval_step_losses:    New eval step loss values
        train_step_metrics:  New train step metric values (perplexity = exp(loss))
        eval_step_metrics:   New eval step metric values
    """
    current_world_size = get_world_size()
    current_rank = get_rank()

    # Load the current on-disk state so we don't overwrite other scenarios
    current_data = _load_current_reference_data()

    # Preserve existing description if the scenario already exists
    existing_entry = current_data.get(scenario_key, {})
    description = existing_entry.get("description", f"Baseline for {scenario_key}")

    new_values = {
        "train_step_losses": [float(v) for v in train_step_losses],
        "eval_step_losses": [float(v) for v in eval_step_losses],
        "train_step_metrics": [float(v) for v in train_step_metrics],
        "eval_step_metrics": [float(v) for v in eval_step_metrics],
    }

    if current_world_size > 1:
        # Distributed run: store per-rank under rank_data
        logger.info(
            f"Updating reference data for '{scenario_key}' [distributed: rank {current_rank}/{current_world_size}]"
        )
        if scenario_key not in current_data:
            current_data[scenario_key] = {"description": description}

        if "rank_data" not in current_data[scenario_key]:
            current_data[scenario_key]["rank_data"] = {}

        current_data[scenario_key]["rank_data"][str(current_rank)] = new_values

        # Preserve top-level description
        current_data[scenario_key]["description"] = description

    else:
        # Single-device run: update top-level lists
        logger.info(f"Updating reference data for '{scenario_key}' [single-device]")
        current_data[scenario_key] = {
            "description": description,
            **new_values,
        }

    _write_reference_data_file(current_data)
    logger.info(f"✅ Reference data updated for scenario: '{scenario_key}'")


# ============================================================================
# Test Configuration Constants
# ============================================================================

# ... existing dataclasses and config constants unchanged ...


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
    # ... existing implementation unchanged ...
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
            type="sft",
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
        ),
    )


def run_training(trainer, config_name):
    # ... existing implementation unchanged ...
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
    # ... existing implementation unchanged ...
    assert train_result is not None
    assert eval_result is not None
    logger.info(f"Training loss: {train_result:.4f}")
    logger.info(f"Evaluation loss: {eval_result:.4f}")
    assert abs(train_result - eval_result) < TRAIN_EVAL_EPOCH_LOSS_DIFF_THRESHOLD


def run_inference_causal_lm(model, tokenizer):
    # ... existing implementation unchanged ...
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
        self.test_output_dir = tempfile.mkdtemp(prefix="test_ft_causal_lm_")
        logger.info(f"Created test directory: {self.test_output_dir}")

    def teardown_method(self):
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
        master_config = create_master_config(
            model_config=LLAMA_MODEL_CONFIG,
            dataset_config=dataset_config,
            output_dir=self.test_output_dir,
        )
        config_manager = ConfigManager(master_config)
        pipeline = FineTuningPipeline(config_manager)
        model, tokenizer = pipeline.get_model_and_tokenizer()
        trainer = pipeline.get_trainer()

        assert model is not None, "Model should be loaded"
        assert tokenizer is not None, "Tokenizer should be loaded"
        assert hasattr(model, "generate"), "Model should have generate method"
        assert hasattr(tokenizer, "decode"), "Tokenizer should have decode method"
        logger.info(f"Model and tokenizer loaded successfully for {config_name}")

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params:,}")

        final_eval_loss, final_train_loss, train_step_loss, eval_step_loss, train_step_metric, eval_step_metric = (
            run_training(trainer, config_name)
        )

        verify_training_results(final_train_loss, final_eval_loss)
        run_inference_causal_lm(model, tokenizer)

        # ✅ UPDATE MODE: write new baseline and skip assertions
        if UPDATE_REFERENCE:
            logger.info(f"UPDATE_REFERENCE=1 → updating reference data for '{scenario_key}'")
            update_reference_data(
                scenario_key=scenario_key,
                train_step_losses=train_step_loss,
                eval_step_losses=eval_step_loss,
                train_step_metrics=train_step_metric,
                eval_step_metrics=eval_step_metric,
            )
            logger.info(f"✅ Skipping assertions — reference data updated for '{scenario_key}'")
            return

        # ✅ NORMAL MODE: compare against reference
        all_ref_metrices, all_config_spy = get_reference_metrics(scenario_key)

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
