# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
from pathlib import Path

import pytest

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


@pytest.fixture
def config_path() -> Path:
    here = Path(__file__).resolve().parent
    return (here / "test_config.yaml").resolve()


def create_master_config(
    output_dir: str,
) -> MasterConfig:
    """
    Args:
        model_config: Test model configuration
        dataset_config: Test dataset configuration
        output_dir: Output directory for training results

    Returns:
        MasterConfig instance
    """

    return MasterConfig(
        model=ModelConfig(
            model_name="HuggingFaceTB/SmolLM-135M",
            model_type="hf",
            auto_class_name="AutoModelForCausalLM",
            use_peft=True,
            use_cache=False,
            device_map=None,
            peft_config=PeftConfig(
                lora_r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type="CAUSAL_LM",
                peft_type="LORA",
            ),
        ),
        dataset=DatasetConfig(
            tokenizer_name="HuggingFaceTB/SmolLM-135M",
            dataset_type="sft_dataset",
            dataset_name="openai/gsm8k",
            max_seq_length=512,
            train_batch_size=1,
            prompt_template="Question: {question}\nAnswer: ",
            completion_template="{answer}",
            config_name="main",
        ),
        optimizers=OptimizerConfig(
            optimizer_name="adamw",
        ),
        scheduler=SchedulerConfig(
            scheduler_name="cosine",
            warmup_steps=1,
        ),
        training=TrainingConfig(
            type="sft",  # Using the "type" field from TrainingConfig
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
        ),
    )


def test_default_config():
    config_manager = ConfigManager()
    assert config_manager is not None
    assert config_manager.config is not None


def test_config_values(config_path):
    config_manager = ConfigManager(config_path=config_path)
    assert config_manager.config is not None
    assert config_manager.config.model["model_name"] == "HuggingFaceTB/SmolLM-135M"
    assert config_manager.config.model["peft_config"]["lora_dropout"] == 0.1
    assert config_manager.config.model["peft_config"]["lora_r"] == 16
    assert config_manager.config.dataset["dataset_name"] == "knkarthick/samsum"
    assert config_manager.config.training["output_dir"] == "./training_results"
    assert config_manager.config.training["per_device_train_batch_size"] == 1
    assert config_manager.config.training["num_train_epochs"] == 1
    assert not config_manager.config.training["gradient_checkpointing_kwargs"]["use_reentrant"]


def test_config_missing_file():
    with pytest.raises(FileNotFoundError):
        ConfigManager(config_path="non_existent_file.yaml")


def test_config_created_from_obj():
    master_config = create_master_config(output_dir="./test_output")
    config_manager = ConfigManager(master_config)
    config = config_manager.config
    assert config is not None
    assert config.model is not None
    assert config.dataset is not None
    assert config.training is not None
    assert config.optimizers is not None
    assert config.scheduler is not None


def test_config(config_path):
    config_manager = ConfigManager(config_path=config_path)
    assert isinstance(config_manager, ConfigManager)

    # Test that all required fields are present
    missing = [
        a
        for a in ("model", "dataset", "optimizers", "scheduler", "callbacks", "training")
        if not hasattr(config_manager, a)
    ]
    assert not missing, f"Missing attributes: {missing}"
    trainer_config = config_manager.get_training_config()
    assert trainer_config is not None
    assert isinstance(trainer_config, dict)
    assert (hasattr(trainer_config, attr) for attr in ("output_dir", "train_batch_size", "num_epochs", "ddp_config"))
    dataset_config = config_manager.get_dataset_config()
    assert dataset_config is not None
    assert isinstance(dataset_config, dict)
    assert (hasattr(dataset_config, attr) for attr in ("dataset_type", "dataset_name", "tokenizer_name"))
    model_config = config_manager.get_model_config()
    assert model_config is not None
    assert isinstance(model_config, dict)
    assert (hasattr(model_config, attr) for attr in ("model_type", "model_name", "use_peft", "peft_config"))
    scheduler_config = config_manager.get_scheduler_config()
    assert scheduler_config is not None
    assert isinstance(scheduler_config, dict)
    assert (hasattr(scheduler_config, attr) for attr in ("scheduler_name"))
    callback_config = config_manager.get_callback_config()
    assert callback_config is not None
    assert isinstance(callback_config, dict)
    assert (hasattr(callback_config, attr) for attr in ("earlystopping"))
    optimizer_config = config_manager.get_optimizer_config()
    assert optimizer_config is not None
    assert isinstance(optimizer_config, dict)
    assert (hasattr(optimizer_config, attr) for attr in ("optimizer_name", "lr"))


def test_torch_dtype_validation():
    """Test that torch_dtype validation works correctly."""
    # Test with default config - should have torch_dtype set to fp16 by default
    config_manager = ConfigManager()
    training_config = config_manager.get_training_config()
    assert training_config.get("torch_dtype") == "fp16"

    # Validation should pass with default config
    config_manager.validate_config()  # Should not raise


def test_torch_dtype_invalid():
    """Test that invalid torch_dtype raises validation error."""
    from QEfficient.finetune.experimental.core.config_manager import MasterConfig, TrainingConfig

    # Create config with invalid torch_dtype
    training_config = TrainingConfig(torch_dtype="invalid_dtype")
    master_config = MasterConfig(training=training_config)
    config_manager = ConfigManager(config=master_config)

    # Validation should fail
    with pytest.raises(ValueError) as exc_info:
        config_manager.validate_config()

    assert "torch_dtype must be one of" in str(exc_info.value)


def test_parallelism_rejects_tp_plus_pp_combo():
    """TP cannot be combined with PP in supported mode matrix."""
    from QEfficient.finetune.experimental.core.config_manager import MasterConfig, TrainingConfig

    training_config = TrainingConfig(tp_degree=2, pp_degree=2, ddp_degree=1)
    master_config = MasterConfig(training=training_config)
    config_manager = ConfigManager(config=master_config)

    with pytest.raises(ValueError) as exc_info:
        config_manager.validate_config()

    assert "TP cannot be combined with PP" in str(exc_info.value)


def test_parallelism_rejects_ddp_plus_pp_combo():
    """DDP cannot be combined with PP in supported mode matrix."""
    from QEfficient.finetune.experimental.core.config_manager import MasterConfig, TrainingConfig

    training_config = TrainingConfig(tp_degree=1, pp_degree=2, ddp_degree=2)
    master_config = MasterConfig(training=training_config)
    config_manager = ConfigManager(config=master_config)

    with pytest.raises(ValueError) as exc_info:
        config_manager.validate_config()

    assert "DDP cannot be combined with PP" in str(exc_info.value)


def test_parallelism_world_size_product_mismatch(monkeypatch):
    """WORLD_SIZE must match pp*tp*ddp when distributed env is set."""
    from QEfficient.finetune.experimental.core.config_manager import MasterConfig, TrainingConfig

    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")

    training_config = TrainingConfig(tp_degree=2, pp_degree=1, ddp_degree=2)
    master_config = MasterConfig(training=training_config)
    config_manager = ConfigManager(config=master_config)

    with pytest.raises(ValueError) as exc_info:
        config_manager.validate_config()

    assert "must equal WORLD_SIZE" in str(exc_info.value)


def test_parallelism_multi_server_rejects_tp(monkeypatch):
    """TP and TP+DDP are rejected for multi-server launch."""
    from QEfficient.finetune.experimental.core.config_manager import MasterConfig, TrainingConfig

    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")

    training_config = TrainingConfig(tp_degree=2, pp_degree=1, ddp_degree=4)
    master_config = MasterConfig(training=training_config)
    config_manager = ConfigManager(config=master_config)

    with pytest.raises(ValueError) as exc_info:
        config_manager.validate_config()

    assert "TP and TP+DDP are supported only on a single server" in str(exc_info.value)


def test_parallelism_valid_tp_ddp_single_server(monkeypatch):
    """TP+DDP single-server should pass when WORLD_SIZE matches degree product."""
    from QEfficient.finetune.experimental.core.config_manager import MasterConfig, TrainingConfig

    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")

    training_config = TrainingConfig(tp_degree=2, pp_degree=1, ddp_degree=2)
    master_config = MasterConfig(training=training_config)
    config_manager = ConfigManager(config=master_config)

    # Should not raise
    config_manager.validate_config()
