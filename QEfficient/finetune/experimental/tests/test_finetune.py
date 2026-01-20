# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Unit tests for finetune_experimental.py.
Tests for FineTuningPipeline class and main() function.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from QEfficient.cloud.finetune_experimental import FineTuningPipeline, main
from QEfficient.finetune.experimental.core.config_manager import MasterConfig, TrainingConfig


class TestFineTuningPipeline:
    """Test suite for FineTuningPipeline class."""

    @pytest.fixture
    def mock_master_config(self):
        """Create a mock MasterConfig for testing."""
        config = MagicMock(spec=MasterConfig)
        config.training = MagicMock()
        config.training.output_dir = "./test_output"
        config.training.seed = 42
        return config

    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock ConfigManager."""
        config_manager = MagicMock()
        config_manager.get_training_config.return_value = {
            "type": "sft",
            "dtype": "fp16",
            "seed": 42,
        }
        config_manager.get_dataset_config.return_value = {
            "dataset_type": "sft_dataset",
            "dataset_name": "test_dataset",
            "train_split": "train",
            "test_split": "test",
        }
        config_manager.get_model_config.return_value = {
            "model_type": "hf",
            "model_name": "test-model",
            "use_peft": False,
        }
        config_manager.get_optimizer_config.return_value = {
            "optimizer_name": "adamw",
            "lr": 1e-4,
        }
        config_manager.get_callback_config.return_value = MagicMock()
        config_manager.get_callback_config.return_value.callbacks = {}
        config_manager.validate_config = MagicMock()
        return config_manager

    def test_initialization(self, mock_master_config):
        """Test pipeline initialization."""
        with patch("QEfficient.cloud.finetune_experimental.ConfigManager") as mock_cm:
            mock_cm.return_value = MagicMock()

            pipeline = FineTuningPipeline(mock_master_config)

            assert pipeline.config == mock_master_config
            assert isinstance(pipeline.output_dir, Path)
            assert pipeline.output_dir == Path("./test_output")

            # Verify ConfigManager was created with correct config
            assert mock_cm.call_count > 0
            assert mock_cm.call_args[0][0] == mock_master_config

    def test_setup_environment(self, mock_master_config):
        """Test environment variable setup."""
        with patch("QEfficient.cloud.finetune_experimental.ConfigManager") as mock_cm:
            mock_cm.return_value = MagicMock()

            # Clear environment variables
            env_vars = ["OUTPUT_DIR", "TRACKIO_DIR", "TENSORBOARD_LOGGING_DIR"]
            for var in env_vars:
                if var in os.environ:
                    del os.environ[var]

            pipeline = FineTuningPipeline(mock_master_config)

            # Verify environment variables are set
            assert os.environ["OUTPUT_DIR"] == str(pipeline.output_dir)
            assert os.environ["TRACKIO_DIR"] == str(pipeline.output_dir / "trackio_logs")
            assert os.environ["TENSORBOARD_LOGGING_DIR"] == str(pipeline.output_dir)

    def test_prepare_training_config(self, mock_master_config, mock_config_manager):
        """Test training config preparation."""
        with patch("QEfficient.cloud.finetune_experimental.ConfigManager", return_value=mock_config_manager):
            with patch("QEfficient.cloud.finetune_experimental.prepare_training_config") as mock_prepare:
                mock_prepare.return_value = {"fp16": True, "seed": 42}

                pipeline = FineTuningPipeline(mock_master_config)
                result = pipeline._prepare_training_config()

                # Verify prepare_training_config was called with correct overrides
                assert mock_prepare.call_count > 0
                call_kwargs = mock_prepare.call_args[1]
                assert call_kwargs["include_num_input_tokens_seen"] is False
                assert call_kwargs["use_cpu"] is False
                assert call_kwargs["config_manager"] == mock_config_manager

                assert result == {"fp16": True, "seed": 42}

    def test_create_datasets(self, mock_master_config, mock_config_manager):
        """Test dataset creation."""
        with patch("QEfficient.cloud.finetune_experimental.ConfigManager", return_value=mock_config_manager):
            with patch("QEfficient.cloud.finetune_experimental.ComponentFactory") as mock_factory:
                mock_train_dataset = MagicMock()
                mock_eval_dataset = MagicMock()

                def create_dataset_side_effect(*args, **kwargs):
                    split = kwargs.get("split", "")
                    if "train" in split:
                        return mock_train_dataset
                    return mock_eval_dataset

                mock_factory.create_dataset.side_effect = create_dataset_side_effect

                pipeline = FineTuningPipeline(mock_master_config)
                train_dataset, eval_dataset = pipeline._create_datasets()

                # Verify datasets were created
                assert train_dataset == mock_train_dataset
                assert eval_dataset == mock_eval_dataset

                # Verify create_dataset was called twice (train and test)
                assert mock_factory.create_dataset.call_count == 2

                # Verify correct parameters were passed
                calls = mock_factory.create_dataset.call_args_list
                train_call = calls[0]
                assert train_call[1]["split"] == "train"
                assert train_call[1]["seed"] == 42
                assert train_call[1]["dataset_type"] == "sft_dataset"
                assert train_call[1]["dataset_name"] == "test_dataset"

    def test_create_datasets_with_custom_splits(self, mock_master_config, mock_config_manager):
        """Test dataset creation with custom split names."""
        mock_config_manager.get_dataset_config.return_value = {
            "dataset_type": "sft_dataset",
            "dataset_name": "test_dataset",
            "train_split": "training",
            "test_split": "testing",
        }

        with patch("QEfficient.cloud.finetune_experimental.ConfigManager", return_value=mock_config_manager):
            with patch("QEfficient.cloud.finetune_experimental.ComponentFactory") as mock_factory:
                mock_factory.create_dataset.return_value = MagicMock()

                pipeline = FineTuningPipeline(mock_master_config)
                pipeline._create_datasets()

                # Verify custom splits were used
                calls = mock_factory.create_dataset.call_args_list
                assert calls[0][1]["split"] == "training"
                assert calls[1][1]["split"] == "testing"

    def test_create_model(self, mock_master_config, mock_config_manager):
        """Test model creation."""
        mock_config_manager.get_training_config.return_value = {
            "dtype": "fp16",
        }

        mock_model_instance = MagicMock()
        mock_model_instance.model = MagicMock()
        mock_model_instance.tokenizer = MagicMock()

        with patch("QEfficient.cloud.finetune_experimental.ConfigManager", return_value=mock_config_manager):
            with patch("QEfficient.cloud.finetune_experimental.ComponentFactory") as mock_factory:
                mock_factory.create_model.return_value = mock_model_instance

                pipeline = FineTuningPipeline(mock_master_config)
                result = pipeline._create_model()

                assert result == mock_model_instance

                # Verify model was created with correct dtype conversion
                assert mock_factory.create_model.call_count > 0
                call_kwargs = mock_factory.create_model.call_args[1]
                assert call_kwargs["dtype"] == "float16"  # fp16 -> float16

    def test_create_model_bf16(self, mock_master_config, mock_config_manager):
        """Test model creation with bf16 dtype."""
        mock_config_manager.get_training_config.return_value = {
            "dtype": "bf16",
        }

        with patch("QEfficient.cloud.finetune_experimental.ConfigManager", return_value=mock_config_manager):
            with patch("QEfficient.cloud.finetune_experimental.ComponentFactory") as mock_factory:
                mock_factory.create_model.return_value = MagicMock()

                pipeline = FineTuningPipeline(mock_master_config)
                pipeline._create_model()

                call_kwargs = mock_factory.create_model.call_args[1]
                assert call_kwargs["dtype"] == "bfloat16"  # bf16 -> bfloat16

    def test_create_model_auto_dtype(self, mock_master_config, mock_config_manager):
        """Test model creation with auto dtype fallback."""
        mock_config_manager.get_training_config.return_value = {
            "dtype": "unknown",
        }

        with patch("QEfficient.cloud.finetune_experimental.ConfigManager", return_value=mock_config_manager):
            with patch("QEfficient.cloud.finetune_experimental.ComponentFactory") as mock_factory:
                mock_factory.create_model.return_value = MagicMock()

                pipeline = FineTuningPipeline(mock_master_config)
                pipeline._create_model()

                call_kwargs = mock_factory.create_model.call_args[1]
                assert call_kwargs["dtype"] == "auto"  # Unknown dtype -> auto

    def test_create_optimizer(self, mock_master_config, mock_config_manager):
        """Test optimizer creation."""
        mock_optimizer_cls = MagicMock()
        mock_optimizer_kwargs = {"lr": 1e-4}

        with patch("QEfficient.cloud.finetune_experimental.ConfigManager", return_value=mock_config_manager):
            with patch("QEfficient.cloud.finetune_experimental.prepare_optimizer") as mock_prepare:
                mock_prepare.return_value = (mock_optimizer_cls, mock_optimizer_kwargs)

                pipeline = FineTuningPipeline(mock_master_config)
                optimizer_cls, optimizer_kwargs = pipeline._create_optimizer()

                assert optimizer_cls == mock_optimizer_cls
                assert optimizer_kwargs == mock_optimizer_kwargs

                assert mock_prepare.call_count > 0
                assert mock_prepare.call_args[0][0] == mock_config_manager.get_optimizer_config.return_value

    def test_create_callbacks(self, mock_master_config, mock_config_manager):
        """Test callback creation."""
        mock_config_manager.get_callback_config.return_value.callbacks = {
            "early_stopping": {"early_stopping_patience": 3},
            "tensorboard": {},
        }

        mock_callback1 = MagicMock()
        mock_callback2 = MagicMock()

        with patch("QEfficient.cloud.finetune_experimental.ConfigManager", return_value=mock_config_manager):
            with patch("QEfficient.cloud.finetune_experimental.create_callbacks") as mock_create:
                mock_create.side_effect = [mock_callback1, mock_callback2]

                pipeline = FineTuningPipeline(mock_master_config)
                callbacks = pipeline._create_callbacks()

                assert len(callbacks) == 2
                assert mock_callback1 in callbacks
                assert mock_callback2 in callbacks

                # Verify callbacks were created with correct names
                assert mock_create.call_count == 2
                assert mock_create.call_args_list[0][0][0] == "early_stopping"
                assert mock_create.call_args_list[1][0][0] == "tensorboard"

    def test_create_callbacks_with_failure(self, mock_master_config, mock_config_manager):
        """Test callback creation with one failure."""
        mock_config_manager.get_callback_config.return_value.callbacks = {
            "early_stopping": {"early_stopping_patience": 3},
            "invalid_callback": {},
        }

        mock_callback = MagicMock()

        with patch("QEfficient.cloud.finetune_experimental.ConfigManager", return_value=mock_config_manager):
            with patch("QEfficient.cloud.finetune_experimental.create_callbacks") as mock_create:
                with patch("QEfficient.cloud.finetune_experimental.logger") as mock_logger:
                    mock_create.side_effect = [
                        mock_callback,
                        ValueError("Unknown callback"),
                    ]

                    pipeline = FineTuningPipeline(mock_master_config)
                    callbacks = pipeline._create_callbacks()

                    # Should only have the successful callback
                    assert len(callbacks) == 1
                    assert mock_callback in callbacks

                    # Should log warning for failed callback
                    log_calls = [call[0][0] for call in mock_logger.log_rank_zero.call_args_list if call]
                    assert any("Warning" in str(msg) and "invalid_callback" in str(msg) for msg in log_calls)

    def test_create_trainer(self, mock_master_config, mock_config_manager):
        """Test trainer creation."""
        mock_config_manager.get_training_config.return_value = {
            "type": "sft",
            "dtype": "fp16",
        }

        mock_trainer_cls = MagicMock()
        mock_args_cls = MagicMock()
        mock_args_instance = MagicMock()
        mock_args_cls.return_value = mock_args_instance

        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_train_dataset = MagicMock()
        mock_eval_dataset = MagicMock()
        mock_optimizer_cls = MagicMock()
        mock_optimizer_kwargs = {}
        mock_callbacks = [MagicMock()]

        training_config = {"type": "sft", "output_dir": "./output"}

        with patch("QEfficient.cloud.finetune_experimental.ConfigManager", return_value=mock_config_manager):
            with patch("QEfficient.cloud.finetune_experimental.create_trainer_config") as mock_create_trainer:
                with patch("QEfficient.cloud.finetune_experimental.replace_progress_callback") as mock_replace:
                    mock_create_trainer.return_value = (mock_trainer_cls, mock_args_cls, {})

                    pipeline = FineTuningPipeline(mock_master_config)
                    trainer = pipeline._create_trainer(
                        model=mock_model,
                        tokenizer=mock_tokenizer,
                        train_dataset=mock_train_dataset,
                        eval_dataset=mock_eval_dataset,
                        optimizer_cls_and_kwargs=(mock_optimizer_cls, mock_optimizer_kwargs),
                        callbacks=mock_callbacks,
                        training_config=training_config.copy(),
                    )

                    assert trainer == mock_trainer_instance

                    # Verify trainer was created with correct parameters
                    assert mock_trainer_cls.call_count > 0
                    call_kwargs = mock_trainer_cls.call_args[1]
                    assert call_kwargs["model"] == mock_model
                    assert call_kwargs["processing_class"] == mock_tokenizer
                    assert call_kwargs["args"] == mock_args_instance
                    assert call_kwargs["compute_loss_func"] is None
                    assert call_kwargs["train_dataset"] == mock_train_dataset
                    assert call_kwargs["eval_dataset"] == mock_eval_dataset
                    assert call_kwargs["optimizer_cls_and_kwargs"] == (mock_optimizer_cls, mock_optimizer_kwargs)
                    assert call_kwargs["callbacks"] == mock_callbacks

                    # Verify progress callback replacement was called
                    assert mock_replace.call_count > 0
                    replace_call_args = mock_replace.call_args[0]
                    assert replace_call_args[0] == mock_trainer_instance
                    assert replace_call_args[1] == mock_callbacks
                    # Third argument should be logger (can be None or Logger instance)
                    assert len(replace_call_args) >= 3

    def test_create_trainer_with_peft(self, mock_master_config, mock_config_manager):
        """Test trainer creation with PEFT enabled."""
        from peft import LoraConfig

        mock_config_manager.get_training_config.return_value = {
            "type": "sft",
            "dtype": "fp16",
        }
        mock_config_manager.get_model_config.return_value = {
            "model_type": "hf",
            "model_name": "test-model",
            "use_peft": True,
            "peft_config": MagicMock(),
        }

        mock_lora_config = MagicMock(spec=LoraConfig)

        with patch("QEfficient.cloud.finetune_experimental.ConfigManager", return_value=mock_config_manager):
            with patch("QEfficient.cloud.finetune_experimental.convert_peft_config_to_lora_config") as mock_convert:
                with patch("QEfficient.cloud.finetune_experimental.create_trainer_config") as mock_create_trainer:
                    mock_convert.return_value = mock_lora_config
                    mock_create_trainer.return_value = (MagicMock(), MagicMock(), {})

                    pipeline = FineTuningPipeline(mock_master_config)
                    training_config = {"type": "sft"}

                    pipeline._create_trainer(
                        model=MagicMock(),
                        tokenizer=MagicMock(),
                        train_dataset=MagicMock(),
                        eval_dataset=MagicMock(),
                        optimizer_cls_and_kwargs=(MagicMock(), {}),
                        callbacks=[],
                        training_config=training_config,
                    )

                    # Verify PEFT config was converted
                    assert mock_convert.call_count > 0

                    # Verify dependencies were passed to create_trainer_config
                    assert mock_create_trainer.call_count > 0
                    call_kwargs = mock_create_trainer.call_args[1]
                    assert "peft_config" in call_kwargs
                    assert call_kwargs["peft_config"] == mock_lora_config

    def test_create_trainer_without_peft(self, mock_master_config, mock_config_manager):
        """Test trainer creation without PEFT."""
        mock_config_manager.get_training_config.return_value = {
            "type": "sft",
            "dtype": "fp16",
        }
        mock_config_manager.get_model_config.return_value = {
            "model_type": "hf",
            "model_name": "test-model",
            "use_peft": False,
        }

        with patch("QEfficient.cloud.finetune_experimental.ConfigManager", return_value=mock_config_manager):
            with patch("QEfficient.cloud.finetune_experimental.create_trainer_config") as mock_create_trainer:
                mock_create_trainer.return_value = (MagicMock(), MagicMock(), {})

                pipeline = FineTuningPipeline(mock_master_config)
                training_config = {"type": "sft"}

                pipeline._create_trainer(
                    model=MagicMock(),
                    tokenizer=MagicMock(),
                    train_dataset=MagicMock(),
                    eval_dataset=MagicMock(),
                    optimizer_cls_and_kwargs=(MagicMock(), {}),
                    callbacks=[],
                    training_config=training_config,
                )

                # Verify no PEFT config in dependencies
                call_kwargs = mock_create_trainer.call_args[1]
                assert "peft_config" not in call_kwargs or call_kwargs.get("peft_config") is None

    def test_run_full_pipeline(self, mock_master_config, mock_config_manager):
        """Test full pipeline execution."""
        mock_train_dataset = MagicMock()
        mock_eval_dataset = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.model = MagicMock()
        mock_model_instance.tokenizer = MagicMock()
        mock_optimizer_cls = MagicMock()
        mock_optimizer_kwargs = {}
        mock_callbacks = [MagicMock()]
        mock_trainer = MagicMock()

        with patch("QEfficient.cloud.finetune_experimental.ConfigManager", return_value=mock_config_manager):
            with patch.object(FineTuningPipeline, "_prepare_training_config", return_value={"type": "sft"}):
                with patch.object(
                    FineTuningPipeline, "_create_datasets", return_value=(mock_train_dataset, mock_eval_dataset)
                ):
                    with patch.object(FineTuningPipeline, "_create_model", return_value=mock_model_instance):
                        with patch.object(
                            FineTuningPipeline,
                            "_create_optimizer",
                            return_value=(mock_optimizer_cls, mock_optimizer_kwargs),
                        ):
                            with patch.object(FineTuningPipeline, "_create_callbacks", return_value=mock_callbacks):
                                with patch.object(FineTuningPipeline, "_create_trainer", return_value=mock_trainer):
                                    with patch("QEfficient.cloud.finetune_experimental.logger") as mock_logger:
                                        pipeline = FineTuningPipeline(mock_master_config)
                                        pipeline.run()

                                        # Verify all steps were executed
                                        assert mock_config_manager.validate_config.call_count > 0
                                        assert pipeline._prepare_training_config.call_count > 0
                                        assert pipeline._create_datasets.call_count > 0
                                        assert pipeline._create_model.call_count > 0
                                        assert pipeline._create_optimizer.call_count > 0
                                        assert pipeline._create_callbacks.call_count > 0
                                        assert pipeline._create_trainer.call_count > 0
                                        assert mock_trainer.train.call_count > 0

                                        # Verify logging occurred
                                        log_messages = [
                                            call[0][0] for call in mock_logger.log_rank_zero.call_args_list if call
                                        ]
                                        assert any("Creating datasets" in msg for msg in log_messages)
                                        assert any("Loading model" in msg for msg in log_messages)
                                        assert any("Preparing optimizer" in msg for msg in log_messages)
                                        assert any("Creating callbacks" in msg for msg in log_messages)
                                        assert any("Initializing trainer" in msg for msg in log_messages)
                                        assert any("Starting training" in msg for msg in log_messages)

    def test_run_with_validation_error(self, mock_master_config, mock_config_manager):
        """Test pipeline run with validation error."""
        mock_config_manager.validate_config.side_effect = ValueError("Invalid config")

        with patch("QEfficient.cloud.finetune_experimental.ConfigManager", return_value=mock_config_manager):
            pipeline = FineTuningPipeline(mock_master_config)

            with pytest.raises(ValueError, match="Invalid config"):
                pipeline.run()

    def test_output_dir_path_handling(self, mock_master_config):
        """Test output directory path handling."""
        mock_master_config.training.output_dir = "/absolute/path/to/output"

        with patch("QEfficient.cloud.finetune_experimental.ConfigManager"):
            pipeline = FineTuningPipeline(mock_master_config)

            assert isinstance(pipeline.output_dir, Path)
            assert str(pipeline.output_dir) == "/absolute/path/to/output"

    def test_output_dir_relative_path(self, mock_master_config):
        """Test output directory with relative path."""
        mock_master_config.training.output_dir = "./relative/output"

        with patch("QEfficient.cloud.finetune_experimental.ConfigManager"):
            pipeline = FineTuningPipeline(mock_master_config)

            assert isinstance(pipeline.output_dir, Path)
            assert pipeline.output_dir == Path("./relative/output")


class TestMainFunction:
    """Test suite for main() function."""

    def test_main_function(self):
        """Test main function execution."""
        mock_config = MagicMock()
        mock_pipeline = MagicMock()

        with patch("QEfficient.cloud.finetune_experimental.parse_arguments", return_value=mock_config):
            with patch("QEfficient.cloud.finetune_experimental.FineTuningPipeline", return_value=mock_pipeline):
                main()

                # Verify pipeline was created and run
                from QEfficient.cloud.finetune_experimental import FineTuningPipeline

                assert FineTuningPipeline.call_count > 0
                assert FineTuningPipeline.call_args[0][0] == mock_config
                assert mock_pipeline.run.call_count > 0

    def test_main_with_parse_error(self):
        """Test main function with parse error."""
        with patch("QEfficient.cloud.finetune_experimental.parse_arguments", side_effect=ValueError("Parse error")):
            with pytest.raises(ValueError, match="Parse error"):
                main()

    def test_main_with_pipeline_error(self):
        """Test main function with pipeline error."""
        mock_config = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = RuntimeError("Training failed")

        with patch("QEfficient.cloud.finetune_experimental.parse_arguments", return_value=mock_config):
            with patch("QEfficient.cloud.finetune_experimental.FineTuningPipeline", return_value=mock_pipeline):
                with pytest.raises(RuntimeError, match="Training failed"):
                    main()
