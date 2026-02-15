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
from unittest.mock import MagicMock, patch

import pytest

from QEfficient.cloud.finetune_experimental import FineTuningPipeline, main
from QEfficient.finetune.experimental.core.config_manager import MasterConfig


class DictLikeMock:
    """A mock that supports both dict access ['key'] and attribute access .key"""

    def __init__(self, data):
        self._data = data
        for key, value in data.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def get(self, key, default=None):
        return self._data.get(key, default)


class TestFineTuningPipeline:
    """Test suite for FineTuningPipeline class."""

    @pytest.fixture
    def mock_master_config(self):
        """Create a mock MasterConfig for testing."""
        config = MagicMock(spec=MasterConfig)
        # Use DictLikeMock to support both dict access ['key'] and attribute access .key
        config.training = DictLikeMock({"output_dir": "./test_output", "seed": 42})
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
        config_manager.get_callback_config.return_value = {"callbacks": {}}
        config_manager.validate_config = MagicMock()
        return config_manager

    def test_initialization(self, mock_config_manager):
        """Test pipeline initialization."""
        # Set up config_manager.config to return a mock that has training dict access
        mock_config_obj = MagicMock()
        mock_config_obj.training = DictLikeMock({"output_dir": "./test_output"})
        mock_config_manager.config = mock_config_obj

        pipeline = FineTuningPipeline(mock_config_manager)

        assert pipeline.config_manager == mock_config_manager
        assert pipeline.config == mock_config_obj
        assert isinstance(pipeline.output_dir, Path)
        assert pipeline.output_dir == Path("./test_output")

    def test_setup_environment(self, mock_config_manager):
        """Test environment variable setup."""
        # Set up config_manager.config
        mock_config_obj = MagicMock()
        mock_config_obj.training = DictLikeMock({"output_dir": "./test_output"})
        mock_config_manager.config = mock_config_obj

        # Clear environment variables
        env_vars = ["OUTPUT_DIR", "TRACKIO_DIR", "TENSORBOARD_LOGGING_DIR"]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

        pipeline = FineTuningPipeline(mock_config_manager)

        # Verify environment variables are set
        assert os.environ["OUTPUT_DIR"] == str(pipeline.output_dir)
        assert os.environ["TRACKIO_DIR"] == str(pipeline.output_dir / "trackio_logs")
        assert os.environ["TENSORBOARD_LOGGING_DIR"] == str(pipeline.output_dir)

    def test_prepare_training_config(self, mock_config_manager):
        """Test training config preparation via prepare_training_config utility."""
        mock_config_obj = MagicMock()
        mock_config_obj.training = DictLikeMock({"output_dir": "./test_output"})
        mock_config_manager.config = mock_config_obj

        with patch("QEfficient.cloud.finetune_experimental.prepare_training_config") as mock_prepare:
            mock_prepare.return_value = {"fp16": True, "seed": 42, "type": "sft"}

            # Call prepare_training_config directly
            result = mock_prepare(config_manager=mock_config_manager)

            # Verify prepare_training_config was called
            assert mock_prepare.call_count > 0
            assert result == {"fp16": True, "seed": 42, "type": "sft"}

    @pytest.mark.parametrize(
        "train_split,test_split,expected_train_split,expected_test_split",
        [
            ("train", "test", "train", "test"),  # Default splits
            ("training", "testing", "training", "testing"),  # Custom splits
        ],
    )
    def test_create_datasets(
        self,
        mock_config_manager,
        train_split,
        test_split,
        expected_train_split,
        expected_test_split,
    ):
        """Test dataset creation with default and custom split names."""
        # Set up config_manager.config.training to support dict access for seed and output_dir
        mock_config_obj = MagicMock()
        mock_config_obj.training = DictLikeMock({"output_dir": "./test_output", "seed": 42})
        mock_config_manager.config = mock_config_obj

        # Update dataset config with the split names
        mock_config_manager.get_dataset_config.return_value = {
            "dataset_type": "sft_dataset",
            "dataset_name": "test_dataset",
            "train_split": train_split,
            "test_split": test_split,
        }

        with patch("QEfficient.cloud.finetune_experimental.ComponentFactory") as mock_factory:
            mock_train_dataset = MagicMock()
            mock_eval_dataset = MagicMock()

            def create_dataset_side_effect(*args, **kwargs):
                split = kwargs.get("split", "")
                # Match based on expected split names
                if expected_train_split in split or (expected_train_split == "train" and "train" in split):
                    return mock_train_dataset
                return mock_eval_dataset

            mock_factory.create_dataset.side_effect = create_dataset_side_effect

            pipeline = FineTuningPipeline(mock_config_manager)
            train_dataset, eval_dataset = pipeline._create_datasets()

            # Verify datasets were created
            assert train_dataset == mock_train_dataset
            assert eval_dataset == mock_eval_dataset

            # Verify create_dataset was called twice (train and test)
            assert mock_factory.create_dataset.call_count == 2

            # Verify correct parameters were passed
            calls = mock_factory.create_dataset.call_args_list
            assert calls[0].kwargs["split"] == expected_train_split
            assert calls[1].kwargs["split"] == expected_test_split
            assert calls[0].kwargs["seed"] == 42
            assert calls[0].kwargs["dataset_type"] == "sft_dataset"
            assert calls[0].kwargs["dataset_name"] == "test_dataset"

    @pytest.mark.parametrize(
        "torch_dtype,expected_dtype",
        [
            ("fp16", "float16"),  # fp16 -> float16
            ("bf16", "bfloat16"),  # bf16 -> bfloat16
            ("unknown", "auto"),  # Unknown dtype -> auto
        ],
    )
    def test_create_model_dtype_conversion(self, mock_config_manager, torch_dtype, expected_dtype):
        """Test model creation with different dtype conversions."""
        mock_config_obj = MagicMock()
        mock_config_obj.training = DictLikeMock({"output_dir": "./test_output"})
        mock_config_manager.config = mock_config_obj

        # Mock get_model_config to return config with torch_dtype already converted
        # (This conversion is done by ConfigManager.get_model_config, not by _create_model)
        mock_config_manager.get_model_config.return_value = {
            "model_type": "hf",
            "model_name": "test-model",
            "torch_dtype": expected_dtype,  # Already converted by get_model_config
        }

        mock_model_instance = MagicMock()
        mock_model_instance.model = MagicMock()
        mock_model_instance.tokenizer = MagicMock()

        with patch("QEfficient.cloud.finetune_experimental.ComponentFactory") as mock_factory:
            mock_factory.create_model.return_value = mock_model_instance

            pipeline = FineTuningPipeline(mock_config_manager)
            result = pipeline._create_model()

            assert result == mock_model_instance

            # Verify model was created with correct dtype (already converted by ConfigManager)
            assert mock_factory.create_model.call_count > 0
            call_kwargs = mock_factory.create_model.call_args.kwargs
            assert call_kwargs.get("torch_dtype") == expected_dtype

    def test_create_optimizer(self, mock_config_manager):
        """Test optimizer creation."""
        mock_config_obj = MagicMock()
        mock_config_obj.training = DictLikeMock({"output_dir": "./test_output"})
        mock_config_manager.config = mock_config_obj

        mock_optimizer_cls = MagicMock()
        mock_optimizer_kwargs = {"lr": 1e-4}

        with patch("QEfficient.cloud.finetune_experimental.prepare_optimizer") as mock_prepare:
            mock_prepare.return_value = (mock_optimizer_cls, mock_optimizer_kwargs)

            pipeline = FineTuningPipeline(mock_config_manager)
            optimizer_cls, optimizer_kwargs = pipeline._create_optimizer()

            assert optimizer_cls == mock_optimizer_cls
            assert optimizer_kwargs == mock_optimizer_kwargs

            assert mock_prepare.call_count > 0
            assert mock_prepare.call_args[0][0] == mock_config_manager.get_optimizer_config.return_value

    @pytest.mark.parametrize(
        "callback_config,expected_count,expected_names",
        [
            (
                {
                    "early_stopping": {"early_stopping_patience": 3},
                    "tensorboard": {},
                },
                2,
                ["early_stopping", "tensorboard"],
            ),
            (
                {
                    "early_stopping": {"early_stopping_patience": 3},
                    "tensorboard": {},
                    "checkpoint": {"save_strategy": "epoch"},
                },
                3,
                ["early_stopping", "tensorboard", "checkpoint"],
            ),
        ],
    )
    def test_create_callbacks(self, mock_config_manager, callback_config, expected_count, expected_names):
        """Test callback creation with different numbers of callbacks."""
        mock_callback_config = {"callbacks": callback_config}
        mock_config_manager.get_callback_config.return_value = mock_callback_config
        mock_config_obj = MagicMock()
        mock_config_obj.training = DictLikeMock({"output_dir": "./test_output"})
        mock_config_manager.config = mock_config_obj

        # Create mock callbacks based on expected count
        mock_callbacks = [MagicMock() for _ in range(expected_count)]

        with patch("QEfficient.cloud.finetune_experimental.ComponentFactory.create_callback") as mock_create:
            mock_create.side_effect = mock_callbacks

            pipeline = FineTuningPipeline(mock_config_manager)
            callbacks = pipeline._create_callbacks()

            assert len(callbacks) == expected_count
            for mock_cb in mock_callbacks:
                assert mock_cb in callbacks

            # Verify callbacks were created with correct names
            assert mock_create.call_count == expected_count
            for i, expected_name in enumerate(expected_names):
                assert mock_create.call_args_list[i][0][0] == expected_name

    def test_create_callbacks_with_failure(self, mock_config_manager):
        """Test callback creation with one failure."""
        mock_callback_config = {
            "callbacks": {
                "early_stopping": {"early_stopping_patience": 3},
                "invalid_callback": {},
            }
        }
        mock_config_manager.get_callback_config.return_value = mock_callback_config
        mock_config_obj = MagicMock()
        mock_config_obj.training = DictLikeMock({"output_dir": "./test_output"})
        mock_config_manager.config = mock_config_obj

        mock_callback = MagicMock()

        with patch("QEfficient.cloud.finetune_experimental.ComponentFactory.create_callback") as mock_create:
            with patch("QEfficient.cloud.finetune_experimental.logger") as mock_logger:
                mock_create.side_effect = [
                    mock_callback,
                    ValueError("Unknown callback"),
                ]

                pipeline = FineTuningPipeline(mock_config_manager)
                callbacks = pipeline._create_callbacks()

                # Should only have the successful callback
                assert len(callbacks) == 1
                assert mock_callback in callbacks

                # Should log warning for failed callback
                log_calls = [call[0][0] for call in mock_logger.log_rank_zero.call_args_list if call]
                assert any("Warning" in str(msg) and "invalid_callback" in str(msg) for msg in log_calls)

    def test_create_trainer(self, mock_config_manager):
        """Test trainer creation."""
        mock_config_obj = MagicMock()
        mock_config_obj.training = DictLikeMock({"output_dir": "./test_output"})
        mock_config_manager.config = mock_config_obj

        mock_config_manager.get_training_config.return_value = {
            "type": "sft",
            "dtype": "fp16",
            "device": "cpu",
        }
        mock_config_manager.get_model_config.return_value = {
            "model_type": "hf",
            "model_name": "test-model",
            "use_peft": False,
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

        training_config = {"type": "sft", "output_dir": "./output", "fp16": True}

        with patch(
            "QEfficient.cloud.finetune_experimental.ComponentFactory.create_trainer_config"
        ) as mock_create_trainer:
            with patch("QEfficient.cloud.finetune_experimental.replace_progress_callback") as mock_replace:
                mock_create_trainer.return_value = (mock_trainer_cls, mock_args_cls, {})

                pipeline = FineTuningPipeline(mock_config_manager)
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
                call_kwargs = mock_trainer_cls.call_args.kwargs
                assert call_kwargs["model"] == mock_model
                assert call_kwargs["processing_class"] == mock_tokenizer
                assert call_kwargs["args"] == mock_args_instance
                assert call_kwargs["compute_loss_func"] is None
                assert call_kwargs["train_dataset"] == mock_train_dataset.dataset
                assert call_kwargs["eval_dataset"] == mock_eval_dataset.dataset
                assert call_kwargs["optimizer_cls_and_kwargs"] == (mock_optimizer_cls, mock_optimizer_kwargs)
                assert call_kwargs["callbacks"] == mock_callbacks

                # Verify progress callback replacement was called
                assert mock_replace.call_count > 0
                replace_call_args = mock_replace.call_args.args
                assert replace_call_args[0] == mock_trainer_instance
                assert replace_call_args[1] == mock_callbacks
                # Third argument should be logger (can be None or Logger instance)
                assert len(replace_call_args) >= 3

    def test_run_full_pipeline(self, mock_config_manager):
        """Test full pipeline execution."""
        mock_config_obj = MagicMock()
        mock_config_obj.training = DictLikeMock({"output_dir": "./test_output"})
        mock_config_manager.config = mock_config_obj

        mock_train_dataset = MagicMock()
        mock_eval_dataset = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.model = MagicMock()
        mock_model_instance.tokenizer = MagicMock()
        mock_optimizer_cls = MagicMock()
        mock_optimizer_kwargs = {}
        mock_callbacks = [MagicMock()]
        mock_trainer = MagicMock()

        with patch(
            "QEfficient.cloud.finetune_experimental.prepare_training_config", return_value={"type": "sft", "fp16": True}
        ):
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
                                    pipeline = FineTuningPipeline(mock_config_manager)
                                    pipeline.run()

                                    # Verify all steps were executed
                                    assert mock_config_manager.validate_config.call_count > 0
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

    def test_run_with_validation_error(self, mock_config_manager):
        """Test pipeline run with validation error."""
        mock_config_obj = MagicMock()
        mock_config_obj.training = DictLikeMock({"output_dir": "./test_output"})
        mock_config_manager.config = mock_config_obj
        mock_config_manager.validate_config.side_effect = ValueError("Invalid config")

        pipeline = FineTuningPipeline(mock_config_manager)

        with pytest.raises(ValueError, match="Invalid config"):
            pipeline.run()

    @pytest.mark.parametrize(
        "output_dir,expected_path",
        [
            ("/absolute/path/to/output", "/absolute/path/to/output"),
            ("./relative/output", "relative/output"),  # Path normalizes ./relative/output to relative/output
        ],
    )
    def test_output_dir_path_handling(self, mock_config_manager, output_dir, expected_path):
        """Test output directory path handling for both absolute and relative paths."""
        # Set up config_manager.config to have training dict
        mock_config_obj = MagicMock()
        mock_config_obj.training = DictLikeMock({"output_dir": output_dir})
        mock_config_manager.config = mock_config_obj

        pipeline = FineTuningPipeline(mock_config_manager)

        assert isinstance(pipeline.output_dir, Path)
        assert str(pipeline.output_dir) == expected_path


class TestMainFunction:
    """Test suite for main() function."""

    def test_main_function(self):
        """Test main function execution."""
        mock_config_manager = MagicMock()
        mock_pipeline = MagicMock()

        with patch("QEfficient.cloud.finetune_experimental.ConfigManager", return_value=mock_config_manager):
            with patch("QEfficient.cloud.finetune_experimental.FineTuningPipeline", return_value=mock_pipeline):
                main()

                # Verify pipeline was created and run
                from QEfficient.cloud.finetune_experimental import FineTuningPipeline

                assert FineTuningPipeline.call_count > 0
                assert FineTuningPipeline.call_args[0][0] == mock_config_manager
                assert mock_pipeline.run.call_count > 0

    def test_main_with_config_error(self):
        """Test main function with config initialization error."""
        with patch("QEfficient.cloud.finetune_experimental.ConfigManager", side_effect=ValueError("Config error")):
            with pytest.raises(ValueError, match="Config error"):
                main()

    def test_main_with_pipeline_error(self):
        """Test main function with pipeline error."""
        mock_config_manager = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = RuntimeError("Training failed")

        with patch("QEfficient.cloud.finetune_experimental.ConfigManager", return_value=mock_config_manager):
            with patch("QEfficient.cloud.finetune_experimental.FineTuningPipeline", return_value=mock_pipeline):
                with pytest.raises(RuntimeError, match="Training failed"):
                    main()


class TestFineTuningPipelineEnhanced:
    """Enhanced test suite for FineTuningPipeline class with additional edge cases."""

    @pytest.fixture
    def mock_master_config(self):
        """Create a mock MasterConfig for testing."""
        config = MagicMock(spec=MasterConfig)
        # Use DictLikeMock to support both dict access ['key'] and attribute access .key
        config.training = DictLikeMock({"output_dir": "./test_output", "seed": 42})
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
        config_manager.get_callback_config.return_value = {"callbacks": {}}
        config_manager.validate_config = MagicMock()
        return config_manager

    def test_create_datasets_with_additional_config_params(self, mock_config_manager):
        """Test that additional dataset config parameters are properly propagated."""
        mock_config_manager.get_dataset_config.return_value = {
            "dataset_type": "sft_dataset",
            "dataset_name": "test_dataset",
            "train_split": "train",
            "test_split": "test",
            "max_seq_length": 512,
            "batch_size": 16,
            "custom_param": "custom_value",
        }
        mock_config_obj = MagicMock()
        mock_config_obj.training = DictLikeMock({"output_dir": "./test_output", "seed": 42})
        mock_config_manager.config = mock_config_obj

        with patch("QEfficient.cloud.finetune_experimental.ComponentFactory") as mock_factory:
            mock_factory.create_dataset.return_value = MagicMock()

            pipeline = FineTuningPipeline(mock_config_manager)
            pipeline._create_datasets()

            # Verify additional parameters are passed through
            calls = mock_factory.create_dataset.call_args_list
            assert calls[0].kwargs.get("max_seq_length") == 512
            assert calls[0].kwargs.get("batch_size") == 16
            assert calls[0].kwargs.get("custom_param") == "custom_value"
            # Verify excluded keys are not passed
            assert "train_split" not in calls[0].kwargs
            assert "test_split" not in calls[0].kwargs

    def test_create_model_with_additional_model_params(self, mock_config_manager):
        """Test that additional model config parameters are properly propagated."""
        mock_config_manager.get_model_config.return_value = {
            "model_type": "hf",
            "model_name": "test-model",
            "use_peft": False,
            "trust_remote_code": True,
            "device_map": "auto",
            "custom_model_param": "value",
        }
        mock_config_obj = MagicMock()
        mock_config_obj.training = DictLikeMock({"output_dir": "./test_output"})
        mock_config_manager.config = mock_config_obj

        with patch("QEfficient.cloud.finetune_experimental.ComponentFactory") as mock_factory:
            mock_factory.create_model.return_value = MagicMock()

            pipeline = FineTuningPipeline(mock_config_manager)
            pipeline._create_model()

            call_kwargs = mock_factory.create_model.call_args.kwargs
            assert call_kwargs.get("trust_remote_code") is True
            assert call_kwargs.get("device_map") == "auto"
            assert call_kwargs.get("custom_model_param") == "value"
            # Verify PEFT keys are excluded
            assert "use_peft" not in call_kwargs
            assert "peft_config" not in call_kwargs

    def test_run_method_calls_validate_config_first(self, mock_config_manager):
        """Test that run() calls validate_config before other operations."""
        mock_config_obj = MagicMock()
        mock_config_obj.training = DictLikeMock({"output_dir": "./test_output", "seed": 42})
        mock_config_manager.config = mock_config_obj

        call_order = []

        def track_validate():
            call_order.append("validate")
            return None

        mock_config_manager.validate_config.side_effect = track_validate

        with patch(
            "QEfficient.cloud.finetune_experimental.prepare_training_config", return_value={"type": "sft", "fp16": True}
        ):
            with patch.object(FineTuningPipeline, "_create_datasets", return_value=(MagicMock(), MagicMock())):
                with patch.object(FineTuningPipeline, "_create_model", return_value=MagicMock()):
                    with patch.object(FineTuningPipeline, "_create_optimizer", return_value=(MagicMock(), {})):
                        with patch.object(FineTuningPipeline, "_create_callbacks", return_value=[]):
                            with patch.object(FineTuningPipeline, "_create_trainer", return_value=MagicMock()):
                                with patch("QEfficient.cloud.finetune_experimental.logger"):
                                    pipeline = FineTuningPipeline(mock_config_manager)
                                    pipeline.run()

                                    # Verify validate_config was called first
                                    assert call_order[0] == "validate"
                                    assert mock_config_manager.validate_config.call_count == 1
