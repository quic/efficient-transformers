from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# 👇 CHANGE THIS to the module path where FineTuningPipeline is defined
MODULE = "QEfficient.cloud.finetune_experimental"

# Import the class under test
FineTuningPipeline = __import__(MODULE, fromlist=["FineTuningPipeline"]).FineTuningPipeline


# ---------- Fixtures ----------


@pytest.fixture
def tmp_outdir(tmp_path):
    return tmp_path / "out"


@pytest.fixture
def mock_config_manager(mocker, tmp_outdir):
    """
    Minimal ConfigManager double:
      - .config.training is dict-like with 'output_dir'
    """
    cm = mocker.MagicMock(name="ConfigManager")
    cm.config = mocker.MagicMock()
    cm.config.training = {"output_dir": str(tmp_outdir)}
    return cm


@pytest.fixture
def mock_logger(mocker):
    """
    Patch the module-level logger used inside the pipeline.
    """
    logger = __import__(MODULE, fromlist=["logger"]).logger
    # Ensure log_rank_zero exists and is mockable
    mocker.patch.object(logger, "log_rank_zero", autospec=True)
    return logger


@pytest.fixture
def training_config_stub(mocker):
    """
    Patch prepare_training_config to avoid side effects and make it assertable.
    """
    return_value = {"some_training_key": "some_training_value"}
    patcher = mocker.patch(
        f"{MODULE}.prepare_training_config",
        autospec=True,
        return_value=return_value,
    )
    return patcher, return_value


@pytest.fixture
def model_bundle(mocker):
    """
    A tiny 'model instance' object that the pipeline expects from _create_model().
    Must have .model and .tokenizer attributes.
    """
    bundle = mocker.MagicMock(name="ModelBundle")
    bundle.model = mocker.MagicMock(name="model")
    bundle.tokenizer = mocker.MagicMock(name="tokenizer")
    return bundle


# ---------- Tests ----------


def test_initialization(
    mocker,
    mock_config_manager,
    mock_logger,
    training_config_stub,
    model_bundle,
):
    # Arrange: patch all internal factory steps to isolate the constructor
    patch_prepare_training_config, training_cfg = training_config_stub

    mock_setup_env = mocker.patch.object(FineTuningPipeline, "_setup_environment", autospec=True)

    train_ds = mocker.MagicMock(name="train_dataset")
    eval_ds = mocker.MagicMock(name="eval_dataset")
    mock_create_datasets = mocker.patch.object(
        FineTuningPipeline,
        "_create_datasets",
        autospec=True,
        return_value=(train_ds, eval_ds),
    )

    mock_create_model = mocker.patch.object(
        FineTuningPipeline,
        "_create_model",
        autospec=True,
        return_value=model_bundle,
    )

    optim_cls = mocker.MagicMock(name="OptimizerClass")
    optim_kwargs = {"lr": 1e-4}
    mock_create_optimizer = mocker.patch.object(
        FineTuningPipeline,
        "_create_optimizer",
        autospec=True,
        return_value=(optim_cls, optim_kwargs),
    )

    callbacks = [mocker.MagicMock(name="Callback")]
    mock_create_callbacks = mocker.patch.object(
        FineTuningPipeline,
        "_create_callbacks",
        autospec=True,
        return_value=callbacks,
    )

    trainer_obj = mocker.MagicMock(name="Trainer")
    mock_create_trainer = mocker.patch.object(
        FineTuningPipeline,
        "_create_trainer",
        autospec=True,
        return_value=trainer_obj,
    )
    pipeline = FineTuningPipeline(mock_config_manager)

    # Assert: environment + training config prepared
    mock_setup_env.assert_called_once()
    patch_prepare_training_config.assert_called_once_with(config_manager=mock_config_manager)
    assert pipeline.training_config == training_cfg

    # Assert: datasets created and assigned
    mock_create_datasets.assert_called_once()
    assert pipeline.train_dataset is train_ds
    assert pipeline.eval_dataset is eval_ds

    # Assert: model/tokenizer assigned
    mock_create_model.assert_called_once()
    assert pipeline.model is model_bundle.model
    assert pipeline.tokenizer is model_bundle.tokenizer

    # Assert: optimizer + callbacks
    mock_create_optimizer.assert_called_once()
    mock_create_callbacks.assert_called_once()
    assert pipeline.optimizer_cls_and_kwargs == (optim_cls, optim_kwargs)
    assert pipeline.callbacks == callbacks

    # Assert: trainer constructed with expected wiring
    mock_create_trainer.assert_called_once_with(
        mocker.ANY,  # self (bound by autospec)
        model=model_bundle.model,
        tokenizer=model_bundle.tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        optimizer_cls_and_kwargs=(optim_cls, optim_kwargs),
        callbacks=callbacks,
        training_config=training_cfg,
    )
    assert pipeline.trainer is trainer_obj

    # Assert: logger calls (optional but nice to verify sequence)
    lr0 = mock_logger.log_rank_zero
    # We only assert messages were emitted; you can also assert exact call order if desired.
    expected_msgs = [
        mocker.call("Creating datasets..."),
        mocker.call("Loading model and tokenizer..."),
        mocker.call("Preparing optimizer..."),
        mocker.call("Creating callbacks..."),
        mocker.call("Initializing trainer..."),
    ]
    lr0.assert_has_calls(expected_msgs, any_order=False)


# ---------- Tests: individual steps / behaviors ----------


def test_setup_environment_called_and_output_dir_set(mocker, mock_config_manager, tmp_outdir):

    mocker.patch.object(FineTuningPipeline, "_setup_environment", autospec=True)
    mocker.patch.object(FineTuningPipeline, "_create_datasets", autospec=True, return_value=(None, None))
    mocker.patch.object(
        FineTuningPipeline, "_create_model", autospec=True, return_value=mocker.MagicMock(model=None, tokenizer=None)
    )
    mocker.patch.object(FineTuningPipeline, "_create_optimizer", autospec=True, return_value=(None, {}))
    mocker.patch.object(FineTuningPipeline, "_create_callbacks", autospec=True, return_value=[])
    mocker.patch.object(FineTuningPipeline, "_create_trainer", autospec=True, return_value=mocker.MagicMock())
    mocker.patch(f"{MODULE}.prepare_training_config", autospec=True, return_value={})

    pipe = FineTuningPipeline(mock_config_manager)

    # Assert
    assert Path(pipe.output_dir) == Path(tmp_outdir)


@pytest.mark.parametrize(
    "train_split,test_split,expected_train_split,expected_test_split",
    [
        ("train", "test", "train", "test"),  # Default splits
        ("training", "testing", "training", "testing"),  # Custom splits
    ],
)
def test_create_datasets_called_and_assigned(
    mocker,
    mock_config_manager,
    train_split,
    test_split,
    expected_train_split,
    expected_test_split,
):
    """Test dataset creation with default and custom split names."""
    mocker.patch(
        f"{MODULE}.prepare_training_config",
        autospec=True,
        return_value={"fp16": True, "torch_dtype": "fp16"},
    )

    mock_config_manager.config.training = {
        "output_dir": "tmp_outdir",
        "seed": 42,
    }

    mock_config_manager.get_dataset_config.return_value = {
        "dataset_type": "sft_dataset",
        "dataset_name": "test_dataset",
        "train_split": train_split,
        "test_split": test_split,
    }

    train_ds = MagicMock(name="train_ds")
    eval_ds = MagicMock(name="eval_ds")

    def create_dataset_side_effect(*args, **kwargs):
        split = kwargs.get("split")
        if split is None and args:
            split = args[0]
        split = split or ""
        return train_ds if expected_train_split in split else eval_ds

    with patch(f"{MODULE}.ComponentFactory") as mock_factory:
        mock_factory.create_dataset.side_effect = create_dataset_side_effect
        mocker.patch.object(FineTuningPipeline, "_setup_environment", autospec=True)
        bundle = MagicMock(model=mocker.MagicMock(), tokenizer=mocker.MagicMock())
        mocker.patch.object(FineTuningPipeline, "_create_model", autospec=True, return_value=bundle)
        mocker.patch.object(FineTuningPipeline, "_create_optimizer", autospec=True, return_value=(None, {}))
        mocker.patch.object(FineTuningPipeline, "_create_callbacks", autospec=True, return_value=[])
        mocker.patch.object(FineTuningPipeline, "_create_trainer", autospec=True, return_value=mocker.MagicMock())

        pipeline = FineTuningPipeline(mock_config_manager)
        assert pipeline.train_dataset == train_ds
        assert pipeline.eval_dataset == eval_ds
        calls = mock_factory.create_dataset.call_args_list
        assert len(calls) == 2, f"Expected two calls (train/test), got {len(calls)}: {calls}"
        assert calls[0].kwargs["split"] == expected_train_split
        assert calls[1].kwargs["split"] == expected_test_split
        assert calls[0].kwargs["seed"] == 42
        assert calls[0].kwargs["dataset_type"] == "sft_dataset"
        assert calls[0].kwargs["dataset_name"] == "test_dataset"


def test_create_model_failure_stops_pipeline(mocker, mock_config_manager):
    # Arrange
    mocker.patch(f"{MODULE}.prepare_training_config", autospec=True, return_value={})
    mocker.patch.object(FineTuningPipeline, "_setup_environment", autospec=True)
    mocker.patch.object(FineTuningPipeline, "_create_datasets", autospec=True, return_value=(None, None))

    mock_create_model = mocker.patch.object(
        FineTuningPipeline, "_create_model", autospec=True, side_effect=RuntimeError("model load failed")
    )
    mock_create_optimizer = mocker.patch.object(FineTuningPipeline, "_create_optimizer", autospec=True)
    mock_create_callbacks = mocker.patch.object(FineTuningPipeline, "_create_callbacks", autospec=True)
    mock_create_trainer = mocker.patch.object(FineTuningPipeline, "_create_trainer", autospec=True)

    with pytest.raises(RuntimeError, match="model load failed"):
        _ = FineTuningPipeline(mock_config_manager)

    mock_create_model.assert_called_once()
    mock_create_optimizer.assert_not_called()
    mock_create_callbacks.assert_not_called()
    mock_create_trainer.assert_not_called()


def test_trainer_receives_expected_arguments(mocker, mock_config_manager, model_bundle):
    training_cfg = {"epochs": 1}
    mocker.patch(f"{MODULE}.prepare_training_config", autospec=True, return_value=training_cfg)
    mocker.patch.object(FineTuningPipeline, "_setup_environment", autospec=True)

    train_ds = mocker.MagicMock(name="T")
    eval_ds = mocker.MagicMock(name="E")
    mocker.patch.object(FineTuningPipeline, "_create_datasets", autospec=True, return_value=(train_ds, eval_ds))
    mocker.patch.object(FineTuningPipeline, "_create_model", autospec=True, return_value=model_bundle)

    optim_cls = object()
    optim_kwargs = {"weight_decay": 0.01}
    mocker.patch.object(FineTuningPipeline, "_create_optimizer", autospec=True, return_value=(optim_cls, optim_kwargs))

    callbacks = [mocker.MagicMock()]
    mocker.patch.object(FineTuningPipeline, "_create_callbacks", autospec=True, return_value=callbacks)

    trainer_obj = mocker.MagicMock(name="Trainer")
    mocked_trainer = mocker.patch.object(FineTuningPipeline, "_create_trainer", autospec=True, return_value=trainer_obj)

    pipe = FineTuningPipeline(mock_config_manager)

    # Assert: _create_trainer wiring
    mocked_trainer.assert_called_once_with(
        mocker.ANY, 
        model=model_bundle.model,
        tokenizer=model_bundle.tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        optimizer_cls_and_kwargs=(optim_cls, optim_kwargs),
        callbacks=callbacks,
        training_config=training_cfg,
    )
    assert pipe.trainer is trainer_obj


def test_create_datasets_failure_stops_pipeline(mocker, mock_config_manager):
    """
    If _create_datasets raises, pipeline should not proceed to model/optimizer/trainer.
    """
    # Arrange
    mocker.patch(f"{MODULE}.prepare_training_config", autospec=True, return_value={})
    mocker.patch.object(FineTuningPipeline, "_setup_environment", autospec=True)

    mock_create_datasets = mocker.patch.object(
        FineTuningPipeline,
        "_create_datasets",
        autospec=True,
        side_effect=RuntimeError("dataset failure"),
    )

    mock_create_model = mocker.patch.object(FineTuningPipeline, "_create_model", autospec=True)
    mock_create_optimizer = mocker.patch.object(FineTuningPipeline, "_create_optimizer", autospec=True)
    mock_create_callbacks = mocker.patch.object(FineTuningPipeline, "_create_callbacks", autospec=True)
    mock_create_trainer = mocker.patch.object(FineTuningPipeline, "_create_trainer", autospec=True)

    with pytest.raises(RuntimeError, match="dataset failure"):
        _ = FineTuningPipeline(mock_config_manager)

    # Downstream steps must not run
    mock_create_datasets.assert_called_once()
    mock_create_model.assert_not_called()
    mock_create_optimizer.assert_not_called()
    mock_create_callbacks.assert_not_called()
    mock_create_trainer.assert_not_called()


def test_create_trainer_failure_stops_pipeline(mocker, mock_config_manager):
    """
    If _create_trainer raises, ensure earlier steps ran and no further actions are taken.
    """
    # Arrange
    mocker.patch(f"{MODULE}.prepare_training_config", autospec=True, return_value={})
    mocker.patch.object(FineTuningPipeline, "_setup_environment", autospec=True)

    train_ds = mocker.MagicMock(name="train_ds")
    eval_ds = mocker.MagicMock(name="eval_ds")
    mocker.patch.object(FineTuningPipeline, "_create_datasets", autospec=True, return_value=(train_ds, eval_ds))

    bundle = mocker.MagicMock(name="ModelBundle")
    bundle.model = mocker.MagicMock(name="model")
    bundle.tokenizer = mocker.MagicMock(name="tokenizer")
    mocker.patch.object(FineTuningPipeline, "_create_model", autospec=True, return_value=bundle)

    optim_cls = mocker.MagicMock(name="OptimClass")
    optim_kwargs = {"lr": 1e-4}
    mocker.patch.object(FineTuningPipeline, "_create_optimizer", autospec=True, return_value=(optim_cls, optim_kwargs))

    callbacks = [mocker.MagicMock(name="Callback")]
    mocker.patch.object(FineTuningPipeline, "_create_callbacks", autospec=True, return_value=callbacks)

    mock_create_trainer = mocker.patch.object(
        FineTuningPipeline,
        "_create_trainer",
        autospec=True,
        side_effect=RuntimeError("trainer init failed"),
    )

    with pytest.raises(RuntimeError, match="trainer init failed"):
        _ = FineTuningPipeline(mock_config_manager)

    mock_create_trainer.assert_called_once()


def test_config_manager_used_and_output_dir_set(mocker, mock_config_manager, tmp_outdir):
    """
    Ensure prepare_training_config is called with the provided config_manager
    and that output_dir is read from config.training.
    """
    training_cfg = {"epochs": 1}
    patch_prep = mocker.patch(f"{MODULE}.prepare_training_config", autospec=True, return_value=training_cfg)
    mocker.patch.object(FineTuningPipeline, "_setup_environment", autospec=True)

    # Keep the rest minimal to complete __init__
    mocker.patch.object(FineTuningPipeline, "_create_datasets", autospec=True, return_value=(None, None))
    bundle = mocker.MagicMock(model=None, tokenizer=None)
    mocker.patch.object(FineTuningPipeline, "_create_model", autospec=True, return_value=bundle)
    mocker.patch.object(FineTuningPipeline, "_create_optimizer", autospec=True, return_value=(None, {}))
    mocker.patch.object(FineTuningPipeline, "_create_callbacks", autospec=True, return_value=[])
    mocker.patch.object(FineTuningPipeline, "_create_trainer", autospec=True, return_value=mocker.MagicMock())

    pipe = FineTuningPipeline(mock_config_manager)

    patch_prep.assert_called_once_with(config_manager=mock_config_manager)
    assert pipe.training_config == training_cfg
    assert Path(pipe.output_dir) == Path(tmp_outdir)


def test_complete_run_calls_trainer_train(mocker, mock_config_manager):
    """
    Tests trainer.train() is called during run().
    This is a basic smoke test for the main execution flow.
    """
    # Arrange a fully-initialized pipeline first
    mocker.patch.object(FineTuningPipeline, "_setup_environment", autospec=True)
    mocker.patch(f"{MODULE}.prepare_training_config", autospec=True, return_value={})
    mocker.patch.object(FineTuningPipeline, "_create_datasets", autospec=True, return_value=(None, None))
    bundle = mocker.MagicMock(model=mocker.MagicMock(), tokenizer=mocker.MagicMock())
    mocker.patch.object(FineTuningPipeline, "_create_model", autospec=True, return_value=bundle)
    mocker.patch.object(FineTuningPipeline, "_create_optimizer", autospec=True, return_value=(None, {}))
    mocker.patch.object(FineTuningPipeline, "_create_callbacks", autospec=True, return_value=[])
    trainer_obj = mocker.MagicMock()
    mocker.patch.object(FineTuningPipeline, "_create_trainer", autospec=True, return_value=trainer_obj)

    pipe = FineTuningPipeline(mock_config_manager)
    pipe.run()
    trainer_obj.train.assert_called_once()
