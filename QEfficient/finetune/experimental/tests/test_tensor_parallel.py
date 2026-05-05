# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Tensor Parallelism (TP) tests for experimental finetuning pipeline.
Covers TP-only and TP+DDP enablement paths for non-MoE models.
"""

from unittest.mock import patch

import pytest
import torch
from accelerate.utils import ParallelismConfig

MODULE = "QEfficient.cloud.finetune_experimental"

FineTuningPipeline = __import__(MODULE, fromlist=["FineTuningPipeline"]).FineTuningPipeline


# ---------------------------------------------------------------------------
# 1. Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_outdir(tmp_path):
    return tmp_path / "out"


@pytest.fixture
def mock_config_manager(mocker, tmp_outdir):
    cm = mocker.MagicMock(name="ConfigManager")
    cm.config = mocker.MagicMock()
    cm.config.training = {"output_dir": str(tmp_outdir)}
    return cm


@pytest.fixture
def model_bundle(mocker):
    bundle = mocker.MagicMock(name="ModelBundle")
    bundle.model = mocker.MagicMock(name="model")
    bundle.tokenizer = mocker.MagicMock(name="tokenizer")
    return bundle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tp_training_cfg(ddp_degree: int) -> dict:
    """Create a minimal training config for TP-only or TP+DDP scenarios."""
    return {
        "tp_degree": 2,
        "ddp_degree": ddp_degree,
        "device": "qaic",
        "torch_dtype": "bf16",
        "parallelism_config": ParallelismConfig(tp_size=2, dp_replicate_size=ddp_degree),
        "type": "sft",
    }


# ---------------------------------------------------------------------------
# 2. FineTuningPipeline integration – TP enablement and flow
# ---------------------------------------------------------------------------


@pytest.mark.finetune
@pytest.mark.parametrize(
    "scenario_name,ddp_degree",
    [
        ("tp_only", 1),
        ("tp_ddp", 2),
    ],
)
def test_tp_non_moe_enablement_functionality(mocker, mock_config_manager, model_bundle, scenario_name, ddp_degree):
    """
    Functional test: TP enablement path for non-MoE models
    in both TP-only and TP+DDP training configurations.
    """
    training_cfg = _make_tp_training_cfg(ddp_degree)

    mocker.patch(f"{MODULE}.prepare_training_config", autospec=True, return_value=training_cfg)
    mock_init_dist = mocker.patch.object(FineTuningPipeline, "_initialize_dist_tp", autospec=True)
    mocker.patch.object(FineTuningPipeline, "_setup_environment", autospec=True)
    mocker.patch.object(FineTuningPipeline, "_create_datasets", autospec=True, return_value=(None, None))
    mocker.patch.object(FineTuningPipeline, "_create_model", autospec=True, return_value=model_bundle)
    mocker.patch.object(FineTuningPipeline, "_create_optimizer", autospec=True, return_value=(None, {}))
    mocker.patch.object(FineTuningPipeline, "_create_callbacks", autospec=True, return_value=[])
    mocker.patch.object(FineTuningPipeline, "_create_trainer", autospec=True, return_value=mocker.MagicMock())

    pipe = FineTuningPipeline(mock_config_manager)

    assert pipe.tp_enabled is True, f"TP should be enabled for scenario '{scenario_name}'"
    mock_init_dist.assert_called_once_with(pipe)


@pytest.mark.finetune
@pytest.mark.parametrize(
    "scenario_name,ddp_degree",
    [
        ("tp_only", 1),
        ("tp_ddp", 2),
    ],
)
def test_tp_non_moe_full_pipeline_check(mocker, mock_config_manager, model_bundle, scenario_name, ddp_degree):
    """
    Full pipeline check: constructor + run() flow
    for TP-only and TP+DDP with non-MoE model path.
    """
    training_cfg = _make_tp_training_cfg(ddp_degree)

    mocker.patch(f"{MODULE}.prepare_training_config", autospec=True, return_value=training_cfg)
    mock_init_dist = mocker.patch.object(FineTuningPipeline, "_initialize_dist_tp", autospec=True)
    mocker.patch.object(FineTuningPipeline, "_setup_environment", autospec=True)
    mocker.patch.object(FineTuningPipeline, "_create_datasets", autospec=True, return_value=(None, None))
    mocker.patch.object(FineTuningPipeline, "_create_model", autospec=True, return_value=model_bundle)
    mocker.patch.object(FineTuningPipeline, "_create_optimizer", autospec=True, return_value=(None, {}))
    mocker.patch.object(FineTuningPipeline, "_create_callbacks", autospec=True, return_value=[])

    trainer_obj = mocker.MagicMock(name=f"trainer_{scenario_name}")
    mocker.patch.object(FineTuningPipeline, "_create_trainer", autospec=True, return_value=trainer_obj)

    pipe = FineTuningPipeline(mock_config_manager)
    pipe.run()

    mock_init_dist.assert_called_once_with(pipe)
    trainer_obj.train.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Unit tests – model TP kwargs injection for non-MoE
# ---------------------------------------------------------------------------


@pytest.mark.finetune
def test_create_model_non_moe_tp_only_injects_tp_plan(mocker):
    """
    Unit test: non-MoE model creation under TP-only configuration
    injects TP arguments and does not apply PEFT when use_peft=False.
    """
    pipe = FineTuningPipeline.__new__(FineTuningPipeline)
    pipe.config_manager = mocker.MagicMock(name="ConfigManager")

    pc = ParallelismConfig(tp_size=2)
    mocker.patch.object(pc, "build_device_mesh", autospec=True, return_value={"tp": "tp_mesh"})

    pipe.training_config = {
        "tp_degree": 2,
        "device": "cpu",
        "parallelism_config": pc,
    }
    pipe.config_manager.get_model_config.return_value = {
        "model_type": "hf",
        "model_name": "non-moe-model",
        "use_peft": False,
    }

    model_instance = mocker.MagicMock(name="ModelInstance")
    original_weight = torch.nn.Parameter(torch.randn(4, 4))
    model_instance.model.lm_head.weight = original_weight

    with patch(f"{MODULE}.ComponentFactory") as mock_factory:
        mock_factory.create_model.return_value = model_instance
        mock_get_peft_model = mocker.patch(f"{MODULE}.get_peft_model", autospec=True)

        returned = pipe._create_model()

    assert returned is model_instance
    mock_get_peft_model.assert_not_called()
    assert model_instance.model.lm_head.weight is not original_weight
    assert isinstance(model_instance.model.lm_head.weight, torch.nn.Parameter)

    kwargs = mock_factory.create_model.call_args.kwargs
    assert kwargs["tp_plan"] == "auto"
    assert kwargs["tp_size"] == 2
    assert kwargs["device_mesh"] == "tp_mesh"


@pytest.mark.finetune
def test_create_model_non_moe_tp_ddp_injects_tp_plan(mocker):
    """
    Unit test: non-MoE model creation under TP+DDP still uses
    TP mesh injection path for model construction.
    """
    pipe = FineTuningPipeline.__new__(FineTuningPipeline)
    pipe.config_manager = mocker.MagicMock(name="ConfigManager")

    pc = ParallelismConfig(tp_size=2, dp_replicate_size=2)
    mocker.patch.object(pc, "build_device_mesh", autospec=True, return_value={"tp": "tp_mesh", "dp": "dp_mesh"})

    pipe.training_config = {
        "tp_degree": 2,
        "ddp_degree": 2,
        "device": "cpu",
        "parallelism_config": pc,
    }
    pipe.config_manager.get_model_config.return_value = {
        "model_type": "hf",
        "model_name": "non-moe-model",
        "use_peft": False,
    }

    model_instance = mocker.MagicMock(name="ModelInstance")
    model_instance.model.lm_head.weight = torch.nn.Parameter(torch.randn(2, 2))

    with patch(f"{MODULE}.ComponentFactory") as mock_factory:
        mock_factory.create_model.return_value = model_instance
        pipe._create_model()

    kwargs = mock_factory.create_model.call_args.kwargs
    assert kwargs["tp_plan"] == "auto"
    assert kwargs["tp_size"] == 2
    assert kwargs["device_mesh"] == "tp_mesh"


@pytest.mark.finetune
@pytest.mark.parametrize(
    "scenario_name,ddp_degree",
    [
        ("tp_only", 1),
        ("tp_ddp", 2),
    ],
)
def test_create_model_non_moe_tp_peft_uses_get_peft_model(mocker, scenario_name, ddp_degree):
    """
    Under TP (TP-only and TP+DDP), PEFT must be applied via get_peft_model
    instead of the legacy apply_peft_to_model pathway.
    """
    pipe = FineTuningPipeline.__new__(FineTuningPipeline)
    pipe.config_manager = mocker.MagicMock(name="ConfigManager")

    pc = ParallelismConfig(tp_size=2, dp_replicate_size=ddp_degree)
    mesh = {"tp": "tp_mesh"}
    if ddp_degree > 1:
        mesh["dp"] = "dp_mesh"
    mocker.patch.object(pc, "build_device_mesh", autospec=True, return_value=mesh)

    pipe.training_config = {
        "tp_degree": 2,
        "ddp_degree": ddp_degree,
        "device": "cpu",
        "parallelism_config": pc,
    }

    peft_cfg_dataclass = mocker.MagicMock(name=f"peft_cfg_dataclass_{scenario_name}")
    pipe.config_manager.get_model_config.return_value = {
        "model_type": "hf",
        "model_name": "non-moe-model",
        "use_peft": True,
        "peft_config": peft_cfg_dataclass,
    }

    model_instance = mocker.MagicMock(name=f"ModelInstance_{scenario_name}")
    model_instance.model.lm_head.weight = torch.nn.Parameter(torch.randn(2, 2))
    base_model_before_peft = model_instance.model

    converted_peft_cfg = object()
    mock_convert = mocker.patch(
        f"{MODULE}.convert_peft_config_to_lora_config", autospec=True, return_value=converted_peft_cfg
    )
    peft_wrapped_model = mocker.MagicMock(name=f"PeftWrappedModel_{scenario_name}")
    mock_get_peft_model = mocker.patch(f"{MODULE}.get_peft_model", autospec=True, return_value=peft_wrapped_model)

    with patch(f"{MODULE}.ComponentFactory") as mock_factory:
        mock_factory.create_model.return_value = model_instance
        returned = pipe._create_model()

    assert returned is model_instance
    mock_convert.assert_called_once_with(peft_cfg_dataclass)
    mock_get_peft_model.assert_called_once_with(base_model_before_peft, converted_peft_cfg)
    assert model_instance.model is peft_wrapped_model
