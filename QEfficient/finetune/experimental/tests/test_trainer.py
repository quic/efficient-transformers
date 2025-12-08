# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import pytest
from peft import IA3Config, LoraConfig

from QEfficient.finetune.experimental.core.component_registry import registry
from QEfficient.finetune.experimental.core.config_manager import PeftConfig
from QEfficient.finetune.experimental.core.trainer.base_trainer import BaseTrainer
from QEfficient.finetune.experimental.core.trainer.sft_trainer import (
    SFTTrainerModule,
    get_default_peft_config,
)


class TestBaseTrainer:
    """Test suite for BaseTrainer class."""

    def test_base_trainer_registered(self):
        """Test that BaseTrainer is registered in the registry."""
        trainer_list = registry.list_trainer_modules()
        assert "base" in trainer_list

    def test_base_trainer_info_structure(self):
        """Test that BaseTrainer registration has correct structure."""
        trainer_info = registry.get_trainer_module("base")

        assert isinstance(trainer_info, dict)
        assert "trainer_cls" in trainer_info
        assert "args_cls" in trainer_info
        assert "required_kwargs" in trainer_info

    def test_base_trainer_class(self):
        """Test that BaseTrainer class is correct."""
        from transformers import Trainer, TrainingArguments

        trainer_info = registry.get_trainer_module("base")
        trainer_cls = trainer_info["trainer_cls"]

        # The decorator returns the dict, but BaseTrainer is the original class
        assert trainer_cls.__name__ == "BaseTrainer"
        assert issubclass(trainer_cls, Trainer)
        assert trainer_info["args_cls"] == TrainingArguments


class TestSFTTrainerModule:
    """Test suite for SFTTrainerModule class."""

    def test_sft_trainer_registered(self):
        """Test that SFTTrainerModule is registered in the registry."""
        trainer_list = registry.list_trainer_modules()
        assert "sft" in trainer_list

    def test_sft_trainer_info_structure(self):
        """Test that SFTTrainerModule registration has correct structure."""
        trainer_info = registry.get_trainer_module("sft")

        assert isinstance(trainer_info, dict)
        assert "trainer_cls" in trainer_info
        assert "args_cls" in trainer_info
        assert "required_kwargs" in trainer_info

    def test_sft_trainer_class(self):
        """Test that SFTTrainerModule class is correct."""
        from trl import SFTConfig, SFTTrainer

        trainer_info = registry.get_trainer_module("sft")
        trainer_cls = trainer_info["trainer_cls"]

        assert trainer_cls == SFTTrainerModule["trainer_cls"]
        assert issubclass(trainer_cls, SFTTrainer)
        assert trainer_info["args_cls"] == SFTConfig

    def test_sft_trainer_required_kwargs(self):
        """Test that SFTTrainerModule has peft_config in required_kwargs."""
        trainer_info = registry.get_trainer_module("sft")

        assert "peft_config" in trainer_info["required_kwargs"]
        assert callable(trainer_info["required_kwargs"]["peft_config"])


class TestGetDefaultPeftConfig:
    """Test suite for get_default_peft_config function."""

    def test_returns_lora_config(self):
        """Test that get_default_peft_config returns a LoraConfig instance."""
        peft_config = get_default_peft_config()
        assert isinstance(peft_config, LoraConfig)

    def test_has_correct_defaults(self):
        """Test that the returned config has the expected default values."""
        peft_config = get_default_peft_config()

        assert peft_config.r == 8
        assert peft_config.lora_alpha == 16
        assert peft_config.lora_dropout == 0.1
        # target_modules might be a set or list depending on peft version
        target_modules = peft_config.target_modules
        if isinstance(target_modules, set):
            assert target_modules == {"q_proj", "v_proj"}
        else:
            assert set(target_modules) == {"q_proj", "v_proj"}
        assert peft_config.bias == "none"
        assert peft_config.task_type == "CAUSAL_LM"

    def test_is_callable(self):
        """Test that get_default_peft_config is callable."""
        assert callable(get_default_peft_config)

    def test_creates_new_instance_each_call(self):
        """Test that each call creates a new config instance."""
        config1 = get_default_peft_config()
        config2 = get_default_peft_config()

        # They should be different instances
        assert config1 is not config2
        # But have the same values
        assert config1.r == config2.r
        assert config1.lora_alpha == config2.lora_alpha


class TestPeftConfigConversion:
    """Test suite for PeftConfig conversion methods."""

    def test_to_lora_config(self):
        """Test that PeftConfig.to_peft_config() correctly converts to LoraConfig."""
        peft_config_params = PeftConfig(
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj"],
            bias="all",
            task_type="CAUSAL_LM",
            peft_type="LORA",
        )

        lora_config = peft_config_params.to_peft_config()

        assert isinstance(lora_config, LoraConfig)
        assert lora_config.r == 16
        assert lora_config.lora_alpha == 32
        assert lora_config.lora_dropout == 0.05
        # target_modules might be converted to set
        target_modules = lora_config.target_modules
        if isinstance(target_modules, set):
            assert target_modules == {"q_proj", "k_proj", "v_proj"}
        else:
            assert set(target_modules) == {"q_proj", "k_proj", "v_proj"}
        assert lora_config.bias == "all"
        assert lora_config.task_type == "CAUSAL_LM"

    def test_to_ia3_config(self):
        """Test that PeftConfig.to_peft_config() correctly converts to IA3Config."""
        peft_config_params = PeftConfig(
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
            peft_type="IA3",
        )

        ia3_config = peft_config_params.to_peft_config()

        assert isinstance(ia3_config, IA3Config)
        # target_modules might be converted to set
        target_modules = ia3_config.target_modules
        if isinstance(target_modules, set):
            assert target_modules == {"q_proj", "v_proj"}
        else:
            assert set(target_modules) == {"q_proj", "v_proj"}
        assert ia3_config.task_type == "CAUSAL_LM"

    def test_unsupported_type_raises_error(self):
        """Test that unsupported peft_type raises ValueError."""
        peft_config_params = PeftConfig(peft_type="UNSUPPORTED")

        with pytest.raises(ValueError) as exc_info:
            peft_config_params.to_peft_config()

        assert "Unsupported peft_type: UNSUPPORTED" in str(exc_info.value)
        assert "Supported types: 'LORA', 'IA3'" in str(exc_info.value)


class TestPeftConfigDefaults:
    """Test suite for PeftConfig default values."""

    def test_default_values(self):
        """Test that PeftConfig has correct default values."""
        peft_config = PeftConfig()

        assert peft_config.lora_r == 8
        assert peft_config.lora_alpha == 16
        assert peft_config.lora_dropout == 0.1
        assert peft_config.target_modules == ["q_proj", "v_proj"]
        assert peft_config.bias == "none"
        assert peft_config.task_type == "CAUSAL_LM"
        assert peft_config.peft_type == "LORA"

    def test_custom_values(self):
        """Test that PeftConfig accepts custom values."""
        peft_config = PeftConfig(
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="all",
            task_type="SEQ_2_SEQ_LM",
            peft_type="LORA",
        )

        assert peft_config.lora_r == 32
        assert peft_config.lora_alpha == 64
        assert peft_config.lora_dropout == 0.2
        assert peft_config.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
        assert peft_config.bias == "all"
        assert peft_config.task_type == "SEQ_2_SEQ_LM"
        assert peft_config.peft_type == "LORA"


class TestTrainerRegistry:
    """Test suite for trainer registration in the component registry."""

    def test_both_trainers_registered(self):
        """Test that both base and sft trainers are registered."""
        trainer_list = registry.list_trainer_modules()

        assert "base" in trainer_list
        assert "sft" in trainer_list
        assert len(trainer_list) >= 2

    def test_registry_returns_dict(self):
        """Test that registry returns dict for trainer modules."""
        base_info = registry.get_trainer_module("base")
        sft_info = registry.get_trainer_module("sft")

        assert isinstance(base_info, dict)
        assert isinstance(sft_info, dict)

    def test_trainer_classes_correct(self):
        """Test that trainer classes are correctly stored."""
        base_info = registry.get_trainer_module("base")
        sft_info = registry.get_trainer_module("sft")
        assert base_info["trainer_cls"] == BaseTrainer["trainer_cls"]
        assert sft_info["trainer_cls"] == SFTTrainerModule["trainer_cls"]
