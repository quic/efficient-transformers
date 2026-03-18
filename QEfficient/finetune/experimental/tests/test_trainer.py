# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import shutil

import pytest
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import Trainer, TrainingArguments
from trl import SFTConfig, SFTTrainer

from QEfficient.finetune.experimental.core.component_registry import ComponentFactory, registry
from QEfficient.finetune.experimental.core.model import HFModel  # noqa: F401 - needed for registration
from QEfficient.finetune.experimental.core.trainer.base_trainer import BaseTrainer
from QEfficient.finetune.experimental.core.trainer.sft_trainer import (
    SFTTrainerModule,
)

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
MAX_LENGTH = 128


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

        trainer_info = registry.get_trainer_module("base")
        trainer_cls = trainer_info["trainer_cls"]

        # The decorator returns the dict, but BaseTrainer is the original class
        assert trainer_cls.__name__ == "BaseTrainer"
        assert issubclass(trainer_cls, Trainer)
        assert trainer_info["args_cls"] == TrainingArguments

    def test_base_trainer_required_kwargs(self):
        """Test that BaseTrainer has peft_config in required_kwargs."""
        trainer_info = registry.get_trainer_module("base")

        assert "peft_config" in trainer_info["required_kwargs"]
        assert callable(trainer_info["required_kwargs"]["peft_config"])


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


class TestBaseTrainerWithModel:
    """Test suite for BaseTrainer integration with model loading and PEFT."""

    @pytest.fixture(autouse=True)
    def cleanup_output_dirs(self):
        """Fixture to clean up test output directories after each test."""
        # Setup: yield control to the test
        yield

        # Teardown: clean up output directories
        output_dirs = ["./test_output", "./test_output_peft", "./test_output_base", "./test_output_base_peft"]
        for output_dir in output_dirs:
            if os.path.exists(output_dir):
                try:
                    shutil.rmtree(output_dir)
                    print(f"\nCleaned up: {output_dir}")
                except Exception as e:
                    print(f"\nWarning: Failed to clean up {output_dir}: {e}")

    @pytest.fixture
    def model_config(self):
        """Fixture for basic model configuration."""
        return {
            "model_name": "HuggingFaceTB/SmolLM-135M",
            "auto_class_name": "AutoModelForCausalLM",
            "use_cache": False,
            "torch_dtype": "float16",
            "attn_implementation": "eager",
            "device_map": None,
            "num_hidden_layers": 1,
        }

    @pytest.fixture
    def peft_model_config(self):
        """Fixture for PEFT configuration."""
        return {
            "r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
        }

    @pytest.fixture
    def dummy_dataset(self):
        """Fixture for creating a dummy dataset."""
        data = {
            "text": [
                "This is a test sentence for training.",
                "Another example text for the model.",
                "Third sample to ensure proper batching.",
            ]
        }
        return Dataset.from_dict(data)

    def test_base_trainer_instantiation_with_model(self, model_config, dummy_dataset):
        """Test that BaseTrainer can be instantiated with a loaded model."""
        # Load model and tokenizer
        model_name = model_config.pop("model_name")
        hf_model = ComponentFactory.create_model("hf", model_name, **model_config)
        model = hf_model.model
        tokenizer = hf_model.tokenizer

        # Create training config
        training_args = TrainingArguments(
            output_dir="./test_output_base",
            per_device_train_batch_size=1,
            num_train_epochs=1,
            logging_steps=1,
            save_strategy="no",
            bf16=False,
            fp16=True,
        )

        # Get BaseTrainer from registry
        trainer_info = registry.get_trainer_module("base")
        trainer_cls = trainer_info["trainer_cls"]

        # Instantiate trainer without PEFT
        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=dummy_dataset,
            processing_class=tokenizer,
        )

        assert trainer is not None
        assert trainer.model is not None
        assert trainer.processing_class is not None

    def test_base_trainer_with_peft_model(self, model_config, peft_model_config, dummy_dataset):
        """Test that BaseTrainer works with PEFT-enabled models."""
        # Load model and tokenizer
        model_name = model_config.pop("model_name")
        hf_model = ComponentFactory.create_model("hf", model_name, **model_config)
        model = hf_model.model
        tokenizer = hf_model.tokenizer

        # Load PEFT Config
        peft_config = LoraConfig(**peft_model_config)

        # Create training config
        training_args = TrainingArguments(
            output_dir="./test_output_base_peft",
            per_device_train_batch_size=1,
            num_train_epochs=1,
            logging_steps=1,
            save_strategy="no",
            bf16=False,
            fp16=True,
        )

        # Get BaseTrainer from registry
        trainer_info = registry.get_trainer_module("base")
        trainer_cls = trainer_info["trainer_cls"]

        # Instantiate trainer with PEFT config
        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=dummy_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

        assert trainer is not None
        assert trainer.model is not None

        # Verify that the model is now a PEFT model
        assert isinstance(trainer.model, PeftModel), "Model should be wrapped as a PeftModel"

        # Verify that the model has the expected PEFT config
        assert hasattr(trainer.model, "peft_config"), "Model should have peft_config attribute"
        assert trainer.model.peft_config is not None, "PEFT config should not be None"

        # Verify trainable parameters are reduced (PEFT should make only a subset trainable)
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in trainer.model.parameters())

        assert trainable_params < total_params, "PEFT should reduce the number of trainable parameters"
        print(f"\nTrainable params: {trainable_params:,} / Total params: {total_params:,}")

    def test_base_trainer_without_peft_config(self, model_config, dummy_dataset):
        """Test that BaseTrainer works without PEFT config (standard training)."""
        # Load model and tokenizer
        model_name = model_config.pop("model_name")
        hf_model = ComponentFactory.create_model("hf", model_name, **model_config)
        model = hf_model.model
        tokenizer = hf_model.tokenizer

        # Create training config
        training_args = TrainingArguments(
            output_dir="./test_output_base",
            per_device_train_batch_size=1,
            num_train_epochs=1,
            logging_steps=1,
            save_strategy="no",
            bf16=False,
            fp16=True,
        )

        # Get BaseTrainer from registry
        trainer_info = registry.get_trainer_module("base")
        trainer_cls = trainer_info["trainer_cls"]

        # Instantiate trainer without PEFT config
        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=dummy_dataset,
            processing_class=tokenizer,
            peft_config=None,  # Explicitly pass None
        )

        assert trainer is not None
        assert trainer.model is not None

        # Verify that the model is NOT a PEFT model
        assert not isinstance(trainer.model, PeftModel), (
            "Model should not be wrapped as a PeftModel when peft_config is None"
        )


class TestSFTTrainerWithModel:
    """Test suite for SFTTrainer integration with model loading."""

    @pytest.fixture(autouse=True)
    def cleanup_output_dirs(self):
        """Fixture to clean up test output directories after each test."""
        # Setup: yield control to the test
        yield

        # Teardown: clean up output directories
        output_dirs = ["./test_output", "./test_output_peft"]
        for output_dir in output_dirs:
            if os.path.exists(output_dir):
                try:
                    shutil.rmtree(output_dir)
                    print(f"\nCleaned up: {output_dir}")
                except Exception as e:
                    print(f"\nWarning: Failed to clean up {output_dir}: {e}")

    @pytest.fixture
    def model_config(self):
        """Fixture for basic model configuration."""
        return {
            "model_name": "HuggingFaceTB/SmolLM-135M",
            "auto_class_name": "AutoModelForCausalLM",
            "use_cache": False,
            "torch_dtype": "float16",
            "attn_implementation": "eager",
            "device_map": None,
            "num_hidden_layers": 1,
        }

    @pytest.fixture
    def peft_model_config(self):
        """Fixture for PEFT configuration."""
        return {
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
        }

    @pytest.fixture
    def dummy_dataset(self):
        """Fixture for creating a dummy dataset."""

        data = {
            "text": [
                "This is a test sentence for training.",
                "Another example text for the model.",
                "Third sample to ensure proper batching.",
            ]
        }
        return Dataset.from_dict(data)

    def test_model_forward_pass(self, model_config):
        """Test that the loaded model can perform a forward pass."""

        model_name = model_config.pop("model_name")
        hf_model = ComponentFactory.create_model("hf", model_name, **model_config)
        loaded_model = hf_model.model
        tokenizer = hf_model.tokenizer

        # Prepare input
        text = "This is a test."
        inputs = tokenizer(text, return_tensors="pt")

        # Perform forward pass
        with torch.no_grad():
            outputs = loaded_model(**inputs)

        assert outputs is not None
        assert hasattr(outputs, "logits")
        assert outputs.logits.shape[0] == 1  # batch size

    def test_sft_trainer_instantiation_with_model(self, model_config, dummy_dataset):
        """Test that SFTTrainer can be instantiated with a loaded model."""

        # Load model and tokenizer
        model_name = model_config.pop("model_name")
        hf_model = ComponentFactory.create_model("hf", model_name, **model_config)
        model = hf_model.model
        tokenizer = hf_model.tokenizer

        # Create SFT config
        sft_config = SFTConfig(
            output_dir="./test_output",
            max_length=MAX_LENGTH,
            per_device_train_batch_size=1,
            num_train_epochs=1,
            logging_steps=1,
            save_strategy="no",
            bf16=False,
            fp16=True,
        )

        # Get SFTTrainer from registry
        trainer_info = registry.get_trainer_module("sft")
        trainer_cls = trainer_info["trainer_cls"]

        # Instantiate trainer
        trainer = trainer_cls(
            model=model,
            args=sft_config,
            train_dataset=dummy_dataset,
            processing_class=tokenizer,
        )

        assert trainer is not None
        assert trainer.model is not None
        assert trainer.tokenizer is not None

    def test_sft_trainer_with_peft_model(self, model_config, peft_model_config, dummy_dataset):
        """Test that SFTTrainer works with PEFT-enabled models."""

        # Load model and tokenizer
        model_name = model_config.pop("model_name")
        hf_model = ComponentFactory.create_model("hf", model_name, **model_config)
        model = hf_model.model
        # Load PEFT Config
        peft_config = LoraConfig(peft_model_config)
        tokenizer = hf_model.tokenizer

        # Create SFT config
        sft_config = SFTConfig(
            output_dir="./test_output_peft",
            max_length=MAX_LENGTH,
            per_device_train_batch_size=1,
            num_train_epochs=1,
            logging_steps=1,
            save_strategy="no",
            bf16=False,
            fp16=True,
        )

        # Get SFTTrainer from registry
        trainer_info = registry.get_trainer_module("sft")
        trainer_cls = trainer_info["trainer_cls"]

        # Instantiate trainer with PEFT config
        trainer = trainer_cls(
            model=model,
            args=sft_config,
            train_dataset=dummy_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

        assert trainer is not None
        assert trainer.model is not None

    def test_sft_trainer_train_dataset_required(self, model_config):
        """Test that SFTTrainer requires a training dataset."""

        # Load model and tokenizer
        model_name = model_config.pop("model_name")
        hf_model = ComponentFactory.create_model("hf", model_name, **model_config)
        model = hf_model.model
        tokenizer = hf_model.tokenizer

        # Create SFT config
        sft_config = SFTConfig(
            output_dir="./test_output",
            max_length=MAX_LENGTH,
            per_device_train_batch_size=1,
            num_train_epochs=1,
            bf16=False,
            fp16=True,
        )

        # Get SFTTrainer from registry
        trainer_info = registry.get_trainer_module("sft")
        trainer_cls = trainer_info["trainer_cls"]

        # Attempt to instantiate without dataset should raise TypeError
        with pytest.raises(TypeError, match="'NoneType' object is not iterable"):
            trainer_cls(
                model=model,
                args=sft_config,
                processing_class=tokenizer,
            )
