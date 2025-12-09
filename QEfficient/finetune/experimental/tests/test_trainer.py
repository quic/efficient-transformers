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
from peft import LoraConfig
from transformers import Trainer, TrainingArguments
from trl import SFTConfig, SFTTrainer

from QEfficient.finetune.experimental.core.component_registry import registry
from QEfficient.finetune.experimental.core.model import HFModel
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
            "torch_dtype": "float32",
            "attn_implementation": "eager",
            "device_map": None,
            "use_peft": False,
        }

    @pytest.fixture
    def peft_model_config(self):
        """Fixture for model configuration with PEFT."""
        return {
            "model_name": "HuggingFaceTB/SmolLM-135M",
            "auto_class_name": "AutoModelForCausalLM",
            "use_cache": False,
            "torch_dtype": "float32",
            "attn_implementation": "eager",
            "device_map": None,
            "use_peft": True,
            "peft_config": {
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "target_modules": ["q_proj", "v_proj"],
                "bias": "none",
            },
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

    def test_hf_model_initialization(self, model_config):
        """Test that HFModel can be initialized properly."""

        model = HFModel(**model_config)
        assert model is not None
        assert model.model_name == model_config["model_name"]
        assert model.auto_class_name == model_config["auto_class_name"]

    def test_hf_model_load_model(self, model_config):
        """Test that HFModel can load the underlying model."""

        model = HFModel(**model_config)
        loaded_model = model.load_model()

        assert loaded_model is not None
        assert hasattr(loaded_model, "forward")
        assert hasattr(loaded_model, "config")

    def test_hf_model_load_tokenizer(self, model_config):
        """Test that HFModel can load the tokenizer."""

        model = HFModel(**model_config)
        tokenizer = model.load_tokenizer()

        assert tokenizer is not None
        assert hasattr(tokenizer, "encode")
        assert hasattr(tokenizer, "decode")
        assert tokenizer.pad_token is not None

    def test_hf_model_with_peft_config(self, peft_model_config):
        """Test that HFModel can be initialized with PEFT configuration."""

        model = HFModel(**peft_model_config)
        assert model.use_peft is True
        assert model.lora_config is not None
        assert model.lora_config.r == LORA_R
        assert model.lora_config.lora_alpha == LORA_ALPHA

    def test_model_forward_pass(self, model_config):
        """Test that the loaded model can perform a forward pass."""

        model = HFModel(**model_config)
        loaded_model = model.load_model()
        tokenizer = model.load_tokenizer()

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
        hf_model = HFModel(**model_config)
        model = hf_model.load_model()
        tokenizer = hf_model.load_tokenizer()

        # Create SFT config
        sft_config = SFTConfig(
            output_dir="./test_output",
            max_length=MAX_LENGTH,
            per_device_train_batch_size=1,
            num_train_epochs=1,
            logging_steps=1,
            save_strategy="no",
            bf16=False,
            fp16=False,
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

    def test_sft_trainer_with_peft_model(self, peft_model_config, dummy_dataset):
        """Test that SFTTrainer works with PEFT-enabled models."""

        # Load model and tokenizer
        hf_model = HFModel(**peft_model_config)
        model = hf_model.load_model()
        tokenizer = hf_model.load_tokenizer()

        # Get PEFT config
        peft_config = hf_model.load_peft_config()
        assert peft_config is not None
        assert isinstance(peft_config, LoraConfig)

        # Create SFT config
        sft_config = SFTConfig(
            output_dir="./test_output_peft",
            max_length=MAX_LENGTH,
            per_device_train_batch_size=1,
            num_train_epochs=1,
            logging_steps=1,
            save_strategy="no",
            bf16=False,
            fp16=False,
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

    def test_model_training_mode(self, model_config):
        """Test that model can be set to training and evaluation modes."""

        model = HFModel(**model_config)
        loaded_model = model.load_model()

        # Test training mode
        loaded_model.train()
        assert loaded_model.training is True

        # Test evaluation mode
        loaded_model.eval()
        assert loaded_model.training is False

    def test_model_parameters_accessible(self, model_config):
        """Test that model parameters are accessible."""

        model = HFModel(**model_config)
        loaded_model = model.load_model()

        # Check parameters
        params = list(loaded_model.parameters())
        assert len(params) > 0

        # Check named parameters
        named_params = dict(loaded_model.named_parameters())
        assert len(named_params) > 0

    def test_tokenizer_encoding_decoding(self, model_config):
        """Test that tokenizer can encode and decode text properly."""

        model = HFModel(**model_config)
        tokenizer = model.load_tokenizer()

        # Test encoding
        text = "Hello, world!"
        encoded = tokenizer.encode(text)
        assert len(encoded) > 0

        # Test decoding
        decoded = tokenizer.decode(encoded)
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_model_device_placement(self, model_config):
        """Test that model can be moved to different devices."""

        model = HFModel(**model_config)
        loaded_model = model.load_model()

        # Test CPU placement
        loaded_model = loaded_model.to("cpu")
        assert next(loaded_model.parameters()).device.type == "cpu"

    def test_sft_trainer_train_dataset_required(self, model_config):
        """Test that SFTTrainer requires a training dataset."""

        # Load model and tokenizer
        hf_model = HFModel(**model_config)
        model = hf_model.load_model()
        tokenizer = hf_model.load_tokenizer()

        # Create SFT config
        sft_config = SFTConfig(
            output_dir="./test_output",
            max_length=MAX_LENGTH,
            per_device_train_batch_size=1,
            num_train_epochs=1,
            bf16=False,
            fp16=False,
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

    def test_peft_model_trainable_parameters(self, peft_model_config):
        """Test that PEFT model has correct trainable vs total parameters ratio."""

        # Load model with PEFT configuration
        hf_model = HFModel(**peft_model_config)
        model = hf_model.load_model()

        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Count frozen parameters
        frozen_params = total_params - trainable_params
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Frozen Parameters: {frozen_params:,}")
        # Assertions to ensure PEFT is properly configured
        assert total_params > 0, "Model should have parameters"
        assert trainable_params > 0, "Model should have trainable parameters (PEFT adapters)"
        assert frozen_params > 0, "Model should have frozen parameters (base model)"
        assert trainable_params < total_params, "Trainable parameters should be less than total (PEFT efficiency)"
        # Print info for debugging (will show in pytest output with -v flag)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {frozen_params:,}")
