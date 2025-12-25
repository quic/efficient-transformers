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

import os
import shutil
import tempfile

import pytest
import torch
from datasets import load_dataset

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
from QEfficient.finetune.experimental.core.dataset import SFTDataset
from QEfficient.finetune.experimental.core.model import HFModel
from QEfficient.finetune.experimental.core.trainer.sft_trainer import SFTTrainerModule
from QEfficient.utils.logging_utils import logger


class TestParametrizedConfigurations:
    """Parametrized tests for different model and dataset configurations."""

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Setup and cleanup test directories and files."""
        self.test_output_dir = tempfile.mkdtemp(prefix="test_ft_config_")
        self.alpaca_json_path = os.path.join(self.test_output_dir, "alpaca_data.json")
        yield
        # Cleanup
        if os.path.exists(self.test_output_dir):
            try:
                shutil.rmtree(self.test_output_dir)
            except Exception as e:
                logger.warning(f"Warning: Failed to clean up {self.test_output_dir}: {e}")

    @pytest.mark.parametrize(
        "model_name,task_type,max_eval_step,max_train_step,dataset_name,data_path_fixture,use_peft,config_name",
        [
            pytest.param(
                "meta-llama/Llama-3.2-1B",
                "CAUSAL_LM",
                10,
                20,
                "gsm8k",
                None,
                True,
                "llama_3.2_1B_config_gsm8k_single_device",
                id="llama_config_gsm8k",
            ),
            pytest.param(
                "meta-llama/Llama-3.2-1B",
                "CAUSAL_LM",
                10,
                20,
                "alpaca",
                None,
                True,
                "llama_3.2_1B_config_alpaca_single_device",
                id="llama_config_alpaca",
            ),
            # pytest.param(
            #     "google-bert/bert-base-uncased",
            #     "SEQ_CLS",
            #     10,
            #     20,
            #     "imdb",
            #     None,
            #     False,
            #     "bert_base_uncased_config_imdb_single_device",
            #     id="bert_config_imdb",
            # ),
        ],
    )
    def test_parametrized_configurations(
        self,
        model_name,
        task_type,
        max_eval_step,
        max_train_step,
        dataset_name,
        data_path_fixture,
        use_peft,
        config_name,
        request,
    ):
        """
        Parametrized test covering multiple model and dataset configurations.
        Tests Llama with GSM8K, Llama with Alpaca, and BERT with IMDB.
        """
        from trl import SFTConfig

        # # Get data path if fixture is specified
        # data_path = None
        # if data_path_fixture:
        #     data_path = request.getfixturevalue(data_path_fixture)

        # Determine auto_class_name based on task type
        if task_type == "CAUSAL_LM":
            auto_class_name = "AutoModelForCausalLM"
            dataset_type = "seq_completion"
        elif task_type == "SEQ_CLS":
            auto_class_name = "AutoModelForSequenceClassification"
            dataset_type = "seq_classification"
        else:
            pytest.fail(f"Unknown task type: {task_type}")

        # Create configuration
        master_config = MasterConfig(
            model=ModelConfig(
                model_name=model_name,
                model_type="hf",
                auto_class_name=auto_class_name,
                use_peft=use_peft,
                use_cache=False,
                attn_implementation="eager",
                device_map=None,
                peft_config=PeftConfig(
                    lora_r=8,
                    lora_alpha=16,
                    lora_dropout=0.1,
                    target_modules=["q_proj", "v_proj"] if "llama" in model_name.lower() else ["query", "value"],
                    bias="none",
                    task_type=task_type,
                    peft_type="LORA",
                )
                if use_peft
                else None,
            ),
            dataset=DatasetConfig(
                tokenizer_name=model_name,
                dataset_type=dataset_type,
                dataset_name=dataset_name,
                max_seq_length=256 if task_type == "CAUSAL_LM" else 128,
                train_batch_size=1,
                eval_batch_size=1,
                num_workers=1,
            ),
            optimizers=OptimizerConfig(
                optimizer_name="AdamW",
                lr=5e-5,
                weight_decay=0.01,
            ),
            scheduler=SchedulerConfig(
                scheduler_name="cosine",
                warmup_steps=5,
            ),
            training=TrainingConfig(
                type="sft",
                output_dir=self.test_output_dir,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                logging_steps=1,
                save_strategy="no",
                eval_strategy="no",
                seed=42,
                max_steps=5,  # Only train for 5 steps for testing
            ),
        )

        # Validate configuration
        config_manager = ConfigManager(master_config)
        config_manager.validate_config()

        # Load Model
        model_config = config_manager.get_model_config()

        # Adjust model config for faster testing
        model_config_kwargs = {}
        model_config_kwargs["num_hidden_layers"] = 2

        # Prepare peft_config_dict only if use_peft is True
        peft_config_dict = None
        if use_peft and model_config.peft_config:
            peft_config_dict = {
                "lora_r": model_config.peft_config.lora_r,
                "lora_alpha": model_config.peft_config.lora_alpha,
                "lora_dropout": model_config.peft_config.lora_dropout,
                "target_modules": model_config.peft_config.target_modules,
                "bias": model_config.peft_config.bias,
            }

        hf_model = HFModel(
            model_name=model_config.model_name,
            auto_class_name=model_config.auto_class_name,
            use_peft=model_config.use_peft,
            use_cache=model_config.use_cache,
            attn_implementation=model_config.attn_implementation,
            device_map=model_config.device_map,
            peft_config=peft_config_dict,
            model_config_kwargs=model_config_kwargs,
        )

        model = hf_model.load_model()
        tokenizer = hf_model.load_tokenizer()
        peft_config = hf_model.load_peft_config() if use_peft else None

        logger.warning(f"Model loaded: {model_config.model_name}")

        # Load Dataset
        if dataset_name == "alpaca":
            alpaca_hf = load_dataset("tatsu-lab/alpaca", split="train")
            alpaca_hf_subset = alpaca_hf.select(range(10))  # Use small subset for testing
            alpaca_json_path = os.path.join(self.test_output_dir, "alpaca.json")
            alpaca_hf_subset.to_json(alpaca_json_path)
            dataset = SFTDataset(
                dataset_name="alpaca",
                split="train",
                json_file_path=alpaca_json_path,
                prompt_template="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
                completion_template="{output}",
            )

            def formatting_func(example):
                prompt = dataset._preprocess_sample(example)
                return prompt["prompt"] + prompt["completion"]

        elif dataset_name == "gsm8k":
            # Load actual GSM8K dataset from HuggingFace via SFTDataset wrapper

            gsm8k_hf = load_dataset("openai/gsm8k", "main", split="train")
            gsm8k_subset = gsm8k_hf.select(range(10))  # Only use 10 samples for testing

            # Save to temporary JSON file to use with SFTDataset
            gsm8k_json_path = os.path.join(self.test_output_dir, "gsm8k_data.json")
            gsm8k_subset.to_json(gsm8k_json_path)

            dataset = SFTDataset(
                dataset_name="gsm8k",
                split="train",
                json_file_path=gsm8k_json_path,
                prompt_template="Question: {question}\nAnswer: ",
                completion_template="{answer}",
            )

            def formatting_func(example):
                prompt = dataset._preprocess_sample(example)
                return prompt["prompt"] + prompt["completion"]

        elif dataset_name == "imdb":
            imdb_hf = load_dataset("stanfordnlp/imdb", split="train")
            imdb_subset = imdb_hf.select(range(10))  # Use small subset for testing
            imdb_json_path = os.path.join(self.test_output_dir, "imdb.json")
            imdb_subset.to_json(imdb_json_path)

            dataset = SFTDataset(
                dataset_name="imdb",
                split="train",
                json_file_path=imdb_json_path,
                prompt_template="Review: {text}\nSentiment: ",
                completion_template="{label}",
            )

            def formatting_func(example):
                prompt = dataset._preprocess_sample(example)
                return prompt["prompt"] + prompt["completion"]

        else:
            pytest.fail(f"Unknown dataset: {dataset_name}")

        logger.warning(f"Dataset loaded: {len(dataset)} samples")

        # Create SFT Training Config
        training_config = master_config.training
        sft_config = SFTConfig(
            output_dir=training_config.output_dir,
            max_length=master_config.dataset.max_seq_length,
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            num_train_epochs=training_config.num_train_epochs,
            max_steps=training_config.max_steps,
            logging_steps=training_config.logging_steps,
            save_strategy=training_config.save_strategy,
            eval_strategy=training_config.eval_strategy,
            seed=training_config.seed,
            bf16=False,
            fp16=False,
            report_to="none",
        )

        logger.warning("Training config created")

        # Create Trainer
        trainer = SFTTrainerModule["trainer_cls"](
            model=model,
            args=sft_config,
            train_dataset=dataset.dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            formatting_func=formatting_func,
        )
        logger.warning("Trainer instantiated")

        # Run Training
        logger.warning(f"Starting training for {config_name}...")
        train_result = trainer.train()
        logger.warning(f"Training completed for {config_name}!")

        # Verify Training Results
        assert train_result is not None
        assert hasattr(train_result, "training_loss")
        logger.warning(f"Training loss: {train_result.training_loss:.4f}")

        # Test Inference
        if task_type == "CAUSAL_LM":
            test_prompt = "Test prompt for generation"
            inputs = tokenizer(test_prompt, return_tensors="pt")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                )
            # Generated Text from the model
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.warning(f"Generated text: {generated_text}")
