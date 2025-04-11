# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import random
import warnings
from typing import Optional, Any

import fire
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from peft import PeftModel, get_peft_model
from dataclasses import fields
from torch.optim.lr_scheduler import StepLR
from transformers import AutoModelForCausalLM, AutoTokenizer

from QEfficient.finetune.configs.peft_config import LoraConfig
from QEfficient.finetune.configs.training import TrainConfig
from QEfficient.finetune.utils.config_utils import (
    generate_dataset_config,
    generate_peft_config,
    get_dataloader_kwargs,
    load_config_file,
    update_config,
    validate_config,
)
from QEfficient.finetune.utils.dataset_utils import (
    get_custom_data_collator,
    get_preprocessed_dataset,
)
from QEfficient.finetune.utils.train_utils import get_longest_seq_length, print_model_size, train
from QEfficient.utils._utils import login_and_download_hf_lm

# Try importing QAIC-specific module, proceed without it if unavailable
try:
    import torch_qaic  # noqa: F401
except ImportError as e:
    print(f"Warning: {e}. Proceeding without QAIC modules.")


from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

# Suppress all warnings
warnings.filterwarnings("ignore")


def setup_distributed_training(config: TrainConfig) -> None:
    """Initialize distributed training environment if enabled.

    Args:
        config (TrainConfig): Training configuration object.

    Notes:
        - If distributed data parallel (DDP) is disabled, this function does nothing.
        - Ensures the device is not CPU and does not specify an index for DDP compatibility.
        - Initializes the process group using the specified distributed backend.

    Raises:
        AssertionError: If device is CPU or includes an index with DDP enabled.
    """
    if not config.enable_ddp:
        return

    torch_device = torch.device(config.device)
    assert torch_device.type != "cpu", "Host doesn't support single-node DDP"
    assert torch_device.index is None, f"DDP requires only device type, got: {torch_device}"

    dist.init_process_group(backend=config.dist_backend)
    # from here onward "qaic/cuda" will automatically map to "qaic:i/cuda:i", where i = process rank
    getattr(torch, torch_device.type).set_device(dist.get_rank())


def setup_seeds(seed: int) -> None:
    """Set random seeds across libraries for reproducibility.

    Args:
        seed (int): Seed value to set for random number generators.

    Notes:
        - Sets seeds for PyTorch, Python's random module, and NumPy.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model_and_tokenizer(config: TrainConfig) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the pre-trained model and tokenizer from Hugging Face.

    Args:
        config (TrainConfig): Training configuration object containing model and tokenizer names.

    Returns:
        tuple: A tuple containing the loaded model (AutoModelForCausalLM) and tokenizer (AutoTokenizer).

    Notes:
        - Downloads the model if not already cached using login_and_download_hf_lm.
        - Configures the model with FP16 precision and disables caching for training.
        - Resizes model embeddings if tokenizer vocab size exceeds model embedding size.
        - Sets pad_token_id to eos_token_id if not defined in the tokenizer.
    """
    pretrained_model_path = login_and_download_hf_lm(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_path,
        use_cache=False,
        attn_implementation="sdpa",
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name if config.tokenizer_name is None else config.tokenizer_name
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing embedding matrix to match tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    # Note: Need to call this before calling PeftModel.from_pretrained or get_peft_model.
    # Because, both makes model.is_gradient_checkpointing = True which is used in peft library to
    # apply gradient checkpointing related hooks to the input embeddings. Without this we will get
    # "No inf checks were recorded for this optimizer." error.
    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        # Note: below attribute and method is only available in HuggingFace Transformer models.
        if hasattr(model, "supports_gradient_checkpointing") and model.supports_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"preserve_rng_state": False})
        else:
            raise RuntimeError("Given model doesn't support gradient checkpointing. Please disable it and run it.")
    
    return model, tokenizer


def apply_peft(model: AutoModelForCausalLM, train_config: TrainConfig, lora_config: LoraConfig) -> PeftModel:
    """Apply Parameter-Efficient Fine-Tuning (PEFT) to the model if enabled."""
    if not train_config.use_peft:
        return model

    # Load the pre-trained peft model checkpoint and setup its configuration
    if train_config.from_peft_checkpoint:
        model = PeftModel.from_pretrained(model, train_config.from_peft_checkpoint, is_trainable=True)
        peft_config = model.peft_config
    # Generate the peft config and start fine-tuning from original model
    else:
        peft_config = generate_peft_config(train_config, lora_config)
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


def setup_dataloaders(
    train_config: TrainConfig, dataset_config, tokenizer: AutoTokenizer, dataset_train, dataset_val
) -> tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """Set up training and validation DataLoaders.

    Args:
        train_config (TrainConfig): Training configuration object.
        dataset_config: Configuration for the dataset (generated from train_config).
        tokenizer (AutoTokenizer): Tokenizer for preprocessing data.
        dataset_train: Preprocessed training dataset.
        dataset_val: Preprocessed validation dataset.

    Returns:
        tuple: A tuple of (train_dataloader, eval_dataloader), where eval_dataloader is None if validation is disabled.

    Raises:
        ValueError: If validation is enabled but the validation set is too small.

    Notes:
        - Applies a custom data collator if provided by get_custom_data_collator.
        - Configures DataLoader kwargs using get_dataloader_kwargs for train and val splits.
    """
    custom_data_collator = get_custom_data_collator(tokenizer, dataset_config)
    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")
    if custom_data_collator:
        train_dl_kwargs["collate_fn"] = custom_data_collator

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )
    print(f"--> Num of Training Set Batches loaded = {len(train_dataloader)}")

    eval_dataloader = None
    if train_config.run_validation:
        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")
        if custom_data_collator:
            val_dl_kwargs["collate_fn"] = custom_data_collator

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )
        print(f"--> Num of Validation Set Batches loaded = {len(eval_dataloader)}")
        if len(eval_dataloader) == 0:
            raise ValueError("Eval set too small to load even one batch.")

    return train_dataloader, eval_dataloader


def main(
    model_name: str = None,
    tokenizer_name: str = None,
    batch_size_training: int = None,
    lr: float = None,
    peft_config_file: str = None,
    **kwargs,
) -> None:
    """
    Fine-tune a model on QAIC hardware with configurable training and LoRA parameters.

    Args:
        model_name (str, optional): Override default model name.
        tokenizer_name (str, optional): Override default tokenizer name.
        batch_size_training (int, optional): Override default training batch size.
        lr (float, optional): Override default learning rate.
        peft_config_file (str, optional): Path to YAML/JSON file containing PEFT (LoRA) config.
        **kwargs: Additional arguments to override TrainConfig.

    Example:
        .. code-block:: bash

            # Using a YAML config file for PEFT
            python -m QEfficient.cloud.finetune \\
                --model_name "meta-llama/Llama-3.2-1B" \\
                --lr 5e-4 \\
                --peft_config_file "lora_config.yaml"

            # Using default LoRA config
            python -m QEfficient.cloud.finetune \\
                --model_name "meta-llama/Llama-3.2-1B" \\
                --lr 5e-4
    """
    train_config = TrainConfig()
    #  local_args = {k: v for k, v in locals().items() if v is not None and k != "peft_config_file" and k != "kwargs"}
    update_config(train_config, **kwargs)

    lora_config = LoraConfig()
    if peft_config_file:
        peft_config_data = load_config_file(peft_config_file)
        validate_config(peft_config_data, config_type="lora")
        lora_config = LoraConfig(**peft_config_data)
    else:
        lora_config = LoraConfig()
        
    update_config(lora_config, **kwargs)

    setup_distributed_training(train_config)
    setup_seeds(train_config.seed)
    model, tokenizer = load_model_and_tokenizer(train_config)
    print_model_size(model, train_config)
    model = apply_peft(model, train_config, lora_config)

    # Pass an empty dict instead of kwargs to avoid irrelevant parameters
    dataset_config = generate_dataset_config(train_config, kwargs)
    dataset_train = get_preprocessed_dataset(
        tokenizer, dataset_config, split="train", context_length=train_config.context_length
    )
    dataset_val = get_preprocessed_dataset(
        tokenizer, dataset_config, split="test", context_length=train_config.context_length
    )
    train_dataloader, eval_dataloader = setup_dataloaders(
        train_config, dataset_config, tokenizer, dataset_train, dataset_val
    )
    dataset_for_seq_length = (
        torch.utils.data.ConcatDataset([train_dataloader.dataset, eval_dataloader.dataset])
        if train_config.run_validation
        else train_dataloader.dataset
    )
    longest_seq_length, _ = get_longest_seq_length(dataset_for_seq_length)
    print(
        f"Longest sequence length: {longest_seq_length}, "
        f"Context length: {train_config.context_length}, "
        f"Model max context: {model.config.max_position_embeddings}"
    )
    model.to(train_config.device)
    optimizer = optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    if train_config.enable_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()])
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        train_config.device,
        dist.get_rank() if train_config.enable_ddp else None,
        None,
    )
    if train_config.enable_ddp:
        dist.destroy_process_group()
    return results


if __name__ == "__main__":
    fire.Fire(main)
