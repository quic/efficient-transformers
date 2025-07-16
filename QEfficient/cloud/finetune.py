# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import logging
import random
import warnings
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from peft import PeftModel, get_peft_model
from torch.optim.lr_scheduler import StepLR
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from QEfficient.finetune.configs.training import TrainConfig
from QEfficient.finetune.utils.config_utils import (
    generate_dataset_config,
    generate_peft_config,
    update_config,
)
from QEfficient.finetune.utils.dataset_utils import get_dataloader, get_longest_seq_length
from QEfficient.finetune.utils.device_map import get_device_map
from QEfficient.finetune.utils.helper import Task_Mode
from QEfficient.finetune.utils.logging_utils import logger
from QEfficient.finetune.utils.parser import get_finetune_parser
from QEfficient.finetune.utils.train_utils import print_model_size, print_trainable_parameters, train
from QEfficient.utils._utils import hf_download

# Try importing QAIC-specific module, proceed without it if unavailable
try:
    import torch_qaic  # noqa: F401
except ImportError as e:
    logger.log_rank_zero(f"{e}. Moving ahead without these qaic modules.", logging.WARNING)


# Suppress all warnings
warnings.filterwarnings("ignore")


def setup_distributed_training(train_config: TrainConfig) -> None:
    """Initialize distributed training environment if enabled.

    Args:
        train_config (TrainConfig): Training configuration object.

    Notes:
        - If distributed data parallel (DDP) is disabled, this function does nothing.
        - Ensures the device is not CPU and does not specify an index for DDP compatibility.
        - Initializes the process group using the specified distributed backend.

    Raises:
        AssertionError: If device is CPU or includes an index with DDP enabled.
    """
    if not train_config.enable_ddp:
        return

    torch_device = torch.device(train_config.device)
    assert torch_device.type != "cpu", "Host doesn't support single-node DDP"
    assert torch_device.index is None, f"DDP requires only device type, got: {torch_device}"
    dist_backend_map = {"cpu": "gloo", "qaic": "qccl", "cuda": "gloo"}
    dist.init_process_group(backend=dist_backend_map[torch_device.type])
    if (not train_config.enable_pp) or (train_config.enable_pp and train_config.num_pp_stages == 1):
        # from here onward "qaic/cuda" will automatically map to "qaic:i/cuda:i", where i = process rank
        getattr(torch, torch_device.type).set_device(dist.get_rank())


def setup_seeds(seed: int) -> None:
    """Set random seeds across libraries for reproducibility.

    Args:
        seed (int): Seed value to set for random number generators.

    Notes:
        - Sets seeds for PyTorch, Python's random module, and NumPy.
    """
    torch.use_deterministic_algorithms(True)
    # With this flag, PP+DDP works only for meta-llama/Llama-3.2-1B and mistralai/Mistral-7B-Instruct-v0.3
    # and throws error during loading model for meta-llama/Llama-3.1-8B and bigger size models.

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model_and_tokenizer(
    train_config: TrainConfig, dataset_config: Any, **kwargs
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the pre-trained model and tokenizer from Hugging Face.

    Args:
        train_config (TrainConfig): Training configuration object containing model and tokenizer names.
        dataset_config (Any): A dataclass object representing dataset configuration.
        kwargs: Additional arguments to override PEFT config.

    Returns:
        tuple: A tuple of two values.
            - Model with pretrained weights loaded.
            - Model's tokenizer (AutoTokenizer).

    Notes:
        - Downloads the model if not already cached using login_and_download_hf_lm.
        - Configures the model with FP16 precision and disables caching for training.
        - Resizes model embeddings if tokenizer vocab size exceeds model embedding size.
        - Sets pad_token_id to eos_token_id if not defined in the tokenizer.
    """
    logger.log_rank_zero(f"Loading HuggingFace model for {train_config.model_name}")
    pretrained_model_path = hf_download(
        train_config.model_name,
        ignore_patterns=["*.txt", "*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.msgpack", "*.h5", "*.pth"],
    )
    if train_config.task_mode == Task_Mode.SEQ_CLASSIFICATION:
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_path,
            num_labels=dataset_config.num_labels,
            attn_implementation="sdpa",
            torch_dtype=torch.float16,
        )

        if not hasattr(model, "base_model_prefix"):
            logger.raise_error("Given huggingface model does not have 'base_model_prefix' attribute.", RuntimeError)

        for param in getattr(model, model.base_model_prefix).parameters():
            param.requires_grad = False

        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)
    else:
        device_map = get_device_map(train_config)
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_path,
            use_cache=False,
            attn_implementation="sdpa",
            torch_dtype=torch.float16,
            device_map=device_map,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        train_config.model_name if train_config.tokenizer_name is None else train_config.tokenizer_name
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # If there is a mismatch between tokenizer vocab size and embedding matrix,
    # throw a warning and then expand the embedding matrix
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        logger.log_rank_zero("Resizing the embedding matrix to match the tokenizer vocab size.", logging.WARNING)
        model.resize_token_embeddings(len(tokenizer))

    print_model_size(model)

    # Note: Need to call this before calling PeftModel.from_pretrained or get_peft_model.
    # Because, both makes model.is_gradient_checkpointing = True which is used in peft library to
    # apply gradient checkpointing related hooks to the input embeddings. Without this we will get
    # "No inf checks were recorded for this optimizer." error.
    # Enable gradient checkpointing
    if train_config.gradient_checkpointing:
        # Note: below attribute and method is only available in HuggingFace Transformer models.
        if hasattr(model, "supports_gradient_checkpointing") and model.supports_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"preserve_rng_state": False})
        else:
            logger.raise_error(
                "Given model doesn't support gradient checkpointing. Please disable it and run it.", RuntimeError
            )

    model = apply_peft(model, train_config, **kwargs)

    return model, tokenizer


def apply_peft(model: AutoModel, train_config: TrainConfig, **kwargs) -> Union[AutoModel, PeftModel]:
    """Apply Parameter-Efficient Fine-Tuning (PEFT) to the model if enabled.

    Args:
        model (AutoModel): Huggingface model.
        train_config (TrainConfig): Training configuration object.
        kwargs: Additional arguments to override PEFT config params.

    Returns:
        Union[AutoModel, PeftModel]: If use_peft in train_config is True
            then PeftModel object is returned else original model object
            (AutoModel) is returned.
    """
    if not train_config.use_peft:
        return model

    # Load the pre-trained peft model checkpoint and setup its configuration
    if train_config.from_peft_checkpoint:
        model = PeftModel.from_pretrained(model, train_config.from_peft_checkpoint, is_trainable=True)
        peft_config = model.peft_config
    # Generate the peft config and start fine-tuning from original model
    else:
        peft_config = generate_peft_config(train_config, **kwargs)
        model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    return model


def setup_dataloaders(
    train_config: TrainConfig,
    dataset_config: Any,
    tokenizer: AutoTokenizer,
) -> tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader], int]:
    """Set up training and validation DataLoaders.

    Args:
        train_config (TrainConfig): Training configuration object.
        dataset_config (Any): Configuration for the dataset (generated from train_config).
        tokenizer (AutoTokenizer): Tokenizer for preprocessing data.

    Returns:
        tuple: A tuple of three values.
            - First value represents train_dataloader
            - Second value represents eval_dataloader. It is None if
              validation is disabled.
            - Length of longest sequence in the dataset.

    Raises:
        RuntimeError: If validation is enabled but the validation set is too small.

    Notes:
        - Applies a custom data collator if provided by get_custom_data_collator.
        - Configures DataLoader kwargs using get_dataloader_kwargs for train and val splits.
    """

    train_dataloader = get_dataloader(tokenizer, dataset_config, train_config, split="train")
    logger.log_rank_zero(f"Number of Training Set Batches loaded = {len(train_dataloader)}")

    eval_dataloader = None
    if train_config.run_validation:
        eval_dataloader = get_dataloader(tokenizer, dataset_config, train_config, split="val")
        if len(eval_dataloader) == 0:
            logger.raise_error(
                f"The eval set size is too small for dataloader to load even one batch. Please increase the size of eval set. ({len(eval_dataloader)=})",
                ValueError,
            )
        else:
            logger.log_rank_zero(f"Number of Validation Set Batches loaded = {len(eval_dataloader)}")

        longest_seq_length, _ = get_longest_seq_length(
            torch.utils.data.ConcatDataset([train_dataloader.dataset, eval_dataloader.dataset])
        )
    else:
        longest_seq_length, _ = get_longest_seq_length(train_dataloader.dataset)

    return train_dataloader, eval_dataloader, longest_seq_length


def main(**kwargs) -> None:
    """
    Fine-tune a model on QAIC hardware with configurable training and LoRA parameters.

    Args:
        kwargs: Additional arguments to override TrainConfig.

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
    # TODO:Remove TrainConfig() and update_config() as all params are passed in kwargs by parser
    train_config = TrainConfig()
    update_config(train_config, **kwargs)
    dataset_config = generate_dataset_config(train_config.dataset)
    update_config(dataset_config, **kwargs)

    logger.prepare_for_logs(train_config.output_dir, train_config.dump_logs, train_config.log_level)

    setup_distributed_training(train_config)
    setup_seeds(train_config.seed)
    model, tokenizer = load_model_and_tokenizer(train_config, dataset_config, **kwargs)

    # Create DataLoaders for the training and validation dataset
    train_dataloader, eval_dataloader, longest_seq_length = setup_dataloaders(train_config, dataset_config, tokenizer)
    logger.log_rank_zero(
        f"The longest sequence length in the train data is {longest_seq_length}, "
        f"passed context length is {train_config.context_length} and overall model's context length is "
        f"{model.config.max_position_embeddings}"
    )
    if (not train_config.enable_pp) or (train_config.enable_pp and train_config.num_pp_stages == 1):
        model.to(train_config.device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    if train_config.enable_ddp:
        ignore_names = set()
        for name, param in model.named_parameters():
            if not param.requires_grad:
                ignore_names.add(name)
        # Adding params in ignore list will enforce DDP to ignore them during synchronization,
        # which will further reduce the tensor exchange across devices.
        torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model, ignore_names)
        model = nn.parallel.DistributedDataParallel(model)

    results = train(
        model,
        tokenizer,
        train_dataloader,
        eval_dataloader,
        optimizer,
        scheduler,
        train_config,
    )
    if train_config.enable_ddp:
        dist.destroy_process_group()
    return results


if __name__ == "__main__":
    parser = get_finetune_parser()
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
