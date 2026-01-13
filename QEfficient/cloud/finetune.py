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
from QEfficient.finetune.utils.helper import Task_Mode, get_local_rank, get_local_world_size, get_rank, get_world_size
from QEfficient.finetune.utils.logging_utils import logger
from QEfficient.finetune.utils.parser import get_finetune_parser
from QEfficient.finetune.utils.train_utils import print_model_size, print_trainable_parameters, train
from QEfficient.utils._utils import hf_download

# Try importing QAIC-specific module, proceed without it if unavailable
try:
    import torch_qaic  # noqa: F401
except ImportError as e:
    logger.log_rank_zero(
        f"Unable to import 'torch_qaic' package due to exception: {e}. Moving ahead without the torch_qaic extension.",
        logging.WARNING,
    )


# Suppress all warnings
warnings.filterwarnings("ignore")


def setup_distributed_training(train_config: TrainConfig) -> None:
    """
    Initialize the distributed training environment if Distributed Data Parallel (DDP) is enabled.

    Supports single-node and multi-node training launched via torchrun
    (uses WORLD_SIZE, RANK, LOCAL_RANK, LOCAL_WORLD_SIZE environment variables).
    Parameters
    ----------
    train_config : TrainConfig
        Training configuration object containing settings for distributed training.

    Raises
    ------
    AssertionError
        If the number of required devices exceeds the total available devices.
        If pipeline parallelism (`num_pp_stages`) is enabled but set to 1.
        If DDP is enabled with a CPU device or with a specific device index (DDP requires device type only).
    Notes
    -----
    - If `train_config.enable_ddp` is False, this function performs no action.
    - Sets the appropriate device for each process in a distributed setup.
    """

    torch_device = torch.device(train_config.device)

    # Validate pipeline parallelism settings
    if train_config.enable_pp:
        assert train_config.num_pp_stages > 1, (
            f"For pipeline parallelism, num_pp_stages should be greater than 1. Got {train_config.num_pp_stages}"
        )

    # If DDP is disabled, nothing to initialize here
    if not train_config.enable_ddp:
        # Non-DDP path: allow explicit device index, just set it if present
        if torch_device.type != "cpu" and torch_device.index is not None:
            getattr(torch, torch_device.type).set_device(torch_device.index)
        return

    # ---- DDP path (single- or multi-node) ----
    assert torch_device.type != "cpu", "Host doesn't support single-node DDP"
    assert torch_device.index is None, f"DDP requires only device type (qaic/cuda), got: {torch_device}"

    # Torchrun-provided env vars
    world_size = get_world_size()
    rank = get_rank()
    local_rank = get_local_rank()
    local_world_size = get_local_world_size()

    # Per-node device validation
    num_available_devices = getattr(torch, torch_device.type).device_count()
    assert local_world_size * train_config.num_pp_stages <= num_available_devices, (
        "Number of devices required per node (LOCAL_WORLD_SIZE * num_pp_stages) should be <= locally available devices."
    )

    dist_backend_map = {"cpu": "gloo", "qaic": "qccl", "cuda": "gloo"}
    dist.init_process_group(dist_backend_map[torch_device.type], rank=rank, world_size=world_size)

    # Set the base device index for this process on this node
    # For PP: each process controls num_pp_stages devices starting from base_device_index
    base_device_index = local_rank * train_config.num_pp_stages
    # from here onward "qaic/cuda" will automatically map to "qaic:i/cuda:i", where i = process rank
    getattr(torch, torch_device.type).set_device(base_device_index)

    # persist rank info in the config
    train_config.rank = rank
    train_config.local_rank = local_rank
    train_config.world_size = world_size
    train_config.local_world_size = local_world_size


def setup_seeds(seed: int) -> None:
    """
    Set random seeds across multiple libraries for reproducibility.

    This function ensures that random number generation is deterministic across PyTorch,
    Python's built-in `random` module, and NumPy for consistent experiment results.

    Parameters
    ----------
    seed : int
        The seed value to set for all random number generators.
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
    """
    Load the pre-trained Hugging Face model and its corresponding tokenizer.

    This function handles model download, configuration (e.g., precision, caching),
    and tokenizer setup. It also applies PEFT if enabled in the training configuration.

    Parameters
    ----------
    train_config : TrainConfig
        Training configuration object containing model and tokenizer names, task mode, etc.
    dataset_config : Any
        A dataclass object representing the dataset configuration, used for task-specific
        model setup (e.g., number of labels for sequence classification).
    **kwargs :
        Additional arguments to override PEFT configuration parameters.

    Returns
    -------
    tuple[Union[AutoModelForCausalLM, AutoModelForSequenceClassification], AutoTokenizer]
        A tuple containing:
        - The loaded model (either `AutoModelForCausalLM` or `AutoModelForSequenceClassification`).
        - The model's tokenizer (`AutoTokenizer`).

    Raises
    ------
    RuntimeError
        If the Hugging Face model for sequence classification does not have
        a `base_model_prefix` attribute when `task_mode` is `SEQ_CLASSIFICATION`.
        If gradient checkpointing is enabled but the model does not support it.
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
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"preserve_rng_state": True})
        else:
            logger.raise_error(
                "Given model doesn't support gradient checkpointing. Please disable it and run it.", RuntimeError
            )

    model = apply_peft(model, train_config, **kwargs)

    return model, tokenizer


def apply_peft(model: AutoModel, train_config: TrainConfig, **kwargs) -> Union[AutoModel, PeftModel]:
    """
    Apply Parameter-Efficient Fine-Tuning (PEFT) to the model if enabled in the training configuration.

    This function configures and applies PEFT methods (e.g., LoRA) to the base model,
    either from a pre-trained PEFT checkpoint or by generating a new PEFT configuration.

    Parameters
    ----------
    model : AutoModel
        The Hugging Face model to which PEFT will be applied.
    train_config : TrainConfig
        Training configuration object, specifying whether to use PEFT and if a checkpoint exists.
    **kwargs :
        Additional arguments to override PEFT configuration parameters.

    Returns
    -------
    Union[AutoModel, PeftModel]
        If `train_config.use_peft` is True, a `PeftModel` object is returned.
        Otherwise, the original `AutoModel` object is returned.
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
    """
    Set up training and optional validation DataLoaders based on the provided configurations.

    This function prepares `DataLoader` instances for both training and validation datasets,
    applying necessary preprocessing and batching. It also determines the longest sequence
    length in the combined dataset.

    Parameters
    ----------
    train_config : TrainConfig
        Training configuration object containing DataLoader settings (batch size, etc.)
        and validation preferences.
    dataset_config : Any
        Configuration for the dataset, used to fetch and prepare splits.
    tokenizer : AutoTokenizer
        Tokenizer for preprocessing and tokenizing the dataset samples.

    Returns
    -------
    tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader], int]
        A tuple containing:
        - `train_dataloader`: The DataLoader for the training dataset.
        - `eval_dataloader`: The DataLoader for the validation dataset, or `None` if validation is disabled.
        - `longest_seq_length`: The length of the longest sequence found in the dataset(s).

    Raises
    ------
    ValueError
        If validation is enabled but the resulting validation DataLoader is empty.
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
    Fine-tune a Hugging Face model on Qualcomm AI 100 hardware with configurable training
    and Parameter-Efficient Fine-Tuning (PEFT) parameters.

    This is the main entry point for the fine-tuning script. It orchestrates the
    setup of distributed training, model and tokenizer loading, DataLoader creation,
    optimizer and scheduler initialization, and the training loop.

    Parameters
    ----------
    **kwargs :
        Additional arguments used to override default parameters in `TrainConfig`
        and PEFT configuration. These are typically parsed from command-line arguments.

    Example
    -------
    To fine-tune a model using a YAML configuration file for PEFT:

    .. code-block:: bash

        python -m QEfficient.cloud.finetune \\
            --model_name "meta-llama/Llama-3.2-1B" \\
            --lr 5e-4 \\
            --peft_config_file "lora_config.yaml"

    To fine-tune a model using a default LoRA configuration:

    .. code-block:: bash

        python -m QEfficient.cloud.finetune \\
            --model_name "meta-llama/Llama-3.2-1B" \\
            --lr 5e-4
    """
    train_config = TrainConfig()
    update_config(train_config, **kwargs)
    custom_dataset_config_file = kwargs.pop("custom_dataset_config", None)
    dataset_config = generate_dataset_config(train_config.dataset, custom_dataset_config_file)

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

    # Figure out the concrete device for this process
    torch_device = torch.device(train_config.device)
    if train_config.enable_ddp and torch_device.type != "cpu":
        # setup_distributed_training has already set the current device based on LOCAL_RANK
        current_idx = getattr(torch, torch_device.type).current_device()
        device = torch.device(torch_device.type, current_idx)
    else:
        device = torch_device

    if not train_config.enable_pp:
        model.to(device)

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