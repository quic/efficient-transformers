# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import random
import warnings

import fire
import numpy as np
import torch
import torch.optim as optim
from peft import PeftModel, get_peft_model
from torch.optim.lr_scheduler import StepLR

from QEfficient.finetune.configs.training import train_config as TRAIN_CONFIG
from QEfficient.finetune.utils.config_utils import (
    generate_dataset_config,
    generate_peft_config,
    get_dataloader_kwargs,
    update_config,
)
from QEfficient.finetune.utils.dataset_utils import (
    get_custom_data_collator,
    get_preprocessed_dataset,
)
from QEfficient.finetune.utils.train_utils import print_model_size, train

try:
    import torch_qaic  # noqa: F401
except ImportError as e:
    print(f"Warning: {e}. Moving ahead without these qaic modules.")


from transformers import AutoModelForCausalLM, AutoTokenizer

# Suppress all warnings
warnings.filterwarnings("ignore")


def main(**kwargs):
    # update the configuration for the training process
    train_config = TRAIN_CONFIG()
    update_config(train_config, **kwargs)

    # Set the seeds for reproducibility
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)

    # Load the pre-trained model and setup its configuration
    # config = AutoConfig.from_pretrained(train_config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        train_config.model_name,
        use_cache=False,
        attn_implementation="sdpa",
        torch_dtype=torch.float16,
    )

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(
        train_config.model_name if train_config.tokenizer_name is None else train_config.tokenizer_name
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # If there is a mismatch between tokenizer vocab size and embedding matrix,
    # throw a warning and then expand the embedding matrix
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    print_model_size(model, train_config)
    # print the datatype of the model parameters
    # print(get_parameter_dtypes(model))

    if train_config.use_peft:
        # Load the pre-trained peft model checkpoint and setup its configuration
        if train_config.from_peft_checkpoint:
            model = PeftModel.from_pretrained(model, train_config.from_peft_checkpoint, is_trainable=True)
            peft_config = model.peft_config
        # Generate the peft config and start fine-tuning from original model
        else:
            peft_config = generate_peft_config(train_config, kwargs)
            model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Get the dataset utils
    dataset_config = generate_dataset_config(train_config, kwargs)
    dataset_processer = tokenizer

    # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        dataset_processer,
        dataset_config,
        split="train",
    )

    dataset_val = get_preprocessed_dataset(
        dataset_processer,
        dataset_config,
        split="test",
    )

    # TODO: vbaddi, check if its necessary to do this?
    # dataset_train = ConcatDataset(
    #             dataset_train, chunk_size=train_config.context_length
    #         )
    ##
    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, dataset_processer, "train")
    print("length of dataset_train", len(dataset_train))
    custom_data_collator = get_custom_data_collator(dataset_processer, dataset_config)
    if custom_data_collator:
        print("custom_data_collator is used")
        train_dl_kwargs["collate_fn"] = custom_data_collator
    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )
    print(f"--> Num of Training Set Batches loaded = {len(train_dataloader)}")

    eval_dataloader = None
    if train_config.run_validation:
        # if train_config.batching_strategy == "packing":
        #     dataset_val = ConcatDataset(
        #         dataset_val, chunk_size=train_config.context_length
        #     )

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, dataset_processer, "val")
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
            raise ValueError(
                f"The eval set size is too small for dataloader to load even one batch. Please increase the size of eval set. ({len(eval_dataloader)=})"
            )
        else:
            print(f"--> Num of Validation Set Batches loaded = {len(eval_dataloader)}")

    model.to(train_config.device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    _ = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        train_config.device,
        None,
        None,
    )


if __name__ == "__main__":
    fire.Fire(main)
