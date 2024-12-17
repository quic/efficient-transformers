# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import inspect
from dataclasses import asdict

from peft import (
    AdaptionPromptConfig,
    LoraConfig,
    PrefixTuningConfig,
)
from transformers import default_data_collator

import QEfficient.finetune.configs.dataset_config as datasets
from QEfficient.finetune.configs.peft_config import llama_adapter_config, lora_config, prefix_config
from QEfficient.finetune.configs.training import train_config
from QEfficient.finetune.dataset.dataset_config import DATASET_PREPROC


def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                # allow --some_config.some_param=True
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        # In case of specialized config we can warn user
                        print(f"Warning: {config_name} does not accept parameter: {k}")
            elif isinstance(config, train_config):
                print(f"Warning: unknown parameter {k}")


def generate_peft_config(train_config, kwargs):
    configs = (lora_config, llama_adapter_config, prefix_config)
    peft_configs = (LoraConfig, AdaptionPromptConfig, PrefixTuningConfig)
    names = tuple(c.__name__.rstrip("_config") for c in configs)

    if train_config.peft_method not in names:
        raise RuntimeError(f"Peft config not found: {train_config.peft_method}")

    config = configs[names.index(train_config.peft_method)]()

    update_config(config, **kwargs)
    params = asdict(config)
    peft_config = peft_configs[names.index(train_config.peft_method)](**params)

    return peft_config


def generate_dataset_config(train_config, kwargs):
    names = tuple(DATASET_PREPROC.keys())

    assert train_config.dataset in names, f"Unknown dataset: {train_config.dataset}"

    dataset_config = {k: v for k, v in inspect.getmembers(datasets)}[train_config.dataset]()

    update_config(dataset_config, **kwargs)

    return dataset_config


# def get_dataloader_kwargs(train_config, dataset, dataset_processer, mode):
#     kwargs = {}
#     batch_size = (
#         train_config.batch_size_training
#         if mode == "train"
#         else train_config.val_batch_size
#     )
#     if train_config.batching_strategy == "padding":
#         kwargs["batch_sampler"] = LengthBasedBatchSampler(
#             dataset, batch_size, drop_last=True, shuffle=mode == "train"
#         )
#         kwargs["collate_fn"] = DataCollatorForSeq2Seq(dataset_processer)
#         # kwargs["collate_fn"] = default_data_collator
#     return kwargs


def get_dataloader_kwargs(train_config, dataset, dataset_processer, mode):
    kwargs = {}
    batch_size = train_config.batch_size_training if mode == "train" else train_config.val_batch_size
    kwargs["batch_size"] = batch_size
    kwargs["drop_last"] = True
    kwargs["collate_fn"] = default_data_collator
    return kwargs
