# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
import torch.distributed as dist
from transformers.data import DataCollatorForSeq2Seq

from QEfficient.finetune.data.sampler import DistributedLengthBasedBatchSampler
from QEfficient.finetune.dataset.dataset_config import DATALOADER_COLLATE_FUNC, DATASET_PREPROC
from QEfficient.finetune.utils.config_utils import pad_dataset
from QEfficient.finetune.utils.helper import get_num_ddp_devices


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train", context_length: int = None
) -> torch.utils.data.Dataset:
    if dataset_config.dataset not in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    def get_split():
        return dataset_config.train_split if split == "train" else dataset_config.test_split

    return DATASET_PREPROC[dataset_config.dataset](dataset_config, tokenizer, get_split(), context_length)


def get_custom_data_collator(dataset_processer, dataset_config) -> torch.utils.data.Dataset:
    if dataset_config.dataset not in DATALOADER_COLLATE_FUNC:
        return None

    return DATALOADER_COLLATE_FUNC[dataset_config.dataset](dataset_processer, dataset_config)


def get_dataloader_kwargs(train_config, dataset, dataset_processer, split):
    kwargs = {}
    batch_size = train_config.train_batch_size if split == "train" else train_config.val_batch_size
    if train_config.enable_ddp:
        if train_config.enable_sorting_for_ddp:
            if train_config.context_length:
                raise ValueError(
                    "Sorting cannot be done with padding, Please disable sorting or pass context_length as None to disable padding"
                )
            else:
                kwargs["batch_sampler"] = DistributedLengthBasedBatchSampler(
                    dataset,
                    batch_size=batch_size,
                    rank=dist.get_rank(),
                    num_replicas=dist.get_world_size(),
                    shuffle=False,
                )
        else:
            kwargs["sampler"] = torch.utils.data.DistributedSampler(
                dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False
            )
            kwargs["batch_size"] = batch_size
            kwargs["drop_last"] = True
    else:
        kwargs["batch_size"] = batch_size
        kwargs["drop_last"] = True
    kwargs["collate_fn"] = DataCollatorForSeq2Seq(dataset_processer)
    return kwargs


def get_dataloader(tokenizer, dataset_config, train_config, split: str = "train"):
    dataset = get_preprocessed_dataset(tokenizer, dataset_config, split, context_length=train_config.context_length)

    total_devices = get_num_ddp_devices()
    dataset = pad_dataset(dataset, train_config.train_batch_size, total_devices)

    dl_kwargs = get_dataloader_kwargs(train_config, dataset, tokenizer, split)

    # FIXME (Meet): Add custom data collator registration from the outside by the user.
    custom_data_collator = get_custom_data_collator(tokenizer, dataset_config)
    if custom_data_collator:
        print("custom_data_collator is used")
        dl_kwargs["collate_fn"] = custom_data_collator

    print(f"length of dataset_{split}", len(dataset))

    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **dl_kwargs,
    )
    return dataloader
