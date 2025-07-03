# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import datasets
import torch
import torch.distributed as dist
from transformers.data import DataCollatorForSeq2Seq

from QEfficient.finetune.data.sampler import DistributedLengthBasedBatchSampler
from QEfficient.finetune.dataset.dataset_config import DATALOADER_COLLATE_FUNC, DATASET_PREPROC
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
            kwargs["drop_last"] = False
    else:
        kwargs["batch_size"] = batch_size
        kwargs["drop_last"] = False
    kwargs["collate_fn"] = DataCollatorForSeq2Seq(dataset_processer)
    return kwargs


def padding_dataset(train_config, dataset, batch_size):
    if train_config.enable_ddp and train_config.enable_sorting_for_ddp:
        if isinstance(dataset, datasets.Dataset):
            # Hugging Face Dataset transformation
            dataset = dataset.map(lambda x: {"input_length": len(x["input_ids"])})
            dataset = dataset.sort("input_length")

        else:
            dataset = sorted(dataset, key=lambda x: len(x["input_ids"]))

    dummy_row = next(iter(dataset))
    dummy_row["labels"] = torch.tensor([-100] * len(dummy_row["labels"]))
    padding_size = 0
    num_replicas = get_num_ddp_devices()
    remainder = len(dataset) % (num_replicas * batch_size)
    padding_size = (num_replicas * batch_size) - remainder

    dummy_data = [dummy_row.copy() for _ in range(padding_size)]
    dummy_dataset = datasets.Dataset.from_list(dummy_data)
    if isinstance(dataset, datasets.Dataset):
        combined_dataset = datasets.concatenate_datasets([dataset, dummy_dataset])
    else:
        combined_dataset = dataset + list(dummy_dataset)
    return combined_dataset


def get_dataloader(tokenizer, dataset_config, train_config, split: str = "train"):
    dataset = get_preprocessed_dataset(tokenizer, dataset_config, split, context_length=train_config.context_length)

    batch_size = train_config.train_batch_size if split == "train" else train_config.val_batch_size
    dataset = padding_dataset(train_config, dataset, batch_size)

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
