# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch

# from QEfficient.finetune.data.concatenator import ConcatDataset
from QEfficient.finetune.dataset.dataset_config import DATALOADER_COLLATE_FUNC, DATASET_PREPROC
from QEfficient.finetune.utils.config_utils import get_dataloader_kwargs


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


def get_dataloader(tokenizer, dataset_config, train_config, split: str = "train"):
    dataset = get_preprocessed_dataset(tokenizer, dataset_config, split)
    dl_kwargs = get_dataloader_kwargs(train_config, dataset, tokenizer, split)

    # if split == "train" and train_config.batching_strategy == "packing":
    #    dataset = ConcatDataset(dataset, chunk_size=train_config.context_length)

    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **dl_kwargs,
    )
    return dataloader
