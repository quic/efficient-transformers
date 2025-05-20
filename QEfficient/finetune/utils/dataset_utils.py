# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch

# from QEfficient.finetune.data.concatenator import ConcatDataset
from QEfficient.finetune.dataset.dataset_config import DATALOADER_COLLATE_FUNC, DATASET_PREPROC


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
