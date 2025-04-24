# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


from itertools import chain

import datasets


def get_preprocessed_imdb(dataset_config, tokenizer, split, context_length=None):
    dataset = datasets.load_dataset("stanfordnlp/imdb", split=split, trust_remote_code=True)

    if split == "test":
        # Test set contains 15000 samples. Not all are required.
        # 0-12499 are 0 labeled samples, 12500-24999 are 1 labeled samples.
        dataset = dataset.select(chain(range(0, 500), range(12500, 13000)))

    # Need to shuffle dataset as all the 0 labeled data is organized first and then all the 1 labeled data.
    dataset = dataset.shuffle(seed=42)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def tokenize_add_label(sample):
        data = tokenizer(
            sample["text"],
            add_special_tokens=True,
            max_length=tokenizer.model_max_length,
        )

        data["labels"] = [sample["label"]]
        return data

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    return dataset
