# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
def insert_pad_token(tokenizer):
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        # Try to use existing special token as pad token
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.bos_token is not None:
            tokenizer.pad_token = tokenizer.bos_token
        elif tokenizer.sep_token is not None:
            tokenizer.pad_token = tokenizer.sep_token
        else:
            # Add a new pad token
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def apply_train_test_split(dataset, split_ratio, split, seed):
    """
    Apply train/test split to the dataset based on split_ratio.
    """
    splitted_dataset = dataset.train_test_split(test_size=(1 - split_ratio), seed=seed)
    if split == "test":
        dataset = splitted_dataset["test"]
    else:
        dataset = splitted_dataset["train"]
    return dataset
