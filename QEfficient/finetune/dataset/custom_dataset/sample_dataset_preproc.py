# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import datasets
from transformers.data import DataCollatorForSeq2Seq


def get_data_collator(tokenizer):
    return DataCollatorForSeq2Seq(tokenizer)


def get_preprocessed_disc(dataset_config, tokenizer, split, context_length=None):
    dataset = datasets.load_dataset("hallisky/DiSC")

    # Considering 'train' split as this dataset has only one split.
    dataset = dataset["train"]

    test_split_ratio = dataset_config.test_split_ratio
    disc_style = dataset_config.disc_style

    # Only collect the samples for a given style.
    available_styles = set(dataset["category"])
    if disc_style not in available_styles:
        raise RuntimeError(f"For DiSC dataset the provided disc_style '{disc_style}' is not supported.")

    dataset = dataset.filter(lambda example: example["category"] == disc_style)

    # Shuffle the dataset before splitting
    dataset = dataset.shuffle(seed=42)

    # Split the data in train and test split.
    total_samples = len(dataset)
    test_size = int(total_samples * test_split_ratio)
    train_size = total_samples - test_size

    if split == "test":
        indices = range(train_size, total_samples)
    else:
        indices = range(0, train_size)

    dataset = dataset.select(indices)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Below is the template of the DiSC dataset.
    # <bos>### Original:{original} \n ### Rewrite: {rewrite} <eos>
    template = "### Original:{original} \n ### Rewrite: "

    def apply_prompt_template(sample):
        return {
            "input": template.format(original=sample["original"]),
            "label": sample["generation"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        if context_length is not None:
            padding_type = "max_length"
        else:
            padding_type = True

        input = tokenizer.encode(
            tokenizer.bos_token + sample["input"],
            add_special_tokens=False,
            max_length=context_length,
            padding=padding_type,
        )
        label = tokenizer.encode(
            sample["label"] + tokenizer.pad_token + tokenizer.eos_token,
            add_special_tokens=False,
            max_length=context_length,
            padding=padding_type,
        )

        sample = {
            "input_ids": (input + label),
            "attention_mask": [1] * (len(input) + len(label)),
            "labels": [-100] * len(input) + label,
        }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
