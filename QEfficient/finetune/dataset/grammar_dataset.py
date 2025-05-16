# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from pathlib import Path

from datasets import load_dataset
from torch.utils.data import Dataset


class grammar(Dataset):
    def __init__(self, tokenizer, csv_name=None, context_length=None):
        try:
            self.dataset = load_dataset(
                "csv",
                data_files={"train": [csv_name]},  # "eval": "grammar_validation.csv"},
                delimiter=",",
            )
        except Exception as e:
            print(
                "Loading of grammar dataset failed! Please see [here](https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/datasets/grammar_dataset/grammar_dataset_process.ipynb) for details on how to download the dataset."
            )
            raise e

        self.context_length = context_length
        self.tokenizer = tokenizer
        self.print_text = False  # print_text

    def __len__(self):
        return self.dataset["train"].shape[0]

    def convert_to_features(self, example_batch):
        # Create prompt and tokenize contexts and questions

        if self.print_text:
            print("Input Text: ", self.clean_text(example_batch["text"]))

        input_ = example_batch["input"]
        target_ = example_batch["target"]

        prompt = f"Correct this to standard English: {input_}\n---\nCorrected: "
        prompt_ids = self.tokenizer.encode(
            self.tokenizer.bos_token + prompt,
            add_special_tokens=False,
            max_length=self.context_length,
            pad_to_max_length=True,
        )
        label_ids = self.tokenizer.encode(
            target_ + self.tokenizer.eos_token,
            add_special_tokens=False,
            max_length=self.context_length,
            pad_to_max_length=True,
        )

        sample = {
            "input_ids": prompt_ids + label_ids,
            "attention_mask": [1] * len(prompt_ids + label_ids),
            "labels": [-100] * len(prompt_ids) + label_ids,
        }

        return sample

    def __getitem__(self, index):
        return self.convert_to_features(self.dataset["train"][int(index)])


def get_dataset(dataset_config, tokenizer, csv_name=None, context_length=None):
    """cover function for handling loading the working dataset"""
    """dataset loading"""
    currPath = Path.cwd() / "datasets_grammar" / "grammar_train.csv"
    print(f"Loading dataset {currPath}")
    csv_name = str(currPath)
    print(csv_name)
    dataset = grammar(tokenizer=tokenizer, csv_name=csv_name, context_length=context_length)

    return dataset
