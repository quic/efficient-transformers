# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from pathlib import Path

from datasets import load_dataset
from torch.utils.data import Dataset

from QEfficient.finetune.utils.logging_utils import logger


class grammar(Dataset):
    def __init__(self, tokenizer, csv_name=None, context_length=None):
        try:
            self.dataset = load_dataset(
                "csv",
                data_files={"train": [csv_name]},  # "eval": "grammar_validation.csv"},
                delimiter=",",
            )
        except FileNotFoundError:
            logger.raise_error(
                "Loading of grammar dataset failed! Please check (https://drive.google.com/drive/folders/1kKlGcinD_FhGXC0LztN4Ts605YXzMEVA) to download the c4_200m_550k.csv. Copy-paste the path of this downloaded csv in the grammar_dataset_preprocess.py and run this file",
                FileNotFoundError,
            )

        self.context_length = context_length
        self.tokenizer = tokenizer
        self.print_text = False  # print_text

    def __len__(self):
        return self.dataset["train"].shape[0]

    def convert_to_features(self, example_batch):
        # Create prompt and tokenize contexts and questions

        if self.print_text:
            logger.log_rank_zero("Input Text: ", self.clean_text(example_batch["text"]))

        input_ = example_batch["input"]
        target_ = example_batch["target"]

        prompt = f"Correct this to standard English: {input_}\n---\nCorrected: "

        if self.context_length is not None:
            padding_type = "max_length"
        else:
            padding_type = True

        prompt_ids = self.tokenizer.encode(
            self.tokenizer.bos_token + prompt,
            add_special_tokens=False,
            max_length=self.context_length,
            padding=padding_type,
        )
        label_ids = self.tokenizer.encode(
            target_ + self.tokenizer.eos_token,
            add_special_tokens=False,
            max_length=self.context_length,
            padding=padding_type,
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
    dataset = grammar(tokenizer=tokenizer, csv_name=str(currPath), context_length=context_length)

    return dataset
