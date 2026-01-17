# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy
import json

import torch
from torch.utils.data import Dataset

from QEfficient.finetune.utils.logging_utils import logger

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", context_length=None):
        try:
            self.ann = json.load(open(dataset_config.data_path))
        except FileNotFoundError:
            logger.raise_error(
                "Loading of alpaca dataset failed! Please use (wget -c https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json -P dataset/) to download the alpaca dataset.",
                FileNotFoundError,
            )
        # Use 5% of the dataset for evaluation
        total_len = len(self.ann)
        eval_length = max(1, int(total_len / 20))
        if partition == "train":
            self.ann = self.ann[eval_length:]
        else:
            self.ann = self.ann[:eval_length]

        self.tokenizer = tokenizer
        self.context_length = context_length

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting

        ann = self.ann[index]
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        example = prompt + ann["output"]

        if self.context_length is not None:
            padding_type = "max_length"
        else:
            padding_type = True
        prompt = torch.tensor(
            self.tokenizer.encode(prompt, max_length=self.context_length, padding=padding_type), dtype=torch.int64
        )
        example = self.tokenizer.encode(example, max_length=self.context_length, padding=padding_type)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask": example_mask.tolist(),
        }
