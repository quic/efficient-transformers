# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import datasets
import torch
from torch.nn.utils.rnn import pad_sequence


def get_preprocessed_samsum(dataset_config, tokenizer, split, context_length=None):
    dataset = datasets.load_dataset("Samsung/samsum", split=split, trust_remote_code=True)

    prompt = "Summarize this dialog:\n{dialog}\n---\nSummary:\n"

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(dialog=sample["dialogue"]),
            "summary": sample["summary"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(
            tokenizer.bos_token + sample["prompt"],
            add_special_tokens=False,
            max_length=context_length,
            pad_to_max_length=True,
        )
        summary = tokenizer.encode(
            sample["summary"] + tokenizer.eos_token,
            add_special_tokens=False,
            max_length=context_length,
            pad_to_max_length=True,
        )

        sample = {
            "input_ids": prompt + summary,
            "attention_mask": [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
        }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset


def collate_fn(batch):
    eos_token = batch[0]["input_ids"][-1]

    input_ids = pad_sequence(
        [torch.tensor(b["input_ids"], dtype=torch.int32) for b in batch], batch_first=True, padding_value=eos_token
    )
    attn_mask = pad_sequence(
        [torch.tensor(b["attention_mask"], dtype=torch.int32) for b in batch], batch_first=True, padding_value=0
    )
    labels = pad_sequence(
        [torch.tensor(b["labels"], dtype=torch.long) for b in batch], batch_first=True, padding_value=eos_token
    )
    return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}


def get_samsum_collate_fn(dataset_processer, dataset_config):
    return collate_fn
