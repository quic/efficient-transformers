# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import datasets


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
