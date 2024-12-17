# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from unittest.mock import patch

import datasets


@patch("builtins.input", return_value="N")
def load_samsum(split, _):
    try:
        ds = datasets.load_dataset("Samsung/samsum", split=split)
    except ValueError as e:
        if "trust_remote_code" in str(e):
            raise ValueError(
                "Loading Samsung/samsum requires you to execute the dataset script in that repo on your local machine. Make sure you have read the code there to avoid malicious use, then set HF_DATASETS_TRUST_REMOTE_CODE env variable to True."
            ) from e
        else:
            raise e
    return ds


def get_preprocessed_samsum(dataset_config, tokenizer, split):
    dataset = load_samsum(split)

    prompt = "Summarize this dialog:\n{dialog}\n---\nSummary:\n"

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(dialog=sample["dialogue"]),
            "summary": sample["summary"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"] + tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + summary,
            "attention_mask": [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
        }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
