# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Dict

from datasets import Dataset, load_dataset

default_instruction = """### Instruction: Solve the math question using a basic calculator.
Calculator can be invoked using the format: <<expression=answer>>.
"expression" can be one of the 4 arithmetic operations, and "answer" will be filled in for you.
Example: <<20+30=50>>

### Question: {question}

### Answer: """


def tokenize_and_mask(row: Dict[str, str], *, tokenizer, instruction) -> Dict[str, list]:
    start_tokens = {tokenizer(x, add_special_tokens=False)["input_ids"][0] for x in ["<<", " <<"]}
    equal_tokens = {tokenizer(x, add_special_tokens=False)["input_ids"][0] for x in ["=", " ="]}
    end_tokens = {tokenizer(x, add_special_tokens=False)["input_ids"][0] for x in [">>"]}

    input_str = tokenizer.bos_token + instruction.format(**row)
    ques_ids = tokenizer(input_str, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    ans_ids = tokenizer(row["answer"] + tokenizer.eos_token, add_special_tokens=False, return_attention_mask=False)[
        "input_ids"
    ]
    input_ids = ques_ids + ans_ids

    # State machine to recognize <<expression=answer>> and mask answer
    mode = 0
    for i, token in enumerate(ans_ids):
        if mode == 0 and token in start_tokens:
            mode = 1
        elif mode == 1 and token in equal_tokens:
            mode = 2
        elif mode == 2:
            ans_ids[i] = -100
            if token in end_tokens:
                mode = 0

    labels = [-100] * len(ques_ids) + ans_ids

    inputs = {"input_ids": input_ids, "labels": labels, "length": len(input_ids)}
    return inputs


def pad_to_max_length(row: Dict[str, list], *, tokenizer, max_length: int) -> Dict[str, list]:
    length = row["length"]
    return {
        "input_ids": row["input_ids"] + [tokenizer.pad_token_id] * (max_length - length),
        "attention_mask": [1] * length + [0] * (max_length - length),
        "labels": row["labels"] + [-100] * (max_length - length),
    }


def get_gsm8k_dataset(
    dataset_config,
    tokenizer,
    split,
    context_length=None,
    instruction: str = default_instruction,
) -> Dataset:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    ds = ds.map(
        tokenize_and_mask,
        fn_kwargs={"tokenizer": tokenizer, "instruction": instruction},
        remove_columns=["question", "answer"],
    )

    if context_length is not None:
        ds = ds.filter(lambda x: x["length"] <= context_length)
        ds = ds.map(
            pad_to_max_length,
            fn_kwargs={"tokenizer": tokenizer, "max_length": context_length},
            remove_columns=["length"],
        )

    ds.set_format("torch")

    return ds
