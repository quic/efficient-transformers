# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class samsum_dataset:
    dataset: str = "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"


@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "train"
    test_split: str = "validation"


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "dataset/alpaca_data.json"


@dataclass
class gsm8k_dataset:
    dataset: str = "gsm8k_dataset"
    train_split: str = "train"
    test_split: str = "test"


@dataclass
class imdb_dataset:
    dataset: str = "imdb_dataset"
    train_split: str = "train"
    test_split: str = "test"
    num_labels: int = 2


@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "dataset/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"
    data_path: str = ""
