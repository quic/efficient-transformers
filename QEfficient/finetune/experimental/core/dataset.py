# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Dataset components for the training system.
"""

import importlib
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

from datasets import load_dataset, load_dataset_builder
from torch.utils.data import Dataset

from QEfficient.finetune.experimental.core.component_registry import registry
from QEfficient.finetune.experimental.core.utils.dataset_utils import (
    apply_train_test_split,
)


class BaseDataset(Dataset, ABC):
    """Base class for all datasets to ensure consistent interface."""

    def __init__(self, dataset_name: str, split: str, seed: int = 42, **kwargs):
        self.dataset_name = dataset_name
        self.split = split
        self.seed = seed
        self.kwargs = kwargs
        self._initialize_dataset()

    @abstractmethod
    def _initialize_dataset(self):
        """Subclasses should implement this to load and prepare the dataset."""
        pass

    @abstractmethod
    def __len__(self):
        """Return the number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """Should return a dictionary with 'input_ids', 'attention_mask', and 'labels'."""
        pass


@registry.dataset("sft_dataset")
class SFTDataset(BaseDataset):
    """
    A Supervised Fine-Tuning (SFT) dataset class for text data.

    This class handles loading data from Hugging Face datasets or custom JSON files,
    filtering out invalid samples, and applying a prompt/completion templating for SFT tasks.

    Args:
        dataset_name (str): The name of the dataset to load from Hugging Face datasets.
                           Ignored if json_file_path is provided.
        split (str): The dataset split to use (e.g., "train", "validation", "test").
        split_ratio (float): Ratio for train/test split when only one split is available.
        seed (int): Random seed for reproducibility.
        json_file_path (str, optional): Path to a custom JSON file containing the dataset.
                                       If provided, this takes precedence over dataset_name.
        prompt_template (str): A string template for constructing the prompt. Variables in the
                                template should be enclosed in curly braces, e.g., "Answer the question: {question}".
        completion_template (str): A string template for constructing the completion (target).
                                   Variables should be enclosed in curly braces, e.g., "{answer}".

    Raises:
        RuntimeError: If any variables specified in `prompt_template` or `completion_template`
                      are not found as columns in the loaded dataset.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        split_ratio: float = 0.8,
        seed: int = 42,
        **kwargs,
    ):
        self.split_ratio = split_ratio
        self.json_file_path = kwargs.get("json_file_path", None)
        self.prompt_template = kwargs.get("prompt_template", None)
        self.completion_template = kwargs.get("completion_template", None)
        self.prompt_func_path = kwargs.get("prompt_func", None)
        self.completion_func_path = kwargs.get("completion_func", None)
        self.remove_samples_with_empty_columns = kwargs.get("remove_samples_with_empty_columns", True)

        if self.json_file_path not in (None, ""):
            if not os.path.isfile(self.json_file_path):
                raise FileNotFoundError(f"JSON file not found or invalid: '{self.json_file_path}'")
        if (self.prompt_template is None and self.prompt_func_path is None) or (
            self.prompt_template is not None and self.prompt_func_path is not None
        ):
            raise RuntimeError("Either provide prompt_template or prompt_func in the config.")
        if (self.completion_template is None and self.completion_func_path is None) or (
            self.completion_template is not None and self.completion_func_path is not None
        ):
            raise RuntimeError("Either provide completion_template or completion_func in the config.")

        # Call parent class __init__ which will call _initialize_dataset
        super().__init__(dataset_name, split, seed, **kwargs)

    def _initialize_dataset(self):
        """
        Initialize the dataset from either HuggingFace or a custom JSON file.

        This method loads the dataset, applies splitting if necessary, and prepares
        it for preprocessing with prompt/completion templates.
        """
        if self.json_file_path:
            # Load dataset from JSON file
            self.dataset = load_dataset("json", data_files=self.json_file_path, split="train")

            # Apply train/test split if needed
            if self.split in ["train", "test"]:
                self.dataset = apply_train_test_split(self.dataset, self.split_ratio, self.split, self.seed)
        else:
            # Load dataset from HuggingFace
            db = load_dataset_builder(self.dataset_name)
            available_splits = []
            if db.info.splits is not None:
                available_splits = list(db.info.splits.keys())

            if self.split not in available_splits:
                raise ValueError(f"Split {self.split} is not available for dataset {self.dataset_name}.")

            # FIXME: Add streaming support for larger datasets.
            self.dataset = load_dataset(self.dataset_name, split=self.split)

            if len(available_splits) == 1:
                self.dataset = apply_train_test_split(self.dataset, self.split_ratio, self.split, self.seed)

        self.dataset = self._setup_templates(self.dataset, self.dataset.column_names)

    def _setup_templates(self, dataset, dataset_columns):
        """
        Set up prompt/completion templates or functions and apply preprocessing.
        """
        if self.prompt_template:
            self.prompt_func = None
            # Extract variables from templates and check if they exist in dataset columns
            prompt_variables = re.findall(r"\{(.*?)\}", self.prompt_template)
            for var in prompt_variables:
                if var not in dataset_columns:
                    raise RuntimeError(
                        f"Prompt template variable '{var}' not found in dataset columns: {dataset_columns}."
                    )
        else:
            prompt_variables = dataset_columns
            self.prompt_func = self.import_func(self.prompt_func_path)

        if self.completion_template:
            self.completion_func = None
            # Extract variables from templates and check if they exist in dataset columns
            completion_variables = re.findall(r"\{(.*?)\}", self.completion_template)
            for var in completion_variables:
                if var not in dataset_columns:
                    raise RuntimeError(
                        f"Completion template variable '{var}' not found in dataset columns: {dataset_columns}."
                    )
        else:
            completion_variables = dataset_columns
            self.completion_func = self.import_func(self.completion_func_path)

        # Filter out samples with None or empty strings in relevant columns
        relevant_columns = list(set(prompt_variables + completion_variables))
        if self.remove_samples_with_empty_columns:
            dataset = dataset.filter(lambda example: self._filter_empty_or_none_samples(example, relevant_columns))
        return dataset

    def import_func(self, func_path: str) -> Callable:
        if ":" not in func_path:
            raise ValueError("func_path must be in the format 'module_file_path:function_name'.")
        module_file_path, function_name = func_path.split(":")

        try:
            module = importlib.import_module(module_file_path)
        except Exception:
            raise RuntimeError(f"Unable to import module : {module_file_path}.")
        if not hasattr(module, function_name):
            raise ValueError(f"Function {function_name} not found in module {module_file_path}.")
        return getattr(module, function_name)

    def _filter_empty_or_none_samples(self, example: Dict[str, Any], relevant_columns: list) -> bool:
        """
        Filters out samples where any of the relevant columns are None or contain only whitespace.

        Args:
            example (Dict[str, Any]): A single sample from the dataset.
            relevant_columns (list): List of column names to check for empty or None values.

        Returns:
            bool: True if the sample should be kept, False otherwise.
        """
        for column in relevant_columns:
            value = example.get(column)
            if value is None or (isinstance(value, str) and not value.strip()):
                return False
        return True

    def _preprocess_sample(self, example: Dict[str, Any]) -> Dict[str, str]:
        """
        Applies the prompt and completion templates to a single example.

        Args:
            example (Dict[str, Any]): A single sample from the dataset.

        Returns:
            Dict[str, str]: A dictionary containing the 'prompt' and 'completion' strings.
        """
        prompt_text = (
            self.prompt_func(example) if self.prompt_func is not None else self.prompt_template.format(**example)
        )
        completion_text = (
            self.completion_func(example)
            if self.completion_func is not None
            else self.completion_template.format(**example)
        )
        return {
            "prompt": prompt_text,
            "completion": completion_text,
        }

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return self.dataset.num_rows

    def __getitem__(self, idx: int) -> Dict[str, str]:
        """
        Retrieves a processed sample from the dataset at the given index.
        This method doesn't tokenize the input items, it is expected that the SFTTrainer will handle tokenization.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Dict[str, str]: A dictionary containing the processed 'prompt' and 'completion' for the sample.
        """
        # Get the raw example using .select and access the first element
        example = self.dataset.select(indices=[int(idx)])[0]

        # Apply preprocessing (templating) on the fly
        processed_example = self._preprocess_sample(example)

        return processed_example
