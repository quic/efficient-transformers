# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import os
import re
import importlib
from typing import Any, Callable, Dict

from datasets import load_dataset, load_dataset_builder

from QEfficient.finetune.experimental.core.component_registry import registry
from QEfficient.finetune.experimental.core.dataset import BaseDataset
from QEfficient.finetune.experimental.core.utils.dataset_utils import (
    validate_json_structure,
    apply_train_test_split,
)

import logging
logger = logging.getLogger(__name__)


@registry.dataset("seq_completion")
class SeqCompletionDataset(BaseDataset):
    """
    A Sequence Completion dataset class for autoregressive (next-token prediction) training.

    Unlike SFTDataset, there is NO prompt/completion split — loss is computed on ALL tokens.
    The entire text is treated as both input and label.

    Supports loading from HuggingFace datasets or local JSON files.

    Args:
        dataset_name (str): The name of the dataset to load from HuggingFace datasets.
                            Ignored if json_file_path is provided.
        split (str): The dataset split to use (e.g., "train", "validation", "test").
        split_ratio (float): Ratio for train/test split when only one split is available.
        seed (int): Random seed for reproducibility.
        json_file_path (str, optional): Path to a custom JSON file containing the dataset.
                                        If provided, this takes precedence over dataset_name.
        prompt_template (str): A string template for constructing the full input text.
                               Variables should be enclosed in curly braces, e.g., "{text}"
                               or "{question} {answer}".
        prompt_func (str, optional): Path to a custom function for constructing input text,
                                    in the format "module_path:function_name".
                                    Used if input_template is not provided.

    Raises:
        RuntimeError: If any variables specified in `input_template` are not found
                      as columns in the loaded dataset.
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
        self.input_template = kwargs.get("prompt_template", None)
        self.input_func_path = kwargs.get("prompt_func", None)
        self.remove_samples_with_empty_columns = kwargs.get("remove_samples_with_empty_columns", True)
        self.config_name = kwargs.get("config_name", None)

        # Validate json_file_path if provided
        if self.json_file_path not in (None, ""):
            if not os.path.isfile(self.json_file_path):
                raise FileNotFoundError(f"JSON file not found or invalid: '{self.json_file_path}'")

        # Warn if both template and func are provided
        if self.input_template and self.input_func_path:
            logger.warning(
                "Both input_template and input_func are provided. Using input_template for preprocessing."
            )

        # Must have at least one way to build the input text
        if self.input_template is None and self.input_func_path is None:
            raise RuntimeError("Either provide input_template or input_func in the config.")

        # Call parent __init__ which triggers _initialize_dataset()
        super().__init__(dataset_name, split, seed, **kwargs)

    # ------------------------------------------------------------------
    # Dataset Initialization
    # ------------------------------------------------------------------

    def _initialize_dataset(self):
        """
        Initialize the dataset from either HuggingFace or a custom JSON file.

        Mirrors SFTDataset._initialize_dataset() — same loading logic,
        same split handling. Difference: calls _setup_input_column()
        instead of _setup_templates(), and _add_text_field() only
        builds a single 'text' field (no prompt/completion split).
        """
        if self.json_file_path:
            # Load from local JSON file
            validate_json_structure(self.json_file_path)
            self.dataset = load_dataset("json", data_files=self.json_file_path, split="train")
            # Apply train/test split if needed
            if self.split in ["train", "test"]:
                self.dataset = apply_train_test_split(self.dataset, self.split_ratio, self.split, self.seed)
        else:
            # Load from HuggingFace hub
            load_kwargs = {}
            if self.config_name is not None:
                load_kwargs["name"] = self.config_name

            db = load_dataset_builder(self.dataset_name, **load_kwargs)
            available_splits = []
            if db.info.splits is not None:
                available_splits = list(db.info.splits.keys())

            if self.split not in available_splits and self.split == "train":
                raise ValueError(f"Split {self.split} is not available for dataset {self.dataset_name}.")

            load_split = self.split
            if self.split not in available_splits:
                load_split = "train"

            self.dataset = load_dataset(self.dataset_name, split=load_split, **load_kwargs)

            if len(available_splits) == 1:
                self.dataset = apply_train_test_split(self.dataset, self.split_ratio, self.split, self.seed)

        # Validate template variables and filter empty samples
        self.dataset = self._setup_input_column(self.dataset, self.dataset.column_names)

        # Add 'text' field — required by TRL SFTTrainer
        self.dataset = self._add_text_field(self.dataset)

    # ------------------------------------------------------------------
    # Template / Function Setup  (mirrors _setup_templates in SFTDataset)
    # ------------------------------------------------------------------

    def _setup_input_column(self, dataset, dataset_columns):
        """
        Validate input_template variables exist in dataset columns,
        set up input_func if template is not provided, and filter
        out empty/None samples.

        Mirrors SFTDataset._setup_templates() but for a single
        input column instead of prompt + completion.
        """
        if self.input_template:
            self.input_func = None
            # Extract {variable} names from the template
            input_variables = re.findall(r"\{(.*?)\}", self.input_template)
            for var in input_variables:
                if var not in dataset_columns:
                    raise RuntimeError(
                        f"Input template variable '{var}' not found in dataset columns: {dataset_columns}."
                    )
        else:
            input_variables = dataset_columns
            self.input_func = self.import_func(self.input_func_path)

        # Filter out samples with empty/None values in relevant columns
        if self.remove_samples_with_empty_columns:
            dataset = dataset.filter(
                lambda example: self._filter_empty_or_none_samples(example, input_variables)
            )
        return dataset

    def _add_text_field(self, dataset):
        """
        Add 'text' field to the dataset by applying the input template
        or input function to each sample.

        Mirrors SFTDataset._add_text_field() — but only builds ONE
        field ('text') instead of three ('text', 'prompt', 'completion').
        """

        def add_text(example):
            processed = self._preprocess_sample(example)
            example["text"] = processed["text"]
            return example

        dataset = dataset.map(add_text, desc="Adding text field")
        return dataset

    # ------------------------------------------------------------------
    # Per-Sample Preprocessing  (mirrors _preprocess_sample in SFTDataset)
    # ------------------------------------------------------------------

    def _preprocess_sample(self, example: Dict[str, Any]) -> Dict[str, str]:
        """
        Applies the input template or input function to a single example
        to produce the full text string.

        Mirrors SFTDataset._preprocess_sample() — but returns only
        {'text'} instead of {'prompt', 'completion'}.

        Args:
            example (Dict[str, Any]): A single sample from the dataset.

        Returns:
            Dict[str, str]: A dictionary containing the 'text' string.
        """
        input_text = (
            self.input_func(example)
            if self.input_func is not None
            else self.input_template.format(**example)
        )
        return {"text": input_text}

    # ------------------------------------------------------------------
    # Helpers  (identical to SFTDataset)
    # ------------------------------------------------------------------

    def import_func(self, func_path: str) -> Callable:
        """
        Dynamically import a function from a module path string.
        Format: "module_path:function_name"
        Identical to SFTDataset.import_func().
        """
        if ":" not in func_path:
            raise ValueError("func_path must be in the format 'module_file_path:function_name'.")
        module_file_path, function_name = func_path.split(":")

        try:
            module = importlib.import_module(module_file_path)
        except Exception:
            raise RuntimeError(f"Unable to import module: {module_file_path}.")

        if not hasattr(module, function_name):
            raise ValueError(f"Function {function_name} not found in module {module_file_path}.")
        return getattr(module, function_name)

    def _filter_empty_or_none_samples(self, example: Dict[str, Any], relevant_columns: list) -> bool:
        """
        Filter out samples where any relevant column is None or whitespace-only.
        Identical to SFTDataset._filter_empty_or_none_samples().
        """
        for column in relevant_columns:
            value = example.get(column)
            if value is None or (isinstance(value, str) and not value.strip()):
                return False
        return True

    # ------------------------------------------------------------------
    # Dataset Protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.dataset.num_rows

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a processed sample at the given index.

        Mirrors SFTDataset.__getitem__() — but returns only {'text'}
        in the raw format (no prompt/completion split).

        For seq_completion, labels = input_ids (set by the trainer/collator).
        """
        if hasattr(self.dataset, "__getitem__"):
            example = self.dataset[int(idx)]
        else:
            example = self.dataset.select(indices=[int(idx)])[0]

        if not isinstance(example, dict):
            example = dict(example)

        if "input_ids" in example:
            # TRL has already tokenized — return as-is
            return example

        # Return raw text format
        return {
            "text": example.get("text", ""),
        }