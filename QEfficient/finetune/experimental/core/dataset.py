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
import re
from typing import Any, Callable, Dict

from datasets import load_dataset, load_dataset_builder
from torch.utils.data import Dataset

from QEfficient.finetune.experimental.core.component_registry import registry


class BaseDataset(Dataset):
    """Base class for all datasets to ensure consistent interface."""

    def __init__(self, dataset_name: str, split: str, seed: int = 42, **kwargs):
        self.dataset_name = dataset_name
        self.split = split
        self.seed = seed
        self.kwargs = kwargs
        self._initialize_dataset()

    def _initialize_dataset(self):
        """Subclasses should implement this to load and prepare the dataset."""
        raise NotImplementedError

    @property
    def hf_dataset(self):
        """Return the underlying Hugging Face dataset object."""
        return self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Should return a dictionary with 'input_ids', 'attention_mask', and 'labels'."""
        raise NotImplementedError


# Used for pretraining
@registry.dataset("seq_completion")
class SentenceCompletionDataset(BaseDataset):
    """
    A dataset class for autoregressive sequence completion tasks (e.g., pretraining).

    This class handles loading data from Hugging Face datasets, filtering out invalid samples,
    and preparing text for autoregressive training where the input serves as both prompt and completion.

    Args:
        dataset_name (str): The name of the dataset to load from Hugging Face datasets.
        split (str): The dataset split to use (e.g., "train", "validation", "test").
        split_ratio (float): Ratio for train/test split if only one split is available. Default: 0.8.
        seed (int): Random seed for reproducibility. Default: 42.
        **kwargs: Additional arguments including:
            - input_column (str): Column name containing the text data. Default: "text".
            - text_func (str): Optional custom function path for text extraction (format: "module:function").
            - remove_samples_with_empty_columns (bool): Whether to filter empty samples. Default: True.

    Raises:
        ValueError: If the specified split is not available for the dataset.
        RuntimeError: If input_column is not found in dataset columns.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        split_ratio: float = 0.8,
        seed: int = 42,
        **kwargs,
    ):
        input_column = kwargs.get("input_column", "text")
        text_func = kwargs.get("text_func", None)
        remove_samples_with_empty_columns = kwargs.get("remove_samples_with_empty_columns", True)

        db = load_dataset_builder(dataset_name)
        available_splits = []
        if db.info.splits is not None:
            available_splits = list(db.info.splits.keys())

        if split not in available_splits and split == "train":
            raise ValueError(f"Split {split} is not available for dataset {dataset_name}.")

        load_split = split
        if split not in available_splits:
            load_split = "train"

        # Load the dataset
        self.dataset = load_dataset(dataset_name, split=load_split)

        # Handle single split datasets
        if len(available_splits) == 1:
            splitted_dataset = self.dataset.train_test_split(test_size=(1 - split_ratio), seed=seed)
            if split == "test":
                self.dataset = splitted_dataset["test"]
            else:
                self.dataset = splitted_dataset["train"]

        self.dataset_columns = self.dataset.column_names

        # Setup text extraction
        if text_func:
            self.text_func = self.import_func(text_func)
            self.input_column = None
            self.relevant_columns = self.dataset_columns
        else:
            self.text_func = None
            self.input_column = input_column
            if self.input_column not in self.dataset_columns:
                raise RuntimeError(
                    f"Input column '{self.input_column}' not found in dataset columns: {self.dataset_columns}."
                )
            self.relevant_columns = [self.input_column]

        # Filter out samples with None or empty strings
        if remove_samples_with_empty_columns:
            self.dataset = self.dataset.filter(self._filter_empty_or_none_samples)

        # Preprocess samples to create prompt/completion format
        self.dataset = self.dataset.map(self._preprocess_sample)

    def import_func(self, func_path: str) -> Callable:
        """
        Dynamically imports a function from a module path.

        Args:
            func_path (str): Path in format 'module_file_path:function_name'.

        Returns:
            Callable: The imported function.

        Raises:
            ValueError: If func_path format is invalid or function not found.
            RuntimeError: If module cannot be imported.
        """
        if ":" not in func_path:
            raise ValueError("func_path must be in the format 'module_file_path:function_name'.")
        module_file_path, function_name = func_path.split(":")

        try:
            module = importlib.import_module(module_file_path)
        except Exception as e:
            raise RuntimeError(f"Unable to import module: {module_file_path}. Error: {e}")

        if not hasattr(module, function_name):
            raise ValueError(f"Function {function_name} not found in module {module_file_path}.")

        return getattr(module, function_name)

    def _filter_empty_or_none_samples(self, example: Dict[str, Any]) -> bool:
        """
        Filters out samples where any of the relevant columns are None or contain only whitespace.

        Args:
            example (Dict[str, Any]): A single sample from the dataset.

        Returns:
            bool: True if the sample should be kept, False otherwise.
        """
        for column in self.relevant_columns:
            value = example.get(column)
            if value is None or (isinstance(value, str) and not value.strip()):
                return False
        return True

    def _preprocess_sample(self, example: Dict[str, Any]) -> Dict[str, str]:
        """
        Extracts text from the example and creates prompt/completion format.
        For autoregressive training, the text serves as both prompt and completion.

        Args:
            example (Dict[str, Any]): A single sample from the dataset.

        Returns:
            Dict[str, str]: A dictionary containing 'prompt' and 'completion' strings.
        """
        if self.text_func is not None:
            text = self.text_func(example)
        else:
            text = example[self.input_column]

        # For autoregressive training, the text is both prompt and completion
        return {
            "prompt": text,
            "completion": text,
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

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Dict[str, str]: A dictionary containing the processed 'prompt' and 'completion' for the sample.
        """
        example = self.dataset.select(indices=[int(idx)])[0]
        processed_example = self._preprocess_sample(example)
        return processed_example


# Used for SFT
@registry.dataset("chatml_instruction_following")
class ChatMLInstructionFollowingDataset(BaseDataset):
    """
    A dataset class for instruction-following tasks using ChatML format.

    This class handles loading data from Hugging Face datasets, filtering out invalid samples,
    and preparing instruction-response pairs for supervised fine-tuning. It supports flexible
    prompt construction from multiple input columns and separate completion columns.

    Args:
        dataset_name (str): The name of the dataset to load from Hugging Face datasets.
        split (str): The dataset split to use (e.g., "train", "validation", "test").
        split_ratio (float): Ratio for train/test split if only one split is available. Default: 0.8.
        seed (int): Random seed for reproducibility. Default: 42.
        **kwargs: Additional arguments including:
            - prompt_template (str): Template string for constructing prompts from input columns.
            - completion_column (str): Column name containing the completion/response text.
            - prompt_func (str): Optional custom function path for prompt construction (format: "module:function").
            - completion_func (str): Optional custom function path for completion extraction (format: "module:function").
            - remove_samples_with_empty_columns (bool): Whether to filter empty samples. Default: True.

    Raises:
        ValueError: If the specified split is not available for the dataset.
        RuntimeError: If required template variables or columns are not found in dataset.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        split_ratio: float = 0.8,
        seed: int = 42,
        **kwargs,
    ):
        prompt_template = kwargs.get("prompt_template", None)
        completion_column = kwargs.get("completion_column", None)
        prompt_func = kwargs.get("prompt_func", None)
        completion_func = kwargs.get("completion_func", None)
        remove_samples_with_empty_columns = kwargs.get("remove_samples_with_empty_columns", True)

        # Validate that either template or function is provided (but not both)
        if (prompt_template is None and prompt_func is None) or (
            prompt_template is not None and prompt_func is not None
        ):
            raise RuntimeError("Either provide prompt_template or prompt_func in the config.")
        if (completion_column is None and completion_func is None) or (
            completion_column is not None and completion_func is not None
        ):
            raise RuntimeError("Either provide completion_column or completion_func in the config.")

        db = load_dataset_builder(dataset_name)
        available_splits = []
        if db.info.splits is not None:
            available_splits = list(db.info.splits.keys())

        if split not in available_splits and split == "train":
            raise ValueError(f"Split {split} is not available for dataset {dataset_name}.")

        load_split = split
        if split not in available_splits:
            load_split = "train"

        # Load the dataset
        self.dataset = load_dataset(dataset_name, split=load_split)

        # Handle single split datasets
        if len(available_splits) == 1:
            splitted_dataset = self.dataset.train_test_split(test_size=(1 - split_ratio), seed=seed)
            if split == "test":
                self.dataset = splitted_dataset["test"]
            else:
                self.dataset = splitted_dataset["train"]

        self.dataset_columns = self.dataset.column_names

        # Setup prompt construction
        if prompt_template:
            self.prompt_template = prompt_template
            self.prompt_func = None
            # Extract variables from template and validate they exist in dataset
            prompt_variables = re.findall(r"\{(.*?)\}", self.prompt_template)
            for var in prompt_variables:
                if var not in self.dataset_columns:
                    raise RuntimeError(
                        f"Prompt template variable '{var}' not found in dataset columns: {self.dataset_columns}."
                    )
        else:
            prompt_variables = self.dataset_columns
            self.prompt_func = self.import_func(prompt_func)
            self.prompt_template = None

        # Setup completion extraction
        if completion_column:
            self.completion_column = completion_column
            self.completion_func = None
            if self.completion_column not in self.dataset_columns:
                raise RuntimeError(
                    f"Completion column '{self.completion_column}' not found in dataset columns: {self.dataset_columns}."
                )
            completion_variables = [self.completion_column]
        else:
            completion_variables = self.dataset_columns
            self.completion_func = self.import_func(completion_func)
            self.completion_column = None

        # Filter out samples with None or empty strings in relevant columns
        self.relevant_columns = list(set(prompt_variables + completion_variables))
        if remove_samples_with_empty_columns:
            self.dataset = self.dataset.filter(self._filter_empty_or_none_samples)

        # Preprocess samples to create prompt/completion format
        self.dataset = self.dataset.map(self._preprocess_sample)

    def import_func(self, func_path: str) -> Callable:
        """
        Dynamically imports a function from a module path.

        Args:
            func_path (str): Path in format 'module_file_path:function_name'.

        Returns:
            Callable: The imported function.

        Raises:
            ValueError: If func_path format is invalid or function not found.
            RuntimeError: If module cannot be imported.
        """
        if ":" not in func_path:
            raise ValueError("func_path must be in the format 'module_file_path:function_name'.")
        module_file_path, function_name = func_path.split(":")

        try:
            module = importlib.import_module(module_file_path)
        except Exception as e:
            raise RuntimeError(f"Unable to import module: {module_file_path}. Error: {e}")

        if not hasattr(module, function_name):
            raise ValueError(f"Function {function_name} not found in module {module_file_path}.")

        return getattr(module, function_name)

    def _filter_empty_or_none_samples(self, example: Dict[str, Any]) -> bool:
        """
        Filters out samples where any of the relevant columns are None or contain only whitespace.

        Args:
            example (Dict[str, Any]): A single sample from the dataset.

        Returns:
            bool: True if the sample should be kept, False otherwise.
        """
        for column in self.relevant_columns:
            value = example.get(column)
            if value is None or (isinstance(value, str) and not value.strip()):
                return False
        return True

    def _preprocess_sample(self, example: Dict[str, Any]) -> Dict[str, str]:
        """
        Constructs the prompt and extracts the completion from the example.

        Args:
            example (Dict[str, Any]): A single sample from the dataset.

        Returns:
            Dict[str, str]: A dictionary containing 'prompt' and 'completion' strings.
        """
        # Construct prompt
        if self.prompt_func is not None:
            prompt_text = self.prompt_func(example)
        else:
            prompt_text = self.prompt_template.format(**example)
            # Cleanup whitespace if needed
            prompt_text = prompt_text.replace("\n ", " ")

        # Extract completion
        if self.completion_func is not None:
            completion_text = self.completion_func(example)
        else:
            completion_text = example[self.completion_column]

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

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Dict[str, str]: A dictionary containing the processed 'prompt' and 'completion' for the sample.
        """
        example = self.dataset.select(indices=[int(idx)])[0]
        processed_example = self._preprocess_sample(example)
        return processed_example


@registry.dataset("sft_dataset")
class SFTDataset(BaseDataset):
    """
    A Supervised Fine-Tuning (SFT) dataset class for text data.

    This class handles loading data from Hugging Face datasets, filtering out invalid samples,
    and applying a prompt/completion templating for SFT tasks.

    Args:
        dataset_name (str): The name of the dataset to load from Hugging Face datasets.
        split (str): The dataset split to use (e.g., "train", "validation", "test").
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
        prompt_template = kwargs.get("prompt_template", None)
        completion_template = kwargs.get("completion_template", None)
        prompt_func = kwargs.get("prompt_func", None)
        completion_func = kwargs.get("completion_func", None)
        remove_samples_with_empty_columns = kwargs.get("remove_samples_with_empty_columns", True)

        if (prompt_template is None and prompt_func is None) and (
            prompt_template is not None and prompt_func is not None
        ):
            raise RuntimeError("Either provide prompt_template or prompt_func in the config.")
        if (completion_template is None and completion_func is None) and (
            completion_template is not None and completion_func is not None
        ):
            raise RuntimeError("Either provide completion_template or completion_func in the config.")

        db = load_dataset_builder(dataset_name)
        available_splits = []
        if db.info.splits is not None:
            available_splits = list(db.info.splits.keys())

        if split not in available_splits and split == "train":
            raise ValueError(f"Split {split} is not available for dataset {dataset_name}.")

        load_split = split
        if split not in available_splits:
            load_split = "train"

        # FIXME: Add streaming support for larger datasets.
        self.dataset = load_dataset(dataset_name, split=load_split)
        if len(available_splits) == 1:
            split_ratio = split_ratio
            splitted_dataset = self.dataset.train_test_split(test_size=(1 - split_ratio), seed=seed)
            if split == "test":
                self.dataset = splitted_dataset["test"]
            else:
                self.dataset = splitted_dataset["train"]

        self.dataset_columns = self.dataset.column_names
        if prompt_template:
            self.prompt_template = prompt_template
            self.prompt_func = None
            # Extract variables from templates and check if they exist in dataset columns
            prompt_variables = re.findall(r"\{(.*?)\}", self.prompt_template)
            for var in prompt_variables:
                if var not in self.dataset_columns:
                    raise RuntimeError(
                        f"Prompt template variable '{var}' not found in dataset columns: {self.dataset_columns}."
                    )
        else:
            prompt_variables = self.dataset_columns
            self.prompt_func = self.import_func(prompt_func)

        if completion_template:
            self.completion_template = completion_template
            self.completion_func = None
            # Extract variables from templates and check if they exist in dataset columns
            completion_variables = re.findall(r"\{(.*?)\}", self.completion_template)
            for var in completion_variables:
                if var not in self.dataset_columns:
                    raise RuntimeError(
                        f"Completion template variable '{var}' not found in dataset columns: {self.dataset_columns}."
                    )
        else:
            completion_variables = self.dataset_columns
            self.completion_func = self.import_func(completion_func)

        # Filter out samples with None or empty strings in relevant columns
        # Only filter columns that are actually used in the templates
        self.relevant_columns = list(set(prompt_variables + completion_variables))
        if remove_samples_with_empty_columns:
            self.dataset = self.dataset.filter(self._filter_empty_or_none_samples)
        self.dataset = self.dataset.map(self._preprocess_sample)

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

    def _filter_empty_or_none_samples(self, example: Dict[str, Any]) -> bool:
        """
        Filters out samples where any of the relevant columns are None or contain only whitespace.

        Args:
            example (Dict[str, Any]): A single sample from the dataset.

        Returns:
            bool: True if the sample should be kept, False otherwise.
        """
        for column in self.relevant_columns:
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
