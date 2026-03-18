# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Tests for dataset components.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from QEfficient.finetune.experimental.core.dataset import BaseDataset, SFTDataset

SEED = 42
SPLIT_RATIO = 0.8


class TestBaseDataset(unittest.TestCase):
    """Tests for BaseDataset abstract class."""

    def test_base_dataset_cannot_be_instantiated(self):
        """Test that BaseDataset cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseDataset(dataset_name="test", split="train")


class TestSFTDataset(unittest.TestCase):
    """Tests for SFTDataset class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.json_file_path = os.path.join(self.test_dir, "test_dataset.json")

        # Create a dummy JSON dataset
        self.dummy_data = [
            {"question": "What is AI?", "answer": "Artificial Intelligence"},
            {"question": "What is ML?", "answer": "Machine Learning"},
            {"question": "What is DL?", "answer": "Deep Learning"},
            {"question": "What is NLP?", "answer": "Natural Language Processing"},
            {"question": "", "answer": "Empty question"},  # Empty question
            {"question": "Valid question", "answer": ""},  # Empty answer
            {"question": None, "answer": "None question"},  # None question
            {"question": "Valid question 2", "answer": None},  # None answer
        ]

        with open(self.json_file_path, "w") as f:
            json.dump(self.dummy_data, f)

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary files and directories
        import shutil

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch("QEfficient.finetune.experimental.core.dataset.load_dataset")
    @patch("QEfficient.finetune.experimental.core.dataset.load_dataset_builder")
    def test_sft_dataset_with_huggingface_dataset_and_templates(self, mock_builder, mock_load):
        """Test loading from HuggingFace dataset with templates using mocked data."""
        # Create mock dataset with dummy data
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text", "label"]
        mock_dataset.num_rows = 3

        # Mock the select method to return individual samples
        def mock_select(indices):
            sample_data = [
                {"text": "Sample text 1", "label": "Label 1"},
                {"text": "Sample text 2", "label": "Label 2"},
                {"text": "Sample text 3", "label": "Label 3"},
            ]
            return [sample_data[indices[0]]]

        mock_dataset.select = mock_select
        mock_dataset.filter = lambda func: mock_dataset  # Return self for filtering

        # Mock train_test_split to return a dict with train/test splits
        mock_split_result = {"train": mock_dataset, "test": mock_dataset}
        mock_dataset.train_test_split = lambda test_size, seed: mock_split_result

        # Mock the dataset builder to indicate multiple splits are available
        mock_info = MagicMock()
        mock_info.splits = {"train": MagicMock(), "test": MagicMock()}
        mock_builder.return_value.info = mock_info

        # Mock load_dataset to return our mock dataset
        mock_load.return_value = mock_dataset

        # Create the dataset
        dataset = SFTDataset(
            dataset_name="dummy_hf_dataset",
            split="train",
            prompt_template="Text: {text}",
            completion_template="Label: {label}",
        )

        self.assertIsNotNone(dataset)
        self.assertEqual(len(dataset), 3)

        # Test __getitem__
        sample = dataset[0]
        self.assertIn("prompt", sample)
        self.assertIn("completion", sample)
        self.assertTrue(sample["prompt"].startswith("Text:"))
        self.assertTrue(sample["completion"].startswith("Label:"))

    def test_sft_dataset_with_json_file_and_templates(self):
        """Test loading from JSON file with templates."""
        dataset = SFTDataset(
            dataset_name="dummy",  # Ignored when json_file_path is provided
            split="train",
            json_file_path=self.json_file_path,
            prompt_template="Q: {question}",
            completion_template="A: {answer}",
        )

        self.assertIsNotNone(dataset)
        # After filtering empty/None values and applying train split (default 0.8)
        # we get a subset of the 4 valid samples
        self.assertGreater(len(dataset), 0)
        self.assertLessEqual(len(dataset), 4)

        # Test __getitem__
        sample = dataset[0]
        self.assertIn("prompt", sample)
        self.assertIn("completion", sample)
        self.assertTrue(sample["prompt"].startswith("Q:"))
        self.assertTrue(sample["completion"].startswith("A:"))

    def test_sft_dataset_json_file_without_filtering(self):
        """Test loading from JSON file without filtering empty samples."""
        dataset = SFTDataset(
            dataset_name="dummy",
            split="train",
            json_file_path=self.json_file_path,
            prompt_template="Q: {question}",
            completion_template="A: {answer}",
            remove_samples_with_empty_columns=False,
        )

        # When filtering is disabled and split="train" is used, it still applies train/test split
        # So we get ~80% of 8 samples = ~6 samples
        self.assertGreater(len(dataset), 0)
        self.assertLessEqual(len(dataset), 8)

    def test_sft_dataset_train_test_split_from_json(self):
        """Test train/test split when loading from JSON file."""
        train_dataset = SFTDataset(
            dataset_name="dummy",
            split="train",
            split_ratio=SPLIT_RATIO,
            json_file_path=self.json_file_path,
            prompt_template="Q: {question}",
            completion_template="A: {answer}",
            seed=SEED,
        )

        test_dataset = SFTDataset(
            dataset_name="dummy",
            split="test",
            split_ratio=SPLIT_RATIO,
            json_file_path=self.json_file_path,
            prompt_template="Q: {question}",
            completion_template="A: {answer}",
            seed=SEED,
        )

        # After filtering, we have 4 valid samples
        # With split ratio, train should have ~3 samples, test should have ~1 sample
        self.assertGreater(len(train_dataset), 0)
        self.assertGreater(len(test_dataset), 0)
        # Total should equal the filtered dataset size
        self.assertEqual(len(train_dataset) + len(test_dataset), 4)

    def test_sft_dataset_with_custom_prompt_function(self):
        """Test loading with custom prompt function."""
        # Create a temporary module file with custom functions
        func_file_path = os.path.join(self.test_dir, "custom_funcs.py")
        with open(func_file_path, "w") as f:
            f.write("""
def custom_prompt(example):
    return f"Custom prompt: {example['question']}"

def custom_completion(example):
    return f"Custom completion: {example['answer']}"
""")

        # Add the test directory to sys.path temporarily
        import sys

        sys.path.insert(0, self.test_dir)

        try:
            dataset = SFTDataset(
                dataset_name="dummy",
                split="train",
                json_file_path=self.json_file_path,
                prompt_func="custom_funcs:custom_prompt",
                completion_func="custom_funcs:custom_completion",
            )

            self.assertIsNotNone(dataset)
            self.assertGreater(len(dataset), 0)

            # Test that custom functions are applied
            sample = dataset[0]
            self.assertTrue(sample["prompt"].startswith("Custom prompt:"))
            self.assertTrue(sample["completion"].startswith("Custom completion:"))
        finally:
            # Clean up
            sys.path.remove(self.test_dir)
            if os.path.exists(func_file_path):
                os.remove(func_file_path)

    def test_sft_dataset_missing_template_variable(self):
        """Test error when template variable is not in dataset columns."""
        with self.assertRaises(RuntimeError) as context:
            SFTDataset(
                dataset_name="dummy",
                split="train",
                json_file_path=self.json_file_path,
                prompt_template="Q: {nonexistent_column}",
                completion_template="A: {answer}",
            )

        self.assertIn("not found in dataset columns", str(context.exception))

    def test_sft_dataset_missing_completion_template_variable(self):
        """Test error when completion template variable is not in dataset columns."""
        with self.assertRaises(RuntimeError) as context:
            SFTDataset(
                dataset_name="dummy",
                split="train",
                json_file_path=self.json_file_path,
                prompt_template="Q: {question}",
                completion_template="A: {nonexistent_column}",
            )

        self.assertIn("not found in dataset columns", str(context.exception))

    def test_sft_dataset_no_prompt_template_or_func(self):
        """Test error when neither prompt_template nor prompt_func is provided."""
        with self.assertRaises(RuntimeError) as context:
            SFTDataset(
                dataset_name="dummy",
                split="train",
                json_file_path=self.json_file_path,
                completion_template="A: {answer}",
            )

        self.assertIn("Either provide prompt_template or prompt_func", str(context.exception))

    def test_sft_dataset_both_prompt_template_and_func(self):
        """Test error when both prompt_template and prompt_func are provided."""
        with self.assertRaises(RuntimeError) as context:
            SFTDataset(
                dataset_name="dummy",
                split="train",
                json_file_path=self.json_file_path,
                prompt_template="Q: {question}",
                prompt_func="module:function",
                completion_template="A: {answer}",
            )

        self.assertIn("Either provide prompt_template or prompt_func", str(context.exception))

    def test_sft_dataset_no_completion_template_or_func(self):
        """Test error when neither completion_template nor completion_func is provided."""
        with self.assertRaises(RuntimeError) as context:
            SFTDataset(
                dataset_name="dummy",
                split="train",
                json_file_path=self.json_file_path,
                prompt_template="Q: {question}",
            )

        self.assertIn(
            "Either provide completion_template or completion_func",
            str(context.exception),
        )

    def test_sft_dataset_both_completion_template_and_func(self):
        """Test error when both completion_template and completion_func are provided."""
        with self.assertRaises(RuntimeError) as context:
            SFTDataset(
                dataset_name="dummy",
                split="train",
                json_file_path=self.json_file_path,
                prompt_template="Q: {question}",
                completion_template="A: {answer}",
                completion_func="module:function",
            )

        self.assertIn(
            "Either provide completion_template or completion_func",
            str(context.exception),
        )

    def test_sft_dataset_invalid_func_path_format(self):
        """Test error when func_path doesn't contain colon separator."""
        with self.assertRaises(ValueError) as context:
            SFTDataset(
                dataset_name="dummy",
                split="train",
                json_file_path=self.json_file_path,
                prompt_func="invalid_format",
                completion_template="A: {answer}",
            )

        self.assertIn("must be in the format", str(context.exception))

    def test_sft_dataset_invalid_module_import(self):
        """Test error when module cannot be imported."""
        with self.assertRaises(RuntimeError) as context:
            SFTDataset(
                dataset_name="dummy",
                split="train",
                json_file_path=self.json_file_path,
                prompt_func="nonexistent_module:function",
                completion_template="A: {answer}",
            )

        self.assertIn("Unable to import module", str(context.exception))

    def test_sft_dataset_invalid_function_name(self):
        """Test error when function doesn't exist in module."""
        # Create a temporary module file without the expected function
        func_file_path = os.path.join(self.test_dir, "test_module.py")
        with open(func_file_path, "w") as f:
            f.write("def some_other_function():\n    pass\n")

        import sys

        sys.path.insert(0, self.test_dir)

        try:
            with self.assertRaises(ValueError) as context:
                SFTDataset(
                    dataset_name="dummy",
                    split="train",
                    json_file_path=self.json_file_path,
                    prompt_func="test_module:nonexistent_function",
                    completion_template="A: {answer}",
                )

            self.assertIn("not found in module", str(context.exception))
        finally:
            sys.path.remove(self.test_dir)
            if os.path.exists(func_file_path):
                os.remove(func_file_path)

    def test_sft_dataset_filter_empty_or_none_samples(self):
        """Test filtering of samples with empty or None values."""
        dataset = SFTDataset(
            dataset_name="dummy",
            split="train",
            json_file_path=self.json_file_path,
            prompt_template="Q: {question}",
            completion_template="A: {answer}",
            remove_samples_with_empty_columns=True,
        )

        # Verify that all samples have valid (non-empty) questions and answers
        for i in range(len(dataset)):
            sample = dataset[i]
            # Extract the actual question and answer from the formatted strings
            question = sample["prompt"].replace("Q: ", "").strip()
            answer = sample["completion"].replace("A: ", "").strip()
            # Verify neither is empty
            self.assertTrue(len(question) > 0, f"Question should not be empty: {sample['prompt']}")
            self.assertTrue(len(answer) > 0, f"Answer should not be empty: {sample['completion']}")

    def test_sft_dataset_getitem_returns_correct_format(self):
        """Test that __getitem__ returns the correct format."""
        dataset = SFTDataset(
            dataset_name="dummy",
            split="train",
            json_file_path=self.json_file_path,
            prompt_template="Q: {question}",
            completion_template="A: {answer}",
        )

        sample = dataset[0]

        # Check that sample is a dictionary
        self.assertIsInstance(sample, dict)

        # Check that it has the required keys
        self.assertIn("prompt", sample)
        self.assertIn("completion", sample)

        # Check that values are strings
        self.assertIsInstance(sample["prompt"], str)
        self.assertIsInstance(sample["completion"], str)

    def test_sft_dataset_len(self):
        """Test __len__ method."""
        dataset = SFTDataset(
            dataset_name="dummy",
            split="train",
            json_file_path=self.json_file_path,
            prompt_template="Q: {question}",
            completion_template="A: {answer}",
        )

        # Check that len returns an integer
        self.assertIsInstance(len(dataset), int)

        # Check that len is positive
        self.assertGreater(len(dataset), 0)

        # Check that we can iterate through all samples
        for i in range(len(dataset)):
            sample = dataset[i]
            self.assertIsNotNone(sample)

    def test_sft_dataset_with_multiple_template_variables(self):
        """Test templates with multiple variables."""
        # Create a more complex JSON dataset
        complex_data = [
            {"context": "The sky", "question": "What color?", "answer": "Blue"},
            {"context": "Math", "question": "What is 2+2?", "answer": "4"},
        ]

        complex_json_path = os.path.join(self.test_dir, "complex_dataset.json")
        with open(complex_json_path, "w") as f:
            json.dump(complex_data, f)

        try:
            dataset = SFTDataset(
                dataset_name="dummy",
                split="train",
                json_file_path=complex_json_path,
                prompt_template="Context: {context}\nQuestion: {question}",
                completion_template="Answer: {answer}",
            )

            # With split="train", it applies train/test split, so we get ~80% of 2 samples
            self.assertGreater(len(dataset), 0)
            self.assertLessEqual(len(dataset), 2)

            sample = dataset[0]
            self.assertIn("Context:", sample["prompt"])
            self.assertIn("Question:", sample["prompt"])
            self.assertIn("Answer:", sample["completion"])
        finally:
            if os.path.exists(complex_json_path):
                os.remove(complex_json_path)

    def test_sft_dataset_seed_reproducibility(self):
        """Test that using the same seed produces the same split."""
        dataset1 = SFTDataset(
            dataset_name="dummy",
            split="train",
            split_ratio=SPLIT_RATIO,
            json_file_path=self.json_file_path,
            prompt_template="Q: {question}",
            completion_template="A: {answer}",
            seed=SEED,
        )

        dataset2 = SFTDataset(
            dataset_name="dummy",
            split="train",
            split_ratio=SPLIT_RATIO,
            json_file_path=self.json_file_path,
            prompt_template="Q: {question}",
            completion_template="A: {answer}",
            seed=SEED,
        )

        # Both datasets should have the same length
        self.assertEqual(len(dataset1), len(dataset2))

        # Both datasets should have the same samples
        for i in range(len(dataset1)):
            sample1 = dataset1[i]
            sample2 = dataset2[i]
            self.assertEqual(sample1["prompt"], sample2["prompt"])
            self.assertEqual(sample1["completion"], sample2["completion"])

    @patch("QEfficient.finetune.experimental.core.dataset.load_dataset")
    @patch("QEfficient.finetune.experimental.core.dataset.load_dataset_builder")
    def test_sft_dataset_invalid_split(self, mock_builder, mock_load):
        """Test error when requesting an invalid split."""
        # Mock the dataset builder to return specific splits
        mock_info = MagicMock()
        mock_info.splits = {"train": MagicMock(), "validation": MagicMock()}
        mock_builder.return_value.info = mock_info

        with self.assertRaises(ValueError) as context:
            SFTDataset(
                dataset_name="dummy_dataset",
                split="nonexistent_split",
                prompt_template="Q: {question}",
                completion_template="A: {answer}",
            )

        self.assertIn("not available", str(context.exception))

    def test_sft_dataset_invalid_json_path(self):
        """Test error when an invalid JSON file path is provided."""
        invalid_path = "/path/to/nonexistent/file.json"

        with self.assertRaises(FileNotFoundError) as context:
            SFTDataset(
                dataset_name="dummy",
                split="train",
                json_file_path=invalid_path,
                prompt_template="Q: {question}",
                completion_template="A: {answer}",
            )

        self.assertIn("JSON file not found or invalid", str(context.exception))
        self.assertIn(invalid_path, str(context.exception))


if __name__ == "__main__":
    unittest.main()
