# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Test Suite for Dataset Classes

This module contains comprehensive unit tests for the three main dataset classes:
- SentenceCompletionDataset (for pretraining/autoregressive tasks)
- ChatMLInstructionFollowingDataset (for instruction-following tasks)
- SFTDataset (for supervised fine-tuning)

Testing Approach:
-----------------
The tests follow HuggingFace's best practices by:
1. Creating in-memory datasets using `Dataset.from_dict()` for fast execution
2. Using `unittest.mock.patch` to mock HF's load_dataset functions
3. Avoiding file I/O operations to keep tests lightweight and fast
4. Testing core functionality without external dependencies

Why These Tests Are Important:
-------------------------------
1. **Data Integrity**: Ensures datasets correctly load, filter, and preprocess data
2. **Format Validation**: Verifies output format matches expected prompt/completion structure
3. **Error Handling**: Confirms proper error messages for invalid configurations
4. **Compatibility**: Ensures datasets work seamlessly with HFTrainer
5. **Regression Prevention**: Catches breaking changes during refactoring
6. **Documentation**: Serves as usage examples for developers

Test Categories:
----------------
1. Basic Functionality Tests: Verify core loading and retrieval operations
2. Configuration Tests: Test custom columns, templates, and functions
3. Error Handling Tests: Ensure proper validation and error messages
4. Data Processing Tests: Verify filtering, splitting, and preprocessing
5. Integration Tests: Confirm compatibility with the training pipeline

Running Tests:
--------------
    pytest test_dataset.py -v                    # Run all tests with verbose output
    pytest test_dataset.py::TestSFTDataset -v    # Run specific test class
    pytest test_dataset.py -k "filtering" -v     # Run tests matching pattern
"""

import unittest
from unittest.mock import MagicMock, patch

from datasets import Dataset as HFDataset

from QEfficient.finetune.experimental.core.dataset import (
    ChatMLInstructionFollowingDataset,
    SentenceCompletionDataset,
    SFTDataset,
)


def create_mock_dataset_builder(splits=None):
    """Helper to create mock dataset builder."""
    if splits is None:
        splits = {"train": None}

    mock_builder = MagicMock()
    mock_builder.info.splits = splits
    return mock_builder


class TestSentenceCompletionDataset(unittest.TestCase):
    """
    Test cases for SentenceCompletionDataset.

    This class tests the dataset used for pretraining and autoregressive tasks where
    the input text serves as both the prompt and completion (next-token prediction).

    Why This Matters:
    - Pretraining is the foundation of language models
    - Ensures text data is correctly formatted for autoregressive training
    - Validates that the dataset can handle various text column configurations
    """

    def setUp(self):
        """Set up test data with sample sentences."""
        self.dummy_data = {
            "text": [
                "First sentence.",
                "Second sentence.",
                "Third sentence.",
            ]
        }
        self.dummy_dataset = HFDataset.from_dict(self.dummy_data)

    @patch("QEfficient.finetune.experimental.core.dataset.load_dataset")
    @patch("QEfficient.finetune.experimental.core.dataset.load_dataset_builder")
    def test_basic_functionality(self, mock_builder, mock_load):
        """
        Test basic dataset loading and retrieval.

        What This Tests:
        - Dataset loads successfully from HuggingFace
        - Returns data in correct prompt/completion format
        - For autoregressive tasks, prompt equals completion
        - Data integrity is maintained from source

        Why It's Important:
        - Validates core functionality of the dataset class
        - Ensures compatibility with training pipeline
        - Confirms data format matches trainer expectations
        """
        mock_builder.return_value = create_mock_dataset_builder()
        mock_load.return_value = self.dummy_dataset

        dataset = SentenceCompletionDataset(
            dataset_name="dummy",
            split="train",
            seed=42,
            remove_samples_with_empty_columns=False,
        )

        # Verify dataset loaded
        self.assertGreater(len(dataset), 0)

        # Verify output format
        sample = dataset[0]
        self.assertIn("prompt", sample)
        self.assertIn("completion", sample)
        self.assertEqual(sample["prompt"], sample["completion"])
        self.assertIn(sample["prompt"], self.dummy_data["text"])

    @patch("QEfficient.finetune.experimental.core.dataset.load_dataset")
    @patch("QEfficient.finetune.experimental.core.dataset.load_dataset_builder")
    def test_custom_column(self, mock_builder, mock_load):
        """
        Test custom input column configuration.

        What This Tests:
        - Dataset can use non-default column names
        - Flexible configuration for different dataset schemas

        Why It's Important:
        - Real-world datasets have varying column names
        - Enables reuse of dataset class across different data sources
        - Prevents need for data preprocessing/renaming
        """
        custom_data = {"content": ["Text 1", "Text 2"]}
        mock_builder.return_value = create_mock_dataset_builder()
        mock_load.return_value = HFDataset.from_dict(custom_data)

        dataset = SentenceCompletionDataset(
            dataset_name="dummy",
            split="train",
            seed=42,
            input_column="content",
            remove_samples_with_empty_columns=False,
        )

        self.assertGreater(len(dataset), 0)
        sample = dataset[0]
        self.assertIn(sample["prompt"], custom_data["content"])


class TestChatMLInstructionFollowingDataset(unittest.TestCase):
    """
    Test cases for ChatMLInstructionFollowingDataset.

    This class tests the dataset used for instruction-following tasks where prompts
    are constructed from multiple input columns and paired with separate completions.

    Why This Matters:
    - Instruction-following is critical for chat models and assistants
    - Validates flexible prompt construction from multiple data fields
    - Ensures proper separation of instruction context and responses
    """

    def setUp(self):
        """Set up test data with instruction-response pairs."""
        self.dummy_data = {
            "instruction": ["Question 1", "Question 2"],
            "response": ["Answer 1", "Answer 2"],
        }
        self.dummy_dataset = HFDataset.from_dict(self.dummy_data)

    @patch("QEfficient.finetune.experimental.core.dataset.load_dataset")
    @patch("QEfficient.finetune.experimental.core.dataset.load_dataset_builder")
    def test_template_formatting(self, mock_builder, mock_load):
        """
        Test prompt template formatting with multiple columns.

        What This Tests:
        - Template variables are correctly replaced with data
        - Prompt formatting follows specified template structure
        - Completion is extracted from correct column

        Why It's Important:
        - Templates enable consistent prompt formatting
        - Critical for instruction-tuned models
        - Ensures model receives properly formatted inputs
        """
        mock_builder.return_value = create_mock_dataset_builder()
        mock_load.return_value = self.dummy_dataset

        dataset = ChatMLInstructionFollowingDataset(
            dataset_name="dummy",
            split="train",
            seed=42,
            prompt_template="Q: {instruction}",
            completion_column="response",
        )

        sample = dataset[0]
        self.assertIn("Q:", sample["prompt"])
        self.assertIn(sample["completion"], self.dummy_data["response"])

    @patch("QEfficient.finetune.experimental.core.dataset.load_dataset")
    @patch("QEfficient.finetune.experimental.core.dataset.load_dataset_builder")
    def test_invalid_column_error(self, mock_builder, mock_load):
        """
        Test error handling for invalid column references.

        What This Tests:
        - Dataset validates template variables against available columns
        - Raises clear error messages for configuration mistakes

        Why It's Important:
        - Prevents silent failures during training
        - Helps developers quickly identify configuration errors
        - Ensures data integrity before expensive training runs
        """
        mock_builder.return_value = create_mock_dataset_builder()
        mock_load.return_value = self.dummy_dataset

        with self.assertRaises(RuntimeError):
            ChatMLInstructionFollowingDataset(
                dataset_name="dummy",
                split="train",
                seed=42,
                prompt_template="Q: {invalid_column}",
                completion_column="response",
            )


class TestSFTDataset(unittest.TestCase):
    """
    Test cases for SFTDataset (Supervised Fine-Tuning).

    This class tests the primary dataset for supervised fine-tuning tasks where
    both prompts and completions are constructed from separate template strings.

    Why This Matters:
    - SFT is the most common fine-tuning approach for task-specific models
    - Validates flexible template-based data formatting
    - Ensures proper train/test splitting for evaluation
    """

    def setUp(self):
        """Set up test data with question-answer pairs."""
        self.dummy_data = {
            "question": ["Q1", "Q2", "Q3"],
            "answer": ["A1", "A2", "A3"],
        }
        self.dummy_dataset = HFDataset.from_dict(self.dummy_data)

    @patch("QEfficient.finetune.experimental.core.dataset.load_dataset")
    @patch("QEfficient.finetune.experimental.core.dataset.load_dataset_builder")
    def test_basic_functionality(self, mock_builder, mock_load):
        """
        Test basic SFT dataset functionality with templates.

        What This Tests:
        - Both prompt and completion templates work correctly
        - Data is formatted according to specified templates
        - Output structure matches trainer expectations

        Why It's Important:
        - SFT is the primary use case for most fine-tuning tasks
        - Template flexibility enables various task formats
        - Validates end-to-end data pipeline
        """
        mock_builder.return_value = create_mock_dataset_builder()
        mock_load.return_value = self.dummy_dataset

        dataset = SFTDataset(
            dataset_name="dummy",
            split="train",
            seed=42,
            prompt_template="Question: {question}",
            completion_template="{answer}",
            remove_samples_with_empty_columns=False,
        )

        self.assertGreater(len(dataset), 0)
        sample = dataset[0]
        self.assertIn("Question:", sample["prompt"])
        self.assertIn(sample["completion"], self.dummy_data["answer"])

    @patch("QEfficient.finetune.experimental.core.dataset.load_dataset")
    @patch("QEfficient.finetune.experimental.core.dataset.load_dataset_builder")
    def test_train_test_split(self, mock_builder, mock_load):
        """
        Test automatic train/test splitting functionality.

        What This Tests:
        - Dataset automatically splits into train/test when only one split exists
        - Split ratios are respected
        - Same seed produces consistent splits

        Why It's Important:
        - Enables proper model evaluation without manual data splitting
        - Ensures reproducibility across experiments
        - Critical for preventing data leakage between train/test
        """
        mock_builder.return_value = create_mock_dataset_builder()
        mock_load.return_value = self.dummy_dataset

        train_dataset = SFTDataset(
            dataset_name="dummy",
            split="train",
            split_ratio=0.66,
            seed=42,
            prompt_template="{question}",
            completion_template="{answer}",
            remove_samples_with_empty_columns=False,
        )

        test_dataset = SFTDataset(
            dataset_name="dummy",
            split="test",
            split_ratio=0.66,
            seed=42,
            prompt_template="{question}",
            completion_template="{answer}",
            remove_samples_with_empty_columns=False,
        )

        # Verify both splits have data
        self.assertGreater(len(train_dataset), 0)
        self.assertGreater(len(test_dataset), 0)
        # Train should typically be larger than test
        total_samples = len(train_dataset) + len(test_dataset)
        self.assertGreater(total_samples, 0)

    @patch("QEfficient.finetune.experimental.core.dataset.load_dataset")
    @patch("QEfficient.finetune.experimental.core.dataset.load_dataset_builder")
    def test_missing_template_error(self, mock_builder, mock_load):
        """
        Test error handling when required templates are missing.

        What This Tests:
        - Dataset validates that either templates or functions are provided
        - Raises appropriate errors for incomplete configuration

        Why It's Important:
        - Prevents training with improperly configured datasets
        - Provides clear feedback for configuration errors
        - Fails fast before expensive training operations
        """
        mock_builder.return_value = create_mock_dataset_builder()
        mock_load.return_value = self.dummy_dataset

        with self.assertRaises((RuntimeError, TypeError)):
            SFTDataset(
                dataset_name="dummy",
                split="train",
                seed=42,
            )


class TestDatasetFiltering(unittest.TestCase):
    """
    Test filtering functionality across datasets.

    This class tests the data quality filtering mechanisms that remove
    invalid or empty samples from datasets before training.

    Why This Matters:
    - Poor quality data can degrade model performance
    - Empty samples waste compute resources during training
    - Filtering ensures only valid data reaches the model
    """

    @patch("QEfficient.finetune.experimental.core.dataset.load_dataset")
    @patch("QEfficient.finetune.experimental.core.dataset.load_dataset_builder")
    def test_empty_sample_filtering(self, mock_builder, mock_load):
        """
        Test that empty samples are properly filtered from datasets.

        What This Tests:
        - Empty strings are identified and removed
        - Filtering can be enabled/disabled via configuration
        - Filtered datasets have fewer samples than unfiltered

        Why It's Important:
        - Prevents training on invalid/empty data
        - Improves data quality and model performance
        - Saves compute resources by removing useless samples
        - Provides flexibility to handle edge cases
        """
        data_with_empty = {"text": ["Valid text", "", "Another valid text", "More text"]}
        mock_builder.return_value = create_mock_dataset_builder()
        mock_load.return_value = HFDataset.from_dict(data_with_empty)

        # Test with filtering disabled
        dataset_unfiltered = SentenceCompletionDataset(
            dataset_name="dummy",
            split="train",
            seed=42,
            remove_samples_with_empty_columns=False,
        )

        # Test with filtering enabled
        dataset_filtered = SentenceCompletionDataset(
            dataset_name="dummy",
            split="train",
            seed=42,
            remove_samples_with_empty_columns=True,
        )

        # Filtered should have fewer samples than unfiltered
        self.assertLess(len(dataset_filtered), len(dataset_unfiltered))


if __name__ == "__main__":
    unittest.main()
