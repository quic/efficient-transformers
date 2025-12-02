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
from unittest.mock import MagicMock, Mock, patch

import pytest
from datasets import Dataset as HFDataset

from QEfficient.finetune.experimental.core.dataset import BaseDataset, SFTDataset


class TestBaseDataset:
    """Test suite for BaseDataset class."""

    def test_base_dataset_initialization(self):
        """Test that BaseDataset initializes with correct attributes."""
        with pytest.raises(NotImplementedError):
            dataset = BaseDataset(
                dataset_name="test_dataset",
                split="train",
                seed=42,
                extra_param="value"
            )

    def test_base_dataset_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        
        class TestDataset(BaseDataset):
            def _initialize_dataset(self):
                self.dataset = MagicMock()
                self.dataset.__len__ = MagicMock(return_value=10)
        
        dataset = TestDataset(dataset_name="test", split="train", seed=42)
        
        # Test that __getitem__ is not implemented
        with pytest.raises(NotImplementedError):
            _ = dataset[0]

    def test_base_dataset_hf_dataset_property(self):
        """Test that hf_dataset property returns the underlying dataset."""
        
        class TestDataset(BaseDataset):
            def _initialize_dataset(self):
                self.dataset = MagicMock()
                self.dataset.__len__ = MagicMock(return_value=10)
            
            def __getitem__(self, idx):
                return {"data": "test"}
        
        dataset = TestDataset(dataset_name="test", split="train", seed=42)
        assert dataset.hf_dataset == dataset.dataset

    def test_base_dataset_len(self):
        """Test that __len__ returns correct length."""
        
        class TestDataset(BaseDataset):
            def _initialize_dataset(self):
                self.dataset = MagicMock()
                self.dataset.__len__ = MagicMock(return_value=100)
            
            def __getitem__(self, idx):
                return {"data": "test"}
        
        dataset = TestDataset(dataset_name="test", split="train", seed=42)
        assert len(dataset) == 100


class TestSFTDataset:
    """Test suite for SFTDataset class."""

    @pytest.fixture
    def mock_hf_dataset(self):
        """Create a mock HuggingFace dataset."""
        data = {
            "question": ["What is AI?", "What is ML?", "What is DL?"],
            "answer": ["Artificial Intelligence", "Machine Learning", "Deep Learning"],
            "context": ["AI context", "ML context", "DL context"]
        }
        return HFDataset.from_dict(data)

    @pytest.fixture
    def mock_hf_dataset_with_empty(self):
        """Create a mock HuggingFace dataset with empty values."""
        data = {
            "question": ["What is AI?", "", "What is DL?", None],
            "answer": ["Artificial Intelligence", "Machine Learning", "", None],
        }
        return HFDataset.from_dict(data)

    @pytest.fixture
    def temp_json_file(self):
        """Create a temporary JSON file for testing."""
        data = [
            {"question": "What is AI?", "answer": "Artificial Intelligence"},
            {"question": "What is ML?", "answer": "Machine Learning"},
            {"question": "What is DL?", "answer": "Deep Learning"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_sft_dataset_missing_prompt_config(self):
        """Test that SFTDataset raises error when prompt config is missing."""
        with pytest.raises(RuntimeError, match="Either provide prompt_template or prompt_func"):
            SFTDataset(
                dataset_name="test",
                split="train",
                completion_template="{answer}"
            )

    def test_sft_dataset_missing_completion_config(self):
        """Test that SFTDataset raises error when completion config is missing."""
        with pytest.raises(RuntimeError, match="Either provide completion_template or completion_func"):
            SFTDataset(
                dataset_name="test",
                split="train",
                prompt_template="{question}"
            )

    def test_sft_dataset_both_prompt_configs(self):
        """Test that SFTDataset raises error when both prompt configs are provided."""
        with pytest.raises(RuntimeError, match="Either provide prompt_template or prompt_func"):
            SFTDataset(
                dataset_name="test",
                split="train",
                prompt_template="{question}",
                prompt_func="module:func",
                completion_template="{answer}"
            )

    def test_sft_dataset_both_completion_configs(self):
        """Test that SFTDataset raises error when both completion configs are provided."""
        with pytest.raises(RuntimeError, match="Either provide completion_template or completion_func"):
            SFTDataset(
                dataset_name="test",
                split="train",
                prompt_template="{question}",
                completion_template="{answer}",
                completion_func="module:func"
            )

    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset')
    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset_builder')
    def test_sft_dataset_initialization_with_templates(self, mock_builder, mock_load, mock_hf_dataset):
        """Test SFTDataset initialization with prompt and completion templates."""
        # Setup mocks
        mock_builder.return_value.info.splits = {"train": None}
        mock_load.return_value = mock_hf_dataset
        
        dataset = SFTDataset(
            dataset_name="test_dataset",
            split="train",
            prompt_template="Question: {question}",
            completion_template="Answer: {answer}"
        )
        
        assert dataset.dataset_name == "test_dataset"
        assert dataset.split == "train"
        assert dataset.seed == 42
        assert dataset.prompt_template == "Question: {question}"
        assert dataset.completion_template == "Answer: {answer}"
        assert len(dataset) > 0

    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset')
    def test_sft_dataset_from_json_file(self, mock_load, temp_json_file):
        """Test SFTDataset loading from JSON file."""
        mock_dataset = HFDataset.from_dict({
            "question": ["What is AI?", "What is ML?", "What is DL?"],
            "answer": ["Artificial Intelligence", "Machine Learning", "Deep Learning"]
        })
        mock_load.return_value = mock_dataset
        
        dataset = SFTDataset(
            dataset_name="ignored",
            split="train",
            json_file_path=temp_json_file,
            prompt_template="{question}",
            completion_template="{answer}"
        )
        
        assert dataset.json_file_path == temp_json_file
        assert len(dataset) > 0

    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset')
    def test_sft_dataset_train_test_split_from_json(self, mock_load):
        """Test train/test split when loading from JSON."""
        mock_dataset = HFDataset.from_dict({
            "question": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10"],
            "answer": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]
        })
        mock_load.return_value = mock_dataset
        
        train_dataset = SFTDataset(
            dataset_name="ignored",
            split="train",
            split_ratio=0.8,
            json_file_path="dummy.json",
            prompt_template="{question}",
            completion_template="{answer}"
        )
        
        test_dataset = SFTDataset(
            dataset_name="ignored",
            split="test",
            split_ratio=0.8,
            json_file_path="dummy.json",
            prompt_template="{question}",
            completion_template="{answer}"
        )
        
        # Train should have ~80% and test should have ~20%
        assert len(train_dataset) > len(test_dataset)

    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset')
    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset_builder')
    def test_sft_dataset_invalid_template_variable(self, mock_builder, mock_load, mock_hf_dataset):
        """Test that invalid template variables raise RuntimeError."""
        mock_builder.return_value.info.splits = {"train": None}
        mock_load.return_value = mock_hf_dataset
        
        with pytest.raises(RuntimeError, match="not found in dataset columns"):
            SFTDataset(
                dataset_name="test_dataset",
                split="train",
                prompt_template="{invalid_column}",
                completion_template="{answer}"
            )

    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset')
    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset_builder')
    def test_sft_dataset_filter_empty_samples(self, mock_builder, mock_load, mock_hf_dataset_with_empty):
        """Test filtering of empty or None samples."""
        mock_builder.return_value.info.splits = {"train": None}
        mock_load.return_value = mock_hf_dataset_with_empty
        
        dataset = SFTDataset(
            dataset_name="test_dataset",
            split="train",
            prompt_template="{question}",
            completion_template="{answer}",
            remove_samples_with_empty_columns=True
        )
        
        # Should only have 1 valid sample (first one)
        assert len(dataset) == 1

    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset')
    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset_builder')
    def test_sft_dataset_no_filter_empty_samples(self, mock_builder, mock_load, mock_hf_dataset_with_empty):
        """Test that filtering can be disabled."""
        mock_builder.return_value.info.splits = {"train": None}
        mock_load.return_value = mock_hf_dataset_with_empty
        
        dataset = SFTDataset(
            dataset_name="test_dataset",
            split="train",
            prompt_template="{question}",
            completion_template="{answer}",
            remove_samples_with_empty_columns=False
        )
        
        # Should have 3 samples (the map operation filters out None values during preprocessing)
        # The 4th sample with None values cannot be processed by the template
        assert len(dataset) == 3

    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset')
    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset_builder')
    def test_sft_dataset_getitem(self, mock_builder, mock_load, mock_hf_dataset):
        """Test __getitem__ returns correctly formatted data."""
        mock_builder.return_value.info.splits = {"train": None}
        mock_load.return_value = mock_hf_dataset
        
        dataset = SFTDataset(
            dataset_name="test_dataset",
            split="train",
            prompt_template="Q: {question}",
            completion_template="A: {answer}"
        )
        
        sample = dataset[0]
        
        assert "prompt" in sample
        assert "completion" in sample
        assert sample["prompt"].startswith("Q:")
        assert sample["completion"].startswith("A:")

    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset')
    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset_builder')
    def test_sft_dataset_preprocess_sample(self, mock_builder, mock_load, mock_hf_dataset):
        """Test _preprocess_sample method."""
        mock_builder.return_value.info.splits = {"train": None}
        mock_load.return_value = mock_hf_dataset
        
        dataset = SFTDataset(
            dataset_name="test_dataset",
            split="train",
            prompt_template="Question: {question}",
            completion_template="Answer: {answer}"
        )
        
        example = {"question": "Test Q", "answer": "Test A"}
        processed = dataset._preprocess_sample(example)
        
        assert processed["prompt"] == "Question: Test Q"
        assert processed["completion"] == "Answer: Test A"

    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset')
    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset_builder')
    def test_sft_dataset_with_custom_prompt_func(self, mock_builder, mock_load, mock_hf_dataset):
        """Test SFTDataset with custom prompt function."""
        mock_builder.return_value.info.splits = {"train": None}
        mock_load.return_value = mock_hf_dataset
        
        # Create a mock module with a function
        with patch('QEfficient.finetune.experimental.core.dataset.importlib.import_module') as mock_import:
            mock_module = MagicMock()
            mock_module.custom_prompt = lambda x: f"Custom: {x['question']}"
            mock_import.return_value = mock_module
            
            dataset = SFTDataset(
                dataset_name="test_dataset",
                split="train",
                prompt_func="module:custom_prompt",
                completion_template="{answer}"
            )
            
            sample = dataset[0]
            assert "Custom:" in sample["prompt"]

    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset')
    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset_builder')
    def test_sft_dataset_with_custom_completion_func(self, mock_builder, mock_load, mock_hf_dataset):
        """Test SFTDataset with custom completion function."""
        mock_builder.return_value.info.splits = {"train": None}
        mock_load.return_value = mock_hf_dataset
        
        # Create a mock module with a function
        with patch('QEfficient.finetune.experimental.core.dataset.importlib.import_module') as mock_import:
            mock_module = MagicMock()
            mock_module.custom_completion = lambda x: f"Custom: {x['answer']}"
            mock_import.return_value = mock_module
            
            dataset = SFTDataset(
                dataset_name="test_dataset",
                split="train",
                prompt_template="{question}",
                completion_func="module:custom_completion"
            )
            
            sample = dataset[0]
            assert "Custom:" in sample["completion"]

    def test_sft_dataset_import_func_invalid_format(self):
        """Test import_func with invalid format."""
        with patch('QEfficient.finetune.experimental.core.dataset.load_dataset'):
            with patch('QEfficient.finetune.experimental.core.dataset.load_dataset_builder') as mock_builder:
                mock_builder.return_value.info.splits = {"train": None}
                
                with pytest.raises(ValueError, match="func_path must be in the format"):
                    dataset = SFTDataset(
                        dataset_name="test",
                        split="train",
                        prompt_func="invalid_format",
                        completion_template="{answer}"
                    )

    def test_sft_dataset_import_func_module_not_found(self):
        """Test import_func when module cannot be imported."""
        with patch('QEfficient.finetune.experimental.core.dataset.load_dataset'):
            with patch('QEfficient.finetune.experimental.core.dataset.load_dataset_builder') as mock_builder:
                mock_builder.return_value.info.splits = {"train": None}
                
                with patch('QEfficient.finetune.experimental.core.dataset.importlib.import_module') as mock_import:
                    mock_import.side_effect = Exception("Module not found")
                    
                    with pytest.raises(RuntimeError, match="Unable to import module"):
                        dataset = SFTDataset(
                            dataset_name="test",
                            split="train",
                            prompt_func="nonexistent:func",
                            completion_template="{answer}"
                        )

    def test_sft_dataset_import_func_function_not_found(self):
        """Test import_func when function is not found in module."""
        with patch('QEfficient.finetune.experimental.core.dataset.load_dataset'):
            with patch('QEfficient.finetune.experimental.core.dataset.load_dataset_builder') as mock_builder:
                mock_builder.return_value.info.splits = {"train": None}
                
                with patch('QEfficient.finetune.experimental.core.dataset.importlib.import_module') as mock_import:
                    mock_module = MagicMock()
                    mock_module.other_func = lambda x: x
                    del mock_module.target_func  # Ensure attribute doesn't exist
                    mock_import.return_value = mock_module
                    
                    with pytest.raises(ValueError, match="Function .* not found in module"):
                        dataset = SFTDataset(
                            dataset_name="test",
                            split="train",
                            prompt_func="module:target_func",
                            completion_template="{answer}"
                        )

    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset')
    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset_builder')
    def test_sft_dataset_filter_empty_or_none_samples(self, mock_builder, mock_load):
        """Test _filter_empty_or_none_samples method."""
        mock_builder.return_value.info.splits = {"train": None}
        # Use a larger dataset to avoid split issues
        mock_dataset = HFDataset.from_dict({
            "question": ["Q1", "Q2", "Q3", "Q4", "Q5"],
            "answer": ["A1", "A2", "A3", "A4", "A5"]
        })
        mock_load.return_value = mock_dataset
        
        dataset = SFTDataset(
            dataset_name="test",
            split="train",
            prompt_template="{question}",
            completion_template="{answer}"
        )
        
        # Test valid sample
        assert dataset._filter_empty_or_none_samples({"question": "Q", "answer": "A"}) == True
        
        # Test None value
        assert dataset._filter_empty_or_none_samples({"question": None, "answer": "A"}) == False
        
        # Test empty string
        assert dataset._filter_empty_or_none_samples({"question": "", "answer": "A"}) == False
        
        # Test whitespace only
        assert dataset._filter_empty_or_none_samples({"question": "   ", "answer": "A"}) == False

    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset')
    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset_builder')
    def test_sft_dataset_hf_dataset_property(self, mock_builder, mock_load, mock_hf_dataset):
        """Test hf_dataset property returns underlying dataset."""
        mock_builder.return_value.info.splits = {"train": None}
        mock_load.return_value = mock_hf_dataset
        
        dataset = SFTDataset(
            dataset_name="test",
            split="train",
            prompt_template="{question}",
            completion_template="{answer}"
        )
        
        assert dataset.hf_dataset is not None
        assert hasattr(dataset.hf_dataset, 'num_rows')

    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset')
    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset_builder')
    def test_sft_dataset_split_not_available(self, mock_builder, mock_load, mock_hf_dataset):
        """Test error when requested split is not available for train split."""
        mock_builder.return_value.info.splits = {"test": None}
        mock_load.return_value = mock_hf_dataset
        
        with pytest.raises(ValueError, match="Split train is not available"):
            dataset = SFTDataset(
                dataset_name="test",
                split="train",
                prompt_template="{question}",
                completion_template="{answer}"
            )

    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset')
    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset_builder')
    def test_sft_dataset_single_split_auto_split(self, mock_builder, mock_load, mock_hf_dataset):
        """Test automatic splitting when only one split is available."""
        mock_builder.return_value.info.splits = {"train": None}
        mock_load.return_value = mock_hf_dataset
        
        # Request test split when only train is available
        dataset = SFTDataset(
            dataset_name="test",
            split="test",
            split_ratio=0.8,
            prompt_template="{question}",
            completion_template="{answer}"
        )
        
        # Should create a test split from the train split
        assert len(dataset) > 0

    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset')
    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset_builder')
    def test_sft_dataset_multiple_template_variables(self, mock_builder, mock_load):
        """Test templates with multiple variables."""
        mock_dataset = HFDataset.from_dict({
            "question": ["Q1", "Q2"],
            "context": ["C1", "C2"],
            "answer": ["A1", "A2"]
        })
        mock_builder.return_value.info.splits = {"train": None}
        mock_load.return_value = mock_dataset
        
        dataset = SFTDataset(
            dataset_name="test",
            split="train",
            prompt_template="Context: {context}\nQuestion: {question}",
            completion_template="Answer: {answer}"
        )
        
        sample = dataset[0]
        assert "Context:" in sample["prompt"]
        assert "Question:" in sample["prompt"]
        assert "Answer:" in sample["completion"]

    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset')
    @patch('QEfficient.finetune.experimental.core.dataset.load_dataset_builder')
    def test_sft_dataset_seed_reproducibility(self, mock_builder, mock_load, mock_hf_dataset):
        """Test that same seed produces same splits."""
        mock_builder.return_value.info.splits = {"train": None}
        mock_load.return_value = mock_hf_dataset
        
        dataset1 = SFTDataset(
            dataset_name="test",
            split="train",
            seed=42,
            split_ratio=0.8,
            prompt_template="{question}",
            completion_template="{answer}"
        )
        
        # Reset mock
        mock_load.return_value = mock_hf_dataset
        
        dataset2 = SFTDataset(
            dataset_name="test",
            split="train",
            seed=42,
            split_ratio=0.8,
            prompt_template="{question}",
            completion_template="{answer}"
        )
        
        # Should have same length with same seed
        assert len(dataset1) == len(dataset2)
