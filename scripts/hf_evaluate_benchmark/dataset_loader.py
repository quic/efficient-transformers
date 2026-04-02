#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Dataset Loader for HF Evaluate Benchmark

Loads and formats datasets from HuggingFace for evaluation.
Supports: GSM8K, BoolQ, HellaSwag
"""

from typing import List, Dict, Optional
from datasets import load_dataset
import re
from tqdm import tqdm

# Constants for answer extraction
CLASSIFICATION_SEARCH_WINDOW = 20  # Characters to search for yes/no
MULTIPLE_CHOICE_SEARCH_WINDOW = 50  # Characters to search for A/B/C/D
PROMPT_TRUNCATE_LENGTH = 100  # Characters to show in truncated prompts

# Pre-compiled regex patterns for performance
NUMERICAL_ANSWER_PATTERN = re.compile(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)')
NUMBER_PATTERN = re.compile(r'-?\d+(?:,\d+)*(?:\.\d+)?')
ANSWER_PHRASE_PATTERNS = [
    re.compile(r'output of\s+(-?\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r'answer is\s+(-?\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r'result is\s+(-?\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r'equals?\s+(-?\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r'=\s+(-?\d+(?:\.\d+)?)', re.IGNORECASE),
]
MULTIPLE_CHOICE_PATTERN = re.compile(r'\b([A-D])\b')


class DatasetLoader:
    """Load and format benchmark datasets."""
    
    SUPPORTED_DATASETS = ["gsm8k", "boolq", "hellaswag"]
    
    def __init__(self, dataset_name: str, num_samples: Optional[int] = None):
        """
        Initialize dataset loader.
        
        Args:
            dataset_name: Name of dataset (gsm8k, boolq, hellaswag)
            num_samples: Number of samples to load (None = all)
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset {dataset_name} not supported. "
                f"Choose from {self.SUPPORTED_DATASETS}"
            )
        
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        
    def load(self) -> List[Dict]:
        """
        Load dataset and return formatted examples.
        
        Returns:
            List of dicts with keys: 'prompt', 'reference', 'type', and optionally 'choices'
        """
        if self.dataset_name == "gsm8k":
            return self._load_gsm8k()
        elif self.dataset_name == "boolq":
            return self._load_boolq()
        elif self.dataset_name == "hellaswag":
            return self._load_hellaswag()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_gsm8k(self) -> List[Dict]:
        """Load GSM8K math reasoning dataset."""
        print(f"Loading GSM8K dataset from HuggingFace...")
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        
        if self.num_samples:
            dataset = dataset.select(range(min(self.num_samples, len(dataset))))
        
        examples = []
        print(f"Processing {len(dataset)} GSM8K examples...")
        for item in tqdm(dataset, desc="Loading GSM8K", unit="example"):
            question = item["question"]
            answer = item["answer"]
            
            # Extract numerical answer from the solution
            # GSM8K answers are in format: "Step 1...\n#### 42"
            numerical_answer = self._extract_numerical_answer(answer)
            
            prompt = f"Question: {question}\nAnswer:"
            
            examples.append({
                "prompt": prompt,
                "reference": numerical_answer,
                "full_answer": answer,
                "type": "generation"
            })
        
        print(f"Loaded {len(examples)} GSM8K examples")
        return examples
    
    def _load_boolq(self) -> List[Dict]:
        """Load BoolQ yes/no question dataset."""
        print(f"Loading BoolQ dataset from HuggingFace...")
        dataset = load_dataset("google/boolq", split="validation")
        
        if self.num_samples:
            dataset = dataset.select(range(min(self.num_samples, len(dataset))))
        
        examples = []
        print(f"Processing {len(dataset)} BoolQ examples...")
        for item in tqdm(dataset, desc="Loading BoolQ", unit="example"):
            passage = item["passage"]
            question = item["question"]
            answer = "yes" if item["answer"] else "no"
            
            prompt = f"Passage: {passage}\n\nQuestion: {question}\nAnswer (yes or no):"
            
            examples.append({
                "prompt": prompt,
                "reference": answer,
                "type": "classification"
            })
        
        print(f"Loaded {len(examples)} BoolQ examples")
        return examples
    
    def _load_hellaswag(self) -> List[Dict]:
        """Load HellaSwag common sense reasoning dataset."""
        print(f"Loading HellaSwag dataset from HuggingFace...")
        dataset = load_dataset("Rowan/hellaswag", split="validation")
        
        if self.num_samples:
            dataset = dataset.select(range(min(self.num_samples, len(dataset))))
        
        examples = []
        print(f"Processing {len(dataset)} HellaSwag examples...")
        for item in tqdm(dataset, desc="Loading HellaSwag", unit="example"):
            context = item["ctx"]
            endings = item["endings"]
            label = int(item["label"])
            
            # Format as multiple choice
            choices_text = "\n".join([f"{chr(65+i)}) {ending}" for i, ending in enumerate(endings)])
            prompt = f"Context: {context}\n\nChoose the most likely continuation:\n{choices_text}\n\nAnswer:"
            
            reference = chr(65 + label)  # Convert 0,1,2,3 to A,B,C,D
            
            examples.append({
                "prompt": prompt,
                "reference": reference,
                "choices": endings,
                "type": "multiple_choice"
            })
        
        print(f"Loaded {len(examples)} HellaSwag examples")
        return examples
    
    @staticmethod
    def _extract_numerical_answer(answer_text: str) -> str:
        """
        Extract numerical answer from GSM8K answer format.
        
        GSM8K answers end with "#### NUMBER"
        
        Args:
            answer_text: The full answer text from GSM8K
            
        Returns:
            Extracted numerical answer as string (without commas)
        """
        # Primary method: Look for "#### NUMBER" format
        match = NUMERICAL_ANSWER_PATTERN.search(answer_text)
        if match:
            # Remove commas from numbers
            return match.group(1).replace(',', '')
        
        # Fallback: try to find any number in the text
        numbers = NUMBER_PATTERN.findall(answer_text)
        if numbers:
            return numbers[-1].replace(',', '')
        
        return ""
    
    @staticmethod
    def extract_answer_from_generation(generated_text: str, dataset_type: str) -> str:
        """
        Extract answer from model generation.
        
        This method uses multiple strategies to extract answers based on dataset type:
        - For generation (GSM8K): Tries phrase patterns, then #### format, then last number
        - For classification (BoolQ): Looks for yes/no in first 20 characters
        - For multiple_choice (HellaSwag): Looks for A/B/C/D in first 50 characters
        
        Args:
            generated_text: Raw model output
            dataset_type: Type of dataset (generation, classification, multiple_choice)
        
        Returns:
            Extracted answer string (empty string if extraction fails)
        """
        generated_text = generated_text.strip()
        
        if dataset_type == "generation":
            # For GSM8K, try multiple strategies to extract the final answer
            
            # Strategy 1: Look for "output of X" or "answer is X" patterns
            for pattern in ANSWER_PHRASE_PATTERNS:
                match = pattern.search(generated_text)
                if match:
                    # Remove commas and decimals to get integer
                    return match.group(1).replace(',', '').split('.')[0]
            
            # Strategy 2: Look for "#### NUMBER" format (common in GSM8K)
            match = NUMERICAL_ANSWER_PATTERN.search(generated_text)
            if match:
                return match.group(1).replace(',', '').split('.')[0]
            
            # Strategy 3: Extract last number in the text (often the final answer)
            numbers = NUMBER_PATTERN.findall(generated_text)
            if numbers:
                # Return last number, removing commas and decimals
                return numbers[-1].replace(',', '').split('.')[0]
            
            return ""
        
        elif dataset_type == "classification":
            # For BoolQ, extract yes/no from beginning of response
            text_lower = generated_text.lower()
            search_window = text_lower[:CLASSIFICATION_SEARCH_WINDOW]
            
            if "yes" in search_window:
                return "yes"
            elif "no" in search_window:
                return "no"
            return ""
        
        elif dataset_type == "multiple_choice":
            # For HellaSwag, extract A/B/C/D from beginning of response
            search_window = generated_text[:MULTIPLE_CHOICE_SEARCH_WINDOW]
            match = MULTIPLE_CHOICE_PATTERN.search(search_window)
            if match:
                return match.group(1)
            return ""
        
        return generated_text


def test_loader():
    """Test dataset loader."""
    print("Testing Dataset Loader\n" + "="*60)
    
    for dataset_name in ["gsm8k", "boolq", "hellaswag"]:
        print(f"\n{dataset_name.upper()}:")
        loader = DatasetLoader(dataset_name, num_samples=2)
        examples = loader.load()
        
        for i, ex in enumerate(examples):
            print(f"\nExample {i+1}:")
            print(f"Prompt: {ex['prompt'][:PROMPT_TRUNCATE_LENGTH]}...")
            print(f"Reference: {ex['reference']}")
            print(f"Type: {ex['type']}")


if __name__ == "__main__":
    test_loader()
