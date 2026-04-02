#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Metric Handler for HF Evaluate Benchmark

Computes metrics using HuggingFace Evaluate library.
"""

from typing import List, Dict, Optional
import evaluate
import logging
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)


class MetricHandler:
    """Handle metric computation using HF Evaluate."""
    
    # Cache loaded metrics to avoid repeated loading
    _metric_cache = {}
    
    def __init__(self, dataset_name: str, use_hf_evaluate: bool = False):
        """
        Initialize metric handler.
        
        Args:
            dataset_name: Name of dataset (determines which metrics to use)
            use_hf_evaluate: Whether to load HF Evaluate metrics (default: False)
                           If False, uses fast manual computation instead
        """
        self.dataset_name = dataset_name
        self.use_hf_evaluate = use_hf_evaluate
        self.metric = self._get_metric() if use_hf_evaluate else None
    
    def _get_metric(self):
        """Get appropriate metric for dataset (with caching)."""
        # Determine metric type based on dataset
        if self.dataset_name == "gsm8k":
            metric_name = "exact_match"
        elif self.dataset_name in ["boolq", "hellaswag"]:
            metric_name = "accuracy"
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        # Load from cache if available, otherwise load and cache
        if metric_name not in self._metric_cache:
            print(f"Loading HuggingFace Evaluate metric '{metric_name}'...")
            print("(This is a one-time download and will be cached for future runs)")
            
            # Use tqdm to show that something is happening
            with tqdm(total=1, desc=f"Loading {metric_name} metric", unit="metric") as pbar:
                self._metric_cache[metric_name] = evaluate.load(metric_name)
                pbar.update(1)
            
            print(f"Metric '{metric_name}' loaded successfully!")
        
        return self._metric_cache[metric_name]
    
    def compute(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute metrics.
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
        
        Returns:
            Dictionary with metric results
        """
        # Validate inputs
        if not predictions or not references:
            raise ValueError("Predictions and references cannot be empty")
        
        if len(predictions) != len(references):
            raise ValueError(
                f"Predictions ({len(predictions)}) and references ({len(references)}) "
                "must have same length"
            )
        
        # Clean predictions and references once
        predictions_clean = [str(p).strip().lower() for p in predictions]
        references_clean = [str(r).strip().lower() for r in references]
        
        # Compute statistics manually (always works)
        total = len(predictions)
        correct = sum(1 for p, r in zip(predictions_clean, references_clean) if p == r)
        accuracy = correct / total if total > 0 else 0.0
        
        # Try to use HF Evaluate metric if enabled, but fall back to manual calculation if it fails
        if self.use_hf_evaluate and self.metric is not None:
            try:
                if self.dataset_name == "gsm8k":
                    # For GSM8K, use exact match
                    result = self.metric.compute(
                        predictions=predictions_clean,
                        references=references_clean
                    )
                    accuracy = result["exact_match"]
                else:
                    # For BoolQ and HellaSwag, accuracy metric expects integer labels
                    # We use manual calculation which is equivalent
                    pass
            except (ValueError, TypeError) as e:
                # Log the error but continue with manual calculation
                logger.warning(
                    f"HF Evaluate metric failed for {self.dataset_name}: {e}. "
                    "Using manual calculation."
                )
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "error_rate": 1.0 - accuracy
        }
    
    def compute_detailed(
        self,
        predictions: List[str],
        references: List[str],
        prompts: Optional[List[str]] = None
    ) -> Dict:
        """
        Compute metrics with detailed per-example results.
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            prompts: Optional list of prompts for debugging
        
        Returns:
            Dictionary with overall metrics and per-example results
        """
        # Compute overall metrics (this handles cleaning internally)
        overall = self.compute(predictions, references)
        
        # Clean for per-example comparison
        predictions_clean = [str(p).strip().lower() for p in predictions]
        references_clean = [str(r).strip().lower() for r in references]
        
        per_example = []
        for i, (pred, ref) in enumerate(zip(predictions_clean, references_clean)):
            is_correct = pred == ref
            example_result = {
                "index": i,
                "prediction": predictions[i],  # Original case
                "reference": references[i],    # Original case
                "correct": is_correct
            }
            if prompts and i < len(prompts):
                # Truncate prompt for readability
                example_result["prompt"] = prompts[i][:100] + "..." if len(prompts[i]) > 100 else prompts[i]
            
            per_example.append(example_result)
        
        return {
            "overall": overall,
            "per_example": per_example
        }


def test_metric_handler():
    """Test metric handler."""
    print("Testing Metric Handler\n" + "="*60)
    
    # Test GSM8K (exact match)
    print("\nGSM8K (Exact Match):")
    handler = MetricHandler("gsm8k")
    predictions = ["42", "100", "7"]
    references = ["42", "99", "7"]
    result = handler.compute(predictions, references)
    print(f"Predictions: {predictions}")
    print(f"References: {references}")
    print(f"Results: {result}")
    
    # Test BoolQ (accuracy)
    print("\nBoolQ (Accuracy):")
    handler = MetricHandler("boolq")
    predictions = ["yes", "no", "yes", "no"]
    references = ["yes", "no", "no", "no"]
    result = handler.compute(predictions, references)
    print(f"Predictions: {predictions}")
    print(f"References: {references}")
    print(f"Results: {result}")
    
    # Test HellaSwag (accuracy)
    print("\nHellaSwag (Accuracy):")
    handler = MetricHandler("hellaswag")
    predictions = ["A", "B", "C", "D"]
    references = ["A", "C", "C", "D"]
    result = handler.compute(predictions, references)
    print(f"Predictions: {predictions}")
    print(f"References: {references}")
    print(f"Results: {result}")
    
    # Test detailed results
    print("\nDetailed Results:")
    detailed = handler.compute_detailed(predictions, references)
    print(f"Overall: {detailed['overall']}")
    print(f"Per-example (first 2):")
    for ex in detailed['per_example'][:2]:
        print(f"  {ex}")


if __name__ == "__main__":
    test_metric_handler()
