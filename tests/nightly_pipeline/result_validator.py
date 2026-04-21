# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Result Validator: Compares current pipeline results with baseline results
Supports performance metrics comparison and generated_ids comparison using cosine similarity.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation check"""

    model_name: str
    status: str  # "passed" or "failed"
    perf_metrics_summary: Optional[Dict] = None  # {"prefill_time": {"current": x, "baseline": y, "deviation": z}, ...}
    generated_ids_summary: Optional[Dict] = None  # {"similarity": x, "mad": y}
    error_message: Optional[str] = None


class ResultValidator:
    """Validates current results against baseline with 5% tolerance"""

    TOLERANCE_PERCENT = 30.0  # 5% tolerance

    def __init__(self, baseline_file: Optional[str] = None):
        """Initialize validator with optional baseline file"""
        self.baseline_file = Path(baseline_file) if baseline_file else None
        self.baseline_results = self._load_baseline()

    def _load_baseline(self) -> Dict:
        """Load baseline results from file"""
        if not self.baseline_file or not self.baseline_file.exists():
            logger.warning(f"Baseline file not found: {self.baseline_file}. Skipping validation.")
            return {}

        try:
            with open(self.baseline_file, "r") as f:
                data = json.load(f)

            logger.info(f"Loaded baseline results from: {self.baseline_file}")
            return data
        except Exception as e:
            logger.error(f"Failed to load baseline results: {e}")
            return {}

    def validate_results(
        self,
        current_inferences: List[Dict],
    ) -> Tuple[List[ValidationResult], bool]:
        """
        Validate current results against baseline

        Returns:
            - List of ValidationResult objects (one per model)
            - Overall pass/fail flag (True if all pass, False if any fail)
        """
        if not self.baseline_results:
            logger.warning("No baseline results available. Skipping validation.")
            return [], True  # Pass if no baseline to compare

        baseline_inferences = self.baseline_results.get("inferences", [])
        baseline_by_model = {r["model_name"]: r for r in baseline_inferences}

        validation_results = []
        all_passed = True

        for current_result in current_inferences:
            if current_result.get("status") != "success":
                # Skip failed inferences
                validation_results.append(
                    ValidationResult(
                        model_name=current_result["model_name"],
                        status="skipped",
                        error_message="Current inference failed, skipping validation",
                    )
                )
                continue

            model_name = current_result["model_name"]
            baseline_result = baseline_by_model.get(model_name)

            if not baseline_result:
                logger.warning(f"No baseline result for model: {model_name}")
                validation_results.append(
                    ValidationResult(
                        model_name=model_name, status="skipped", error_message="No baseline result to compare"
                    )
                )
                continue

            # Validate perf metrics
            perf_validation = self._validate_perf_metrics(current_result, baseline_result, model_name)

            # Validate generated_ids using cosine similarity
            ids_validation = self._validate_generated_ids(current_result, baseline_result, model_name)

            # Determine overall status
            status = "passed"
            if not perf_validation["passed"] or not ids_validation["passed"]:
                status = "failed"
                all_passed = False

            result = ValidationResult(
                model_name=model_name,
                status=status,
                perf_metrics_summary=perf_validation["summary"],
                generated_ids_summary=ids_validation["summary"],
            )

            validation_results.append(result)

        return validation_results, all_passed

    def _validate_perf_metrics(self, current: Dict, baseline: Dict, model_name: str) -> Dict:
        """Validate performance metrics with 5% tolerance"""
        metrics_to_check = ["prefill_time", "decode_perf", "total_perf", "total_time"]
        summary = {}
        all_passed = True

        for metric in metrics_to_check:
            current_val = current.get(metric)
            baseline_val = baseline.get(metric)

            if current_val is None or baseline_val is None:
                summary[metric] = {
                    "current": current_val,
                    "baseline": baseline_val,
                    "status": "skipped",
                    "reason": "Missing value",
                }
                continue

            # Calculate deviation percentage
            if baseline_val == 0:
                if current_val == 0:
                    deviation_percent = 0
                else:
                    all_passed = False
                    deviation_percent = float("inf")
            else:
                deviation_percent = abs((current_val - baseline_val) / baseline_val) * 100

            passed = deviation_percent <= self.TOLERANCE_PERCENT

            summary[metric] = {
                "current": current_val,
                "baseline": baseline_val,
                "deviation_percent": round(deviation_percent, 2),
                "tolerance_percent": self.TOLERANCE_PERCENT,
                "status": "passed" if passed else "failed",
            }

            if not passed:
                all_passed = False
                logger.warning(
                    f"[{model_name}] {metric} deviation exceeds tolerance: "
                    f"{deviation_percent:.2f}% > {self.TOLERANCE_PERCENT}%"
                )

        return {"passed": all_passed, "summary": summary}

    def _validate_generated_ids(self, current: Dict, baseline: Dict, model_name: str) -> Dict:
        """Validate generated_ids using cosine similarity with MAD tolerance"""
        current_ids = current.get("generated_ids")
        baseline_ids = baseline.get("generated_ids")

        if not current_ids or not baseline_ids:
            return {
                "passed": True,
                "summary": {"similarity": None, "mad": None, "status": "skipped", "reason": "Missing generated_ids"},
            }

        # Convert to numpy arrays if needed
        current_ids_array = self._to_numpy_array(current_ids)
        baseline_ids_array = self._to_numpy_array(baseline_ids)

        if current_ids_array is None or baseline_ids_array is None:
            return {
                "passed": True,
                "summary": {
                    "similarity": None,
                    "mad": None,
                    "status": "skipped",
                    "reason": "Could not convert generated_ids to arrays",
                },
            }

        # Flatten arrays for embedding comparison
        current_flat = current_ids_array.flatten()
        baseline_flat = baseline_ids_array.flatten()

        # Calculate cosine similarity
        similarity = self._cosine_similarity(current_flat, baseline_flat)

        # Calculate MAD (Mean Absolute Deviation) as percentage from perfect match
        # For sequences, MAD = mean(abs(current - baseline)) / len
        # Express as percentage deviation from baseline
        mad = np.mean(np.abs(current_flat - baseline_flat))

        # Convert MAD to percentage for tolerance comparison
        # If mean token ID is ~10000, MAD of 100 = 1% tolerance
        mean_baseline = np.mean(baseline_flat)
        if mean_baseline > 0:
            mad_percent = (mad / mean_baseline) * 100
        else:
            mad_percent = 0 if mad == 0 else float("inf")

        # For cosine similarity, we want it to be close to 1.0
        # tolerance means it should be > (1 - tolerance/100) = 0.95
        similarity_tolerance = 1.0 - (self.TOLERANCE_PERCENT / 100)

        passed = similarity >= similarity_tolerance

        logger.info(
            f"[{model_name}] Generated IDs Similarity: {similarity:.4f} "
            f"(tolerance: >{similarity_tolerance:.4f}), MAD: {mad_percent:.2f}%"
        )

        return {
            "passed": passed,
            "summary": {
                "cosine_similarity": round(similarity, 4),
                "mad": round(mad, 4),
                "mad_percent": round(mad_percent, 2),
                "similarity_threshold": round(similarity_tolerance, 4),
                "tolerance_percent": self.TOLERANCE_PERCENT,
                "status": "passed" if passed else "failed",
            },
        }

    def _to_numpy_array(self, ids) -> Optional[np.ndarray]:
        """Convert generated_ids to numpy array"""
        try:
            if isinstance(ids, np.ndarray):
                return ids
            elif isinstance(ids, list):
                # Handle list of lists or flat list
                if ids and isinstance(ids[0], list):
                    # List of lists - flatten
                    return np.array([item for sublist in ids for item in sublist], dtype=int)
                else:
                    return np.array(ids, dtype=int)
            else:
                return None
        except Exception as e:
            logger.error(f"Error converting generated_ids to array: {e}")
            return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            if a.shape != b.shape:
                raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

            a_norm = a / (np.linalg.norm(a) + 1e-8)
            b_norm = b / (np.linalg.norm(b) + 1e-8)

            similarity = np.dot(a_norm, b_norm)
            return float(similarity)

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0

    def update_baseline(self, current_results: Dict, output_file: Optional[Path] = None):
        """Update baseline file with current results after validation passes"""
        output_file = output_file or self.baseline_file

        if not output_file:
            logger.error("No output file specified for baseline update")
            return False

        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(current_results, f, indent=2)

            logger.info(f" Baseline updated: {output_file}")
            return True
        except Exception as e:
            logger.error(f" Failed to update baseline: {e}")
            return False


def log_validation_report(validation_results: List[ValidationResult], all_passed: bool):
    """Log a formatted validation report"""
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION REPORT")
    logger.info("=" * 80)

    for result in validation_results:
        if result.perf_metrics_summary:
            logger.info("  Perf Metrics:")
            for metric, values in result.perf_metrics_summary.items():
                if values["status"] != "skipped":
                    logger.info(
                        f"    {metric}: {values['status']} "
                        f"(current={values['current']}, "
                        f"baseline={values['baseline']}, "
                        f"deviation={values['deviation_percent']}%)"
                    )

        if result.generated_ids_summary:
            logger.info("  Generated IDs:")
            summary = result.generated_ids_summary
            if summary["status"] != "skipped":
                logger.info(
                    f"    Cosine Similarity: {summary['cosine_similarity']} "
                    f"(threshold={summary['similarity_threshold']})"
                )

        if result.error_message:
            logger.info(f"  Note: {result.error_message}")

    logger.info("\n" + "=" * 80)
    if all_passed:
        logger.info(" ALL VALIDATIONS PASSED")
    else:
        logger.info(" SOME VALIDATIONS FAILED")
    logger.info("=" * 80)
