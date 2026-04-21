# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Nightly Causal LM Pipeline: Sequential Export → Parallel Compilation → Sequential Inference
This script orchestrates the three-phase pipeline for all causal language models.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from result_validator import ResultValidator, log_validation_report
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ExportResult:
    """Result of export phase"""

    model_name: str
    status: str  # "success" or "failed"
    onnx_path: Optional[str] = None
    model_object: Optional[object] = None  # Keep model object for reuse
    error_message: Optional[str] = None
    export_time: float = 0.0
    export_params: Optional[Dict] = None


@dataclass
class CompilationResult:
    """Result of compilation phase"""

    model_name: str
    onnx_path: str
    status: str  # "success" or "failed"
    qpc_path: Optional[str] = None
    model_object: Optional[object] = None  # Keep model object for reuse
    error_message: Optional[str] = None
    compile_time: float = 0.0
    compile_params: Optional[Dict] = None


@dataclass
class InferenceResult:
    """Result of inference phase"""

    model_name: str
    qpc_path: str
    status: str  # "success" or "failed"
    batch_size: int = 1  # Batch size of the QPC compilation
    generated_texts: Optional[List[str]] = None  # All generated texts
    generated_ids: Optional[Union[List[np.ndarray], np.ndarray]] = None  # Generated token IDs
    # Performance metrics from CloudAI100ExecInfo
    prefill_time: Optional[float] = None  # Time to first token (TTFT)
    decode_perf: Optional[float] = None  # Decode performance (tokens/sec per token)
    total_perf: Optional[float] = None  # Total performance (tokens/sec)
    total_time: Optional[float] = None  # Total E2E inference time
    error_message: Optional[str] = None
    inference_time: float = 0.0
    generation_params: Optional[Dict] = None


class NightlyPipeline:
    """Orchestrates the three-phase nightly pipeline"""

    def __init__(
        self,
        output_dir: str = "./nightly_pipeline_outputs",
        causal_lm_models: Optional[List[str]] = None,
        num_export_workers: int = 2,
        num_compile_workers: int = 4,
        export_params: Optional[Dict] = None,
        compile_params: Optional[Dict] = None,
        generation_params: Optional[Dict] = None,
        baseline_dir: Optional[str] = None,
    ):
        """Initialize pipeline with configuration

        Supports:
        1. Param dictionaries (recommended): Pass export_params, compile_params, generation_params

        Args:
            baseline_dir: Optional path to baseline inference results for validation
        """
        self.output_dir = Path(output_dir)
        self.CAUSAL_LM_MODELS = causal_lm_models
        self.num_export_workers = num_export_workers
        self.num_compile_workers = num_compile_workers
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_file = self.baseline_dir / "baseline_inference_results.json"
        self.validator = ResultValidator(baseline_file=self.baseline_file)

        self._export_params = export_params or {}

        if compile_params is None:
            self._compile_params = {
                "prefill_seq_len": 32,
                "ctx_len": 128,
                "num_cores": 16,
                "num_devices": 1,
                "aic_hw_version": "ai100",
            }
        else:
            self._compile_params = compile_params

        if generation_params is None:
            self._generation_params = {
                "generation_len": 24,
                "prompt": "My Name is",
            }
        else:
            self._generation_params = generation_params

        self.results_dir = self.output_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Pipeline initialized with output_dir: {self.output_dir}")
        logger.info(f"Models to test: {len(self.CAUSAL_LM_MODELS)}")
        logger.info(f"Export workers: {self.num_export_workers}, Compile workers: {self.num_compile_workers}")

    def _get_export_params(self) -> Dict:
        """Get all export phase parameters"""
        return self._export_params

    def _get_compile_params(self) -> Dict:
        """Get all compilation phase parameters"""
        return self._compile_params

    def _get_generation_params(self) -> Dict:
        """Get all generation/inference phase parameters"""
        return self._generation_params

    def _export_single_model(self, model_name: str) -> ExportResult:
        """
        Export a single model (called in parallel)
        Loads model once and keeps the object for reuse
        """
        export_start = time.time()
        logger.info(f"[EXPORT] Starting export: {model_name}")

        result = ExportResult(model_name=model_name, status="failed")

        try:
            # Load model once
            logger.info("[EXPORT] Loading model from HuggingFace/QEFF...")
            model = QEFFAutoModelForCausalLM.from_pretrained(model_name)

            # Export model
            logger.info(f"[EXPORT] Exporting {model_name} to ONNX...")
            onnx_output_path = model.export(**self._get_export_params())

            result.onnx_path = str(onnx_output_path)
            result.model_object = model  # Keep model object for reuse
            result.export_params = self._get_export_params()
            result.status = "success"
            result.export_time = time.time() - export_start

            logger.info(f"[EXPORT] {model_name} exported in {result.export_time:.2f}s")
            logger.info(f"[EXPORT]  ONNX path: {onnx_output_path}")

        except Exception as e:
            result.error_message = str(e)
            result.export_time = time.time() - export_start
            logger.error(f"[EXPORT] {model_name} failed: {e}")

        return result

    def phase_1_export_parallel(self) -> List[ExportResult]:
        """
        PHASE 1: Parallel Export
        Process causal LM models in parallel using ThreadPoolExecutor
        Loads each model once and keeps it for compilation phase
        """
        logger.info("=" * 80)
        logger.info("PHASE 1: Parallel Export")
        logger.info("=" * 80)

        export_results = []
        start_time = time.time()

        logger.info(f"Starting parallel export with {self.num_export_workers} workers...")

        # Use ThreadPoolExecutor for parallel export
        with ThreadPoolExecutor(max_workers=self.num_export_workers) as executor:
            # Submit all export jobs
            futures = {
                executor.submit(self._export_single_model, model_name): model_name
                for model_name in self.CAUSAL_LM_MODELS
            }

            # Collect results as they complete
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result = future.result()
                    export_results.append(result)
                except Exception as e:
                    logger.error(f"[EXPORT] Unexpected error for {model_name}: {e}")
                    export_results.append(ExportResult(model_name=model_name, status="failed", error_message=str(e)))

        total_export_time = time.time() - start_time
        successful_exports = sum(1 for r in export_results if r.status == "success")
        failed_exports = sum(1 for r in export_results if r.status == "failed")

        logger.info(f"\nPhase 1 Complete: {successful_exports} succeeded, {failed_exports} failed")
        logger.info(f"Total export time: {total_export_time:.2f}s")

        return export_results

    def _compile_single_model(self, export_result: ExportResult) -> CompilationResult:
        """
        Compile a single model (called in parallel)
        Reuses the model object loaded in export phase
        """
        compile_start = time.time()
        logger.info(f"[COMPILE] Starting compilation: {export_result.model_name}")

        result = CompilationResult(
            model_name=export_result.model_name, onnx_path=export_result.onnx_path or "", status="failed"
        )

        try:
            # Reuse model object from export phase (already loaded)
            model = export_result.model_object
            if model is None:
                raise RuntimeError(f"Model object not available for {export_result.model_name}")

            logger.info(
                f"[COMPILE] Compiling {export_result.model_name} with {self._compile_params.get('num_devices', 1)} devices..."
            )
            qpc_path = model.compile(**self._get_compile_params())

            result.qpc_path = qpc_path
            result.model_object = model  # Keep model object for inference phase
            result.compile_params = self._get_compile_params()
            result.status = "success"
            result.compile_time = time.time() - compile_start

            logger.info(f"[COMPILE] {export_result.model_name} compiled in {result.compile_time:.2f}s")
            logger.info(f"[COMPILE]  QPC saved to: {qpc_path}")

        except Exception as e:
            result.error_message = str(e)
            result.compile_time = time.time() - compile_start
            logger.error(f"[COMPILE] {export_result.model_name} failed: {e}")

        return result

    def phase_2_compile_parallel(self, export_results: List[ExportResult]) -> List[CompilationResult]:
        """
        PHASE 2: Parallel Compilation
        Compile all exported models in parallel using ThreadPoolExecutor
        Reuses model objects loaded in export phase
        """
        logger.info("=" * 80)
        logger.info("PHASE 2: Parallel Compilation")
        logger.info("=" * 80)

        # Filter only successful exports
        successful_exports = [r for r in export_results if r.status == "success"]
        if not successful_exports:
            logger.error("No successful exports. Skipping compilation phase.")
            return []

        compilation_results = []
        start_time = time.time()

        logger.info(f"Starting parallel compilation with {self.num_compile_workers} workers...")

        # Use ThreadPoolExecutor for parallel compilation
        with ThreadPoolExecutor(max_workers=self.num_compile_workers) as executor:
            # Submit all compilation jobs, passing export results
            futures = {
                executor.submit(self._compile_single_model, export_result): export_result.model_name
                for export_result in successful_exports
            }

            # Collect results as they complete
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result = future.result()
                    compilation_results.append(result)
                except Exception as e:
                    logger.error(f"[COMPILE] Unexpected error for {model_name}: {e}")
                    compilation_results.append(
                        CompilationResult(model_name=model_name, onnx_path="", status="failed", error_message=str(e))
                    )

        total_compile_time = time.time() - start_time
        successful_compiles = sum(1 for r in compilation_results if r.status == "success")
        failed_compiles = sum(1 for r in compilation_results if r.status == "failed")

        logger.info(f"\nPhase 2 Complete: {successful_compiles} succeeded, {failed_compiles} failed")
        logger.info(f"Total compilation time: {total_compile_time:.2f}s")

        return compilation_results

    def phase_3_inference_sequential(self, compile_results: List[CompilationResult]) -> List[InferenceResult]:
        """
        PHASE 3: Sequential Inference
        Run inference tests one by one after all QPCs are ready
        Reuses model objects loaded in export phase
        """
        logger.info("=" * 80)
        logger.info("PHASE 3: Sequential Inference")
        logger.info("=" * 80)

        # Filter only successful compilations
        successful_compiles = [r for r in compile_results if r.status == "success"]
        if not successful_compiles:
            logger.error("No successful compilations. Skipping inference phase.")
            return []

        inference_results = []
        start_time = time.time()

        logger.info(f"Starting sequential inference on {len(successful_compiles)} models...")

        # Use prompt from generation_params or default
        test_prompt = self._generation_params.get("prompt", "What is artificial intelligence?")
        generation_len = self._generation_params.get("generation_len", 100)

        for compile_result in successful_compiles:
            model_name = compile_result.model_name
            qpc_path = compile_result.qpc_path
            inference_start = time.time()

            logger.info(f"\n[INFERENCE] Processing: {model_name}")

            result = InferenceResult(model_name=model_name, qpc_path=qpc_path, status="failed")

            try:
                logger.info(" Loading tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)

                # Reuse model object from compile phase (already loaded and compiled)
                model = compile_result.model_object
                if model is None:
                    raise RuntimeError(f"Model object not available for {model_name}")

                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                logger.info(" Running inference...")
                exec_info = model.generate(
                    tokenizer=tokenizer,
                    prompts=[test_prompt],
                    generation_len=generation_len,
                )

                result.status = "success"

                # Capture all information from CloudAI100ExecInfo
                result.batch_size = getattr(exec_info, "batch_size", 1)
                result.generated_texts = exec_info.generated_texts if hasattr(exec_info, "generated_texts") else None
                result.generated_ids = getattr(exec_info, "generated_ids", None)
                result.generation_params = self._get_generation_params()

                # Extract performance metrics from perf_metrics
                perf = {}
                if hasattr(exec_info, "perf_metrics") and exec_info.perf_metrics:
                    perf = exec_info.perf_metrics
                result.prefill_time = getattr(perf, "prefill_time", None)
                result.decode_perf = getattr(perf, "decode_perf", None)
                result.total_perf = getattr(perf, "total_perf", None)
                result.total_time = getattr(perf, "total_time", None)

                result.inference_time = time.time() - inference_start

                logger.info(f"  Inference succeeded in {result.inference_time:.2f}s")

            except Exception as e:
                result.error_message = str(e)
                result.inference_time = time.time() - inference_start
                logger.error(f"  Inference failed: {e}")

            inference_results.append(result)

        total_inference_time = time.time() - start_time
        successful_inferences = sum(1 for r in inference_results if r.status == "success")
        failed_inferences = sum(1 for r in inference_results if r.status == "failed")

        logger.info(f"\nPhase 3 Complete: {successful_inferences} succeeded, {failed_inferences} failed")
        logger.info(f"Total inference time: {total_inference_time:.2f}s")

        return inference_results

    def save_results(
        self,
        export_results: List[ExportResult],
        compile_results: List[CompilationResult],
        inference_results: List[InferenceResult],
    ) -> Dict:
        """Save results to JSON files and return results dict"""
        results_file = self.results_dir / "pipeline_results.json"

        # Convert dataclasses to dicts, excluding non-serializable fields (model objects)
        def prepare_results_for_json(results_list, fields_to_exclude=None):
            """Convert results to JSON-serializable format"""
            if fields_to_exclude is None:
                fields_to_exclude = {"model_object"}  # Exclude model objects

            prepared = []
            for r in results_list:
                r_dict = asdict(r)

                # Remove non-serializable fields
                for field in fields_to_exclude:
                    r_dict.pop(field, None)

                # Convert all values to JSON-serializable types
                for key, value in r_dict.items():
                    if isinstance(value, Path):
                        r_dict[key] = str(value)
                    elif isinstance(value, np.ndarray):
                        r_dict[key] = value.tolist()
                    elif isinstance(value, list) and value:
                        # Check if list contains numpy arrays or Path objects
                        converted_list = []
                        for item in value:
                            if isinstance(item, np.ndarray):
                                converted_list.append(item.tolist())
                            elif isinstance(item, Path):
                                converted_list.append(str(item))
                            else:
                                converted_list.append(item)
                        r_dict[key] = converted_list

                prepared.append(r_dict)
            return prepared

        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "exports": prepare_results_for_json(export_results),
            "compilations": prepare_results_for_json(compile_results),
            "inferences": prepare_results_for_json(inference_results),
            "summary": {
                "total_models": len(self.CAUSAL_LM_MODELS),
                "successful_exports": sum(1 for r in export_results if r.status == "success"),
                "successful_compilations": sum(1 for r in compile_results if r.status == "success"),
                "successful_inferences": sum(1 for r in inference_results if r.status == "success"),
            },
        }

        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\nResults saved to: {results_file}")
        return summary

    def run(self) -> bool:
        """Execute the complete three-phase pipeline

        Returns:
            True if validation passed or no baseline provided, False if validation failed
        """
        logger.info("\n" + "=" * 80)
        logger.info("STARTING NIGHTLY CAUSAL LM PIPELINE")
        logger.info("=" * 80 + "\n")

        pipeline_start = time.time()

        # Phase 1: Export in parallel
        export_results = self.phase_1_export_parallel()

        # Phase 2: Compile in parallel
        compile_results = self.phase_2_compile_parallel(export_results)

        # Phase 3: Inference sequentially
        inference_results = self.phase_3_inference_sequential(compile_results)

        # Save all results
        results_summary = self.save_results(export_results, compile_results, inference_results)

        # Validate results against baseline
        validation_results, validation_passed = self.validator.validate_results(
            current_inferences=results_summary["inferences"]
        )

        # Log validation report
        log_validation_report(validation_results, validation_passed)

        # Update baseline if validation passed
        if validation_passed:
            logger.info("\n Validation passed. Updating baseline results...")
            baseline_path = (
                Path(self.baseline_file) if self.baseline_file else self.results_dir / "baseline_inference_results.json"
            )
            self.validator.update_baseline(results_summary, output_file=baseline_path)
        else:
            logger.warning("\n Validation failed. Baseline results NOT updated.")

        total_time = time.time() - pipeline_start
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total execution time: {total_time:.2f}s")

        return validation_passed
