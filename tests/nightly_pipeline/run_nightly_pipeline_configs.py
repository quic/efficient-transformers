# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Multi-Config Nightly Pipeline Runner
Loads causal model names and pipeline configurations from JSON files.
Supports single or multiple pipeline configurations.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

from nightly_causal_lm_pipeline import NightlyPipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MultiConfigPipelineRunner:
    """Runs NightlyPipeline with configurations from JSON file"""

    def __init__(self, pipeline_config_file: str, causal_model_config_file: str):
        """Initialize runner with pipeline and model config files"""
        self.pipeline_config_file = Path(pipeline_config_file)
        self.causal_model_config_file = Path(causal_model_config_file)

        # Load configurations
        self.pipeline_configs = self._load_pipeline_configs()
        self.causal_model_configs = self._load_causal_model_configs()

    def _load_pipeline_configs(self) -> List[Dict]:
        """Load pipeline configurations from JSON file"""
        if not self.pipeline_config_file.exists():
            raise FileNotFoundError(f"Pipeline config file not found: {self.pipeline_config_file}")

        with open(self.pipeline_config_file, "r") as f:
            config_data = json.load(f)

        logger.info(f"Loaded pipeline configuration from: {self.pipeline_config_file}")

        configs = []
        if isinstance(config_data, list):
            configs = config_data
        elif isinstance(config_data, dict):
            if "pipeline_configs" in config_data:
                configs = config_data["pipeline_configs"]
                if not isinstance(configs, list):
                    configs = [configs]
            else:
                logger.error(
                    "Unknown config structure. Expected list, dict, or wrapped in 'pipeline_configs'/'configs' key"
                )
                return []
        else:
            logger.error(f"Unexpected config type: {type(config_data)}")
            return []

        logger.info(f"Loaded {len(configs)} pipeline configuration(s)")
        return configs

    def _load_causal_model_configs(self) -> Dict[str, List[Dict]]:
        """Load causal model configurations from JSON file"""
        if not self.causal_model_config_file.exists():
            raise FileNotFoundError(f"Causal model config file not found: {self.causal_model_config_file}")

        with open(self.causal_model_config_file, "r") as f:
            config_data = json.load(f)

        logger.info(f"Loaded causal model configuration from: {self.causal_model_config_file}")
        return config_data

    def extract_model_names(self, section: str = "causal_lm_models") -> List[str]:
        """Extract model names from a causal model config section"""

        models = self.causal_model_configs[section]
        logger.info(f"Extracted {len(models)} models from '{section}' section")
        return models

    def run_all_configs(self):
        """Run pipeline for all configurations with models from config file"""

        model_names = self.extract_model_names()
        if not model_names:
            logger.error("No models found in section")
            return

        for config in self.pipeline_configs:
            self._run_pipeline_with_config(config, model_names)

        logger.info("=" * 80)
        logger.info("All pipeline configurations completed")
        logger.info("=" * 80)

    def _run_pipeline_with_config(self, config: Dict, model_names: List[str]):
        """Execute pipeline with given configuration and models"""

        config_name = config.get("config_name", "unnamed")
        logger.info("=" * 80)
        logger.info(f"Starting pipeline with config: {config_name}")
        logger.info("=" * 80)

        try:
            # Extract required parameters with defaults
            output_dir = config.get("output_dir", "./nightly_pipeline_outputs")
            num_export_workers = config.get("num_export_workers", 1)
            num_compile_workers = config.get("num_compile_workers", 1)
            baseline_dir = os.environ.get("NIGHTLY_DIR", Path("~/.cache/Nightly_Pipeline").expanduser())

            # Extract parameter dictionaries
            export_params = config.get("export_params", {})
            compile_params = config.get(
                "compile_params", {"prefill_seq_len": 32, "ctx_len": 128, "num_cores": 16, "aic_hw_version": "ai100"}
            )
            generation_params = config.get("generation_params", {"generation_len": 100, "prompt": "My name is"})

            # Create pipeline instance with models and params dictionaries
            pipeline = NightlyPipeline(
                output_dir=output_dir,
                causal_lm_models=model_names,
                num_export_workers=num_export_workers,
                num_compile_workers=num_compile_workers,
                export_params=export_params,
                compile_params=compile_params,
                generation_params=generation_params,
                baseline_dir=baseline_dir,
            )

            # Run the pipeline and get validation result
            validation_passed = pipeline.run()

            if validation_passed:
                logger.info(f"Pipeline SUCCEEDED for config: {config_name}")
            else:
                logger.warning(f"Pipeline completed but VALIDATION FAILED for config: {config_name}")

        except Exception as e:
            logger.error(f"Pipeline failed for config {config_name}: {e}")


def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Run Nightly Causal LM Pipeline with configurations from JSON file")
    parser.add_argument(
        "--pipeline-config-file",
        type=str,
        default=str(script_dir / "configs" / "causal_pipeline_config.json"),
        help="Path to pipeline configurations JSON file",
    )
    parser.add_argument(
        "--validated-model-config-file",
        type=str,
        default=str(script_dir / "configs" / "validated_models.json"),
        help="Path to causal model configs JSON file",
    )

    args = parser.parse_args()

    try:
        runner = MultiConfigPipelineRunner(
            pipeline_config_file=args.pipeline_config_file, causal_model_config_file=args.validated_model_config_file
        )

        runner.run_all_configs()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
